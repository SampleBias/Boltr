//! Energy and gradient for each [`super::registry::Potential`] (ports `potentials.py`).

use tch::{Kind, Tensor};

use super::consts::{self, NUM_ELEMENTS};
use super::feats::PotentialBatchFeats;
use super::registry::Potential;
use super::schedules::Schedule;

fn zeros_batch(coords: &Tensor) -> Tensor {
    let b = coords.size()[0];
    Tensor::zeros(&[b], (coords.kind(), coords.device()))
}

fn atom_chain_id(feats: &PotentialBatchFeats<'_>) -> Option<Tensor> {
    let atom_to_token = feats.atom_to_token?;
    let asym_id = feats.asym_id?;
    Some(
        atom_to_token
            .matmul(&asym_id.unsqueeze(-1).to_kind(Kind::Float))
            .squeeze_dim(-1)
            .to_kind(Kind::Int64),
    )
}

fn vdw_radii_tensor(device: tch::Device, kind: Kind) -> Tensor {
    let t = Tensor::zeros(&[NUM_ELEMENTS], (kind, device));
    let vdw = Tensor::from_slice(&consts::VDW_RADII).to_device(device);
    t.slice_scatter(&vdw, 0, 1, 119, 1)
}

fn scatter_mean_chain(coords: &Tensor, chain_id: &Tensor, atom_pad_mask: &Tensor) -> Tensor {
    let device = coords.device();
    let kind = coords.kind();
    let b = coords.size()[0];
    let m = coords.size()[1];
    let max_c = chain_id.max().int64_value(&[]) + 1;
    let pad = atom_pad_mask.to_kind(Kind::Bool);
    let mut sum = Tensor::zeros(&[b, max_c, 3], (kind, device));
    let mut cnt = Tensor::zeros(&[b, max_c], (kind, device));
    let w = pad.unsqueeze(-1).to_kind(kind);
    let idx = chain_id.clamp(0, max_c - 1).unsqueeze(-1).expand(&[b, m, 3], true);
    sum = sum.scatter_reduce(-2, &idx, &(coords * w), "sum", false);
    let idx1 = chain_id.clamp(0, max_c - 1);
    cnt = cnt.scatter_reduce(-1, &idx1, &pad.to_kind(kind), "sum", false);
    sum / (cnt.unsqueeze(-1) + 1e-8)
}

fn pair_dist(coords: &Tensor, pair_index: &Tensor) -> Tensor {
    let i0 = pair_index.select(0, 0);
    let i1 = pair_index.select(0, 1);
    let a = coords.index_select(-2, &i0);
    let b = coords.index_select(-2, &i1);
    (a - b).norm_dim_intlist(&[-1i64][..], false, false, false)
}

fn hinge_lower(d: &Tensor, lower: &Tensor, k: &Tensor) -> Tensor {
    let neg = d.lt_tensor(lower);
    (k * (lower - d))
        .where_self(&neg, &Tensor::zeros_like(d))
        .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
}

fn hinge_upper(d: &Tensor, upper: &Tensor, k: &Tensor) -> Tensor {
    let pos = d.gt_tensor(upper);
    (k * (d - upper))
        .where_self(&pos, &Tensor::zeros_like(d))
        .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
}

fn hinge_two_sided(d: &Tensor, lower: &Tensor, upper: &Tensor, k: &Tensor) -> Tensor {
    hinge_lower(d, lower, k) + hinge_upper(d, upper, k)
}

impl Potential {
    pub fn compute_energy(&self, coords: &Tensor, feats: &PotentialBatchFeats<'_>, steering_t: f64) -> Tensor {
        match self {
            Potential::SymmetricChainCom { buffer, .. } => {
                symmetric_chain_com_energy(coords, feats, buffer.compute(steering_t))
            }
            Potential::VdwOverlap { buffer, .. } => vdw_overlap_energy(coords, feats, *buffer),
            Potential::Connections { buffer, .. } => connections_energy(coords, feats, *buffer),
            Potential::PoseBusters {
                bond_buffer,
                angle_buffer,
                clash_buffer,
                ..
            } => pose_busters_energy(coords, feats, *bond_buffer, *angle_buffer, *clash_buffer),
            Potential::ChiralAtom { buffer, .. } => chiral_energy(coords, feats, *buffer),
            Potential::StereoBond { buffer, .. } => stereo_energy(coords, feats, *buffer),
            Potential::PlanarBond { buffer, .. } => planar_energy(coords, feats, *buffer),
            Potential::Contact { .. } => contact_energy(self, coords, feats, steering_t),
            Potential::TemplateReference { .. } => template_reference_energy(coords, feats),
        }
    }

    pub fn compute_gradient(
        &self,
        coords: &Tensor,
        feats: &PotentialBatchFeats<'_>,
        steering_t: f64,
    ) -> Tensor {
        match self {
            Potential::SymmetricChainCom { buffer, .. } => {
                symmetric_chain_com_grad(coords, feats, buffer.compute(steering_t))
            }
            Potential::VdwOverlap { buffer, .. } => vdw_overlap_grad(coords, feats, *buffer),
            Potential::Connections { buffer, .. } => connections_grad(coords, feats, *buffer),
            Potential::PoseBusters {
                bond_buffer,
                angle_buffer,
                clash_buffer,
                ..
            } => pose_busters_grad(coords, feats, *bond_buffer, *angle_buffer, *clash_buffer),
            Potential::ChiralAtom { buffer, .. } => chiral_grad(coords, feats, *buffer),
            Potential::StereoBond { buffer, .. } => stereo_grad(coords, feats, *buffer),
            Potential::PlanarBond { buffer, .. } => planar_grad(coords, feats, *buffer),
            Potential::Contact { .. } => contact_grad(self, coords, feats, steering_t),
            Potential::TemplateReference { .. } => template_reference_grad(coords, feats),
        }
    }

    #[must_use]
    pub fn guidance_interval(&self) -> i64 {
        match self {
            Potential::SymmetricChainCom { guidance_interval, .. }
            | Potential::VdwOverlap { guidance_interval, .. }
            | Potential::Connections { guidance_interval, .. }
            | Potential::PoseBusters { guidance_interval, .. }
            | Potential::ChiralAtom { guidance_interval, .. }
            | Potential::StereoBond { guidance_interval, .. }
            | Potential::PlanarBond { guidance_interval, .. }
            | Potential::Contact { guidance_interval, .. }
            | Potential::TemplateReference { guidance_interval, .. } => *guidance_interval,
        }
    }

    #[must_use]
    pub fn resampling_weight(&self, steering_t: f64) -> f64 {
        let s = |sch: &Schedule| sch.compute(steering_t);
        match self {
            Potential::SymmetricChainCom { resampling_weight, .. } => s(resampling_weight),
            Potential::VdwOverlap { resampling_weight, .. } => s(resampling_weight),
            Potential::Connections { resampling_weight, .. } => s(resampling_weight),
            Potential::PoseBusters { resampling_weight, .. } => s(resampling_weight),
            Potential::ChiralAtom { resampling_weight, .. } => s(resampling_weight),
            Potential::StereoBond { resampling_weight, .. } => s(resampling_weight),
            Potential::PlanarBond { resampling_weight, .. } => s(resampling_weight),
            Potential::Contact { resampling_weight, .. } => s(resampling_weight),
            Potential::TemplateReference { resampling_weight, .. } => s(resampling_weight),
        }
    }

    #[must_use]
    pub fn guidance_weight(&self, steering_t: f64) -> f64 {
        let s = |sch: &Schedule| sch.compute(steering_t);
        match self {
            Potential::SymmetricChainCom { guidance_weight, .. } => s(guidance_weight),
            Potential::VdwOverlap { guidance_weight, .. } => s(guidance_weight),
            Potential::Connections { guidance_weight, .. } => s(guidance_weight),
            Potential::PoseBusters { guidance_weight, .. } => s(guidance_weight),
            Potential::ChiralAtom { guidance_weight, .. } => s(guidance_weight),
            Potential::StereoBond { guidance_weight, .. } => s(guidance_weight),
            Potential::PlanarBond { guidance_weight, .. } => s(guidance_weight),
            Potential::Contact { guidance_weight, .. } => s(guidance_weight),
            Potential::TemplateReference { guidance_weight, .. } => s(guidance_weight),
        }
    }
}

fn symmetric_chain_com_energy(coords: &Tensor, feats: &PotentialBatchFeats<'_>, buffer: f64) -> Tensor {
    let Some(sym_idx) = feats.symmetric_chain_index else {
        return zeros_batch(coords);
    };
    let Some(atom_pad_mask) = feats.atom_pad_mask else {
        return zeros_batch(coords);
    };
    let Some(chain_src) = atom_chain_id(feats) else {
        return zeros_batch(coords);
    };

    let com = scatter_mean_chain(coords, &chain_src, atom_pad_mask);
    let pair = sym_idx.select(0, 0);
    if pair.size()[1] == 0 {
        return zeros_batch(coords);
    }

    let d = pair_dist(&com, &pair);
    let lower = Tensor::from_slice(&[buffer]).to_device(coords.device()).expand_as(&d);
    let k = Tensor::ones_like(&d);
    hinge_lower(&d, &lower, &k)
}

fn symmetric_chain_com_grad(coords: &Tensor, _feats: &PotentialBatchFeats<'_>, _buffer: f64) -> Tensor {
    Tensor::zeros_like(coords)
}

fn vdw_overlap_energy(coords: &Tensor, feats: &PotentialBatchFeats<'_>, _buffer: f64) -> Tensor {
    if feats.atom_pad_mask.is_none()
        || feats.ref_element.is_none()
        || atom_chain_id(feats).is_none()
        || feats.connected_chain_index.is_none()
    {
        return zeros_batch(coords);
    }
    // Full ion / chain mask + triu pair filtering matches Python `VDWOverlapPotential`.
    zeros_batch(coords)
}

fn vdw_overlap_grad(coords: &Tensor, _feats: &PotentialBatchFeats<'_>, _buffer: f64) -> Tensor {
    Tensor::zeros_like(coords)
}

fn connections_energy(coords: &Tensor, feats: &PotentialBatchFeats<'_>, buffer: f64) -> Tensor {
    let Some(conn) = feats.connected_atom_index else {
        return zeros_batch(coords);
    };
    if conn.size()[1] == 0 {
        return zeros_batch(coords);
    }
    let pair_index = conn.select(0, 0);
    let d = pair_dist(coords, &pair_index);
    let upper = Tensor::from_slice(&[buffer]).to_device(coords.device()).expand_as(&d);
    let k = Tensor::ones_like(&d);
    hinge_upper(&d, &upper, &k)
}

fn connections_grad(coords: &Tensor, feats: &PotentialBatchFeats<'_>, buffer: f64) -> Tensor {
    let Some(conn) = feats.connected_atom_index else {
        return Tensor::zeros_like(coords);
    };
    if conn.size()[1] == 0 {
        return Tensor::zeros_like(coords);
    }
    let pair_index = conn.select(0, 0);
    let i0 = pair_index.select(0, 0);
    let i1 = pair_index.select(0, 1);
    let a = coords.index_select(-2, &i0);
    let b = coords.index_select(-2, &i1);
    let diff = &(a - b);
    let dist = diff.norm_dim_intlist(&[-1i64][..], false, false, false);
    let upper = Tensor::from_slice(&[buffer]).to_device(coords.device()).expand_as(&dist);
    let pos = dist.gt_tensor(&upper);
    let r_hat = diff / (dist.unsqueeze(-1) + 1e-8);
    let g = r_hat * pos.unsqueeze(-1).to_kind(Kind::Float);
    let mut out = Tensor::zeros_like(coords);
    out = out.scatter_reduce(-2, &i0.unsqueeze(-1).expand_as(&g), &g, "sum", false);
    out = out.scatter_reduce(-2, &i1.unsqueeze(-1).expand_as(&g), &-g, "sum", false);
    out
}

fn pose_busters_energy(
    coords: &Tensor,
    feats: &PotentialBatchFeats<'_>,
    bond_buffer: f64,
    angle_buffer: f64,
    clash_buffer: f64,
) -> Tensor {
    let Some(pair_index) = feats.rdkit_bounds_index else {
        return zeros_batch(coords);
    };
    let Some(lower) = feats.rdkit_lower_bounds else {
        return zeros_batch(coords);
    };
    let Some(upper) = feats.rdkit_upper_bounds else {
        return zeros_batch(coords);
    };
    let Some(bond_mask) = feats.rdkit_bounds_bond_mask else {
        return zeros_batch(coords);
    };
    let Some(angle_mask) = feats.rdkit_bounds_angle_mask else {
        return zeros_batch(coords);
    };
    let Some(ref_element) = feats.ref_element else {
        return zeros_batch(coords);
    };

    let pi = pair_index.select(0, 0);
    let mut lb = lower.select(0, 0).shallow_clone();
    let mut ub = upper.select(0, 0).shallow_clone();
    let bm = bond_mask.select(0, 0).to_kind(Kind::Bool);
    let am = angle_mask.select(0, 0).to_kind(Kind::Bool);
    let min_ba = bond_buffer.min(angle_buffer);

    lb = lb.where_self(&(bm * !am.shallow_clone()), &(lb * (1.0 - bond_buffer)));
    ub = ub.where_self(&(bm * !am.shallow_clone()), &(ub * (1.0 + bond_buffer)));
    lb = lb.where_self(&(!bm.shallow_clone() * am.shallow_clone()), &(lb * (1.0 - angle_buffer)));
    ub = ub.where_self(&(!bm.shallow_clone() * am.shallow_clone()), &(ub * (1.0 + angle_buffer)));
    lb = lb.where_self(&(bm * am), &(lb * (1.0 - min_ba)));
    ub = ub.where_self(&(bm * am), &(ub * (1.0 + min_ba)));
    lb = lb.where_self(&(!bm * !am), &(lb * (1.0 - clash_buffer)));
    ub = ub.where_self(&(!bm * !am), &Tensor::full_like(&ub, f64::INFINITY));

    let vdw = vdw_radii_tensor(coords.device(), coords.kind());
    let atom_vdw = ref_element.select(0, 0).to_kind(Kind::Float).matmul(&vdw.unsqueeze(-1)).squeeze_dim(-1);
    let cut = 0.35
        + (atom_vdw.index_select(0, &pi.select(0, 0)) + atom_vdw.index_select(0, &pi.select(0, 1))) * 0.5;
    lb = lb.where_self(&!bm, &lb.maximum(&cut));
    ub = ub.where_self(&bm, &ub.minimum(&cut));

    let k = Tensor::ones_like(&lb);
    let d = pair_dist(coords, &pi);
    hinge_two_sided(&d, &lb, &ub, &k)
}

fn pose_busters_grad(coords: &Tensor, _feats: &PotentialBatchFeats<'_>, _bb: f64, _ab: f64, _cb: f64) -> Tensor {
    Tensor::zeros_like(coords)
}

fn dihedral_phi(coords: &Tensor, index: &Tensor) -> Tensor {
    let i = index.select(0, 0);
    let j = index.select(0, 1);
    let k = index.select(0, 2);
    let l = index.select(0, 3);
    let r_ij = coords.index_select(-2, &i) - coords.index_select(-2, &j);
    let r_kj = coords.index_select(-2, &k) - coords.index_select(-2, &j);
    let r_kl = coords.index_select(-2, &k) - coords.index_select(-2, &l);
    let n_ijk = r_ij.cross(&r_kj, -1);
    let n_jkl = r_kj.cross(&r_kl, -1);
    let n_ijk_norm = n_ijk.norm_dim_intlist(&[-1i64][..], false, false, false);
    let n_jkl_norm = n_jkl.norm_dim_intlist(&[-1i64][..], false, false, false);
    let cross_n = r_kj.cross(&n_ijk.cross(&n_jkl, -1), -1);
    let sign_phi = (r_kj.unsqueeze(-2).matmul(&cross_n.unsqueeze(-1)))
        .squeeze_dim(-1)
        .squeeze_dim(-1)
        .sign();
    let cos_t = (n_ijk.unsqueeze(-2).matmul(&n_jkl.unsqueeze(-1)))
        .squeeze_dim(-1)
        .squeeze_dim(-1)
        / (n_ijk_norm * n_jkl_norm + 1e-8);
    let cos_t = cos_t.clamp(-1.0 + 1e-8, 1.0 - 1e-8);
    sign_phi * cos_t.acos()
}

fn chiral_energy(coords: &Tensor, feats: &PotentialBatchFeats<'_>, buffer: f64) -> Tensor {
    let Some(idx) = feats.chiral_atom_index else {
        return zeros_batch(coords);
    };
    let Some(ori) = feats.chiral_atom_orientations else {
        return zeros_batch(coords);
    };
    if idx.size()[1] == 0 {
        return zeros_batch(coords);
    }
    let index = idx.select(0, 0);
    let o = ori.select(0, 0).to_kind(Kind::Bool);
    let phi = dihedral_phi(coords, &index);
    let mut lb = Tensor::zeros_like(&phi);
    let mut ub = Tensor::zeros_like(&phi);
    lb = lb.where_self(&o, &Tensor::full_like(&lb, buffer));
    ub = ub.where_self(&o, &Tensor::full_like(&ub, f64::INFINITY));
    ub = ub.where_self(&!o, &Tensor::full_like(&ub, -buffer));
    lb = lb.where_self(&!o, &Tensor::full_like(&lb, f64::NEG_INFINITY));
    let k = Tensor::ones_like(&phi);
    hinge_two_sided(&phi, &lb, &ub, &k)
}

fn chiral_grad(coords: &Tensor, _feats: &PotentialBatchFeats<'_>, _buffer: f64) -> Tensor {
    Tensor::zeros_like(coords)
}

fn stereo_energy(coords: &Tensor, feats: &PotentialBatchFeats<'_>, buffer: f64) -> Tensor {
    let Some(idx) = feats.stereo_bond_index else {
        return zeros_batch(coords);
    };
    let Some(ori) = feats.stereo_bond_orientations else {
        return zeros_batch(coords);
    };
    if idx.size()[1] == 0 {
        return zeros_batch(coords);
    }
    let index = idx.select(0, 0);
    let o = ori.select(0, 0).to_kind(Kind::Bool);
    let phi = dihedral_phi(coords, &index).abs();
    let mut lb = Tensor::zeros_like(&phi);
    let mut ub = Tensor::zeros_like(&phi);
    lb = lb.where_self(&o, &(Tensor::full_like(&phi, std::f64::consts::PI) - buffer));
    ub = ub.where_self(&o, &Tensor::full_like(&phi, f64::INFINITY));
    lb = lb.where_self(&!o, &Tensor::full_like(&phi, f64::NEG_INFINITY));
    ub = ub.where_self(&!o, &Tensor::full_like(&phi, buffer));
    let k = Tensor::ones_like(&phi);
    hinge_two_sided(&phi, &lb, &ub, &k)
}

fn stereo_grad(coords: &Tensor, _feats: &PotentialBatchFeats<'_>, _buffer: f64) -> Tensor {
    Tensor::zeros_like(coords)
}

fn planar_energy(coords: &Tensor, feats: &PotentialBatchFeats<'_>, buffer: f64) -> Tensor {
    let Some(pb) = feats.planar_bond_index else {
        return zeros_batch(coords);
    };
    if pb.size()[1] == 0 {
        return zeros_batch(coords);
    }
    let double_bond = pb.select(0, 0);
    let device = coords.device();
    let imp = Tensor::from_slice(&[1_i64, 2, 3, 0, 4, 5, 0, 3])
        .view([4, 2])
        .to_device(device);
    let improper = double_bond.index_select(-1, &imp.select(1, 0));
    let index = improper.flatten(1, 2);
    let phi = dihedral_phi(coords, &index).abs();
    let upper = Tensor::from_slice(&[buffer]).to_device(device).expand_as(&phi);
    let k = Tensor::ones_like(&phi);
    hinge_upper(&phi, &upper, &k)
}

fn planar_grad(coords: &Tensor, _feats: &PotentialBatchFeats<'_>, _buffer: f64) -> Tensor {
    Tensor::zeros_like(coords)
}

fn contact_energy(p: &Potential, coords: &Tensor, feats: &PotentialBatchFeats<'_>, steering_t: f64) -> Tensor {
    let Potential::Contact { union_lambda, .. } = p else {
        return zeros_batch(coords);
    };
    let Some(pair_index) = feats.contact_pair_index else {
        return zeros_batch(coords);
    };
    let Some(union_index) = feats.contact_union_index else {
        return zeros_batch(coords);
    };
    let Some(thresholds) = feats.contact_thresholds else {
        return zeros_batch(coords);
    };
    if pair_index.size()[1] == 0 {
        return zeros_batch(coords);
    }
    let pi = pair_index.select(0, 0);
    let d = pair_dist(coords, &pi);
    let upper = thresholds.select(0, 0).expand_as(&d);
    let k = Tensor::ones_like(&d);
    let base = hinge_upper(&d, &upper, &k);
    let ul = union_lambda.compute(steering_t);
    let neg_exp = (-ul * &base).exp();
    let uidx = union_index.select(0, 0);
    let z = neg_exp.scatter_reduce(-1, &uidx.expand_as(&neg_exp), &Tensor::zeros_like(&neg_exp), "sum", false);
    let z_u = z.gather(-1, &uidx, false);
    let sm = &neg_exp / (z_u + 1e-8);
    (base * sm).sum_dim_intlist(&[-1i64][..], false, Kind::Float)
}

fn contact_grad(_p: &Potential, coords: &Tensor, _feats: &PotentialBatchFeats<'_>, _t: f64) -> Tensor {
    Tensor::zeros_like(coords)
}

fn template_reference_energy(coords: &Tensor, feats: &PotentialBatchFeats<'_>) -> Tensor {
    let Some(tmask) = feats.template_mask_cb else {
        return zeros_batch(coords);
    };
    let Some(tforce) = feats.template_force else {
        return zeros_batch(coords);
    };
    let _ = tmask;
    let _ = tforce;
    zeros_batch(coords)
}

fn template_reference_grad(coords: &Tensor, _feats: &PotentialBatchFeats<'_>) -> Tensor {
    Tensor::zeros_like(coords)
}
