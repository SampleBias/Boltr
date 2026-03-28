//! Port of Boltz `process_token_features` (`data/feature/featurizerv2.py`) for **inference-style** batches.
//!
//! Omits training-only random pocket/contact augmentation. Contact maps start as `UNSELECTED`, then
//! collapse to `UNSPECIFIED` when uniformly unselected (matches Python before one-hot).

use ndarray::{Array1, Array2, Array3};

use crate::boltz_const::{contact_conditioning_id, method_type_id, NUM_TOKENS};
use crate::feature_batch::FeatureBatch;
use crate::pad::pad_1d;
use crate::tokenize::boltz2::{TokenBondV2, TokenData};

/// `len(const.contact_conditioning_info)` in Boltz / [`CONTACT_CONDITIONING_CHANNELS`](boltr_backend_tch::CONTACT_CONDITIONING_CHANNELS).
pub const CONTACT_CONDITIONING_NUM_CLASSES: usize = 5;

fn unselected_contact_id() -> i64 {
    i64::from(contact_conditioning_id("UNSELECTED").expect("UNSELECTED"))
}

fn unspecified_contact_id() -> i64 {
    i64::from(contact_conditioning_id("UNSPECIFIED").expect("UNSPECIFIED"))
}

/// Token-level tensors for one example (no batch axis), aligned with Python `token_features` dict.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenFeatureTensors {
    pub token_index: Array1<i64>,
    pub residue_index: Array1<i64>,
    pub asym_id: Array1<i64>,
    pub entity_id: Array1<i64>,
    pub sym_id: Array1<i64>,
    pub mol_type: Array1<i64>,
    /// One-hot `res_type`, shape `[N, NUM_TOKENS]` float32.
    pub res_type: Array2<f32>,
    pub disto_center: Array2<f32>,
    /// `[N, N, 1]` float32.
    pub token_bonds: Array3<f32>,
    /// `[N, N]` int64 — same encoding as tokenizer `TokenBondV2` type field.
    pub type_bonds: Array2<i64>,
    pub token_pad_mask: Array1<f32>,
    pub token_resolved_mask: Array1<f32>,
    pub token_disto_mask: Array1<f32>,
    /// One-hot pairwise contact conditioning, `[N, N, CONTACT_CONDITIONING_NUM_CLASSES]`.
    pub contact_conditioning: Array3<f32>,
    pub contact_threshold: Array2<f32>,
    pub method_feature: Array1<i64>,
    pub modified: Array1<i64>,
    pub cyclic_period: Array1<f32>,
    pub affinity_token_mask: Array1<f32>,
}

impl TokenFeatureTensors {
    /// Pack into [`FeatureBatch`] with keys matching Python `process_token_features` return dict.
    #[must_use]
    pub fn to_feature_batch(&self) -> FeatureBatch {
        let mut b = FeatureBatch::new();
        b.insert_i64("token_index", self.token_index.clone().into_dyn());
        b.insert_i64("residue_index", self.residue_index.clone().into_dyn());
        b.insert_i64("asym_id", self.asym_id.clone().into_dyn());
        b.insert_i64("entity_id", self.entity_id.clone().into_dyn());
        b.insert_i64("sym_id", self.sym_id.clone().into_dyn());
        b.insert_i64("mol_type", self.mol_type.clone().into_dyn());
        b.insert_f32("res_type", self.res_type.clone().into_dyn());
        b.insert_f32("disto_center", self.disto_center.clone().into_dyn());
        b.insert_f32("token_bonds", self.token_bonds.clone().into_dyn());
        b.insert_i64("type_bonds", self.type_bonds.clone().into_dyn());
        b.insert_f32("token_pad_mask", self.token_pad_mask.clone().into_dyn());
        b.insert_f32(
            "token_resolved_mask",
            self.token_resolved_mask.clone().into_dyn(),
        );
        b.insert_f32("token_disto_mask", self.token_disto_mask.clone().into_dyn());
        b.insert_f32(
            "contact_conditioning",
            self.contact_conditioning.clone().into_dyn(),
        );
        b.insert_f32(
            "contact_threshold",
            self.contact_threshold.clone().into_dyn(),
        );
        b.insert_i64("method_feature", self.method_feature.clone().into_dyn());
        b.insert_i64("modified", self.modified.clone().into_dyn());
        b.insert_f32("cyclic_period", self.cyclic_period.clone().into_dyn());
        b.insert_f32(
            "affinity_token_mask",
            self.affinity_token_mask.clone().into_dyn(),
        );
        b
    }
}

fn one_hot_pairwise(ids: &Array2<i64>, num_classes: usize) -> Array3<f32> {
    let (n0, n1) = ids.dim();
    let mut out = Array3::zeros((n0, n1, num_classes));
    for i in 0..n0 {
        for j in 0..n1 {
            let v = ids[[i, j]];
            if v >= 0 && (v as usize) < num_classes {
                out[[i, j, v as usize]] = 1.0;
            }
        }
    }
    out
}

/// Build token features from tokenizer output.
///
/// * `max_tokens`: if `Some(M)` with `M >= n`, right-pad the token axis to length `M` (Python `pad_to_max_tokens`).
pub fn process_token_features(
    tokens: &[TokenData],
    bonds: &[TokenBondV2],
    max_tokens: Option<usize>,
) -> TokenFeatureTensors {
    let n = tokens.len();
    let m = max_tokens.map_or(n, |cap| cap.max(n));

    let tok_to_idx: std::collections::HashMap<i32, usize> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.token_idx, i))
        .collect();

    let mut token_bonds = Array3::zeros((m, m, 1));
    let mut type_bonds = Array2::zeros((m, m));
    for &(t1, t2, ty) in bonds {
        let Some(&i) = tok_to_idx.get(&t1) else {
            continue;
        };
        let Some(&j) = tok_to_idx.get(&t2) else {
            continue;
        };
        if i < n && j < n {
            token_bonds[[i, j, 0]] = 1.0;
            token_bonds[[j, i, 0]] = 1.0;
            let t = i64::from(ty);
            type_bonds[[i, j]] = t;
            type_bonds[[j, i]] = t;
        }
    }

    let mut contact_ids = Array2::from_elem((m, m), unselected_contact_id());
    if contact_ids.iter().all(|&v| v == unselected_contact_id()) {
        contact_ids.fill(unspecified_contact_id());
    }
    let contact_conditioning = one_hot_pairwise(&contact_ids, CONTACT_CONDITIONING_NUM_CLASSES);

    let residue_index: Vec<i64> = tokens.iter().map(|t| i64::from(t.res_idx)).collect();
    let asym_id: Vec<i64> = tokens.iter().map(|t| i64::from(t.asym_id)).collect();
    let entity_id: Vec<i64> = tokens.iter().map(|t| i64::from(t.entity_id)).collect();
    let sym_id: Vec<i64> = tokens.iter().map(|t| i64::from(t.sym_id)).collect();
    let mol_type: Vec<i64> = tokens.iter().map(|t| i64::from(t.mol_type)).collect();

    let mut res_type = Array2::zeros((m, NUM_TOKENS));
    for (i, t) in tokens.iter().enumerate().take(n) {
        let c = t.res_type as usize;
        if c < NUM_TOKENS {
            res_type[[i, c]] = 1.0;
        }
    }

    let mut disto_center = Array2::zeros((m, 3));
    for (i, t) in tokens.iter().enumerate().take(n) {
        disto_center[[i, 0]] = t.disto_coords[0];
        disto_center[[i, 1]] = t.disto_coords[1];
        disto_center[[i, 2]] = t.disto_coords[2];
    }

    let token_pad_mask_f: Vec<f32> = (0..m).map(|i| if i < n { 1.0 } else { 0.0 }).collect();
    let token_resolved: Vec<f32> = pad_1d(
        &tokens
            .iter()
            .map(|t| f32::from(t.resolved_mask))
            .collect::<Vec<_>>(),
        m,
        0.0,
    );
    let token_disto: Vec<f32> = pad_1d(
        &tokens
            .iter()
            .map(|t| f32::from(t.disto_mask))
            .collect::<Vec<_>>(),
        m,
        0.0,
    );

    let method_id = i64::from(method_type_id("x-ray diffraction"));
    let method_feature: Vec<i64> = vec![method_id; m];

    let modified: Vec<i64> = pad_1d(
        &tokens
            .iter()
            .map(|t| i64::from(u8::from(t.modified)))
            .collect::<Vec<_>>(),
        m,
        0,
    );

    let cyclic: Vec<f32> = pad_1d(
        &tokens
            .iter()
            .map(|t| t.cyclic_period as f32)
            .collect::<Vec<_>>(),
        m,
        0.0,
    );

    let affinity: Vec<f32> = pad_1d(
        &tokens
            .iter()
            .map(|t| if t.affinity_mask { 1.0 } else { 0.0 })
            .collect::<Vec<_>>(),
        m,
        0.0,
    );

    let token_index: Vec<i64> = (0..n as i64).collect();
    let contact_threshold = Array2::zeros((m, m));

    TokenFeatureTensors {
        token_index: Array1::from(pad_1d(&token_index, m, 0)),
        residue_index: Array1::from(pad_1d(&residue_index, m, 0)),
        asym_id: Array1::from(pad_1d(&asym_id, m, 0)),
        entity_id: Array1::from(pad_1d(&entity_id, m, 0)),
        sym_id: Array1::from(pad_1d(&sym_id, m, 0)),
        mol_type: Array1::from(pad_1d(&mol_type, m, 0)),
        res_type,
        disto_center,
        token_bonds,
        type_bonds,
        token_pad_mask: Array1::from(token_pad_mask_f),
        token_resolved_mask: Array1::from(token_resolved),
        token_disto_mask: Array1::from(token_disto),
        contact_conditioning,
        contact_threshold,
        method_feature: Array1::from(method_feature),
        modified: Array1::from(modified),
        cyclic_period: Array1::from(cyclic),
        affinity_token_mask: Array1::from(affinity),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::structure_v2_single_ala;
    use crate::tokenize::boltz2::tokenize_structure;

    #[test]
    fn ala_shapes_match_expected() {
        let s = structure_v2_single_ala();
        let (tokens, bonds) = tokenize_structure(&s, None);
        let f = process_token_features(&tokens, &bonds, None);
        assert_eq!(f.token_index.len(), 1);
        assert_eq!(f.res_type.shape(), [1, NUM_TOKENS]);
        assert_eq!(f.disto_center.shape(), [1, 3]);
        assert_eq!(f.token_bonds.shape(), [1, 1, 1]);
        assert_eq!(f.type_bonds.shape(), [1, 1]);
        assert_eq!(
            f.contact_conditioning.shape(),
            [1, 1, CONTACT_CONDITIONING_NUM_CLASSES]
        );
        assert_eq!(f.contact_threshold.shape(), [1, 1]);
        // Uniform UNSPECIFIED path → one-hot channel 0
        assert!((f.contact_conditioning[[0, 0, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn padding_increases_token_axis() {
        let s = structure_v2_single_ala();
        let (tokens, bonds) = tokenize_structure(&s, None);
        let f = process_token_features(&tokens, &bonds, Some(8));
        assert_eq!(f.token_pad_mask.len(), 8);
        assert_eq!(f.token_pad_mask[0], 1.0);
        assert_eq!(f.token_pad_mask[7], 0.0);
        assert_eq!(f.res_type.shape(), [8, NUM_TOKENS]);
        assert_eq!(f.token_bonds.shape(), [8, 8, 1]);
    }

    #[test]
    fn ligand_bond_fills_pair_matrices() {
        use crate::boltz_const::chain_type_id;
        use crate::structure_v2::{
            AtomV2Row, BondV2AtomRow, ChainRow, EnsembleRow, ResidueRow, StructureV2Tables,
        };

        let np = chain_type_id("NONPOLYMER").expect("NONPOLYMER") as i8;
        let coords = vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let atoms: Vec<_> = coords
            .iter()
            .map(|&c| AtomV2Row {
                name: String::new(),
                coords: c,
                is_present: true,
                bfactor: 0.0,
                plddt: 0.0,
            })
            .collect();
        let s = StructureV2Tables {
            atoms,
            residues: vec![ResidueRow {
                name: "LIG".to_string(),
                res_type: 0,
                res_idx: 0,
                atom_idx: 0,
                atom_num: 2,
                atom_center: 0,
                atom_disto: 1,
                is_standard: false,
                is_present: true,
            }],
            chains: vec![ChainRow {
                name: String::new(),
                mol_type: np,
                sym_id: 0,
                asym_id: 0,
                entity_id: 0,
                atom_idx: 0,
                atom_num: 2,
                res_idx: 0,
                res_num: 1,
                cyclic_period: 0,
            }],
            chain_mask: vec![true],
            coords: coords.clone(),
            ensemble: vec![EnsembleRow {
                atom_coord_idx: 0,
                atom_num: 2,
            }],
            ensemble_atom_coord_idx: 0,
            bonds: vec![BondV2AtomRow {
                atom_1: 0,
                atom_2: 1,
                bond_type: 0,
            }],
        };
        let (tokens, bonds) = tokenize_structure(&s, None);
        assert_eq!(tokens.len(), 2);
        let f = process_token_features(&tokens, &bonds, None);
        assert_eq!(f.token_bonds[[0, 1, 0]], 1.0);
        assert_eq!(f.token_bonds[[1, 0, 0]], 1.0);
        assert_eq!(f.type_bonds[[0, 1]], 1);
    }
}
