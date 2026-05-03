use std::collections::HashSet;

use crate::qc::geometry::{add, distance, normalize, scale, sub};
use crate::qc::protein::{heavy_atoms, protein_chains};
use crate::qc::report::QcThresholds;
use crate::structure_v2::StructureV2Tables;

#[derive(Debug, Clone, Copy)]
struct DistanceRestraint {
    a: usize,
    b: usize,
    target: f32,
    weight: f32,
}

#[derive(Debug, Clone)]
pub struct RelaxationOutcome {
    pub attempted: bool,
    pub iterations: usize,
    pub max_displacement: f32,
    pub message: String,
}

fn restraints(structure: &StructureV2Tables) -> Vec<DistanceRestraint> {
    let mut out = Vec::new();
    for chain in protein_chains(structure) {
        for residue in &chain.residues {
            if !residue.is_standard_protein {
                continue;
            }
            for (a_name, b_name, target, weight) in [
                ("N", "CA", 1.46, 1.0),
                ("CA", "C", 1.53, 1.0),
                ("C", "O", 1.23, 0.8),
            ] {
                if let (Some(a), Some(b)) = (residue.atoms.get(a_name), residue.atoms.get(b_name)) {
                    out.push(DistanceRestraint {
                        a: a.index,
                        b: b.index,
                        target,
                        weight,
                    });
                }
            }
        }
        for pair in chain.residues.windows(2) {
            let a = &pair[0];
            let b = &pair[1];
            if !a.is_standard_protein || !b.is_standard_protein {
                continue;
            }
            if let (Some(c), Some(n)) = (a.atoms.get("C"), b.atoms.get("N")) {
                out.push(DistanceRestraint {
                    a: c.index,
                    b: n.index,
                    target: 1.33,
                    weight: 1.2,
                });
            }
            if let (Some(ca_a), Some(ca_b)) = (a.atoms.get("CA"), b.atoms.get("CA")) {
                out.push(DistanceRestraint {
                    a: ca_a.index,
                    b: ca_b.index,
                    target: 3.8,
                    weight: 0.6,
                });
            }
        }
    }
    out
}

fn bonded_pairs(
    restraints: &[DistanceRestraint],
    structure: &StructureV2Tables,
) -> HashSet<(usize, usize)> {
    let mut out = HashSet::new();
    for r in restraints {
        out.insert((r.a.min(r.b), r.a.max(r.b)));
    }
    for b in &structure.bonds {
        let (Ok(a), Ok(c)) = (usize::try_from(b.atom_1), usize::try_from(b.atom_2)) else {
            continue;
        };
        out.insert((a.min(c), a.max(c)));
    }
    out
}

fn sync_flat_coords(structure: &mut StructureV2Tables) {
    let base = structure.ensemble_base_offset(0);
    if base < 0 {
        return;
    }
    let base = base as usize;
    for i in 0..structure.atoms.len() {
        let j = base + i;
        if j < structure.coords.len() {
            structure.coords[j] = structure.atoms[i].coords;
        }
    }
}

/// Relax coordinates with deterministic bonded-distance restraints and hard-overlap repulsion.
///
/// This does not add missing atoms or change chain/residue metadata. It is intentionally modest:
/// if validation failures are topological rather than coordinate-level, revalidation will still fail.
pub fn relax_structure(
    structure: &mut StructureV2Tables,
    thresholds: QcThresholds,
) -> RelaxationOutcome {
    let rs = restraints(structure);
    if rs.is_empty() {
        return RelaxationOutcome {
            attempted: false,
            iterations: 0,
            max_displacement: 0.0,
            message: "no relaxable protein restraints".to_string(),
        };
    }
    let bonded = bonded_pairs(&rs, structure);
    let initial: Vec<_> = structure.atoms.iter().map(|a| a.coords).collect();
    let iterations = 300;
    let step = 0.045_f32;
    let clash_target = thresholds.hard_overlap.max(1.05);

    for _ in 0..iterations {
        let mut delta = vec![[0.0_f32; 3]; structure.atoms.len()];
        for r in &rs {
            if r.a >= structure.atoms.len() || r.b >= structure.atoms.len() {
                continue;
            }
            let a = structure.atoms[r.a].coords;
            let b = structure.atoms[r.b].coords;
            let ab = sub(b, a);
            let Some(dir) = normalize(ab) else {
                delta[r.a] = add(delta[r.a], [-0.01, 0.0, 0.0]);
                delta[r.b] = add(delta[r.b], [0.01, 0.0, 0.0]);
                continue;
            };
            let d = distance(a, b);
            let err = d - r.target;
            let corr = scale(dir, 0.5 * step * r.weight * err.clamp(-2.0, 2.0));
            delta[r.a] = add(delta[r.a], corr);
            delta[r.b] = sub(delta[r.b], corr);
        }

        let heavy = heavy_atoms(structure);
        for i in 0..heavy.len() {
            for j in (i + 1)..heavy.len() {
                let a = &heavy[i];
                let b = &heavy[j];
                let key = (a.index.min(b.index), a.index.max(b.index));
                if bonded.contains(&key) {
                    continue;
                }
                let ab = sub(b.coords, a.coords);
                let d = distance(a.coords, b.coords);
                if d >= clash_target {
                    continue;
                }
                let dir = normalize(ab).unwrap_or([1.0, 0.0, 0.0]);
                let corr = scale(dir, 0.5 * step * (clash_target - d).clamp(0.0, 1.0));
                delta[a.index] = sub(delta[a.index], corr);
                delta[b.index] = add(delta[b.index], corr);
            }
        }

        for (atom, d) in structure.atoms.iter_mut().zip(delta.iter()) {
            if atom.is_present {
                atom.coords = add(atom.coords, *d);
            }
        }
    }
    sync_flat_coords(structure);

    let mut max_displacement = 0.0_f32;
    for (atom, start) in structure.atoms.iter().zip(initial.iter()) {
        if atom.is_present {
            max_displacement = max_displacement.max(distance(atom.coords, *start));
        }
    }

    RelaxationOutcome {
        attempted: true,
        iterations,
        max_displacement,
        message: "coordinate minimization completed".to_string(),
    }
}
