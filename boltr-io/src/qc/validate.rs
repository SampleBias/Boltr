use std::collections::HashSet;

use crate::qc::geometry::{dihedral_degrees, distance, radius_of_gyration};
use crate::qc::protein::{display_chain_name, heavy_atoms, protein_chains};
use crate::qc::report::{
    BackboneBondDistance, ChainBreak, MissingBackboneAtom, PeptideBondDistance, QcReport,
    QcThresholds, StericClash, TorsionMeasure,
};
use crate::structure_v2::StructureV2Tables;

fn in_range(v: f32, min: f32, max: f32) -> bool {
    v.is_finite() && v >= min && v <= max
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boltz_const::{chain_type_id, token_id};
    use crate::qc::relax_structure;
    use crate::structure_v2::{AtomV2Row, ChainRow, EnsembleRow, ResidueRow, StructureV2Tables};

    fn two_ala_structure() -> StructureV2Tables {
        let protein = chain_type_id("PROTEIN").expect("protein") as i8;
        let ala = token_id("ALA").expect("ALA") as i8;
        let names = ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O", "CB"];
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.46, 0.0, 0.0],
            [2.36, 1.24, 0.0],
            [2.20, 2.46, 0.0],
            [1.46, -1.50, 0.0],
            [3.66, 1.52, 0.0],
            [3.85, 2.95, 0.20],
            [4.75, 4.19, 0.20],
            [4.59, 5.41, 0.20],
            [3.85, 2.95, 1.70],
        ];
        let atoms = names
            .iter()
            .zip(coords.iter())
            .map(|(&name, &coords)| AtomV2Row {
                name: name.to_string(),
                coords,
                is_present: true,
                bfactor: 0.0,
                plddt: 0.0,
            })
            .collect();
        StructureV2Tables {
            atoms,
            residues: vec![
                ResidueRow {
                    name: "ALA".to_string(),
                    res_type: ala,
                    res_idx: 0,
                    atom_idx: 0,
                    atom_num: 5,
                    atom_center: 1,
                    atom_disto: 4,
                    is_standard: true,
                    is_present: true,
                },
                ResidueRow {
                    name: "ALA".to_string(),
                    res_type: ala,
                    res_idx: 1,
                    atom_idx: 5,
                    atom_num: 5,
                    atom_center: 6,
                    atom_disto: 9,
                    is_standard: true,
                    is_present: true,
                },
            ],
            chains: vec![ChainRow {
                name: "A".to_string(),
                mol_type: protein,
                sym_id: 0,
                asym_id: 0,
                entity_id: 0,
                atom_idx: 0,
                atom_num: 10,
                res_idx: 0,
                res_num: 2,
                cyclic_period: 0,
            }],
            chain_mask: vec![true],
            coords: coords.clone(),
            ensemble: vec![EnsembleRow {
                atom_coord_idx: 0,
                atom_num: 10,
            }],
            ensemble_atom_coord_idx: 0,
            bonds: vec![],
        }
    }

    #[test]
    fn valid_two_residue_chain_passes() {
        let s = two_ala_structure();
        let report = validate_structure_qc(
            &s,
            "prediction_model_0.cif",
            QcThresholds::default(),
            false,
            false,
        );
        assert!(report.passed, "{:?}", report.fail_reasons);
        assert_eq!(report.chain_count, 1);
        assert_eq!(report.residue_count, 2);
        assert_eq!(report.peptide_bond_distances.len(), 1);
    }

    #[test]
    fn missing_backbone_atom_fails() {
        let mut s = two_ala_structure();
        s.atoms[1].is_present = false;
        let report = validate_structure_qc(
            &s,
            "prediction_model_0.cif",
            QcThresholds::default(),
            false,
            false,
        );
        assert!(!report.passed);
        assert!(report
            .missing_backbone_atoms
            .iter()
            .any(|m| m.residue_index == 1 && m.atom == "CA"));
    }

    #[test]
    fn hard_chain_break_fails() {
        let mut s = two_ala_structure();
        s.atoms[5].coords = [20.0, 0.0, 0.0];
        let report = validate_structure_qc(
            &s,
            "prediction_model_0.cif",
            QcThresholds::default(),
            false,
            false,
        );
        assert!(!report.passed);
        assert!(!report.detected_chain_breaks.is_empty());
    }

    #[test]
    fn steric_overlap_fails() {
        let mut s = two_ala_structure();
        s.atoms[9].coords = s.atoms[4].coords;
        let report = validate_structure_qc(
            &s,
            "prediction_model_0.cif",
            QcThresholds::default(),
            false,
            false,
        );
        assert!(!report.passed);
        assert!(!report.steric_clashes.is_empty());
    }

    #[test]
    fn relaxation_can_fix_mild_peptide_distance_error() {
        let mut s = two_ala_structure();
        s.atoms[5].coords = [4.05, 1.65, 0.0];
        let initial = validate_structure_qc(
            &s,
            "prediction_model_0.cif",
            QcThresholds::default(),
            false,
            false,
        );
        assert!(!initial.passed);
        let outcome = relax_structure(&mut s, QcThresholds::default());
        assert!(outcome.attempted);
        let relaxed = validate_structure_qc(
            &s,
            "prediction_model_0.cif",
            QcThresholds::default(),
            true,
            true,
        );
        assert!(relaxed.passed, "{:?}", relaxed.fail_reasons);
    }
}

fn push_fail_once(fail_reasons: &mut Vec<String>, reason: &str) {
    if !fail_reasons.iter().any(|r| r == reason) {
        fail_reasons.push(reason.to_string());
    }
}

fn bonded_pair_set(structure: &StructureV2Tables) -> HashSet<(usize, usize)> {
    let mut set = HashSet::new();
    for b in &structure.bonds {
        let Ok(a1) = usize::try_from(b.atom_1) else {
            continue;
        };
        let Ok(a2) = usize::try_from(b.atom_2) else {
            continue;
        };
        if a1 == a2 {
            continue;
        }
        let lo = a1.min(a2);
        let hi = a1.max(a2);
        set.insert((lo, hi));
    }

    for chain in protein_chains(structure) {
        for r in &chain.residues {
            for pair in [("N", "CA"), ("CA", "C"), ("C", "O")] {
                if let (Some(a), Some(b)) = (r.atoms.get(pair.0), r.atoms.get(pair.1)) {
                    set.insert((a.index.min(b.index), a.index.max(b.index)));
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
                set.insert((c.index.min(n.index), c.index.max(n.index)));
            }
        }
    }
    set
}

/// Validate protein geometry in a [`StructureV2Tables`] instance and return a serializable report.
#[allow(clippy::too_many_lines)]
pub fn validate_structure_qc(
    structure: &StructureV2Tables,
    model_filename: impl Into<String>,
    thresholds: QcThresholds,
    relaxation_attempted: bool,
    relaxation_fixed: bool,
) -> QcReport {
    let model_filename = model_filename.into();
    let chains = protein_chains(structure);
    let chain_count = chains.len();
    let residue_count = chains.iter().map(|c| c.residues.len()).sum();
    let mut missing_backbone_atoms = Vec::new();
    let mut backbone_bond_distances = Vec::new();
    let mut peptide_bond_distances = Vec::new();
    let mut ca_ca_distances = Vec::new();
    let mut detected_chain_breaks = Vec::new();
    let mut omega_torsions = Vec::new();
    let mut fail_reasons = Vec::new();

    for chain in &chains {
        let chain_name = display_chain_name(
            &chain.name,
            chain.residues.first().map_or(0, |r| r.chain_idx),
        );
        let mut last_res_idx: Option<i32> = None;
        for residue in &chain.residues {
            if let Some(prev) = last_res_idx {
                if residue.residue_index <= prev {
                    detected_chain_breaks.push(ChainBreak {
                        chain: chain_name.clone(),
                        from_residue_index: prev,
                        to_residue_index: residue.residue_index,
                        c_to_n_distance_angstrom: None,
                        reason: "non-increasing residue order".to_string(),
                    });
                    push_fail_once(
                        &mut fail_reasons,
                        "residue order is not strictly increasing",
                    );
                }
            }
            last_res_idx = Some(residue.residue_index);

            if !residue.is_standard_protein {
                continue;
            }
            for atom in ["N", "CA", "C", "O"] {
                if !residue.atoms.contains_key(atom) {
                    missing_backbone_atoms.push(MissingBackboneAtom {
                        chain: chain_name.clone(),
                        residue_index: residue.residue_index,
                        residue_name: residue.residue_name.clone(),
                        atom: atom.to_string(),
                    });
                    push_fail_once(&mut fail_reasons, "missing required backbone atoms");
                }
            }
            for (bond, min, max) in [
                ("N-CA", thresholds.n_ca_min, thresholds.n_ca_max),
                ("CA-C", thresholds.ca_c_min, thresholds.ca_c_max),
                ("C-O", thresholds.c_o_min, thresholds.c_o_max),
            ] {
                let (a_name, b_name) = bond.split_once('-').expect("bond format");
                let (Some(a), Some(b)) = (residue.atoms.get(a_name), residue.atoms.get(b_name))
                else {
                    continue;
                };
                let d = distance(a.coords, b.coords);
                let passed = in_range(d, min, max);
                if !passed {
                    push_fail_once(&mut fail_reasons, "backbone bond distance out of tolerance");
                }
                backbone_bond_distances.push(BackboneBondDistance {
                    chain: chain_name.clone(),
                    residue_index: residue.residue_index,
                    residue_name: residue.residue_name.clone(),
                    bond: bond.to_string(),
                    distance_angstrom: d,
                    passed,
                });
            }
        }

        for pair in chain.residues.windows(2) {
            let a = &pair[0];
            let b = &pair[1];
            if !a.is_standard_protein || !b.is_standard_protein {
                continue;
            }
            let chain_name = display_chain_name(&a.chain_name, a.chain_idx);
            let (Some(c), Some(n)) = (a.atoms.get("C"), b.atoms.get("N")) else {
                detected_chain_breaks.push(ChainBreak {
                    chain: chain_name.clone(),
                    from_residue_index: a.residue_index,
                    to_residue_index: b.residue_index,
                    c_to_n_distance_angstrom: None,
                    reason: "missing C or N for peptide continuity".to_string(),
                });
                push_fail_once(&mut fail_reasons, "chain continuity cannot be evaluated");
                continue;
            };
            let cn = distance(c.coords, n.coords);
            let cn_passed = in_range(cn, thresholds.peptide_cn_min, thresholds.peptide_cn_max);
            if !cn_passed {
                push_fail_once(&mut fail_reasons, "peptide C-N distance out of tolerance");
            }
            peptide_bond_distances.push(PeptideBondDistance {
                chain: chain_name.clone(),
                from_residue_index: a.residue_index,
                to_residue_index: b.residue_index,
                distance_angstrom: cn,
                passed: cn_passed,
            });
            if cn > thresholds.hard_chain_break_cn {
                detected_chain_breaks.push(ChainBreak {
                    chain: chain_name.clone(),
                    from_residue_index: a.residue_index,
                    to_residue_index: b.residue_index,
                    c_to_n_distance_angstrom: Some(cn),
                    reason: "hard C-N chain break".to_string(),
                });
                push_fail_once(&mut fail_reasons, "hard chain break detected");
            }

            if let (Some(ca_a), Some(ca_b)) = (a.atoms.get("CA"), b.atoms.get("CA")) {
                let d = distance(ca_a.coords, ca_b.coords);
                let passed = in_range(d, thresholds.ca_ca_min, thresholds.ca_ca_max);
                if !passed {
                    push_fail_once(&mut fail_reasons, "CA-CA distance out of tolerance");
                }
                ca_ca_distances.push(PeptideBondDistance {
                    chain: chain_name.clone(),
                    from_residue_index: a.residue_index,
                    to_residue_index: b.residue_index,
                    distance_angstrom: d,
                    passed,
                });
            }

            if let (Some(ca_a), Some(c_a), Some(n_b), Some(ca_b)) = (
                a.atoms.get("CA"),
                a.atoms.get("C"),
                b.atoms.get("N"),
                b.atoms.get("CA"),
            ) {
                let omega = dihedral_degrees(ca_a.coords, c_a.coords, n_b.coords, ca_b.coords);
                let passed = omega.is_finite() && omega.abs() >= thresholds.omega_abs_min_degrees;
                if !passed {
                    push_fail_once(&mut fail_reasons, "omega torsion out of tolerance");
                }
                omega_torsions.push(TorsionMeasure {
                    chain: chain_name,
                    from_residue_index: a.residue_index,
                    to_residue_index: b.residue_index,
                    omega_degrees: omega,
                    passed,
                });
            }
        }
    }

    let bonded = bonded_pair_set(structure);
    let heavy = heavy_atoms(structure);
    let mut steric_clashes = Vec::new();
    for i in 0..heavy.len() {
        for j in (i + 1)..heavy.len() {
            let a = &heavy[i];
            let b = &heavy[j];
            let key = (a.index.min(b.index), a.index.max(b.index));
            if bonded.contains(&key) {
                continue;
            }
            let d = distance(a.coords, b.coords);
            if d < thresholds.hard_overlap {
                steric_clashes.push(StericClash {
                    atom_1_index: a.index,
                    atom_2_index: b.index,
                    atom_1: a.name.clone(),
                    atom_2: b.name.clone(),
                    distance_angstrom: d,
                });
            }
        }
    }
    if !steric_clashes.is_empty() {
        push_fail_once(&mut fail_reasons, "hard steric overlap detected");
    }

    let rg_coords: Vec<_> = heavy.iter().map(|a| a.coords).collect();
    let radius_of_gyration = radius_of_gyration(&rg_coords);
    if chain_count == 0 || residue_count == 0 {
        push_fail_once(&mut fail_reasons, "no visible protein residues found");
    }

    let passed = fail_reasons.is_empty();
    QcReport {
        model_filename,
        chain_count,
        residue_count,
        missing_backbone_atoms,
        backbone_bond_distances,
        peptide_bond_distances,
        ca_ca_distances,
        detected_chain_breaks,
        steric_clashes,
        omega_torsions,
        radius_of_gyration,
        passed,
        fail_reasons,
        relaxation_attempted,
        relaxation_fixed,
    }
}
