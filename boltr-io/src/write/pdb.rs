//! PDB text — Boltz-style ATOM/HETATM + CONECT; coordinates and `--output_format pdb` in
//! [prediction.md](../../../../boltz-reference/docs/prediction.md).
//!
//! Uses [`crate::ambiguous_atoms::resolve_ambiguous_element`] for element symbols; no RDKit.

use crate::ambiguous_atoms::{pdb_atom_key, resolve_ambiguous_element};
use crate::boltz_const::CHAIN_TYPES;
use crate::structure_v2::StructureV2Tables;

fn nonpolymer_mol_type() -> i8 {
    CHAIN_TYPES
        .iter()
        .position(|&c| c == "NONPOLYMER")
        .expect("NONPOLYMER") as i8
}

fn pdb_atom_name_field(name: &str) -> String {
    let n = name.trim();
    let s = if n.len() == 4 {
        n.to_string()
    } else {
        format!(" {}", n)
    };
    format!("{:<4}", s)
}

fn element_symbol(atom_name: &str, res_name_3: &str) -> String {
    let key = pdb_atom_key(atom_name);
    resolve_ambiguous_element(&key, res_name_3)
        .unwrap_or_else(|| key.chars().next().unwrap_or('C').to_string())
        .to_uppercase()
}

fn res_name_3(res_name: &str, hetatm: bool) -> String {
    if hetatm {
        return "LIG".to_string();
    }
    let u = res_name.to_uppercase();
    if u.len() >= 3 {
        u[..3].to_string()
    } else {
        format!("{:<3}", u)
    }
}

fn chain_tag_char(name: &str) -> char {
    name.chars().next().unwrap_or(' ')
}

/// Serialize coordinates to PDB ATOM/HETATM records (80-column lines).
///
/// `per_atom_b` — optional per-atom values in **\[0,1\]** (pLDDT scale); written as `value * 100` in
/// the B-factor column (two decimal places), matching Boltz when a `plddt` tensor is passed. If
/// `None`, uses **100.0** (Boltz when `plddts` is absent). If the slice length does not match
/// `atoms.len()`, missing entries use **100.0**.
#[must_use]
pub fn structure_v2_to_pdb(t: &StructureV2Tables, per_atom_plddt_01: Option<&[f32]>) -> String {
    let nonpoly = nonpolymer_mol_type();
    let visible: Vec<usize> = (0..t.chains.len())
        .filter(|&i| t.chain_mask.get(i).copied().unwrap_or(false))
        .collect();

    let mut lines: Vec<String> = Vec::new();
    let mut atom_index: usize = 1;
    let mut serial_for_global: Vec<usize> = vec![0; t.atoms.len()];

    for (vi, &chain_idx) in visible.iter().enumerate() {
        let chain = &t.chains[chain_idx];
        let chain_tag = chain_tag_char(&chain.name);
        let res_start = chain.res_idx as usize;
        let res_end = res_start + chain.res_num as usize;
        let residues = if res_start < t.residues.len() && res_end <= t.residues.len() {
            &t.residues[res_start..res_end]
        } else {
            continue;
        };

        let mut last_residue_index = 1_i32;
        let mut last_res3 = "ALA".to_string();

        for residue in residues {
            let atom_start = residue.atom_idx as usize;
            let atom_end = atom_start + residue.atom_num as usize;
            if atom_start >= t.atoms.len() || atom_end > t.atoms.len() {
                continue;
            }
            let atoms = &t.atoms[atom_start..atom_end];
            let hetatm = chain.mol_type == nonpoly;
            let res3 = res_name_3(&residue.name, hetatm);
            let record_type = if hetatm { "HETATM" } else { "ATOM" };
            let residue_index = residue.res_idx + 1;
            last_residue_index = residue_index;
            last_res3.clone_from(&res3);

            for (k, atom) in atoms.iter().enumerate() {
                if !atom.is_present {
                    continue;
                }
                let gidx = atom_start + k;
                let name = pdb_atom_name_field(&atom.name);
                let element = element_symbol(&atom.name, &res3);
                let pos = atom.coords;
                let b_factor = per_atom_plddt_01
                    .and_then(|s| s.get(gidx).copied())
                    .map(|v| f64::from(v) * 100.0)
                    .map(|v| (v * 100.0).round() / 100.0)
                    .unwrap_or(100.0);

                let alt_loc = "";
                let ins_code = "";
                let line = format!(
                    "{:<6}{:5} {:4}{:1}{:3} {:1}{:4}{:1}   {:8.3}{:8.3}{:8.3}{:6.2}{:6.2}          {:>2}  ",
                    record_type,
                    atom_index,
                    name,
                    alt_loc,
                    res3,
                    chain_tag,
                    residue_index,
                    ins_code,
                    pos[0],
                    pos[1],
                    pos[2],
                    1.0_f64,
                    b_factor,
                    element
                );
                lines.push(line);
                if gidx < serial_for_global.len() {
                    serial_for_global[gidx] = atom_index;
                }
                atom_index += 1;
            }
        }

        let should_ter = vi + 1 < visible.len();
        if should_ter {
            lines.push(format!(
                "{:<6}{:5}      {:3} {:1}{:4}",
                "TER",
                atom_index,
                last_res3,
                chain_tag,
                last_residue_index
            ));
            atom_index += 1;
        }
    }

    for b in &t.bonds {
        let a1 = b.atom_1 as usize;
        let a2 = b.atom_2 as usize;
        if a1 >= t.atoms.len() || a2 >= t.atoms.len() {
            continue;
        }
        if !t.atoms[a1].is_present || !t.atoms[a2].is_present {
            continue;
        }
        let s1 = serial_for_global.get(a1).copied().unwrap_or(0);
        let s2 = serial_for_global.get(a2).copied().unwrap_or(0);
        if s1 == 0 || s2 == 0 {
            continue;
        }
        lines.push(format!("CONECT{:5}{:5}", s1, s2));
    }

    lines.push("END".to_string());
    lines.push(String::new());

    lines
        .into_iter()
        .map(|l| format!("{:<80}", l))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::structure_v2_single_ala;
    use crate::structure_v2::BondV2AtomRow;

    #[test]
    fn ala_smoke_contains_atom_and_end() {
        let t = structure_v2_single_ala();
        let s = structure_v2_to_pdb(&t, None);
        assert!(s.contains("ATOM"));
        assert!(s.contains("END"));
        assert!(s.lines().all(|l| l.len() <= 80));
    }

    #[test]
    fn conect_two_bonds_share_first_atom() {
        let mut t = structure_v2_single_ala();
        t.bonds.push(BondV2AtomRow {
            atom_1: 0,
            atom_2: 1,
            bond_type: 1,
        });
        t.bonds.push(BondV2AtomRow {
            atom_1: 0,
            atom_2: 2,
            bond_type: 1,
        });
        let s = structure_v2_to_pdb(&t, None);
        let conects: Vec<_> = s.lines().filter(|l| l.starts_with("CONECT")).collect();
        assert_eq!(conects.len(), 2);
        assert!(conects.iter().any(|l| l.contains("    1    2")));
        assert!(conects.iter().any(|l| l.contains("    1    3")));
        assert!(s.lines().all(|l| l.len() <= 80));
    }
}
