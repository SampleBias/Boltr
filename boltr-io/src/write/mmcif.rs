//! Minimal mmCIF (`_atom_site`) — consumer-friendly coordinates without Python `modelcif` / `ihm`.
//!
//! For full **ModelCIF** parity see [`mmcif.py`](../../../boltz-reference/src/boltz/data/write/mmcif.py).

use crate::ambiguous_atoms::{pdb_atom_key, resolve_ambiguous_element};
use crate::boltz_const::CHAIN_TYPES;
use crate::structure_v2::StructureV2Tables;

fn nonpolymer_mol_type() -> i8 {
    CHAIN_TYPES
        .iter()
        .position(|&c| c == "NONPOLYMER")
        .expect("NONPOLYMER") as i8
}

fn chain_asym_id(name: &str) -> String {
    let c = name.chars().next().unwrap_or('.');
    if c.is_whitespace() {
        ".".to_string()
    } else {
        c.to_string()
    }
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

/// Minimal `data_` + `_atom_site` loop (core chemistry columns only).
#[must_use]
pub fn structure_v2_to_mmcif(t: &StructureV2Tables) -> String {
    let nonpoly = nonpolymer_mol_type();
    let mut out = String::new();
    out.push_str("data_BOLTR\n");
    out.push_str("#\n");
    out.push_str("loop_\n");
    out.push_str("_atom_site.group_PDB\n");
    out.push_str("_atom_site.id\n");
    out.push_str("_atom_site.type_symbol\n");
    out.push_str("_atom_site.label_atom_id\n");
    out.push_str("_atom_site.label_alt_id\n");
    out.push_str("_atom_site.label_comp_id\n");
    out.push_str("_atom_site.label_asym_id\n");
    out.push_str("_atom_site.label_entity_id\n");
    out.push_str("_atom_site.label_seq_id\n");
    out.push_str("_atom_site.Cartn_x\n");
    out.push_str("_atom_site.Cartn_y\n");
    out.push_str("_atom_site.Cartn_z\n");
    out.push_str("_atom_site.occupancy\n");
    out.push_str("_atom_site.B_iso_or_equiv\n");

    let mut serial: usize = 1;
    for (chain_idx, chain) in t.chains.iter().enumerate() {
        if !t.chain_mask.get(chain_idx).copied().unwrap_or(false) {
            continue;
        }
        let asym = chain_asym_id(&chain.name);
        let res_start = chain.res_idx as usize;
        let res_end = res_start + chain.res_num as usize;
        let residues = if res_start < t.residues.len() && res_end <= t.residues.len() {
            &t.residues[res_start..res_end]
        } else {
            continue;
        };

        for residue in residues {
            let atom_start = residue.atom_idx as usize;
            let atom_end = atom_start + residue.atom_num as usize;
            if atom_start >= t.atoms.len() || atom_end > t.atoms.len() {
                continue;
            }
            let atoms = &t.atoms[atom_start..atom_end];
            let hetatm = chain.mol_type == nonpoly;
            let res3 = res_name_3(&residue.name, hetatm);
            let record = if hetatm { "HETATM" } else { "ATOM" };
            let seq_id = residue.res_idx + 1;

            for atom in atoms {
                if !atom.is_present {
                    continue;
                }
                let elem = element_symbol(&atom.name, &res3);
                let n = atom.name.trim();
                let pos = atom.coords;
                let b_iso = if atom.plddt > 0.0 {
                    f64::from(atom.plddt) * 100.0
                } else {
                    100.0
                };
                out.push_str(&format!(
                    "{} {} {} {} . {} {} {} {} {:.3} {:.3} {:.3} 1.0 {:.2}\n",
                    record,
                    serial,
                    elem,
                    n,
                    res3,
                    asym,
                    chain.entity_id,
                    seq_id,
                    pos[0],
                    pos[1],
                    pos[2],
                    b_iso
                ));
                serial += 1;
            }
        }
    }

    out.push_str("#\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::structure_v2_single_ala;

    #[test]
    fn ala_smoke_has_atom_site_loop() {
        let t = structure_v2_single_ala();
        let s = structure_v2_to_mmcif(&t);
        assert!(s.contains("loop_"));
        assert!(s.contains("_atom_site.Cartn_x"));
        assert!(s.contains("ATOM"));
    }
}
