//! Boltz2 structure tokenization — port of `boltz.data.tokenize.boltz2` (`tokenize_structure`, `compute_frame`).

use crate::boltz_const::{chain_type_id, unk_token_id};
use crate::structure_v2::{ChainRow, ResidueRow, StructureV2Tables};

/// One token row (logical `TokenData` / `TokenV2` before numpy packing).
#[derive(Clone, Debug, PartialEq)]
pub struct TokenData {
    pub token_idx: i32,
    pub atom_idx: i32,
    pub atom_num: i32,
    pub res_idx: i32,
    pub res_type: i32,
    pub res_name: String,
    pub sym_id: i32,
    pub asym_id: i32,
    pub entity_id: i32,
    pub mol_type: i32,
    pub center_idx: i32,
    pub disto_idx: i32,
    pub center_coords: [f32; 3],
    pub disto_coords: [f32; 3],
    pub resolved_mask: bool,
    pub disto_mask: bool,
    pub modified: bool,
    pub frame_rot: [f32; 9],
    pub frame_t: [f32; 3],
    pub frame_mask: bool,
    pub cyclic_period: i32,
    pub affinity_mask: bool,
}

/// `(token_1, token_2, type)` with `type == bond_type + 1` per Python `TokenBondV2`.
pub type TokenBondV2 = (i32, i32, i8);

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm(v: [f32; 3]) -> f32 {
    dot(v, v).sqrt()
}

#[inline]
fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Backbone local frame (rotation flattened column-major columns `[e1|e2|e3]`, then C-order flatten).
///
/// Matches `compute_frame` in `boltz2.py` (`n`, `ca`, `c` atom coordinates).
#[must_use]
pub fn compute_frame(n: [f32; 3], ca: [f32; 3], c: [f32; 3]) -> ([f32; 9], [f32; 3]) {
    const EPS: f32 = 1e-10;
    let v1 = sub(c, ca);
    let v2 = sub(n, ca);
    let n1 = norm(v1) + EPS;
    let e1 = scale(v1, 1.0 / n1);
    let proj = dot(e1, v2);
    let u2 = sub(v2, scale(e1, proj));
    let n2 = norm(u2) + EPS;
    let e2 = scale(u2, 1.0 / n2);
    let e3 = cross(e1, e2);
    let rot = [
        e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], e3[0], e3[1], e3[2],
    ];
    (rot, ca)
}

#[inline]
fn unk_res_type_for_mol_type(mol_type: i8) -> i32 {
    let dna = chain_type_id("DNA").expect("DNA chain id") as i8;
    let rna = chain_type_id("RNA").expect("RNA chain id") as i8;
    if mol_type == dna {
        unk_token_id("DNA").expect("DN token")
    } else if mol_type == rna {
        unk_token_id("RNA").expect("N token")
    } else {
        unk_token_id("PROTEIN").expect("UNK token")
    }
}

#[inline]
fn nonpolymer_mol_type() -> i8 {
    chain_type_id("NONPOLYMER").expect("NONPOLYMER id") as i8
}

#[inline]
fn protein_mol_type() -> i8 {
    chain_type_id("PROTEIN").expect("PROTEIN id") as i8
}

#[allow(clippy::too_many_arguments)] // Mirrors `tokenize_structure` branches; splitting obscures parity.
fn push_standard_or_modified_token(
    out: &mut Vec<TokenData>,
    struct_: &StructureV2Tables,
    chain: &ChainRow,
    res: &ResidueRow,
    token_idx: i32,
    res_type: i32,
    modified: bool,
    affinity_mask: bool,
) -> Option<()> {
    let center = struct_.atoms.get(usize::try_from(res.atom_center).ok()?)?;
    let disto = struct_.atoms.get(usize::try_from(res.atom_disto).ok()?)?;
    let is_present = res.is_present && center.is_present;
    let is_disto_present = res.is_present && disto.is_present;
    let c_coords = struct_.ensemble_coords(res.atom_center)?;
    let d_coords = struct_.ensemble_coords(res.atom_disto)?;

    let is_protein = chain.mol_type == protein_mol_type();
    let mut frame_rot = [0.0_f32; 9];
    frame_rot[0] = 1.0;
    frame_rot[4] = 1.0;
    frame_rot[8] = 1.0;
    let mut frame_t = [0.0_f32; 3];
    let mut frame_mask = false;

    if is_protein {
        let atom_st = usize::try_from(res.atom_idx).ok()?;
        let atom_en = atom_st + usize::try_from(res.atom_num).ok()?;
        if atom_en - atom_st >= 3 {
            let atom_n = &struct_.atoms[atom_st];
            let atom_ca = &struct_.atoms[atom_st + 1];
            let atom_c = &struct_.atoms[atom_st + 2];
            frame_mask = atom_ca.is_present && atom_c.is_present && atom_n.is_present;
            if frame_mask {
                let (r, t) = compute_frame(atom_n.coords, atom_ca.coords, atom_c.coords);
                frame_rot = r;
                frame_t = t;
            }
        }
    }

    out.push(TokenData {
        token_idx,
        atom_idx: res.atom_idx,
        atom_num: res.atom_num,
        res_idx: res.res_idx,
        res_type,
        res_name: res.name.clone(),
        sym_id: chain.sym_id,
        asym_id: chain.asym_id,
        entity_id: chain.entity_id,
        mol_type: i32::from(chain.mol_type),
        center_idx: res.atom_center,
        disto_idx: res.atom_disto,
        center_coords: c_coords,
        disto_coords: d_coords,
        resolved_mask: is_present,
        disto_mask: is_disto_present,
        modified,
        frame_rot,
        frame_t,
        frame_mask,
        cyclic_period: chain.cyclic_period,
        affinity_mask,
    });
    Some(())
}

/// Tokenize a structure (Boltz `tokenize_structure`).
///
/// `affinity_asym_id`: when `Some`, sets `affinity_mask` on tokens whose chain `asym_id` matches.
#[must_use]
pub fn tokenize_structure(
    struct_: &StructureV2Tables,
    affinity_asym_id: Option<i32>,
) -> (Vec<TokenData>, Vec<TokenBondV2>) {
    let mut token_data: Vec<TokenData> = Vec::new();
    let mut atom_to_token: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
    let mut token_idx: i32 = 0;

    for (ci, chain) in struct_.chains.iter().enumerate() {
        if !struct_.chain_mask.get(ci).copied().unwrap_or(false) {
            continue;
        }

        let res_start = chain.res_idx as usize;
        let res_end = res_start + chain.res_num as usize;
        let Some(res_slice) = struct_.residues.get(res_start..res_end) else {
            continue;
        };

        let affinity_mask = affinity_asym_id.is_some_and(|a| a == chain.asym_id);

        for res in res_slice {
            let atom_start = res.atom_idx;
            let atom_end = res.atom_idx + res.atom_num;

            if res.is_standard {
                let rt = i32::from(res.res_type);
                if push_standard_or_modified_token(
                    &mut token_data,
                    struct_,
                    chain,
                    res,
                    token_idx,
                    rt,
                    false,
                    affinity_mask,
                )
                .is_some()
                {
                    for a in atom_start..atom_end {
                        atom_to_token.insert(a, token_idx);
                    }
                    token_idx += 1;
                }
            } else if chain.mol_type == nonpolymer_mol_type() {
                let unk_id = unk_token_id("PROTEIN").expect("UNK");
                let atom_st = usize::try_from(atom_start).unwrap_or(0);
                let atom_en = usize::try_from(atom_end).unwrap_or(0);
                for (i, atom) in struct_
                    .atoms
                    .get(atom_st..atom_en)
                    .unwrap_or(&[])
                    .iter()
                    .enumerate()
                {
                    let index = atom_start + i as i32;
                    let is_present = res.is_present && atom.is_present;
                    let ac = struct_.ensemble_coords(index).unwrap_or([0.0; 3]);
                    token_data.push(TokenData {
                        token_idx,
                        atom_idx: index,
                        atom_num: 1,
                        res_idx: res.res_idx,
                        res_type: unk_id,
                        res_name: res.name.clone(),
                        sym_id: chain.sym_id,
                        asym_id: chain.asym_id,
                        entity_id: chain.entity_id,
                        mol_type: i32::from(chain.mol_type),
                        center_idx: index,
                        disto_idx: index,
                        center_coords: ac,
                        disto_coords: ac,
                        resolved_mask: is_present,
                        disto_mask: is_present,
                        modified: false,
                        frame_rot: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        frame_t: [0.0; 3],
                        frame_mask: false,
                        cyclic_period: chain.cyclic_period,
                        affinity_mask,
                    });
                    atom_to_token.insert(index, token_idx);
                    token_idx += 1;
                }
            } else {
                let rt = unk_res_type_for_mol_type(chain.mol_type);
                if push_standard_or_modified_token(
                    &mut token_data,
                    struct_,
                    chain,
                    res,
                    token_idx,
                    rt,
                    true,
                    affinity_mask,
                )
                .is_some()
                {
                    for a in atom_start..atom_end {
                        atom_to_token.insert(a, token_idx);
                    }
                    token_idx += 1;
                }
            }
        }
    }

    let mut token_bonds: Vec<TokenBondV2> = Vec::new();
    for bond in &struct_.bonds {
        let Some(&t1) = atom_to_token.get(&bond.atom_1) else {
            continue;
        };
        let Some(&t2) = atom_to_token.get(&bond.atom_2) else {
            continue;
        };
        token_bonds.push((t1, t2, bond.bond_type.wrapping_add(1)));
    }

    (token_data, token_bonds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boltz_const::token_id;
    use crate::fixtures::structure_v2_single_ala;
    use crate::structure_v2::{AtomV2Row, BondV2AtomRow, ChainRow, ResidueRow, StructureV2Tables};

    #[test]
    fn compute_frame_matches_analytic_example() {
        let n = [0.0_f32, 0.0, 0.0];
        let ca = [1.0, 0.0, 0.0];
        let c = [1.0, 1.0, 0.0];
        let (rot, t) = compute_frame(n, ca, c);
        let expected = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        for (a, b) in rot.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "rot {rot:?} != {expected:?}");
        }
        assert!((t[0] - 1.0).abs() < 1e-5 && t[1].abs() < 1e-5 && t[2].abs() < 1e-5);
    }

    #[test]
    fn tokenize_single_standard_ala() {
        let s = structure_v2_single_ala();
        let (tokens, bonds) = tokenize_structure(&s, None);
        assert_eq!(tokens.len(), 1);
        assert!(bonds.is_empty());
        let t = &tokens[0];
        assert_eq!(t.res_type, token_id("ALA").unwrap());
        assert_eq!(t.atom_num, 5);
        assert!(t.frame_mask);
        assert!(!t.modified);
        let expected_rot = [0.0_f32, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        for (a, b) in t.frame_rot.iter().zip(expected_rot.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
        assert!(
            (t.frame_t[0] - 1.0).abs() < 1e-4
                && t.frame_t[1].abs() < 1e-4
                && t.frame_t[2].abs() < 1e-4
        );
    }

    #[test]
    fn tokenize_ligand_two_atoms_and_bond() {
        let np = nonpolymer_mol_type();
        let coords = vec![[0.0_f32; 3], [1.0, 0.0, 0.0]];
        let atoms: Vec<_> = coords
            .iter()
            .map(|&c| AtomV2Row {
                coords: c,
                is_present: true,
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
                atom_disto: 0,
                is_standard: false,
                is_present: true,
            }],
            chains: vec![ChainRow {
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
            ensemble_atom_coord_idx: 0,
            bonds: vec![BondV2AtomRow {
                atom_1: 0,
                atom_2: 1,
                bond_type: 0,
            }],
        };
        let (tokens, bonds) = tokenize_structure(&s, None);
        assert_eq!(tokens.len(), 2);
        assert_eq!(bonds, vec![(0, 1, 1)]);
        assert_eq!(tokens[0].atom_num, 1);
        assert_eq!(tokens[0].res_type, token_id("UNK").unwrap());
    }

    #[test]
    fn affinity_mask_on_matching_asym() {
        let s = structure_v2_single_ala();
        let (tok, _) = tokenize_structure(&s, None);
        assert!(!tok[0].affinity_mask);
        let (tok2, _) = tokenize_structure(&s, Some(0));
        assert!(tok2[0].affinity_mask);
        let (tok3, _) = tokenize_structure(&s, Some(1));
        assert!(!tok3[0].affinity_mask);
    }
}
