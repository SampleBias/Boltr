//! Boltz `process_symmetry_features` (`featurizerv2.py`) + `symmetry.py` (`get_*_symmetries`).
//!
//! Produces the same structure as Python: float/bool/int tensors for coords and crop maps, and
//! ragged lists for symmetry swaps (Python keeps these as Python objects; see
//! [`crate::feature_batch::INFERENCE_COLLATE_EXCLUDED_KEYS`]).
//!
//! **Ligand symmetries** from CCD pickle (`get_ligand_symmetries`) are only populated when a
//! Boltz-style `symmetries` map is supplied; otherwise [`SymmetryFeatures::ligand_symmetries`] is
//! empty (matches an empty dict in Python).

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;

use std::collections::HashMap;

use itertools::Itertools;

use crate::boltz_const::chain_type_id;
use crate::feature_batch::FeatureBatch;
use crate::ref_atoms::{ref_atoms_key_from_token, ref_symmetry_groups};
use crate::structure_v2::StructureV2Tables;
use crate::tokenize::boltz2::TokenData;

/// One chain swap tuple: `(start1, end1, start2, end2, chainidx1, chainidx2)` in the concatenated
/// atom layout used by [`SymmetryFeatures::all_coords`].
pub type ChainSwap = (usize, usize, usize, usize, usize, usize);

#[derive(Clone, Debug, PartialEq)]
pub struct SymmetryFeatures {
    pub all_coords: Array2<f32>,
    pub all_resolved_mask: Array1<bool>,
    pub crop_to_all_atom_map: Array1<i64>,
    /// Each element is one valid global symmetry assignment (product of per-chain swap choices).
    pub chain_symmetries: Vec<Vec<ChainSwap>>,
    /// Per residue token: inner groups are alternative symmetry swaps (atom index pairs in crop space).
    pub amino_acids_symmetries: Vec<Vec<Vec<(usize, usize)>>>,
    /// Per ligand / non-standard molecule: groups of `(i, j)` swaps in crop space (empty when no CCD symmetries).
    pub ligand_symmetries: Vec<Vec<Vec<(usize, usize)>>>,
}

impl SymmetryFeatures {
    /// Convert tensor fields into a [`FeatureBatch`].
    ///
    /// Note: `chain_symmetries`, `amino_acids_symmetries`, and `ligand_symmetries`
    /// are ragged Python objects and cannot be stored as tensors. They go into
    /// the "excluded" batch for per-example collation (not stacked).
    #[must_use]
    pub fn to_feature_batch(&self) -> FeatureBatch {
        let mut b = FeatureBatch::new();
        b.insert_f32("all_coords", self.all_coords.clone().into_dyn());
        // all_resolved_mask: bool → i64
        b.insert_i64(
            "all_resolved_mask",
            self.all_resolved_mask
                .mapv(|v| if v { 1_i64 } else { 0 })
                .into_dyn(),
        );
        b.insert_i64(
            "crop_to_all_atom_map",
            self.crop_to_all_atom_map.clone().into_dyn(),
        );
        b
    }
}

fn all_different_after_swap(combo: &[ChainSwap]) -> bool {
    let finals: Vec<usize> = combo.iter().map(|s| s.5).collect();
    let uniq: std::collections::HashSet<usize> = finals.iter().copied().collect();
    uniq.len() == finals.len()
}

/// `get_amino_acids_symmetries` from `symmetry.py`.
#[must_use]
pub fn get_amino_acids_symmetries(tokens: &[TokenData]) -> Vec<Vec<Vec<(usize, usize)>>> {
    let mut swaps: Vec<Vec<Vec<(usize, usize)>>> = Vec::new();
    let mut start_index_crop: usize = 0;
    for token in tokens {
        let key = ref_atoms_key_from_token(&token.res_name);
        let syms = ref_symmetry_groups(key);
        if !syms.is_empty() {
            let mut residue_swaps: Vec<Vec<(usize, usize)>> = Vec::new();
            for sym in syms {
                let sym_new_idx: Vec<(usize, usize)> = sym
                    .iter()
                    .map(|&(i, j)| (i + start_index_crop, j + start_index_crop))
                    .collect();
                residue_swaps.push(sym_new_idx);
            }
            swaps.push(residue_swaps);
        }
        start_index_crop += token.atom_num as usize;
    }
    swaps
}

/// `get_ligand_symmetries` — full parity requires RDKit/CCD pickle data. With `symmetries == None`,
/// returns an empty list (same as Python with an empty dict).
#[must_use]
pub fn get_ligand_symmetries_empty() -> Vec<Vec<Vec<(usize, usize)>>> {
    Vec::new()
}

/// Map CCD-style residue keys (see [`ref_atoms_key_from_token`]) to symmetry swap groups, then lift
/// atom indices into **crop** space (same convention as [`get_amino_acids_symmetries`]).
///
/// One entry is appended per **NONPOLYMER** token (Boltz ligand); use an empty inner `Vec` when the
/// ligand has no entry in `symmetries`.
#[must_use]
pub fn get_ligand_symmetries_for_tokens(
    tokens: &[TokenData],
    symmetries: &HashMap<String, Vec<Vec<(usize, usize)>>>,
) -> Vec<Vec<Vec<(usize, usize)>>> {
    let nonpoly = chain_type_id("NONPOLYMER").expect("NONPOLYMER");
    let mut out: Vec<Vec<Vec<(usize, usize)>>> = Vec::new();
    let mut crop_start: usize = 0;
    for token in tokens {
        let n = token.atom_num as usize;
        if token.mol_type == nonpoly {
            let key = ref_atoms_key_from_token(&token.res_name);
            let groups = symmetries.get(key).map(|v| v.as_slice()).unwrap_or(&[]);
            let mapped: Vec<Vec<(usize, usize)>> = groups
                .iter()
                .map(|g| {
                    g.iter()
                        .map(|&(i, j)| (i + crop_start, j + crop_start))
                        .collect()
                })
                .collect();
            out.push(mapped);
        }
        crop_start += n;
    }
    out
}

/// `get_chain_symmetries` from `symmetry.py` (structure + tokens only; uses first conformer coords).
#[allow(clippy::cast_possible_wrap)]
pub fn get_chain_symmetries(
    structure: &StructureV2Tables,
    tokens: &[TokenData],
    max_n_symmetries: usize,
    rng: &mut impl Rng,
) -> SymmetryFeatures {
    let mut all_coords: Vec<f32> = Vec::new();
    let mut all_resolved_mask: Vec<bool> = Vec::new();
    let mut original_atom_idx: Vec<i32> = Vec::new();
    let mut chain_atom_idx: Vec<usize> = Vec::new();
    let mut chain_atom_num: Vec<usize> = Vec::new();
    let mut chain_in_crop: Vec<bool> = Vec::new();
    let mut chain_asym_id: Vec<i32> = Vec::new();

    let mut new_atom_idx: usize = 0;

    for chain in &structure.chains {
        let atom_idx = chain.atom_idx as usize;
        let atom_num = chain.atom_num as usize;
        let resolved_mask: Vec<bool> = structure
            .atoms
            .get(atom_idx..atom_idx + atom_num)
            .map(|slice| slice.iter().map(|a| a.is_present).collect())
            .unwrap_or_default();
        let coords: Vec<[f32; 3]> = structure
            .atoms
            .get(atom_idx..atom_idx + atom_num)
            .map(|slice| slice.iter().map(|a| a.coords).collect())
            .unwrap_or_default();

        let in_crop = tokens.iter().any(|t| t.asym_id == chain.asym_id);

        for c in &coords {
            all_coords.extend_from_slice(c);
        }
        all_resolved_mask.extend(resolved_mask);
        original_atom_idx.push(chain.atom_idx);
        chain_atom_idx.push(new_atom_idx);
        chain_atom_num.push(atom_num);
        chain_in_crop.push(in_crop);
        chain_asym_id.push(chain.asym_id);

        new_atom_idx += atom_num;
    }

    let mut crop_to_all_atom_map: Vec<i64> = Vec::new();
    for token in tokens {
        let chain_idx = chain_asym_id
            .iter()
            .position(|&a| a == token.asym_id)
            .expect("token asym_id must match a structure chain");
        let start = chain_atom_idx[chain_idx] as i64 - i64::from(original_atom_idx[chain_idx])
            + i64::from(token.atom_idx);
        let n = token.atom_num as usize;
        for k in 0..n {
            crop_to_all_atom_map.push(start + k as i64);
        }
    }

    let mut swaps: Vec<Vec<ChainSwap>> = Vec::new();
    for (i, chain) in structure.chains.iter().enumerate() {
        let start = chain_atom_idx[i];
        let end = start + chain_atom_num[i];
        if !chain_in_crop[i] {
            continue;
        }
        let mut possible_swaps: Vec<ChainSwap> = Vec::new();
        for (j, chain2) in structure.chains.iter().enumerate() {
            let start2 = chain_atom_idx[j];
            let end2 = start2 + chain_atom_num[j];
            if chain.entity_id == chain2.entity_id && end - start == end2 - start2 {
                possible_swaps.push((start, end, start2, end2, i, j));
            }
        }
        swaps.push(possible_swaps);
    }

    let max_consider = max_n_symmetries.saturating_mul(10).max(1);
    let mut combinations: Vec<Vec<ChainSwap>> = swaps
        .iter()
        .map(|v| v.iter().cloned())
        .multi_cartesian_product()
        .filter(|c| all_different_after_swap(c))
        .take(max_consider)
        .collect();

    if combinations.len() > max_n_symmetries {
        combinations.shuffle(rng);
        combinations.truncate(max_n_symmetries);
    }
    if combinations.is_empty() {
        combinations.push(Vec::new());
    }

    let n_atoms = new_atom_idx;
    let all_coords_arr =
        Array2::from_shape_vec((n_atoms, 3), all_coords).unwrap_or_else(|_| Array2::zeros((0, 3)));
    let all_resolved_arr = Array1::from(all_resolved_mask);
    let crop_map = Array1::from(crop_to_all_atom_map);

    SymmetryFeatures {
        all_coords: all_coords_arr,
        all_resolved_mask: all_resolved_arr,
        crop_to_all_atom_map: crop_map,
        chain_symmetries: combinations,
        amino_acids_symmetries: get_amino_acids_symmetries(tokens),
        ligand_symmetries: get_ligand_symmetries_empty(),
    }
}

/// Like [`process_symmetry_features`]; when `ligand_symmetry_map` is `Some`, fills
/// [`SymmetryFeatures::ligand_symmetries`] via [`get_ligand_symmetries_for_tokens`].
#[inline]
pub fn process_symmetry_features_with_ligand_symmetries(
    structure: &StructureV2Tables,
    tokens: &[TokenData],
    ligand_symmetry_map: Option<&HashMap<String, Vec<Vec<(usize, usize)>>>>,
) -> SymmetryFeatures {
    let mut rng = StdRng::seed_from_u64(0);
    let mut f = get_chain_symmetries(structure, tokens, 100, &mut rng);
    if let Some(m) = ligand_symmetry_map {
        f.ligand_symmetries = get_ligand_symmetries_for_tokens(tokens, m);
    }
    f
}

/// Boltz `process_symmetry_features(cropped, symmetries)`.
#[inline]
pub fn process_symmetry_features(
    structure: &StructureV2Tables,
    tokens: &[TokenData],
) -> SymmetryFeatures {
    process_symmetry_features_with_ligand_symmetries(structure, tokens, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::structure_v2_single_ala;
    use crate::tokenize::boltz2::{tokenize_structure, TokenData};

    #[test]
    fn single_chain_ala_amino_swaps_empty() {
        let s = structure_v2_single_ala();
        let (tok, _) = tokenize_structure(&s, None);
        let aa = get_amino_acids_symmetries(&tok);
        assert!(aa.is_empty(), "ALA has no ref_symmetry groups");
    }

    #[test]
    fn ligand_symmetry_map_maps_atom_indices() {
        use std::collections::HashMap;

        let mut m = HashMap::new();
        m.insert(
            "LIG".to_string(),
            vec![vec![(0_usize, 1_usize), (1_usize, 0_usize)]],
        );
        let nonpoly = chain_type_id("NONPOLYMER").unwrap();
        let lig = TokenData {
            token_idx: 0,
            atom_idx: 0,
            atom_num: 5,
            res_idx: 0,
            res_type: 2,
            res_name: "LIG".to_string(),
            sym_id: 0,
            asym_id: 0,
            entity_id: 0,
            mol_type: nonpoly,
            center_idx: 0,
            disto_idx: 0,
            center_coords: [0.0; 3],
            disto_coords: [0.0; 3],
            resolved_mask: true,
            disto_mask: true,
            modified: false,
            frame_rot: [0.0; 9],
            frame_t: [0.0; 3],
            frame_mask: true,
            cyclic_period: 0,
            affinity_mask: false,
        };
        let sy = get_ligand_symmetries_for_tokens(&[lig], &m);
        assert_eq!(sy.len(), 1);
        assert_eq!(sy[0], vec![vec![(0, 1), (1, 0)]]);
    }

    #[test]
    fn single_chain_ala_chain_symmetry_identity_combo() {
        let s = structure_v2_single_ala();
        let (tok, _) = tokenize_structure(&s, None);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let f = get_chain_symmetries(&s, &tok, 100, &mut rng);
        assert_eq!(f.all_coords.nrows(), 5);
        assert_eq!(f.all_resolved_mask.len(), 5);
        assert_eq!(f.crop_to_all_atom_map.len(), 5);
        assert!(
            f.chain_symmetries.iter().any(|c| c.len() == 1),
            "expected at least one non-empty swap assignment"
        );
        let id = f.chain_symmetries.iter().find(|c| c.len() == 1).unwrap();
        assert_eq!(id[0], (0, 5, 0, 5, 0, 0));
    }
}
