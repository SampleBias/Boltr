//! Affinity cropping for Boltz2 inference.
//!
//! Port of [`boltz.data.crop.affinity.AffinityCropper`](https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/crop/affinity.py):
//! crop tokenized structure around affinity ligand tokens subject to `neighborhood_size`,
//! `max_tokens_protein`, and optional `max_atoms`.

use std::collections::HashSet;

use rand::Rng;

use crate::boltz_const::chain_type_id;
use crate::tokenize::boltz2::{TokenBondV2, TokenData};

/// Main-chain tokenization slice used for affinity cropping (no template token maps).
#[derive(Clone, Debug, PartialEq)]
pub struct AffinityTokenized {
    pub tokens: Vec<TokenData>,
    pub bonds: Vec<TokenBondV2>,
}

/// Affinity cropper for Boltz2 affinity inference (Python `AffinityCropper`).
#[derive(Debug, Clone)]
pub struct AffinityCropper {
    /// Expands contiguous neighborhoods along `res_idx` when the chain is larger than this.
    pub neighborhood_size: usize,
    /// Maximum protein (non-ligand) tokens to retain.
    pub max_tokens_protein: usize,
    /// Maximum total atoms (`None` = unlimited).
    pub max_atoms: Option<usize>,
}

impl Default for AffinityCropper {
    fn default() -> Self {
        Self {
            neighborhood_size: 10,
            max_tokens_protein: 200,
            max_atoms: None,
        }
    }
}

impl AffinityCropper {
    #[must_use]
    pub fn new(
        neighborhood_size: usize,
        max_tokens_protein: usize,
        max_atoms: Option<usize>,
    ) -> Self {
        Self {
            neighborhood_size,
            max_tokens_protein,
            max_atoms,
        }
    }

    /// Crop tokenized data toward affinity ligand neighborhoods (Python `AffinityCropper.crop`).
    ///
    /// Returns a clone when there are no resolved tokens, no ligand `affinity_mask` hits among
    /// resolved tokens, or when cropping fails constraints before adding anything (then identity).
    #[must_use]
    pub fn crop(
        &self,
        data: &AffinityTokenized,
        max_tokens: usize,
        max_atoms: Option<usize>,
        _rng: &mut impl Rng,
    ) -> AffinityTokenized {
        let token_data = &data.tokens;
        let max_atoms = max_atoms.or(self.max_atoms);

        let valid: Vec<usize> = token_data
            .iter()
            .enumerate()
            .filter(|(_, t)| t.resolved_mask)
            .map(|(i, _)| i)
            .collect();

        if valid.is_empty() {
            return data.clone();
        }

        let ligand_coords: Vec<[f32; 3]> = valid
            .iter()
            .filter(|&&vi| token_data[vi].affinity_mask)
            .map(|&vi| token_data[vi].center_coords)
            .collect();

        if ligand_coords.is_empty() {
            return data.clone();
        }

        let nonpoly = chain_type_id("NONPOLYMER").expect("NONPOLYMER id");
        let ligand_ids: HashSet<i32> = valid
            .iter()
            .filter(|&&vi| token_data[vi].mol_type == nonpoly)
            .map(|&vi| token_data[vi].token_idx)
            .collect();

        let dists: Vec<f32> = valid
            .iter()
            .map(|&vi| {
                let c = token_data[vi].center_coords;
                ligand_coords
                    .iter()
                    .map(|lc| {
                        let dx = c[0] - lc[0];
                        let dy = c[1] - lc[1];
                        let dz = c[2] - lc[2];
                        (dx * dx + dy * dy + dz * dz).sqrt()
                    })
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();

        let mut order: Vec<usize> = (0..valid.len()).collect();
        order.sort_by(|&a, &b| {
            dists[a]
                .partial_cmp(&dists[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut cropped: HashSet<i32> = HashSet::new();
        let mut cropped_protein: HashSet<i32> = HashSet::new();
        let mut total_atoms: usize = 0;

        for &valid_row in &order {
            let vi = valid[valid_row];
            let token = &token_data[vi];
            let asym = token.asym_id;

            let chain_indices: Vec<usize> = token_data
                .iter()
                .enumerate()
                .filter(|(_, t)| t.asym_id == asym)
                .map(|(i, _)| i)
                .collect();

            let new_token_idx_set: HashSet<i32> = if chain_indices.len() <= self.neighborhood_size {
                chain_indices
                    .iter()
                    .map(|&i| token_data[i].token_idx)
                    .collect()
            } else {
                let min_init = token.res_idx - self.neighborhood_size as i32;
                let max_init = token.res_idx + self.neighborhood_size as i32;
                let max_token_set: Vec<usize> = chain_indices
                    .iter()
                    .copied()
                    .filter(|&i| {
                        let r = token_data[i].res_idx;
                        r >= min_init && r <= max_init
                    })
                    .collect();

                let mut min_r = token.res_idx;
                let mut max_r = token.res_idx;
                let mut new_sel: Vec<usize> = max_token_set
                    .iter()
                    .copied()
                    .filter(|&i| token_data[i].res_idx == token.res_idx)
                    .collect();

                let mut guard = 0usize;
                while new_sel.len() < self.neighborhood_size && guard < 1_000_000 {
                    guard += 1;
                    min_r -= 1;
                    max_r += 1;
                    new_sel = max_token_set
                        .iter()
                        .copied()
                        .filter(|&i| {
                            let r = token_data[i].res_idx;
                            r >= min_r && r <= max_r
                        })
                        .collect();
                }
                new_sel.iter().map(|&i| token_data[i].token_idx).collect()
            };

            let new_indices: HashSet<i32> =
                new_token_idx_set.difference(&cropped).copied().collect();
            if new_indices.is_empty() {
                continue;
            }

            let new_atoms: usize = new_indices
                .iter()
                .map(|&tid| token_data[tid as usize].atom_num as usize)
                .sum();

            let new_protein: HashSet<i32> = new_indices.difference(&ligand_ids).copied().collect();
            let union_protein = cropped_protein.union(&new_protein).count();

            if new_indices.len() > max_tokens.saturating_sub(cropped.len())
                || max_atoms
                    .map(|ma| total_atoms + new_atoms > ma)
                    .unwrap_or(false)
                || union_protein > self.max_tokens_protein
            {
                break;
            }

            cropped.extend(new_indices.iter().copied());
            total_atoms += new_atoms;
            cropped_protein.extend(new_protein);
        }

        let mut cropped_ids: Vec<i32> = cropped.into_iter().collect();
        cropped_ids.sort_unstable();

        let out_tokens: Vec<TokenData> = cropped_ids
            .iter()
            .map(|&tid| token_data[tid as usize].clone())
            .collect();

        let kept: HashSet<i32> = cropped_ids.iter().copied().collect();
        let out_bonds: Vec<TokenBondV2> = data
            .bonds
            .iter()
            .copied()
            .filter(|&(a, b, _)| kept.contains(&a) && kept.contains(&b))
            .collect();

        AffinityTokenized {
            tokens: out_tokens,
            bonds: out_bonds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn minimal_token(
        idx: i32,
        asym: i32,
        res_idx: i32,
        mol_type: i32,
        center: [f32; 3],
        affinity: bool,
    ) -> TokenData {
        TokenData {
            token_idx: idx,
            atom_idx: idx * 10,
            atom_num: 5,
            res_idx,
            res_type: 2,
            res_name: "ALA".to_string(),
            sym_id: 0,
            asym_id: asym,
            entity_id: asym,
            mol_type,
            center_idx: 0,
            disto_idx: 0,
            center_coords: center,
            disto_coords: center,
            resolved_mask: true,
            disto_mask: true,
            modified: false,
            frame_rot: [0.0; 9],
            frame_t: [0.0; 3],
            frame_mask: true,
            cyclic_period: 0,
            affinity_mask: affinity,
        }
    }

    #[test]
    fn no_ligand_affinity_returns_clone() {
        let cropper = AffinityCropper::default();
        let t = chain_type_id("PROTEIN").unwrap();
        let data = AffinityTokenized {
            tokens: vec![minimal_token(0, 0, 0, t, [0.0, 0.0, 0.0], false)],
            bonds: vec![],
        };
        let mut rng = StdRng::seed_from_u64(0);
        let out = cropper.crop(&data, 50, None, &mut rng);
        assert_eq!(out.tokens.len(), 1);
    }

    #[test]
    fn keeps_ligand_and_nearest_protein() {
        let cropper = AffinityCropper {
            neighborhood_size: 50,
            max_tokens_protein: 10,
            max_atoms: None,
        };
        let prot = chain_type_id("PROTEIN").unwrap();
        let lig = chain_type_id("NONPOLYMER").unwrap();
        // Ligand far from protein token — protein at origin, ligand at (10,0,0)
        let data = AffinityTokenized {
            tokens: vec![
                minimal_token(0, 0, 0, prot, [0.0, 0.0, 0.0], false),
                minimal_token(1, 1, 0, lig, [10.0, 0.0, 0.0], true),
            ],
            bonds: vec![],
        };
        let mut rng = StdRng::seed_from_u64(1);
        let out = cropper.crop(&data, 50, None, &mut rng);
        assert!(out.tokens.len() >= 1);
        assert!(out.tokens.iter().any(|t| t.affinity_mask));
    }

    #[test]
    fn default_matches_python_ctor_numbers() {
        let c = AffinityCropper::default();
        assert_eq!(c.neighborhood_size, 10);
        assert_eq!(c.max_tokens_protein, 200);
        assert!(c.max_atoms.is_none());
    }
}
