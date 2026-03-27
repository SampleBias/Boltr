//! Port of `construct_paired_msa` + `prepare_msa_arrays` from
//! [`featurizerv2.py`](../../../boltz-reference/src/boltz/data/feature/featurizerv2.py).

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use ndarray::Array2;
use rand::seq::index::sample;
use rand::Rng;

use crate::a3m::{A3mMsa, A3mSequenceMeta};
use crate::structure_v2::StructureV2Tables;
use crate::tokenize::boltz2::TokenData;

/// Inner loop equivalent to `_prepare_msa_arrays_inner` (Numba) in Boltz.
pub fn prepare_msa_arrays_inner(
    token_asym_ids: &[i64],
    token_res_idxs: &[i64],
    token_asym_ids_idx: &[i64],
    pairing: &Array2<i64>,
    is_paired: &Array2<i64>,
    deletions: &HashMap<(i64, i64, i64), i64>,
    msa_sequences: &Array2<i64>,
    msa_residues: &Array2<i64>,
    gap_token: i64,
) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n_tokens = token_asym_ids.len();
    let n_pairs = pairing.nrows();
    let mut msa_data = Array2::from_elem((n_tokens, n_pairs), gap_token);
    let mut paired_data = Array2::zeros((n_tokens, n_pairs));
    let mut del_data = Array2::zeros((n_tokens, n_pairs));

    for token_idx in 0..n_tokens {
        let chain_id_idx = token_asym_ids_idx[token_idx] as usize;
        let chain_id = token_asym_ids[token_idx];
        let res_idx = token_res_idxs[token_idx];

        for pair_idx in 0..n_pairs {
            let seq_idx = pairing[[pair_idx, chain_id_idx]];
            paired_data[[token_idx, pair_idx]] = is_paired[[pair_idx, chain_id_idx]];

            if seq_idx != -1 {
                let res_start = msa_sequences[[chain_id_idx, seq_idx as usize]];
                let res_type = msa_residues[[chain_id_idx, (res_start + res_idx) as usize]];
                let k = (chain_id, seq_idx, res_idx);
                if let Some(&dv) = deletions.get(&k) {
                    del_data[[token_idx, pair_idx]] = dv;
                }
                msa_data[[token_idx, pair_idx]] = res_type;
            }
        }
    }

    (msa_data, del_data, paired_data)
}

fn chain_ids_unique_sorted(tokens: &[TokenData]) -> Vec<i32> {
    tokens
        .iter()
        .map(|t| t.asym_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn dummy_msa_for_chain(structure: &StructureV2Tables, chain: &crate::structure_v2::ChainRow) -> A3mMsa {
    let res_start = chain.res_idx as usize;
    let res_end = res_start + chain.res_num as usize;
    let residues: Vec<i32> = structure.residues[res_start..res_end]
        .iter()
        .map(|r| i32::from(r.res_type))
        .collect();
    let sequences = vec![A3mSequenceMeta {
        seq_idx: 0,
        taxonomy_id: -1,
        res_start: 0,
        res_end: residues.len(),
        del_start: 0,
        del_end: 0,
    }];
    A3mMsa {
        residues,
        deletions: vec![],
        sequences,
    }
}

fn prepare_msa_arrays(
    tokens: &[TokenData],
    pairing: &[HashMap<i32, i64>],
    is_paired: &[HashMap<i32, i64>],
    deletions: &HashMap<(i64, i64, i64), i64>,
    msa: &HashMap<i32, A3mMsa>,
    chain_ids: &[i32],
) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n_pairs = pairing.len();
    let chain_id_to_idx: HashMap<i32, usize> = chain_ids
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    let token_asym_ids: Vec<i64> = tokens.iter().map(|t| i64::from(t.asym_id)).collect();
    let token_res_idxs: Vec<i64> = tokens.iter().map(|t| i64::from(t.res_idx)).collect();
    let token_asym_ids_idx: Vec<i64> = token_asym_ids
        .iter()
        .map(|&a| chain_id_to_idx[&(a as i32)] as i64)
        .collect();

    let mut pairing_arr = Array2::zeros((n_pairs, chain_ids.len()));
    for (i, row) in pairing.iter().enumerate() {
        for &cid in chain_ids {
            pairing_arr[[i, chain_id_to_idx[&cid]]] = row.get(&cid).copied().unwrap_or(-1);
        }
    }
    let mut is_paired_arr = Array2::zeros((n_pairs, chain_ids.len()));
    for (i, row) in is_paired.iter().enumerate() {
        for &cid in chain_ids {
            is_paired_arr[[i, chain_id_to_idx[&cid]]] = row.get(&cid).copied().unwrap_or(0);
        }
    }

    let max_seq_len = chain_ids
        .iter()
        .map(|c| msa[c].sequences.len())
        .max()
        .unwrap_or(0);
    let mut msa_sequences = Array2::from_elem((chain_ids.len(), max_seq_len), -1i64);
    for (ci, &chain_id) in chain_ids.iter().enumerate() {
        let cm = &msa[&chain_id];
        for (i, seq) in cm.sequences.iter().enumerate() {
            msa_sequences[[ci, i]] = i64::try_from(seq.res_start).unwrap_or(0);
        }
    }

    let max_residues_len = chain_ids
        .iter()
        .map(|c| msa[c].residues.len())
        .max()
        .unwrap_or(0);
    let mut msa_residues = Array2::from_elem((chain_ids.len(), max_residues_len), -1i64);
    for (ci, &chain_id) in chain_ids.iter().enumerate() {
        let residues = &msa[&chain_id].residues;
        for (j, &r) in residues.iter().enumerate() {
            msa_residues[[ci, j]] = i64::from(r);
        }
    }

    let gap = i64::from(crate::boltz_const::token_id("-").expect("gap token"));
    prepare_msa_arrays_inner(
        &token_asym_ids,
        &token_res_idxs,
        &token_asym_ids_idx,
        &pairing_arr,
        &is_paired_arr,
        deletions,
        &msa_sequences,
        &msa_residues,
        gap,
    )
}

/// Build paired MSA matrices `(n_tokens, n_pairs)` before the transpose in `process_msa_features`.
pub fn construct_paired_msa(
    tokens: &[TokenData],
    structure: &StructureV2Tables,
    msas: &HashMap<i32, A3mMsa>,
    rng: &mut impl Rng,
    max_seqs: usize,
    max_pairs: usize,
    max_total: usize,
    random_subset: bool,
) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    assert!(
        tokens.windows(2).all(|w| w[0].asym_id <= w[1].asym_id),
        "tokens must be sorted by asym_id (Boltz invariant)"
    );

    let chain_ids: Vec<i32> = chain_ids_unique_sorted(tokens);

    let mut msa_map: HashMap<i32, A3mMsa> = HashMap::new();
    for &chain_id in &chain_ids {
        let chain = structure
            .chains
            .iter()
            .find(|c| c.asym_id == chain_id)
            .expect("chain asym_id");

        let res_start = chain.res_idx as usize;
        let res_end = res_start + chain.res_num as usize;
        let residues_slice = &structure.residues[res_start..res_end];

        if let Some(user_msa) = msas.get(&chain_id) {
            let first = &user_msa.sequences[0];
            let msa_residues = &user_msa.residues
                [first.res_start..first.res_end.min(user_msa.residues.len())];
            let mut ok = residues_slice.len() == msa_residues.len();
            if ok {
                for (r, mr) in residues_slice.iter().zip(msa_residues.iter()) {
                    if r.res_type as i32 != *mr {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                msa_map.insert(chain_id, user_msa.clone());
            } else {
                msa_map.insert(chain_id, dummy_msa_for_chain(structure, chain));
            }
        } else {
            msa_map.insert(chain_id, dummy_msa_for_chain(structure, chain));
        }
    }

    // Taxonomy map: taxon -> list of (chain_id, seq_idx)
    let mut taxonomy_map: HashMap<i32, Vec<(i32, i64)>> = HashMap::new();
    for (&chain_id, chain_msa) in &msa_map {
        for seq in &chain_msa.sequences {
            if seq.taxonomy_id == -1 {
                continue;
            }
            taxonomy_map
                .entry(seq.taxonomy_id)
                .or_default()
                .push((chain_id, i64::from(seq.seq_idx)));
        }
    }
    taxonomy_map.retain(|_, v| v.len() > 1);
    let mut taxonomy_sorted: Vec<(i32, Vec<(i32, i64)>)> = taxonomy_map.into_iter().collect();
    taxonomy_sorted.sort_by(|a, b| {
        let ca: HashSet<i32> = a.1.iter().map(|x| x.0).collect();
        let cb: HashSet<i32> = b.1.iter().map(|x| x.0).collect();
        cb.len().cmp(&ca.len())
    });

    let mut visited: HashSet<(i32, i64)> = HashSet::new();
    for (_, items) in &taxonomy_sorted {
        for &(c, s) in items {
            visited.insert((c, s));
        }
    }

    let mut available: HashMap<i32, VecDeque<i64>> = HashMap::new();
    for &c in &chain_ids {
        let nseq = msa_map[&c].sequences.len();
        let dq: VecDeque<i64> = (1..nseq as i64)
            .filter(|&i| !visited.contains(&(c, i)))
            .collect();
        available.insert(c, dq);
    }

    let mut is_paired: Vec<HashMap<i32, i64>> = Vec::new();
    let mut pairing: Vec<HashMap<i32, i64>> = Vec::new();

    let mut row_paired = HashMap::new();
    let mut row_pairing = HashMap::new();
    for &c in &chain_ids {
        row_paired.insert(c, 1);
        row_pairing.insert(c, 0);
    }
    is_paired.push(row_paired);
    pairing.push(row_pairing);

    for (_, pairs) in &taxonomy_sorted {
        let mut chain_occurences: HashMap<i32, Vec<i64>> = HashMap::new();
        for &(chain_id, seq_idx) in pairs {
            chain_occurences.entry(chain_id).or_default().push(seq_idx);
        }
        let max_occurences = chain_occurences
            .values()
            .map(|v| v.len())
            .max()
            .unwrap_or(0);
        for i in 0..max_occurences {
            let mut row_pairing = HashMap::new();
            let mut row_is_paired = HashMap::new();
            for (&chain_id, seq_idxs) in &chain_occurences {
                let idx = i % seq_idxs.len();
                let seq_idx = seq_idxs[idx];
                row_pairing.insert(chain_id, seq_idx);
                row_is_paired.insert(chain_id, 1);
            }
            for &chain_id in &chain_ids {
                if !row_pairing.contains_key(&chain_id) {
                    row_is_paired.insert(chain_id, 0);
                    if let Some(av) = available.get_mut(&chain_id) {
                        if let Some(s) = av.pop_front() {
                            row_pairing.insert(chain_id, s);
                        } else {
                            row_pairing.insert(chain_id, -1);
                        }
                    } else {
                        row_pairing.insert(chain_id, -1);
                    }
                }
            }
            pairing.push(row_pairing);
            is_paired.push(row_is_paired);
            if pairing.len() >= max_pairs {
                break;
            }
        }
        if pairing.len() >= max_pairs {
            break;
        }
    }

    let max_left = chain_ids
        .iter()
        .map(|c| available[c].len())
        .max()
        .unwrap_or(0);
    let n_add = (max_total - pairing.len()).min(max_left);
    for _ in 0..n_add {
        let mut row_pairing = HashMap::new();
        let mut row_is_paired = HashMap::new();
        for &chain_id in &chain_ids {
            row_is_paired.insert(chain_id, 0);
            if let Some(av) = available.get_mut(&chain_id) {
                if let Some(s) = av.pop_front() {
                    row_pairing.insert(chain_id, s);
                } else {
                    row_pairing.insert(chain_id, -1);
                }
            } else {
                row_pairing.insert(chain_id, -1);
            }
        }
        pairing.push(row_pairing);
        is_paired.push(row_is_paired);
        if pairing.len() >= max_total {
            break;
        }
    }

    if random_subset && pairing.len() > max_seqs {
        let num_seqs = pairing.len();
        let need = max_seqs.saturating_sub(1);
        let pool = num_seqs.saturating_sub(1);
        if pool > 0 && need > 0 {
            let k = need.min(pool);
            let idxs = sample(rng, pool, k);
            let mut new_p = vec![pairing[0].clone()];
            let mut new_i = vec![is_paired[0].clone()];
            for i in idxs {
                let j = i + 1;
                new_p.push(pairing[j].clone());
                new_i.push(is_paired[j].clone());
            }
            pairing = new_p;
            is_paired = new_i;
        }
    } else if pairing.len() > max_seqs {
        pairing.truncate(max_seqs);
        is_paired.truncate(max_seqs);
    }

    let mut deletions: HashMap<(i64, i64, i64), i64> = HashMap::new();
    for (&chain_id, chain_msa) in &msa_map {
        let cid = i64::from(chain_id);
        for seq in &chain_msa.sequences {
            let sidx = i64::from(seq.seq_idx);
            let del_slice = &chain_msa.deletions[seq.del_start..seq.del_end.min(chain_msa.deletions.len())];
            for d in del_slice {
                let res_idx = i64::from(d.0);
                let delv = i64::from(d.1);
                deletions.insert((cid, sidx, res_idx), delv);
            }
        }
    }

    prepare_msa_arrays(
        tokens,
        &pairing,
        &is_paired,
        &deletions,
        &msa_map,
        &chain_ids,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_inner_tiny_matches_manual() {
        let token_asym_ids = [0_i64, 0_i64];
        let token_res_idxs = [0_i64, 1_i64];
        let token_asym_ids_idx = [0_i64, 0_i64];
        let pairing = ndarray::arr2(&[[0_i64], [1_i64]]);
        let is_paired = ndarray::arr2(&[[1_i64], [0_i64]]);
        let deletions: HashMap<(i64, i64, i64), i64> = HashMap::new();
        let msa_sequences = ndarray::arr2(&[[0_i64, 1_i64]]);
        // Row 0 needs index res_start + res_idx for seq 1, token res_idx 1 => 1 + 1 = 2
        let msa_residues = ndarray::arr2(&[[5_i64, 6_i64, 7_i64]]);
        let (msa, del, paired) = prepare_msa_arrays_inner(
            &token_asym_ids,
            &token_res_idxs,
            &token_asym_ids_idx,
            &pairing,
            &is_paired,
            &deletions,
            &msa_sequences,
            &msa_residues,
            1,
        );
        assert_eq!(msa[[0, 0]], 5);
        assert_eq!(msa[[1, 0]], 6);
        assert_eq!(msa[[0, 1]], 6);
        assert_eq!(msa[[1, 1]], 7);
        assert_eq!(paired[[0, 0]], 1);
        assert_eq!(paired[[0, 1]], 0);
        assert_eq!(del.sum(), 0);
    }
}
