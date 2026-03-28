//! Port of `process_template_features` / `compute_template_features` from Boltz
//! `featurizerv2.py` (template stacking + per-row features).
//!
//! [`TemplateAlignment`] mirrors manifest `TemplateInfo` without depending on `inference_dataset`.

use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use ndarray::{concatenate, Array2, Array3, Array4, Axis};

use crate::boltz_const::NUM_TOKENS;
use crate::featurizer::dummy_templates::{load_dummy_templates_features, DummyTemplateTensors};
use crate::structure_v2::StructureV2Tables;
use crate::tokenize::boltz2::TokenData;

/// Alignment metadata for one template block (Boltz `TemplateInfo`).
#[derive(Clone, Debug)]
pub struct TemplateAlignment {
    pub name: String,
    pub query_chain: String,
    pub query_st: i32,
    pub query_en: i32,
    pub template_chain: String,
    pub template_st: i32,
    pub template_en: i32,
    pub force: bool,
    pub threshold: Option<f32>,
}

struct RowToken {
    q_idx: usize,
    pdb_id: usize,
    tmpl_token: TokenData,
}

fn asym_id_for_chain_name(structure: &StructureV2Tables, chain_name: &str) -> Result<i32> {
    for (ci, ch) in structure.chains.iter().enumerate() {
        if !structure.chain_mask.get(ci).copied().unwrap_or(false) {
            continue;
        }
        if ch.name == chain_name {
            return Ok(ch.asym_id);
        }
    }
    bail!("chain name {chain_name:?} not found in structure chains")
}

/// Group template rows by unique `name` (insertion order of first occurrence), matching Python
/// `defaultdict` + iteration order.
fn group_templates_by_name(record: &[TemplateAlignment]) -> Vec<(String, Vec<&TemplateAlignment>)> {
    let mut out: Vec<(String, Vec<&TemplateAlignment>)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for t in record {
        if let Some(&i) = index.get(&t.name) {
            out[i].1.push(t);
        } else {
            index.insert(t.name.clone(), out.len());
            out.push((t.name.clone(), vec![t]));
        }
    }
    out
}

fn build_row_tokens(
    template: &TemplateAlignment,
    template_id: usize,
    query_tokens: &[TokenData],
    query_structure: &StructureV2Tables,
    template_structure: &StructureV2Tables,
    template_tokens: &[TokenData],
    num_slots: usize,
) -> Result<Vec<RowToken>> {
    if template.query_chain.is_empty() || template.template_chain.is_empty() {
        return Ok(vec![]);
    }
    let q_asym = asym_id_for_chain_name(query_structure, &template.query_chain)?;
    let tmpl_asym = asym_id_for_chain_name(template_structure, &template.template_chain)?;
    let offset = template.template_st - template.query_st;

    let mut q_indices: HashMap<i32, i32> = HashMap::new();
    for t in query_tokens {
        if t.asym_id == q_asym {
            q_indices.insert(t.res_idx, t.token_idx);
        }
    }

    let mut row_tokens = Vec::new();
    for t in template_tokens {
        if t.asym_id != tmpl_asym {
            continue;
        }
        let ridx = t.res_idx - offset;
        if let Some(&q_idx) = q_indices.get(&ridx) {
            let qi = usize::try_from(q_idx)
                .map_err(|_| anyhow::anyhow!("bad query token index {q_idx}"))?;
            if qi < num_slots {
                row_tokens.push(RowToken {
                    q_idx: qi,
                    pdb_id: template_id,
                    tmpl_token: t.clone(),
                });
            }
        }
    }
    Ok(row_tokens)
}

fn one_hot_token_class(num_classes: usize, class_id: i64) -> Vec<f32> {
    let mut v = vec![0.0f32; num_classes];
    if class_id >= 0 {
        let u = class_id as usize;
        if u < num_classes {
            v[u] = 1.0;
        }
    }
    v
}

/// One template row (`tdim == 1` in output), `num_slots` token positions (padded).
fn compute_template_features_single(
    query_tokens: &[TokenData],
    query_structure: &StructureV2Tables,
    row_tokens: &[RowToken],
    num_slots: usize,
) -> DummyTemplateTensors {
    let c = NUM_TOKENS;
    let mut template_restype = Array3::<f32>::zeros((1, num_slots, c));
    let mut template_frame_rot = Array4::<f32>::zeros((1, num_slots, 3, 3));
    let mut template_frame_t = Array3::<f32>::zeros((1, num_slots, 3));
    let mut template_cb = Array3::<f32>::zeros((1, num_slots, 3));
    let mut template_ca = Array3::<f32>::zeros((1, num_slots, 3));
    let mut template_mask_cb = Array2::<f32>::zeros((1, num_slots));
    let mut template_mask_frame = Array2::<f32>::zeros((1, num_slots));
    let mut template_mask = Array2::<f32>::zeros((1, num_slots));
    let query_to_template = Array2::<i64>::zeros((1, num_slots));
    let mut visibility_ids = Array2::<f32>::zeros((1, num_slots));

    let mut asym_id_to_pdb_id: HashMap<i32, usize> = HashMap::new();
    for rt in row_tokens {
        let q_tok = &query_tokens[rt.q_idx];
        asym_id_to_pdb_id.insert(q_tok.asym_id, rt.pdb_id);
        let tmpl = &rt.tmpl_token;
        let idx = rt.q_idx;
        let oh = one_hot_token_class(c, i64::from(tmpl.res_type));
        for (k, v) in oh.iter().enumerate() {
            template_restype[[0, idx, k]] = *v;
        }
        for i in 0..3 {
            for j in 0..3 {
                template_frame_rot[[0, idx, i, j]] = tmpl.frame_rot[i * 3 + j];
            }
        }
        template_frame_t[[0, idx, 0]] = tmpl.frame_t[0];
        template_frame_t[[0, idx, 1]] = tmpl.frame_t[1];
        template_frame_t[[0, idx, 2]] = tmpl.frame_t[2];
        template_cb[[0, idx, 0]] = tmpl.disto_coords[0];
        template_cb[[0, idx, 1]] = tmpl.disto_coords[1];
        template_cb[[0, idx, 2]] = tmpl.disto_coords[2];
        template_ca[[0, idx, 0]] = tmpl.center_coords[0];
        template_ca[[0, idx, 1]] = tmpl.center_coords[1];
        template_ca[[0, idx, 2]] = tmpl.center_coords[2];
        template_mask_cb[[0, idx]] = if tmpl.disto_mask { 1.0 } else { 0.0 };
        template_mask_frame[[0, idx]] = if tmpl.frame_mask { 1.0 } else { 0.0 };
        template_mask[[0, idx]] = 1.0;
    }

    let n_fill = query_tokens.len().min(num_slots);
    for i in 0..n_fill {
        let asym_id = query_tokens[i].asym_id;
        if let Some(&pdb_id) = asym_id_to_pdb_id.get(&asym_id) {
            visibility_ids[[0, i]] = pdb_id as f32;
        }
    }

    let mut seen_asym: HashSet<i32> = HashSet::new();
    for (ci, ch) in query_structure.chains.iter().enumerate() {
        if !query_structure.chain_mask.get(ci).copied().unwrap_or(false) {
            continue;
        }
        seen_asym.insert(ch.asym_id);
    }
    for asym_id in seen_asym {
        if asym_id_to_pdb_id.contains_key(&asym_id) {
            continue;
        }
        for i in 0..n_fill {
            if query_tokens[i].asym_id == asym_id {
                visibility_ids[[0, i]] = -1.0 - asym_id as f32;
            }
        }
    }

    DummyTemplateTensors {
        template_restype,
        template_frame_rot,
        template_frame_t,
        template_cb,
        template_ca,
        template_mask_cb,
        template_mask_frame,
        template_mask,
        query_to_template,
        visibility_ids,
    }
}

fn concat_templates(a: DummyTemplateTensors, b: DummyTemplateTensors) -> Result<DummyTemplateTensors> {
    Ok(DummyTemplateTensors {
        template_restype: concatenate(
            Axis(0),
            &[a.template_restype.view(), b.template_restype.view()],
        )?
        .into_owned(),
        template_frame_rot: concatenate(
            Axis(0),
            &[a.template_frame_rot.view(), b.template_frame_rot.view()],
        )?
        .into_owned(),
        template_frame_t: concatenate(
            Axis(0),
            &[a.template_frame_t.view(), b.template_frame_t.view()],
        )?
        .into_owned(),
        template_cb: concatenate(Axis(0), &[a.template_cb.view(), b.template_cb.view()])?.into_owned(),
        template_ca: concatenate(Axis(0), &[a.template_ca.view(), b.template_ca.view()])?.into_owned(),
        template_mask_cb: concatenate(
            Axis(0),
            &[a.template_mask_cb.view(), b.template_mask_cb.view()],
        )?
        .into_owned(),
        template_mask_frame: concatenate(
            Axis(0),
            &[a.template_mask_frame.view(), b.template_mask_frame.view()],
        )?
        .into_owned(),
        template_mask: concatenate(Axis(0), &[a.template_mask.view(), b.template_mask.view()])?
            .into_owned(),
        query_to_template: concatenate(
            Axis(0),
            &[a.query_to_template.view(), b.query_to_template.view()],
        )?
        .into_owned(),
        visibility_ids: concatenate(
            Axis(0),
            &[a.visibility_ids.view(), b.visibility_ids.view()],
        )?
        .into_owned(),
    })
}

/// Concatenate along template dimension (axis 0). Fails if `rows` is empty.
pub fn stack_template_feature_rows(rows: Vec<DummyTemplateTensors>) -> Result<DummyTemplateTensors> {
    let mut it = rows.into_iter();
    let first = it.next().ok_or_else(|| anyhow::anyhow!("no template rows"))?;
    let mut acc = first;
    for r in it {
        acc = concat_templates(acc, r)?;
    }
    Ok(acc)
}

/// Pad template axis with zero blocks until `target_tdim` rows exist.
pub fn pad_template_tdim(t: DummyTemplateTensors, target_tdim: usize) -> DummyTemplateTensors {
    let cur = t.template_restype.shape()[0];
    if cur >= target_tdim {
        return t;
    }
    let pad = target_tdim - cur;
    let n = t.template_restype.shape()[1];
    let z = load_dummy_templates_features(pad, n);
    concat_templates(t, z).expect("pad_template_tdim concat")
}

/// Boltz `process_template_features` for one example (no batch axis). Stacks one row per unique
/// template **name** in `record_templates`, in first-seen order.
pub fn process_template_features(
    query_tokens: &[TokenData],
    query_structure: &StructureV2Tables,
    template_structures: &HashMap<String, StructureV2Tables>,
    template_tokens_map: &HashMap<String, Vec<TokenData>>,
    record_templates: &[TemplateAlignment],
    max_tokens: usize,
) -> Result<DummyTemplateTensors> {
    if record_templates.is_empty() {
        bail!("empty record_templates");
    }
    let num_slots = max_tokens.max(query_tokens.len());
    let grouped = group_templates_by_name(record_templates);
    let mut rows: Vec<DummyTemplateTensors> = Vec::new();

    for (template_id, (name, group)) in grouped.iter().enumerate() {
        let tmpl_struct = template_structures
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing template structure for name {name:?}"))?;
        let tmpl_toks = template_tokens_map
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing template tokens for name {name:?}"))?;

        let mut all_row: Vec<RowToken> = Vec::new();
        for template in group.iter().copied() {
            all_row.extend(build_row_tokens(
                template,
                template_id,
                query_tokens,
                query_structure,
                tmpl_struct,
                tmpl_toks,
                num_slots,
            )?);
        }
        rows.push(compute_template_features_single(
            query_tokens,
            query_structure,
            &all_row,
            num_slots,
        ));
    }

    stack_template_feature_rows(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::structure_v2_single_ala;
    use crate::tokenize::boltz2::tokenize_structure;

    #[test]
    fn identity_ala_template_fills_mask_and_coords() {
        let mut s = structure_v2_single_ala();
        s.chains[0].name = "A".to_string();
        let (q_tok, _) = tokenize_structure(&s, None);
        let (tmpl_tok, _) = tokenize_structure(&s, None);
        let mut structures = HashMap::new();
        structures.insert("tmpl1".to_string(), s.clone());
        let mut tok_map = HashMap::new();
        tok_map.insert("tmpl1".to_string(), tmpl_tok);
        let record = vec![TemplateAlignment {
            name: "tmpl1".to_string(),
            query_chain: "A".to_string(),
            query_st: 0,
            query_en: 0,
            template_chain: "A".to_string(),
            template_st: 0,
            template_en: 0,
            force: false,
            threshold: None,
        }];
        let out = process_template_features(
            &q_tok,
            &s,
            &structures,
            &tok_map,
            &record,
            q_tok.len(),
        )
        .expect("process_template_features");
        assert_eq!(out.template_restype.shape()[0], 1);
        assert!(out.template_mask[[0, 0]] > 0.5);
        assert!(out.template_mask_cb[[0, 0]] > 0.5);
        assert!(out.template_mask_frame[[0, 0]] > 0.5);
        assert!(out.visibility_ids[[0, 0]] >= 0.0);
    }
}
