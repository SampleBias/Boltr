# Predict: preprocess extras (CCD mols, constraints, ensemble)

This document matches the `boltr predict` flags added for the folding-accuracy rollout: optional paths into [`load_input`](../boltr-io/src/inference_dataset.rs) and optional multi-conformer atom featurization.

## CLI flags

| Flag | Purpose |
|------|---------|
| `--extra-mols-dir <DIR>` | Directory of CCD `*.json` files (heavy-atom graphs). Passed to `load_input` as `extra_mols_dir`. Improves **ligand reference chemistry** in `process_atom_features` and fills **ligand symmetry** groups (heuristic) when combined with tokenization. |
| `--constraints-dir <DIR>` | Directory containing `{record_id}.npz` residue constraint files (same layout as Boltz preprocess). Passed to `load_input` as `constraints_dir`. Featurizer emits constraint tensors in the trunk `FeatureBatch`. |
| `--preprocess-auto-extras` | When set, if the above are omitted, discover under the YAML parent directory: `mols/` or `extra_mols/` (must contain at least one `*.json`), and `constraints/` or `residue_constraints/` (directory exists). Default is **off** so behavior stays unchanged unless you opt in. |
| `--ensemble-ref single\|multi` | `single` (default): one ensemble index `0` for atom featurization. `multi`: up to five indices, bounded by `StructureV2Tables::num_ensemble_conformers`. **Experimental** — verify tensor shapes against your checkpoint before relying on `multi` in production. |

## Minimal before/after comparison (Phase D)

1. Run a ligand-containing job **without** extras (baseline):

   ```bash
   boltr predict input.yaml --output ./out_a
   ```

2. Place or generate CCD JSON files under `mols/` next to the YAML (or pass `--extra-mols-dir /path/to/mols`).

3. Run again with extras:

   ```bash
   boltr predict input.yaml --output ./out_b --extra-mols-dir ./mols
   # or: --preprocess-auto-extras
   ```

4. Compare outputs (e.g. confidence JSON under `predictions/<id>/`, or structure coordinates). Differences indicate the featurizer path changed; **magnitude** depends on the ligand and checkpoint.

## Residue constraints vs diffusion (Phase C)

- **Rust:** `trunk_smoke_feature_batch_from_inference_input` merges residue-constraint tensors from `process_residue_constraint_features` into the `FeatureBatch`.
- **Torch bridge:** `OwnedPredictTensors` / `predict_step` currently take a **fixed** set of tensors; constraint arrays are **not** passed into `Boltz2Model::predict_step` or diffusion sampling in this codebase.
- **Python Boltz:** constraints may be used in **training** or auxiliary objectives; confirm in `boltz-reference` for your checkpoint before expecting constraint **enforcement** at inference.

Extending the backend to feed constraint tensors into diffusion is a separate change (parity tests + checkpoint alignment).

## See also

- [boltz-reference/docs/prediction.md](../boltz-reference/docs/prediction.md)
- [vybrid_todo.md](../vybrid_todo.md) — backlog for multi-ensemble defaults and diffusion-side constraints
