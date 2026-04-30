# Prediction Parity Baseline

This document tracks the minimum parity matrix for Boltr prediction reliability work.
The goal is to compare Boltr against upstream Boltz/Boltz2 with the same input,
checkpoint family, preprocess artifacts, and inference settings before changing
model behavior.

## Fixture Matrix

| Fixture | Purpose | Preferred preprocess | Required outputs |
| --- | --- | --- | --- |
| `protein_minimal` | Protein-only structure smoke and confidence ranking | `native` and `boltz` | structure, `boltr_predict_complete.txt`, resolved predict args |
| `protein_ligand_affinity` | Boltz2 affinity path, ligand mask, MW correction | `boltz` | structure, affinity JSON, selected sample metadata |
| `template_complex` | Template feature parity | `boltz` | structure, template usage status, confidence metrics |
| `constrained_complex` | Potentials/steering behavior | `boltz` | structure or explicit unsupported status |

## Current Boltr Status

- The primary inference path is `load_input` -> collate -> `Boltz2Model::predict_step`.
- RunPod currently executes the same CLI path over SSH; it should not change model
  numerics except through device, timeout, or transfer failure behavior.
- RunPod applies quality-oriented defaults by default: recycling `3`, sampling steps
  `200`, diffusion samples `2`, max parallel samples `1`, and high-fidelity preprocess.
- Template tensors are not passed into the current Rust predict bridge.
- Potentials/steering are not passed into the current Rust predict bridge.
- Affinity is implemented for the single-output Rust path; paired/ensemble affinity
  fields require an explicit parity decision.

## Comparison Checklist

For each fixture, store the upstream and Boltr command lines plus the following:

- resolved predict args (`recycling_steps`, `sampling_steps`, `diffusion_samples`,
  `max_parallel_samples`)
- preprocess mode and whether upstream Boltz produced the bundle
- selected structure sample and ranking metric
- confidence availability and key confidence scores
- affinity value, binary probability, MW correction status, and selected sample
- completion status (`predict_step_complete` vs fallback/reference status)

## Opt-In Gate

Full numerical comparison is GPU and checkpoint dependent, so the parity gate should
be opt-in. A recommended local convention is:

```bash
BOLTR_RUN_PREDICT_PARITY=1 scripts/cargo-tch test -p boltr-cli --features tch predict_parity_fixture_matrix_is_documented
```

The always-on test should only verify that this baseline stays present and names the
fixtures; the opt-in path can run real upstream Boltz/Boltz2 comparisons when the
environment has the required GPU, checkpoints, and `boltz` CLI.
