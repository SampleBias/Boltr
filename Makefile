# Boltr — checkpoint export, hparams, and safetensors verification (Phase 0 tooling).
# Requires: Python 3 with torch + safetensors for export targets; scripts/cargo-tch for verify.
#
# Examples:
#   make export-safetensors CKPT=~/bolt/boltz2.ckpt OUT=/tmp/m.safetensors
#   make verify-safetensors SAFETENSORS=/tmp/m.safetensors
#   make export-hparams CKPT=~/bolt/boltz2.ckpt HPARAMS_JSON=/tmp/h.json

.PHONY: help export-safetensors export-hparams verify-safetensors compare-ckpt-safetensors-counts verify-constraints-npz

help:
	@echo "Targets:"
	@echo "  export-safetensors CKPT=path OUT=path   - Lightning .ckpt -> .safetensors"
	@echo "  export-hparams CKPT=path HPARAMS_JSON=path - Lightning hyper_parameters -> JSON"
	@echo "  verify-safetensors SAFETENSORS=path    - VarStore keys vs file (boltr-backend-tch)"
	@echo "  compare-ckpt-safetensors-counts CKPT=path SAFETENSORS=path - tensor key counts"

OUT ?= boltz2_export.safetensors
export-safetensors:
	@test -n "$(CKPT)" || (echo "Usage: make export-safetensors CKPT=path/to.ckpt [OUT=out.safetensors]"; exit 1)
	python3 scripts/export_checkpoint_to_safetensors.py "$(CKPT)" "$(OUT)" --strip-prefix model.

HPARAMS_JSON ?= boltz2_hparams.json
export-hparams:
	@test -n "$(CKPT)" || (echo "Usage: make export-hparams CKPT=path/to.ckpt [HPARAMS_JSON=out.json]"; exit 1)
	python3 scripts/export_hparams_from_ckpt.py "$(CKPT)" "$(HPARAMS_JSON)"

verify-safetensors:
	@test -n "$(SAFETENSORS)" || (echo "Usage: make verify-safetensors SAFETENSORS=path/to.safetensors"; exit 1)
	scripts/cargo-tch run -p boltr-backend-tch --bin verify_boltz2_safetensors --features tch-backend -- "$(SAFETENSORS)"

compare-ckpt-safetensors-counts:
	@test -n "$(CKPT)" -a -n "$(SAFETENSORS)" || (echo "Usage: make compare-ckpt-safetensors-counts CKPT=... SAFETENSORS=..."; exit 1)
	python3 scripts/compare_ckpt_safetensors_counts.py "$(CKPT)" "$(SAFETENSORS)"

NPZ ?=
verify-constraints-npz:
	@test -n "$(NPZ)" || (echo "Usage: make verify-constraints-npz NPZ=path/toconstraints.npz"; exit 1)
	python3 scripts/verify_constraints_npz_layout.py "$(NPZ)"
