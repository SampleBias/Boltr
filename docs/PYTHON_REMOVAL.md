# Phased removal of vendored Python (`boltz-reference/`)

The tree under [`boltz-reference/`](../boltz-reference/) is the upstream Boltz implementation used as the **behavioral spec** and for golden tests. Do **not** delete large subtrees until Rust code has tests that prove parity for that slice.

## Gates

| Milestone | Safe action |
|-----------|-------------|
| Featurizer / tokenizer outputs match Python on fixtures | Trim only redundant documentation if desired; **keep** Python for model reference. |
| `boltr predict` matches Python on a fixed fixture set | Replace `boltz-reference/` with a **git submodule** to [jwohlwend/boltz](https://github.com/jwohlwend/boltz), or remove the vendor copy and rely on pinned fixtures + checkpoint hashes. |
| Long term | Keep a **minimal** Python environment (or submodule) only for checkpoint export and occasional regression, not as the shipped user binary. |

## Rationale

Premature deletion breaks the ability to diff behavior, export weights, and regenerate golden tensors. Removal should track **completed** Rust modules, not the roadmap.
