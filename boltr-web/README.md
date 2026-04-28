# boltr-web

Local web UI for Boltr cache status, YAML validation, and **running `boltr predict`** from the browser.

## Security

- Default listen address is **`127.0.0.1:8080`** (local only). Binding a public interface without authentication would expose your machine to **unauthenticated predict execution**; do not expose this server to untrusted networks.
- Predict runs the **`boltr` binary** on the host (resolved via `BOLTR` env, repo `target/release/boltr`, or `PATH`). Ensure that matches your **tch/LibTorch** build.
- Disable predict endpoints with **`BOLTR_WEB_ENABLE_PREDICT=0`** if you only want validation/status.

## Run

```bash
# From repo root, with dev venv on PATH for LibTorch (same as boltr-cli):
scripts/with_dev_venv.sh ./target/release/boltr-web
```

Set **`BOLTR=/path/to/target/release/boltr`** so status checks and predict use the **same** `boltr` binary (required for **`boltr doctor --json`** probes and consistent behavior). The bootstrap script [`Boltr_Boltz_bootstrap`](../Boltr_Boltz_bootstrap) / `./Boltr_go` prints a suggested `export BOLTR=...` line when it finishes.

## RunPod GPU Target

The **RunPod GPU** target supports two modes:

- **Launched on the pod:** if `boltr-web` is started inside your SSH RunPod session and `nvidia-smi` sees a GPU, the RunPod status auto-connects to the local CUDA device. No `BOLTR_RUNPOD_HOST` is needed.
- **Launched elsewhere:** set `BOLTR_RUNPOD_HOST` so the web server can SSH to the pod. Optional variables are `BOLTR_RUNPOD_USER`, `BOLTR_RUNPOD_PORT`, `BOLTR_RUNPOD_KEY`, `BOLTR_RUNPOD_WORKDIR`, `BOLTR_RUNPOD_BOLTR`, and `BOLTR_RUNPOD_CACHE`.

## Preprocess and upstream Boltz

For **`--preprocess boltz`** / **`auto`** (Python upstream Boltz), the server first tries **`boltz` on `PATH`**, then **auto-discovers** a file at common locations (`~/.local/bin/boltz`, `$CONDA_PREFIX/bin/boltz`, `$VIRTUAL_ENV/bin/boltz`, `BOLTR_REPO`’s `.venv/bin/boltz`, and walking parents for `.venv/bin/boltz`). If found, it sets **`--bolt-command`** automatically. Otherwise set **`BOLTR_BOLTZ_COMMAND`** on the server or use the Web UI **Bolt command** field.

The status panel checks this at startup/page load and shows the resolved upstream **Boltz CLI** path, or a missing dependency warning before you submit a prediction. To install it into the repo venv:

```bash
source .venv/bin/activate
pip install boltz
# optional for future bootstrap runs:
BOLTR_INSTALL_BOLTZ=1 bash scripts/bootstrap_dev_venv.sh
```

**Model bootstrap ≠ Boltz CLI:** [`bootstrap_webui_env.sh`](../scripts/bootstrap_webui_env.sh) and **`Boltr_Boltz_bootstrap`** download **weights** into `BOLTZ_CACHE`; they do **not** install the **`boltz`** PyPI package. Install **`boltz`** separately if you rely on **`boltz`** preprocess.

**Form defaults:** The predict form defaults **Preprocess** to **`auto`** (and the API defaults missing `preprocess` to **`auto`**) so a preprocess bundle is generated when possible. Use **`off`** only if you already have **`manifest.json` + `.npz`** next to your YAML.

See the root [**README.md** § Predict: preprocess and structure output](../README.md#predict-preprocess-and-structure-output) for when **mmCIF/PDB** appear and how **`pipeline_complete`** differs from **`predict_step_complete`**.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/status` | Cache file checklist + optional `boltr doctor` |
| POST | `/api/validate` | Multipart YAML validation |
| POST | `/api/predict` | Start predict job; returns `202` + `{ "job_id": "..." }` |
| GET | `/api/predict/:id` | JSON status + log tail + **`structure_output`** (see below) |
| GET | `/api/predict/:id/stream` | SSE log stream (log lines are sanitized so SSE clients never receive raw `\r`/`\n` in a single event) |
| GET | `/api/predict/:id/download` | Tarball of the job **`output/`** directory (`output/` prefix inside the archive). **Success jobs only** (`boltr` exit 0). |

Jobs live under the system temp dir (`…/boltr-web-jobs/<id>/`); completed jobs are removed after **~1 hour**.

### `GET /api/predict/:id` response (when predict is enabled)

Besides `job_id`, `done`, `exit_code`, `success`, and `log_tail`, the response includes **`structure_output`**:

| Field | Meaning |
|-------|---------|
| **`structure_paths`** | Canonical absolute paths to every **`.cif`** / **`.pdb`** found under the job output directory (recursive scan). Empty if none. |
| **`structure_message`** | Short paragraph: either confirmation that structures were found, or **why** no mmCIF/PDB was produced (uses `boltr_predict_complete.txt` when present, e.g. `pipeline_complete` vs `predict_step_complete`). |
| **`completion_status`** | JSON **`status`** field from `boltr_predict_complete.txt` when parsed. |
| **`completion_note`** | JSON **`note`** field from `boltr_predict_complete.txt` when parsed. |

While the job is still running, **`structure_message`** indicates that paths are not final yet.

The web UI shows a **Structure output** panel after the log stream ends (before the download link) so you can see paths and warnings **before** downloading the tarball.

## Tests

```bash
cargo test -p boltr-web
```
