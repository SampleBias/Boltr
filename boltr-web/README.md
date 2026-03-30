# boltr-web

Local web UI for Boltr cache status, YAML validation, and **running `boltr predict`** from the browser.

## Security

- Default listen address is **`127.0.0.1:8080`** (local only). Binding a public interface without authentication would expose your machine to **unauthenticated predict execution**; do not expose this server to untrusted networks.
- Predict runs the **`boltr` binary** on the host (resolved via `BOLTR` env, repo `target/release/boltr`, or `PATH`). Ensure that matches your tch/LibTorch build.
- Disable predict endpoints with **`BOLTR_WEB_ENABLE_PREDICT=0`** if you only want validation/status.

## Run

```bash
# From repo root, with dev venv on PATH for LibTorch (same as boltr-cli):
scripts/with_dev_venv.sh ./target/release/boltr-web
```

Set **`BOLTR=/path/to/target/release/boltr`** so status checks and predict use the same binary.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/status` | Cache file checklist + optional `boltr doctor` |
| POST | `/api/validate` | Multipart YAML validation |
| POST | `/api/predict` | Start predict job; returns `202` + `{ "job_id": "..." }` |
| GET | `/api/predict/:id` | JSON status + log tail |
| GET | `/api/predict/:id/stream` | SSE log stream |
| GET | `/api/predict/:id/download` | `output.tar.gz` (success jobs only) |

Jobs live under the system temp dir (`…/boltr-web-jobs/<id>/`); completed jobs are removed after **~1 hour**.

## Tests

```bash
cargo test -p boltr-web
```
