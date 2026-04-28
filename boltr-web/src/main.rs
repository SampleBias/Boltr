//! Local web UI for Boltr prediction cache status, YAML validation, and `boltr predict`.
//!
//! See `boltr-web/README.md` for security notes (`BOLTR_WEB_ENABLE_PREDICT`, bind address).
//!
//! ```text
//! cargo run -p boltr-web -- --help
//! ```

mod paths;
mod predict_job;
mod prereq;
mod runpod;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Context;
use async_stream::stream;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Multipart, Path as PathParam, Query, State};
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use futures::Stream;
use serde::Deserialize;
use serde::Serialize;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use crate::paths::resolve_boltr_exe;
use crate::predict_job::{
    build_predict_argv, cache_from_form_override, inspect_predict_output, parse_preprocess_mode,
    path_under_job_dir, predict_enabled, preprocess_preflight, probe_boltz_cli, run_predict_job,
    tarball_out_dir, JobStore, PredictCliOptions, PredictJob, PredictOutputInspect,
};
use crate::prereq::{enrich_status, gather_status, resolve_cache_dir, validate_yaml_at};
use crate::runpod::{
    local_cuda_available, status_from_env as runpod_status_from_env, RemotePredictRequest,
    RunPodConfig,
};

#[derive(Clone)]
struct AppState {
    default_cache: PathBuf,
    jobs: std::sync::Arc<JobStore>,
}

#[derive(Parser, Debug)]
#[command(name = "boltr-web")]
#[command(about = "Minimal web UI for Boltr prediction status and YAML validation")]
struct Args {
    /// Listen address (use 127.0.0.1 for local-only).
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: String,

    /// Model / asset cache directory (default: same as `boltr` — `BOLTZ_CACHE` or `~/.cache/boltr`).
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct StatusQuery {
    cache_dir: Option<PathBuf>,
}

async fn get_status(
    State(state): State<AppState>,
    Query(q): Query<StatusQuery>,
) -> Json<serde_json::Value> {
    let cache = q
        .cache_dir
        .as_ref()
        .map(|p| resolve_cache_dir(Some(p.as_path())))
        .unwrap_or_else(|| state.default_cache.clone());
    let s = gather_status(&cache);
    let s = enrich_status(s).await;
    Json(serde_json::to_value(s).unwrap_or(serde_json::Value::Null))
}

async fn get_runpod_status() -> Json<serde_json::Value> {
    let s = runpod_status_from_env().await;
    Json(serde_json::to_value(s).unwrap_or(serde_json::Value::Null))
}

async fn post_validate(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut file_name: Option<String> = None;
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut cache_override: Option<PathBuf> = None;
    let mut job_dir: Option<PathBuf> = None;
    let mut assume_msa_server = false;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            file_name = field.file_name().map(|s| s.to_string());
            let bytes = field
                .bytes()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            file_bytes = Some(bytes.to_vec());
        } else if name == "cache_dir" {
            let t = field
                .text()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            let t = t.trim();
            if !t.is_empty() {
                cache_override = Some(PathBuf::from(t));
            }
        } else if name == "job_dir" {
            let t = field
                .text()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            let t = t.trim();
            if !t.is_empty() {
                job_dir = Some(PathBuf::from(t));
            }
        } else if name == "assume_msa_server" {
            let t = field
                .text()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            let t = t.trim();
            assume_msa_server =
                t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("on");
        }
    }

    let name = file_name.unwrap_or_else(|| "input.yaml".to_string());

    let (yaml_path, text): (PathBuf, String) = if let Some(ref dir) = job_dir {
        let p = dir.join(&name);
        if !p.is_file() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("job_dir: file not found: {}", p.display()),
            ));
        }
        let text = std::fs::read_to_string(&p).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("read {}: {e}", p.display()),
            )
        })?;
        (p, text)
    } else {
        let bytes = file_bytes.ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "missing file field (or set job_dir to validate YAML on disk)".to_string(),
            )
        })?;
        let tmp = tempfile::tempdir()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("temp dir: {e}")))?;
        let yaml_path = tmp.path().join(&name);
        std::fs::write(&yaml_path, &bytes).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("write upload: {e}"),
            )
        })?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("UTF-8: {e}")))?
            .to_string();
        (yaml_path, text)
    };

    let cache = cache_override
        .as_ref()
        .map(|p| resolve_cache_dir(Some(p.as_path())))
        .unwrap_or_else(|| state.default_cache.clone());

    let v = validate_yaml_at(&yaml_path, &text, &cache, assume_msa_server);
    let json =
        serde_json::to_value(&v).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(json))
}

#[derive(Serialize)]
struct PredictStartResponse {
    job_id: String,
}

fn parse_opt_i64(s: &str) -> Option<i64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.parse().ok()
}

fn parse_opt_usize(s: &str) -> Option<usize> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.parse().ok()
}

fn parse_opt_f64(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.parse().ok()
}

async fn post_predict(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<(StatusCode, Json<PredictStartResponse>), (StatusCode, String)> {
    if !predict_enabled() {
        return Err((
            StatusCode::FORBIDDEN,
            "predict disabled (set BOLTR_WEB_ENABLE_PREDICT=1)".to_string(),
        ));
    }

    let mut file_name: Option<String> = None;
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut cache_override: Option<String> = None;
    let mut job_dir: Option<PathBuf> = None;

    let mut opts = PredictCliOptions::default();

    let mut field_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            file_name = field.file_name().map(|s| s.to_string());
            let bytes = field
                .bytes()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            file_bytes = Some(bytes.to_vec());
        } else if name == "cache_dir" || name == "job_dir" {
            let t = field
                .text()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            if name == "cache_dir" {
                cache_override = Some(t);
            } else {
                let t = t.trim();
                if !t.is_empty() {
                    job_dir = Some(PathBuf::from(t));
                }
            }
        } else {
            let t = field.text().await.unwrap_or_default();
            field_map.insert(name, t);
        }
    }

    let name = file_name.unwrap_or_else(|| "input.yaml".to_string());
    let predict_target = field_map
        .get("predict_target")
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "local".to_string());
    if predict_target != "local" && predict_target != "runpod" {
        return Err((
            StatusCode::BAD_REQUEST,
            "predict_target: invalid value (use local or runpod)".to_string(),
        ));
    }

    let job_id = format!("{:016x}", rand::random::<u64>());
    let base = std::env::temp_dir().join("boltr-web-jobs").join(&job_id);
    tokio::fs::create_dir_all(&base)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let out_dir = base.join("out");

    let (input_path, remote_input_root): (PathBuf, PathBuf) = if let Some(ref dir) = job_dir {
        let p = dir.join(&name);
        if !p.is_file() {
            let _ = tokio::fs::remove_dir_all(&base).await;
            return Err((
                StatusCode::BAD_REQUEST,
                format!("job_dir: file not found: {}", p.display()),
            ));
        }
        path_under_job_dir(dir, &p).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        let canon = p
            .canonicalize()
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        let root = dir
            .canonicalize()
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        (canon, root)
    } else {
        let bytes = file_bytes.ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "missing file upload (or set job_dir to use YAML on disk)".to_string(),
            )
        })?;
        let p = base.join(&name);
        tokio::fs::write(&p, &bytes)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let canon = p
            .canonicalize()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let root = canon.parent().map(PathBuf::from).ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "uploaded file has no parent directory".to_string(),
            )
        })?;
        (canon, root)
    };

    // Compute: prefer `compute` (auto|gpu|cpu); else legacy `device` (cpu|cuda|cuda:N|auto|gpu).
    if let Some(c) = field_map.get("compute") {
        let t = c.trim().to_lowercase();
        opts.device = match t.as_str() {
            "auto" => "auto".to_string(),
            "gpu" => "gpu".to_string(),
            "cpu" => "cpu".to_string(),
            _ => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("compute: invalid value {c:?} (use auto, gpu, cpu)"),
                ));
            }
        };
    } else if let Some(d) = field_map.get("device") {
        let t = d.trim();
        if !t.is_empty() {
            opts.device = t.to_string();
        }
    }
    opts.use_msa_server = field_map
        .get("use_msa_server")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    opts.msa_server_url = field_map.get("msa_server_url").cloned();
    opts.msa_pairing_strategy = field_map.get("msa_pairing_strategy").cloned();
    opts.affinity = field_map
        .get("affinity")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    opts.use_potentials = field_map
        .get("use_potentials")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    opts.recycling_steps = field_map
        .get("recycling_steps")
        .and_then(|s| parse_opt_i64(s));
    opts.sampling_steps = field_map
        .get("sampling_steps")
        .and_then(|s| parse_opt_i64(s));
    opts.diffusion_samples = field_map
        .get("diffusion_samples")
        .and_then(|s| parse_opt_i64(s));
    opts.max_parallel_samples = field_map
        .get("max_parallel_samples")
        .and_then(|s| parse_opt_i64(s));
    opts.step_scale = field_map.get("step_scale").and_then(|s| parse_opt_f64(s));
    opts.output_format = field_map.get("output_format").cloned();
    opts.max_msa_seqs = field_map
        .get("max_msa_seqs")
        .and_then(|s| parse_opt_usize(s));
    opts.num_samples = field_map
        .get("num_samples")
        .and_then(|s| parse_opt_usize(s));
    if let Some(p) = field_map.get("checkpoint") {
        let t = p.trim();
        if !t.is_empty() {
            opts.checkpoint = Some(PathBuf::from(t));
        }
    }
    if let Some(p) = field_map.get("affinity_checkpoint") {
        let t = p.trim();
        if !t.is_empty() {
            opts.affinity_checkpoint = Some(PathBuf::from(t));
        }
    }
    opts.affinity_mw_correction = field_map
        .get("affinity_mw_correction")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    opts.sampling_steps_affinity = field_map
        .get("sampling_steps_affinity")
        .and_then(|s| parse_opt_i64(s));
    opts.diffusion_samples_affinity = field_map
        .get("diffusion_samples_affinity")
        .and_then(|s| parse_opt_i64(s));
    opts.preprocessing_threads = field_map
        .get("preprocessing_threads")
        .and_then(|s| parse_opt_usize(s));
    opts.r#override = field_map
        .get("override")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    opts.write_full_pae = field_map
        .get("write_full_pae")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    opts.write_full_pde = field_map
        .get("write_full_pde")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    opts.spike_only = field_map
        .get("spike_only")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let preprocess_raw = field_map
        .get("preprocess")
        .map(String::as_str)
        .unwrap_or("auto");
    opts.preprocess = parse_preprocess_mode(preprocess_raw)
        .map_err(|msg| (StatusCode::BAD_REQUEST, format!("preprocess: {msg}")))?;
    if let Some(p) = field_map.get("bolt_command") {
        let t = p.trim();
        if !t.is_empty() {
            opts.bolt_command = Some(t.to_string());
        }
    }
    if opts.bolt_command.is_none() {
        if let Ok(v) = std::env::var("BOLTR_BOLTZ_COMMAND") {
            let t = v.trim();
            if !t.is_empty() {
                opts.bolt_command = Some(t.to_string());
            }
        }
    }
    if let Some(p) = field_map.get("preprocess_staging") {
        let t = p.trim();
        if !t.is_empty() {
            opts.preprocess_staging = Some(PathBuf::from(t));
        }
    }
    opts.preprocess_keep_staging = field_map
        .get("preprocess_keep_staging")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    opts.preprocess_symlink = field_map
        .get("preprocess_symlink")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if let Some(raw) = field_map.get("preprocess_bolt_arg") {
        opts.preprocess_bolt_arg = raw
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
    }
    if let Some(p) = field_map.get("preprocess_record_id") {
        let t = p.trim();
        if !t.is_empty() {
            opts.preprocess_record_id = Some(t.to_string());
        }
    }
    if let Some(p) = field_map.get("preprocess_cuda_visible_devices") {
        let t = p.trim();
        if !t.is_empty() {
            opts.preprocess_cuda_visible_devices = Some(t.to_string());
        }
    }
    opts.preprocess_boltz_cpu = field_map
        .get("preprocess_boltz_cpu")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    opts.preprocess_auto_boltz_gpu = field_map
        .get("preprocess_auto_boltz_gpu")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    opts.preprocess_post_boltz_empty_cache = field_map
        .get("preprocess_post_boltz_empty_cache")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let quality_preset = field_map
        .get("quality_preset")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("on"))
        .unwrap_or(predict_target == "runpod");
    if quality_preset {
        if opts.recycling_steps.is_none() {
            opts.recycling_steps = Some(3);
        }
        if opts.sampling_steps.is_none() {
            opts.sampling_steps = Some(200);
        }
        if opts.diffusion_samples.is_none() {
            opts.diffusion_samples = Some(2);
        }
        if opts.max_parallel_samples.is_none() {
            opts.max_parallel_samples = Some(1);
        }
        if opts.step_scale.is_none() {
            opts.step_scale = Some(1.638);
        }
        if opts.preprocess == crate::predict_job::PreprocessMode::Auto {
            opts.preprocess = crate::predict_job::PreprocessMode::HighFidelity;
        }
    }

    let runpod_config = if predict_target == "runpod" {
        RunPodConfig::from_env()
    } else {
        None
    };
    let runpod_local_cuda =
        predict_target == "runpod" && runpod_config.is_none() && local_cuda_available().await;

    if predict_target == "local" || runpod_local_cuda {
        if let Err(msg) = preprocess_preflight(&input_path, &mut opts) {
            let _ = tokio::fs::remove_dir_all(&base).await;
            return Err((StatusCode::BAD_REQUEST, msg));
        }
    }

    if predict_target == "runpod" && opts.affinity && opts.diffusion_samples_affinity.is_none() {
        opts.diffusion_samples_affinity = Some(2);
    }

    if predict_target == "runpod" && opts.affinity && opts.sampling_steps_affinity.is_none() {
        opts.sampling_steps_affinity = Some(200);
    }

    if predict_target == "runpod" && !opts.preprocess_auto_boltz_gpu {
        opts.preprocess_boltz_cpu = true;
    }

    if predict_target == "local" && opts.device.trim().is_empty() {
        let _ = tokio::fs::remove_dir_all(&base).await;
        return Err((
            StatusCode::BAD_REQUEST,
            "device cannot be blank".to_string(),
        ));
    }

    let job = std::sync::Arc::new(PredictJob::new(base, out_dir.clone()));
    state
        .jobs
        .insert(job_id.clone(), std::sync::Arc::clone(&job))
        .await;

    let sem = state.jobs.semaphore();
    let jid = job_id.clone();
    if predict_target == "runpod" {
        if let Some(config) = runpod_config {
            let req = RemotePredictRequest {
                config,
                local_input_root: remote_input_root,
                local_input_path: input_path,
                local_out_dir: out_dir,
                opts,
            };
            tokio::spawn(async move {
                crate::runpod::run_remote_predict_job(jid, job, req, sem).await;
            });
        } else if runpod_local_cuda {
            let boltr = resolve_boltr_exe().ok_or_else(|| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "boltr binary not found: set BOLTR or install boltr on PATH".to_string(),
                )
            })?;
            let cache = cache_from_form_override(cache_override.as_deref(), &state.default_cache);
            let argv = build_predict_argv(&input_path, &out_dir, &cache, &opts);
            tokio::spawn(async move {
                run_predict_job(jid, job, boltr, argv, sem).await;
            });
        } else {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "RunPod is not configured: set BOLTR_RUNPOD_HOST and related BOLTR_RUNPOD_* env vars, or launch boltr-web on a host with a visible CUDA GPU"
                    .to_string(),
            ));
        }
    } else {
        let boltr = resolve_boltr_exe().ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "boltr binary not found: set BOLTR or install boltr on PATH".to_string(),
            )
        })?;
        let cache = cache_from_form_override(cache_override.as_deref(), &state.default_cache);
        let argv = build_predict_argv(&input_path, &out_dir, &cache, &opts);
        tokio::spawn(async move {
            run_predict_job(jid, job, boltr, argv, sem).await;
        });
    }

    Ok((StatusCode::ACCEPTED, Json(PredictStartResponse { job_id })))
}

#[derive(Serialize)]
struct PredictStatusJson {
    job_id: String,
    done: bool,
    exit_code: i32,
    success: bool,
    log_tail: Vec<String>,
    /// Paths to `.cif` / `.pdb` plus a human-readable assessment (see [`PredictOutputInspect`]).
    structure_output: PredictOutputInspect,
}

async fn get_predict_status(
    State(state): State<AppState>,
    PathParam(job_id): PathParam<String>,
) -> Result<Json<PredictStatusJson>, (StatusCode, String)> {
    let job = state
        .jobs
        .get(&job_id)
        .await
        .ok_or_else(|| (StatusCode::NOT_FOUND, "unknown job_id".to_string()))?;
    let tail = {
        let g = job.logs.lock().await;
        let n = g.len().min(200);
        g[g.len().saturating_sub(n)..].to_vec()
    };
    let structure_output = if job.done.load(Ordering::SeqCst) {
        inspect_predict_output(&job.out_dir)
    } else {
        PredictOutputInspect::job_running()
    };
    Ok(Json(PredictStatusJson {
        job_id,
        done: job.done.load(Ordering::SeqCst),
        exit_code: job.exit_code.load(Ordering::SeqCst),
        success: job.success.load(Ordering::SeqCst),
        log_tail: tail,
        structure_output,
    }))
}

async fn get_predict_stream(
    State(state): State<AppState>,
    PathParam(job_id): PathParam<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>> + Send>, (StatusCode, String)> {
    let job = state
        .jobs
        .get(&job_id)
        .await
        .ok_or_else(|| (StatusCode::NOT_FOUND, "unknown job_id".to_string()))?;

    let logs = std::sync::Arc::clone(&job.logs);
    let done = std::sync::Arc::clone(&job.done);
    let exit_code = std::sync::Arc::clone(&job.exit_code);

    let s = stream! {
        let mut seen = 0usize;
        loop {
            let snap = {
                let g = logs.lock().await;
                g.clone()
            };
            while seen < snap.len() {
                yield Ok(Event::default().data(snap[seen].clone()));
                seen += 1;
            }
            if done.load(Ordering::SeqCst) {
                let code = exit_code.load(Ordering::SeqCst);
                yield Ok(Event::default().data(format!("[boltr-web] stream end (exit {code})")));
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    };

    Ok(Sse::new(s).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keepalive"),
    ))
}

async fn get_predict_download(
    State(state): State<AppState>,
    PathParam(job_id): PathParam<String>,
) -> Result<Response, (StatusCode, String)> {
    let job = state
        .jobs
        .get(&job_id)
        .await
        .ok_or_else(|| (StatusCode::NOT_FOUND, "unknown job_id".to_string()))?;
    if !job.done.load(Ordering::SeqCst) {
        return Err((StatusCode::BAD_REQUEST, "job still running".to_string()));
    }
    if !job.success.load(Ordering::SeqCst) {
        return Err((
            StatusCode::BAD_REQUEST,
            "job did not succeed; no download".to_string(),
        ));
    }
    if !job.out_dir.is_dir() {
        return Err((
            StatusCode::NOT_FOUND,
            "output directory missing".to_string(),
        ));
    }

    let bytes = tarball_out_dir(&job.out_dir)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("archive: {e}")))?;

    let filename = format!("boltr-predict-{job_id}.tar.gz");
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/gzip")
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{filename}\""),
        )
        .body(Body::from(bytes))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../assets/index.html"))
}

async fn favicon() -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/svg+xml")
        .body(Body::from(
            include_bytes!("../assets/favicon.svg").as_slice(),
        ))
        .unwrap()
}

async fn cleanup_loop(jobs: std::sync::Arc<JobStore>) {
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    loop {
        interval.tick().await;
        jobs.cleanup_expired(Duration::from_secs(60 * 60)).await;
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let args = Args::parse();
    let default_cache = resolve_cache_dir(args.cache_dir.as_deref());

    let jobs = std::sync::Arc::new(JobStore::new());
    let jobs_bg = std::sync::Arc::clone(&jobs);
    tokio::spawn(async move {
        cleanup_loop(jobs_bg).await;
    });

    let state = AppState {
        default_cache,
        jobs,
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/favicon.svg", get(favicon))
        .route("/api/status", get(get_status))
        .route("/api/runpod/status", get(get_runpod_status))
        .route("/api/validate", post(post_validate))
        .route("/api/predict", post(post_predict))
        .route("/api/predict/:id", get(get_predict_status))
        .route("/api/predict/:id/stream", get(get_predict_stream))
        .route("/api/predict/:id/download", get(get_predict_download))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state);

    let addr: SocketAddr = args
        .listen
        .parse()
        .with_context(|| format!("invalid --listen {:?}", args.listen))?;

    let boltz_probe = probe_boltz_cli();
    if boltz_probe.available {
        tracing::info!(
            command = boltz_probe.command.as_deref().unwrap_or("boltz"),
            source = boltz_probe.source.as_deref().unwrap_or("unknown"),
            "upstream boltz CLI available for preprocess"
        );
    } else {
        tracing::warn!(
            error = boltz_probe.error.as_deref().unwrap_or("not found"),
            "upstream boltz CLI unavailable; high-fidelity/Boltz preprocess will fail until configured"
        );
    }
    tracing::info!(%addr, cache = %resolve_cache_dir(args.cache_dir.as_deref()).display(), "boltr-web listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
