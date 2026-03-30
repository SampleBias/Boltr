//! Local web UI for Boltr prediction cache status and YAML validation.
//!
//! ```text
//! cargo run -p boltr-web -- --help
//! ```

mod prereq;

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Context;
use axum::extract::{DefaultBodyLimit, Multipart, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, Json};
use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use serde::Deserialize;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use crate::prereq::{enrich_status, gather_status, resolve_cache_dir, validate_yaml_at};

#[derive(Clone)]
struct AppState {
    default_cache: PathBuf,
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
        let tmp = tempfile::tempdir().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("temp dir: {e}"),
            )
        })?;
        let yaml_path = tmp.path().join(&name);
        std::fs::write(&yaml_path, &bytes).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("write upload: {e}"),
            )
        })?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    format!("UTF-8: {e}"),
                )
            })?
            .to_string();
        (yaml_path, text)
    };

    let cache = cache_override
        .as_ref()
        .map(|p| resolve_cache_dir(Some(p.as_path())))
        .unwrap_or_else(|| state.default_cache.clone());

    let v = validate_yaml_at(&yaml_path, &text, &cache, assume_msa_server);
    let json = serde_json::to_value(&v).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            e.to_string(),
        )
    })?;
    Ok(Json(json))
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../assets/index.html"))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let args = Args::parse();
    let default_cache = resolve_cache_dir(args.cache_dir.as_deref());

    let state = AppState { default_cache };

    let app = Router::new()
        .route("/", get(index))
        .route("/api/status", get(get_status))
        .route("/api/validate", post(post_validate))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state);

    let addr: SocketAddr = args
        .listen
        .parse()
        .with_context(|| format!("invalid --listen {:?}", args.listen))?;

    tracing::info!(%addr, cache = %resolve_cache_dir(args.cache_dir.as_deref()).display(), "boltr-web listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
