//! Download checkpoints and static assets (URLs aligned with `boltz-reference/src/boltz/main.py`).

use std::path::{Path, PathBuf};

use anyhow::Result;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

/// Boltz2 structure confidence checkpoint (primary mirrors).
pub const BOLTZ2_CONF_URLS: &[&str] = &[
    "https://model-gateway.boltz.bio/boltz2_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
];

/// Boltz2 affinity checkpoint.
pub const BOLTZ2_AFF_URLS: &[&str] = &[
    "https://model-gateway.boltz.bio/boltz2_aff.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
];

/// Boltz1 confidence checkpoint.
pub const BOLTZ1_CONF_URLS: &[&str] = &[
    "https://model-gateway.boltz.bio/boltz1_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt",
];

pub const CCD_PKL_URL: &str = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl";
pub const MOLS_TAR_URL: &str =
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar";

async fn try_download(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::Client::builder()
        .user_agent("boltr/0.1 (https://github.com/SampleBias/Boltr)")
        .build()?;
    let res = client.get(url).send().await?.error_for_status()?;
    let bytes = res.bytes().await?;
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut f = File::create(dest).await?;
    f.write_all(&bytes).await?;
    f.flush().await?;
    Ok(())
}

/// Try each mirror until one succeeds.
pub async fn download_first_ok(urls: &[&str], dest: &Path) -> Result<()> {
    let mut last_err = None;
    for url in urls {
        match try_download(url, dest).await {
            Ok(()) => {
                tracing::info!(path = %dest.display(), url, "download ok");
                return Ok(());
            }
            Err(e) => {
                tracing::warn!(url, error = %e, "download failed, trying next mirror");
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no URLs provided")))
}

/// Download a named bundle into `cache_dir`.
pub async fn download_model_assets(version: &str, cache_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    match version.to_lowercase().as_str() {
        "boltz2" | "2" => {
            let p = cache_dir.join("boltz2_conf.ckpt");
            download_first_ok(BOLTZ2_CONF_URLS, &p).await?;
            written.push(p);
            let p = cache_dir.join("boltz2_aff.ckpt");
            download_first_ok(BOLTZ2_AFF_URLS, &p).await?;
            written.push(p);
            let p = cache_dir.join("ccd.pkl");
            try_download(CCD_PKL_URL, &p).await?;
            written.push(p);
            let p = cache_dir.join("mols.tar");
            try_download(MOLS_TAR_URL, &p).await?;
            written.push(p);
        }
        "boltz1" | "1" => {
            let p = cache_dir.join("boltz1_conf.ckpt");
            download_first_ok(BOLTZ1_CONF_URLS, &p).await?;
            written.push(p);
            let p = cache_dir.join("ccd.pkl");
            try_download(CCD_PKL_URL, &p).await?;
            written.push(p);
        }
        other => anyhow::bail!("unknown model version {other:?}; use boltz2 or boltz1"),
    }
    Ok(written)
}
