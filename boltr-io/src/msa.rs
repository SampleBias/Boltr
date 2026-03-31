//! ColabFold-style MSA server client (same protocol as `boltz.data.msa.mmseqs2`).

use std::collections::HashMap;
use std::io::Read;
use std::time::Duration;

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use rand::Rng;
use serde::Deserialize;
use tar::Archive;
use tokio::time::sleep;

#[derive(Debug, Deserialize)]
struct JobResponse {
    status: String,
    #[serde(default)]
    id: Option<String>,
}

/// Fetch MSA(s) for protein sequences via MMseqs2 HTTP API (ColabFold-compatible).
pub struct MsaProcessor {
    /// Base URL only, e.g. `https://api.colabfold.com` (same default as Boltz CLI).
    pub host_url: String,
}

impl MsaProcessor {
    pub fn new(host_url: impl Into<String>) -> Self {
        let mut s = host_url.into();
        while s.ends_with('/') {
            s.pop();
        }
        Self { host_url: s }
    }

    /// Returns one A3M document string per input sequence (order preserved).
    pub async fn fetch_msas(
        &self,
        sequences: &[String],
        use_env: bool,
        use_filter: bool,
    ) -> Result<Vec<String>> {
        let client = reqwest::Client::builder()
            .user_agent("boltr/0.1")
            .timeout(Duration::from_secs(300))
            .build()?;

        let mode = if use_filter {
            if use_env {
                "env"
            } else {
                "all"
            }
        } else if use_env {
            "env-nofilter"
        } else {
            "nofilter"
        };

        let seqs_unique: Vec<String> = {
            let mut u = Vec::new();
            for s in sequences {
                if !u.contains(s) {
                    u.push(s.clone());
                }
            }
            u
        };

        let mut n = 101_i32;
        let mut query = String::new();
        for seq in &seqs_unique {
            query.push('>');
            query.push_str(&n.to_string());
            query.push('\n');
            query.push_str(seq);
            query.push('\n');
            n += 1;
        }

        let bytes = run_job_download(&client, &self.host_url, &query, mode).await?;
        let a3m_by_key = extract_a3m_from_tar_gz(&bytes, use_env)?;

        let ms: Vec<i32> = sequences
            .iter()
            .map(|seq| 101 + seqs_unique.iter().position(|s| s == seq).unwrap() as i32)
            .collect();

        let mut out = Vec::with_capacity(sequences.len());
        for m in ms {
            let block = a3m_by_key
                .get(&m)
                .with_context(|| format!("missing MSA block for query id {m}"))?;
            out.push(block.clone());
        }
        Ok(out)
    }
}

async fn run_job_download(
    client: &reqwest::Client,
    base: &str,
    query: &str,
    mode: &str,
) -> Result<Vec<u8>> {
    let mut rng = rand::thread_rng();
    let submit_url = format!("{base}/ticket/msa");

    let submitted = loop {
        let out = post_ticket_msa(client, &submit_url, query, mode).await?;
        match out.status.as_str() {
            "UNKNOWN" | "RATELIMIT" => {
                let t = Duration::from_secs(5 + rng.gen_range(0..6));
                tracing::warn!(reason = %out.status, "MSA submit throttled");
                sleep(t).await;
            }
            "ERROR" => anyhow::bail!("MSA server ERROR on submit"),
            "MAINTENANCE" => anyhow::bail!("MSA server MAINTENANCE"),
            _ => break out,
        }
    };

    let id = submitted
        .id
        .clone()
        .filter(|_| !submitted.status.is_empty())
        .context("MSA submit response missing id")?;

    let status_url = format!("{base}/ticket/{id}");
    let mut status = submitted.status;
    while status == "UNKNOWN" || status == "RUNNING" || status == "PENDING" {
        let t = Duration::from_secs(5 + rng.gen_range(0..6));
        sleep(t).await;
        let st = get_json::<JobResponse>(client, &status_url).await?;
        status = st.status;
        if status == "ERROR" {
            anyhow::bail!("MSA job failed with ERROR");
        }
    }

    if status != "COMPLETE" {
        anyhow::bail!("unexpected MSA terminal status: {status}");
    }

    let dl = format!("{base}/result/download/{id}");
    let res = client.get(&dl).send().await?.error_for_status()?;
    Ok(res.bytes().await?.to_vec())
}

async fn post_ticket_msa(
    client: &reqwest::Client,
    url: &str,
    query: &str,
    mode: &str,
) -> Result<JobResponse> {
    let form = [("q", query), ("mode", mode)];
    let res = client
        .post(url)
        .form(&form)
        .send()
        .await?
        .error_for_status()?;
    Ok(res.json().await?)
}

async fn get_json<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
) -> Result<T> {
    let res = client.get(url).send().await?.error_for_status()?;
    Ok(res.json().await?)
}

fn extract_a3m_from_tar_gz(bytes: &[u8], use_env: bool) -> Result<HashMap<i32, String>> {
    let gz = GzDecoder::new(bytes);
    let mut archive = Archive::new(gz);
    let mut combined: HashMap<i32, Vec<String>> = HashMap::new();

    for entry in archive.entries().context("tar entries")? {
        let mut entry = entry.context("tar entry")?;
        let path_s = entry
            .path()
            .context("tar path")?
            .to_string_lossy()
            .into_owned();
        let is_uniref = path_s.ends_with("uniref.a3m");
        let is_bfd = path_s.contains("bfd.mgnify30.metaeuk30.smag30.a3m");
        if !is_uniref && !(use_env && is_bfd) {
            continue;
        }
        let mut text = String::new();
        entry
            .read_to_string(&mut text)
            .with_context(|| format!("read tar member {path_s}"))?;
        parse_a3m_into_map(&text, &mut combined);
    }

    Ok(combined.into_iter().map(|(k, v)| (k, v.concat())).collect())
}

/// ColabFold / MMseqs2 tar members use `>101` or `>102|UniRef50_...` style headers.
/// Only the leading numeric id must match our submitted query ids (101..).
fn query_id_from_a3m_header(line: &str) -> Option<i32> {
    let rest = line.strip_prefix('>')?.trim();
    let token = rest
        .split(|c: char| c == '|' || c.is_whitespace())
        .next()?;
    token.parse().ok()
}

fn parse_a3m_into_map(text: &str, out: &mut HashMap<i32, Vec<String>>) {
    let mut current_m: Option<i32> = None;
    let mut update_m = true;
    for mut line in text.lines() {
        if line.contains('\0') {
            line = line.trim_end_matches('\0');
            update_m = true;
        }
        if line.is_empty() {
            continue;
        }
        let line_with_nl = format!("{line}\n");
        if line.starts_with('>') && update_m {
            let id = query_id_from_a3m_header(line).unwrap_or(-1);
            current_m = Some(id);
            update_m = false;
            out.entry(id).or_default().push(line_with_nl);
        } else if let Some(m) = current_m {
            out.entry(m).or_default().push(line_with_nl);
        }
    }
}

/// Write A3M text to path (creates parent dirs).
pub fn write_a3m(path: &std::path::Path, content: &str) -> Result<()> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    std::fs::write(path, content)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_a3m_map() {
        // Matches ColabFold-style headers: one query id block (see mmseqs2.py gather loop).
        let text = ">101\nACDE\nWWW\n";
        let mut m = HashMap::new();
        parse_a3m_into_map(text, &mut m);
        assert!(m.contains_key(&101));
    }

    #[test]
    fn parse_a3m_map_uniref_suffix_header() {
        // API returns `>102|UniRef50_...` — full-line parse must not fail.
        let text = ">102|UniRef50_dummy\nACDE\n";
        let mut m = HashMap::new();
        parse_a3m_into_map(text, &mut m);
        assert!(
            m.contains_key(&102),
            "expected query id 102 when header has pipe-separated description"
        );
    }
}
