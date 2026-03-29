use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[derive(Parser, Debug)]
#[command(name = "boltr")]
#[command(about = "Boltr - Rust Native Boltz Implementation", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Use MSA server for sequence alignment (ColabFold-compatible API)
    #[arg(long)]
    use_msa_server: bool,

    /// MSA server base URL (default matches Boltz: api.colabfold.com)
    #[arg(long, default_value = "https://api.colabfold.com")]
    msa_server_url: String,

    /// Compute device: cpu, cuda, or cuda:N (requires `boltr` built with `--features tch` and CUDA LibTorch for GPU)
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Model / asset cache directory (default: XDG_CACHE_HOME/boltr or ~/.cache/boltr)
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Number of inference samples
    #[arg(long, default_value_t = 1)]
    num_samples: usize,

    /// Boltz2 trunk recycling iterations (forward_trunk); 0 means one pairformer pass per step
    #[arg(long, default_value_t = 0)]
    recycling_steps: i64,

    /// Output directory
    #[arg(short, long, default_value = "./output")]
    output: String,

    /// Verbosity level
    #[command(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

#[derive(Parser, Debug)]
enum Commands {
    /// Run prediction on input files
    Predict {
        /// Input YAML path
        input: String,
        /// Affinity inference (expects preprocess `pre_affinity_*.npz` layout; same flag intent as Boltz)
        #[arg(long)]
        affinity: bool,
        /// Inference-time physical potentials / steering (Boltz `--use_potentials`; disabled on affinity path)
        #[arg(long)]
        use_potentials: bool,
    },
    /// Download model weights and static assets
    Download {
        /// boltz2 | boltz1
        #[arg(short, long, default_value = "boltz2")]
        version: String,
    },
    /// Run model evaluation (not yet implemented)
    Eval { test_dir: String },
    /// Convert an MSA file (`.a3m`, `.a3m.gz`, or Boltz `.csv`) to a Boltz `MSA` `.npz`
    MsaToNpz {
        /// Input path (format from extension: a3m / a3m.gz / csv)
        input: PathBuf,
        /// Output `.npz` path (default: same directory, base name with `.npz`)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Maximum number of sequences (after dedup), like Boltz preprocess
        #[arg(long)]
        max_seqs: Option<usize>,
    },
    /// Tokenize `StructureV2` and write columnar token `.npz` ([`boltr_io::token_npz`])
    TokensToNpz {
        /// Boltz preprocess `StructureV2` `.npz` (atoms/residues/chains/…); mutually exclusive with `DEMO`
        #[arg(short = 'i', long = "structure-npz")]
        structure_npz: Option<PathBuf>,
        /// Output token `.npz` path
        #[arg(short, long)]
        output: PathBuf,
        /// Demo structure when `--structure-npz` is omitted (`ala` = single standard ALA)
        demo: Option<String>,
        /// If set, tokens on chains with this `asym_id` get `affinity_mask` (see `tokenize_structure`)
        #[arg(long)]
        affinity_asym_id: Option<i32>,
    },
}

fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("boltr")
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let level = match cli.verbose.log_level_filter() {
        clap_verbosity_flag::log::LevelFilter::Off => tracing::Level::ERROR,
        clap_verbosity_flag::log::LevelFilter::Error => tracing::Level::ERROR,
        clap_verbosity_flag::log::LevelFilter::Warn => tracing::Level::WARN,
        clap_verbosity_flag::log::LevelFilter::Info => tracing::Level::INFO,
        clap_verbosity_flag::log::LevelFilter::Debug => tracing::Level::DEBUG,
        clap_verbosity_flag::log::LevelFilter::Trace => tracing::Level::TRACE,
    };

    let filter = EnvFilter::builder()
        .with_default_directive(level.into())
        .from_env_lossy();

    tracing_subscriber::registry()
        .with(fmt::layer().with_filter(filter))
        .init();

    tracing::info!("Boltr starting...");

    match cli.command {
        Commands::Predict {
            ref input,
            affinity,
            use_potentials,
        } => {
            predict_flow(&cli, input, affinity, use_potentials).await?;
        }
        Commands::Download { version } => {
            let cache = cli.cache_dir.unwrap_or_else(default_cache_dir);
            tokio::fs::create_dir_all(&cache).await?;
            tracing::info!(dir = %cache.display(), "downloading into cache");
            let paths = boltr_io::download_model_assets(&version, &cache).await?;
            for p in paths {
                tracing::info!(path = %p.display(), "saved");
            }
        }
        Commands::Eval { test_dir } => {
            tracing::info!("Running evaluation on: {}", test_dir);
            anyhow::bail!("Evaluation not yet implemented");
        }
        Commands::MsaToNpz {
            input,
            output,
            max_seqs,
        } => {
            run_msa_to_npz(&input, output.as_deref(), max_seqs)?;
        }
        Commands::TokensToNpz {
            structure_npz,
            output,
            demo,
            affinity_asym_id,
        } => {
            run_tokens_to_npz(
                structure_npz.as_deref(),
                demo.as_deref(),
                &output,
                affinity_asym_id,
            )?;
        }
    }
    Ok(())
}

fn run_tokens_to_npz(
    structure_npz: Option<&Path>,
    demo: Option<&str>,
    output: &Path,
    affinity_asym_id: Option<i32>,
) -> Result<()> {
    let structure = match (structure_npz, demo) {
        (Some(path), None) => boltr_io::read_structure_v2_npz_path(path)
            .with_context(|| format!("read StructureV2 npz {}", path.display()))?,
        (None, Some(d)) => match d.to_ascii_lowercase().as_str() {
            "ala" => boltr_io::structure_v2_single_ala(),
            other => anyhow::bail!("unknown demo {other:?} (supported: ala)"),
        },
        (Some(_), Some(_)) => anyhow::bail!("pass either --structure-npz or DEMO, not both"),
        (None, None) => anyhow::bail!("pass --structure-npz PATH or a DEMO (e.g. ala)"),
    };
    let (tokens, bonds) = boltr_io::tokenize_structure(&structure, affinity_asym_id);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create directory {}", parent.display()))?;
    }
    boltr_io::write_token_batch_npz_compressed(output, &tokens, &bonds)
        .with_context(|| format!("write {}", output.display()))?;
    tracing::info!(
        demo,
        out_path = %output.display(),
        n_tokens = tokens.len(),
        n_bonds = bonds.len(),
        "wrote token batch npz"
    );
    Ok(())
}

fn run_msa_to_npz(input: &Path, output: Option<&Path>, max_seqs: Option<usize>) -> Result<()> {
    let msa = load_msa_for_npz(input, max_seqs)
        .with_context(|| format!("read MSA from {}", input.display()))?;
    let out = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_msa_npz_path(input));
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create directory {}", parent.display()))?;
    }
    boltr_io::write_msa_npz_compressed(&out, &msa)
        .with_context(|| format!("write {}", out.display()))?;
    tracing::info!(
        in_path = %input.display(),
        out_path = %out.display(),
        n_seq = msa.sequences.len(),
        "wrote Boltz MSA npz"
    );
    Ok(())
}

fn load_msa_for_npz(path: &Path, max_seqs: Option<usize>) -> Result<boltr_io::A3mMsa> {
    let lossy = path.to_string_lossy();
    let lower = lossy.to_ascii_lowercase();
    if lower.ends_with(".csv") {
        boltr_io::parse_csv_path(path, max_seqs)
    } else {
        boltr_io::parse_a3m_path(path, max_seqs)
    }
}

/// `foo.a3m` → `foo.npz`; `foo.a3m.gz` → `foo.npz` (strip `.gz` then optional inner extension).
fn default_msa_npz_path(input: &Path) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default();
    let base = Path::new(stem).file_stem().unwrap_or(stem);
    let parent = input.parent().unwrap_or(Path::new("."));
    parent.join(format!("{}.npz", base.to_string_lossy()))
}

async fn predict_flow(cli: &Cli, input: &str, affinity: bool, use_potentials: bool) -> Result<()> {
    let input_path = std::path::Path::new(input);
    let parsed = boltr_io::parse_input_path(input_path)?;
    if affinity {
        tracing::info!("--affinity set: use Boltz affinity preprocess (pre_affinity npz) when wiring full I/O");
    }
    if use_potentials {
        if affinity {
            tracing::warn!(
                "--use-potentials with --affinity: Boltz disables steering on affinity; potentials not applied"
            );
        } else {
            tracing::info!(
                use_potentials,
                "inference potentials / steering requested (Boltz --use_potentials); wired when full predict uses Boltz2Model::predict_step"
            );
        }
    }
    tracing::info!(
        chains = ?parsed.summary_chain_ids(),
        "parsed YAML"
    );

    let out_dir = std::path::Path::new(&cli.output);
    tokio::fs::create_dir_all(out_dir).await?;

    if cli.use_msa_server {
        let need = parsed.protein_sequences_for_msa();
        if !need.is_empty() {
            let proc = boltr_io::MsaProcessor::new(cli.msa_server_url.clone());
            let seqs: Vec<String> = need.iter().map(|(_, s)| s.clone()).collect();
            tracing::info!(n = seqs.len(), "fetching MSAs from server");
            let msas = proc.fetch_msas(&seqs, true, true).await?;
            let msa_dir = out_dir.join("msa");
            for ((chain_id, _), a3m) in need.iter().zip(msas.iter()) {
                let path = msa_dir.join(format!("{chain_id}.a3m"));
                boltr_io::write_a3m(&path, a3m)?;
                tracing::info!(path = %path.display(), "wrote MSA");
            }
        }
    }

    let device_str = std::env::var("BOLTR_DEVICE").unwrap_or_else(|_| cli.device.clone());

    let summary = boltr_io::PredictionRunSummary::from_input(
        input,
        &parsed,
        cli.use_msa_server,
        &device_str,
        cli.num_samples,
        predict_backend_note(),
    );
    let summary_path = out_dir.join("boltr_run_summary.json");
    summary.write_json(&summary_path)?;
    tracing::info!(path = %summary_path.display(), "wrote run summary");

    try_model_spike(
        cli,
        &device_str,
        out_dir,
        cli.recycling_steps,
        use_potentials && !affinity,
    )
    .await?;

    Ok(())
}

fn predict_backend_note() -> &'static str {
    #[cfg(feature = "tch")]
    {
        "tch backend linked; full Boltz2 forward requires remaining trunk/diffusion port"
    }
    #[cfg(not(feature = "tch"))]
    {
        "rebuild with: cargo build -p boltr-cli --features tch (LibTorch required)"
    }
}

#[cfg(feature = "tch")]
async fn try_model_spike(
    cli: &Cli,
    device_str: &str,
    out_dir: &std::path::Path,
    recycling_steps: i64,
    use_potentials_steering: bool,
) -> Result<()> {
    use boltr_backend_tch::{
        cuda_is_available, parse_device_spec, safetensor_names_not_in_var_store, Boltz2Model,
        RelPosFeatures, SteeringParams,
    };

    tch::maybe_init_cuda();
    tracing::info!(
        cuda_available = cuda_is_available(),
        requested_device = %device_str,
        use_potentials_steering,
        steering = ?SteeringParams::from_use_potentials(use_potentials_steering),
        "LibTorch device probe"
    );

    let device = parse_device_spec(device_str)?;
    let cache = cli.cache_dir.clone().unwrap_or_else(default_cache_dir);
    let safetensors_path = cache.join("boltz2_conf.safetensors");
    if safetensors_path.exists() {
        let token_s = 384_i64;
        let mut model = Boltz2Model::new(device, token_s);

        let mut missing_after_partial: Vec<String> = Vec::new();
        let mut partial_load_ok = false;
        match model.load_partial_from_safetensors(&safetensors_path) {
            Ok(missing) => {
                partial_load_ok = true;
                missing_after_partial = missing;
                tracing::info!(
                    model_params = model.var_store().len(),
                    still_missing = missing_after_partial.len(),
                    "safetensors VarStore::load_partial"
                );
                if !missing_after_partial.is_empty() {
                    tracing::warn!(
                        n = missing_after_partial.len(),
                        sample = ?missing_after_partial.iter().take(12).collect::<Vec<_>>(),
                        "checkpoint missing these VarStore keys (left default-init)"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "VarStore::load_partial failed (key/shape mismatch vs Rust graph?); trying s_init only"
                );
                if let Err(e2) = model.load_s_init_from_safetensors(&safetensors_path) {
                    tracing::warn!(error = %e2, "s_init load also failed");
                }
            }
        }

        if tracing::enabled!(tracing::Level::DEBUG) {
            match safetensor_names_not_in_var_store(&safetensors_path, model.var_store()) {
                Ok(extra) if !extra.is_empty() => {
                    tracing::debug!(
                        n = extra.len(),
                        sample = ?extra.iter().take(8).collect::<Vec<_>>(),
                        "safetensors keys not mapped into this Boltz2Model VarStore"
                    );
                }
                _ => {}
            }
        }

        let probe = tch::Tensor::randn(&[2, token_s], (tch::Kind::Float, device));
        let _y = model.forward_s_init(&probe);

        // Exercise full trunk + pairformer + recycling (embedder outputs not yet wired from IO).
        let b = 2_i64;
        let n = 16_i64;
        let s_in = tch::Tensor::randn(&[b, n, token_s], (tch::Kind::Float, device));
        let (s_out, z_out) = model
            .forward_trunk(&s_in, Some(recycling_steps), None)
            .map_err(|e| anyhow::anyhow!("forward_trunk spike: {e}"))?;
        let s_sz = s_out.size();
        let z_sz = z_out.size();
        tracing::info!(
            ?s_sz,
            ?z_sz,
            recycling_steps,
            "forward_trunk (pairformer stack) spike ok"
        );

        let asym_id = tch::Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let residue_index = tch::Tensor::arange(n, (tch::Kind::Int64, device))
            .view_(&[1, n])
            .expand(&[b, n], false);
        let entity_id = tch::Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let token_index = residue_index.shallow_clone();
        let sym_id = tch::Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let cyclic_period = tch::Tensor::zeros(&[b, n], (tch::Kind::Int64, device));
        let rel = RelPosFeatures {
            asym_id: &asym_id,
            residue_index: &residue_index,
            entity_id: &entity_id,
            token_index: &token_index,
            sym_id: &sym_id,
            cyclic_period: &cyclic_period,
        };
        let (s_ps, z_ps) = model
            .predict_step_trunk(&s_in, &rel, None, None, None, Some(recycling_steps), None)
            .map_err(|e| anyhow::anyhow!("predict_step_trunk spike: {e}"))?;
        tracing::info!(
            s_predict = ?s_ps.size(),
            z_predict = ?z_ps.size(),
            recycling_steps,
            "predict_step_trunk (z_init + recycling + pairformer) spike ok"
        );

        let spike_path = out_dir.join("boltr_backend_spike_ok.txt");
        let load_note = if partial_load_ok {
            format!(
                "VarStore partial load ok; keys still missing in checkpoint: {}\n",
                missing_after_partial.len()
            )
        } else {
            "VarStore partial load failed; see logs (optional s_init-only fallback may have run).\n"
                .to_string()
        };
        let msg = format!(
            "Boltz2Model: s_init forward + forward_trunk + predict_step_trunk executed (pairformer + recycling).\n\
             recycling_steps={recycling_steps}\n\
             s_out shape: {s_sz:?}\n\
             z_out shape: {z_sz:?}\n\
             predict_step_trunk s_out: {:?}\n\
             predict_step_trunk z_out: {:?}\n\
             {load_note}",
            s_ps.size(),
            z_ps.size()
        );
        tokio::fs::write(&spike_path, msg).await?;
        tracing::info!(path = %spike_path.display(), "backend spike");
    } else {
        tracing::info!(
            path = %safetensors_path.display(),
            "skip weight spike: place exported safetensors here (scripts/export_checkpoint_to_safetensors.py)"
        );
    }
    Ok(())
}

#[cfg(not(feature = "tch"))]
async fn try_model_spike(
    _cli: &Cli,
    _device_str: &str,
    _out_dir: &std::path::Path,
    _recycling_steps: i64,
    _use_potentials_steering: bool,
) -> Result<()> {
    Ok(())
}
