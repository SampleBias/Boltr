use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

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
    },
    /// Download model weights and static assets
    Download {
        /// boltz2 | boltz1
        #[arg(short, long, default_value = "boltz2")]
        version: String,
    },
    /// Run model evaluation (not yet implemented)
    Eval {
        test_dir: String,
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
        Commands::Predict { ref input } => {
            predict_flow(&cli, input).await?;
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
    }
    Ok(())
}

async fn predict_flow(cli: &Cli, input: &str) -> Result<()> {
    let input_path = std::path::Path::new(input);
    let parsed = boltr_io::parse_input_path(input_path)?;
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

    try_model_spike(cli, &device_str, out_dir).await?;

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
) -> Result<()> {
    use boltr_backend_tch::{cuda_is_available, parse_device_spec, Boltz2Model};

    tch::maybe_init_cuda();
    tracing::info!(
        cuda_available = cuda_is_available(),
        requested_device = %device_str,
        "LibTorch device probe"
    );

    let device = parse_device_spec(device_str)?;
    let cache = cli.cache_dir.clone().unwrap_or_else(default_cache_dir);
    let safetensors_path = cache.join("boltz2_conf.safetensors");
    if safetensors_path.exists() {
        let token_s = 384_i64;
        let mut model = Boltz2Model::new(device, token_s)?;
        if let Err(e) = model.load_s_init_from_safetensors(&safetensors_path) {
            tracing::warn!(
                error = %e,
                "optional s_init load failed (check safetensor key names vs Lightning state_dict)"
            );
        }
        let probe = tch::Tensor::randn(&[2, token_s], (tch::Kind::Float, device));
        let _y = model.forward_s_init(&probe);
        let spike_path = out_dir.join("boltr_backend_spike_ok.txt");
        tokio::fs::write(
            &spike_path,
            "Boltz2Model s_init forward executed (see boltr-backend-tch).\n",
        )
        .await?;
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
) -> Result<()> {
    Ok(())
}
