//! `boltr` CLI — Rust-native Boltz2 inference.
//!
//! Commands: `predict`, `download`, `eval`, `msa-to-npz`, `tokens-to-npz`.
//!
//! The `predict` command mirrors the upstream `boltz predict` interface and output layout
//! ([boltz-reference/docs/prediction.md](../../boltz-reference/docs/prediction.md)).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[cfg(feature = "tch")]
mod predict_tch;

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

/// Boltz model / asset cache directory resolution.
///
/// Priority: `--cache-dir` flag > `BOLTZ_CACHE` env var > XDG default (`~/.cache/boltr`).
fn resolve_cache_dir(cli_cache: Option<&Path>) -> PathBuf {
    if let Some(p) = cli_cache {
        return p.to_path_buf();
    }
    if let Ok(p) = std::env::var("BOLTZ_CACHE") {
        return PathBuf::from(p);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("boltr")
}

/// Output format for predicted structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
pub enum OutputFormat {
    /// mmCIF format (default, matches Boltz upstream).
    Mmcif,
    /// PDB format.
    Pdb,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Mmcif => write!(f, "mmcif"),
            OutputFormat::Pdb => write!(f, "pdb"),
        }
    }
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Mmcif
    }
}

#[derive(Parser, Debug)]
#[command(name = "boltr")]
#[command(about = "Boltr — Rust-native Boltz2 structure prediction", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbosity level (-v, -vv, -qqq, etc.).
    #[command(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

#[derive(Parser, Debug)]
enum Commands {
    /// Run structure prediction on input YAML or FASTA files.
    Predict {
        /// Input path: a `.yaml`/`.fasta` file, or a directory (processes all `.yaml`/`.fasta` inside).
        input: String,

        /// Output directory (default: `./output`).
        #[arg(short, long, default_value = "./output")]
        output: String,

        /// Compute device: cpu, cuda, or cuda:N (requires `--features tch` and LibTorch for GPU).
        #[arg(long, default_value = "cpu")]
        device: String,

        /// Model / asset cache directory.
        /// Falls back to `BOLTZ_CACHE` env var, then `~/.cache/boltr`.
        #[arg(long)]
        cache_dir: Option<PathBuf>,

        // ---- Boltz-compatible flags ----
        /// Use MSA server for sequence alignment (ColabFold-compatible API).
        #[arg(long)]
        use_msa_server: bool,

        /// MSA server base URL.
        #[arg(long, default_value = "https://api.colabfold.com")]
        msa_server_url: String,

        /// MSA pairing strategy (greedy | complete).
        #[arg(long, default_value = "greedy")]
        msa_pairing_strategy: String,

        /// Affinity prediction (expects `pre_affinity_*.npz` layout when preprocessing exists).
        #[arg(long)]
        affinity: bool,

        /// Apply inference-time physical potentials / steering.
        #[arg(long)]
        use_potentials: bool,

        /// Override trunk recycling iterations.
        /// Omit to use YAML `predict_args` / checkpoint hparams / defaults.
        #[arg(long)]
        recycling_steps: Option<i64>,

        /// Diffusion sampler steps (overrides YAML / checkpoint defaults).
        #[arg(long)]
        sampling_steps: Option<i64>,

        /// Number of structure samples per prediction (multiplicity).
        #[arg(long)]
        diffusion_samples: Option<i64>,

        /// Max parallel diffusion samples (steering / potentials path).
        #[arg(long)]
        max_parallel_samples: Option<i64>,

        /// Diffusion step scale (temperature). Recommended range [1, 2]. Default 1.638.
        #[arg(long, default_value_t = 1.638)]
        step_scale: f64,

        /// Output structure format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Mmcif)]
        output_format: OutputFormat,

        /// Max MSA sequences per chain (Boltz `--max_msa_seqs`).
        #[arg(long, default_value_t = 8192)]
        max_msa_seqs: usize,

        /// Record-level default for number of samples when `--diffusion-samples` is omitted.
        #[arg(long, default_value_t = 1)]
        num_samples: usize,

        /// Custom structure confidence checkpoint path (`.ckpt` or `.safetensors`).
        #[arg(long)]
        checkpoint: Option<PathBuf>,

        /// Custom affinity checkpoint path.
        #[arg(long)]
        affinity_checkpoint: Option<PathBuf>,

        /// Apply molecular weight correction to affinity prediction.
        #[arg(long)]
        affinity_mw_correction: bool,

        /// Sampling steps for affinity diffusion.
        #[arg(long)]
        sampling_steps_affinity: Option<i64>,

        /// Diffusion samples for affinity prediction.
        #[arg(long)]
        diffusion_samples_affinity: Option<i64>,

        /// Number of preprocessing threads (default: num_cpus).
        #[arg(long)]
        preprocessing_threads: Option<usize>,

        /// Re-run prediction even if output already exists (Boltz `--override`).
        #[arg(long)]
        r#override: bool,

        /// Save full PAE matrix as `.npz`.
        #[arg(long)]
        write_full_pae: bool,

        /// Save full PDE matrix as `.npz`.
        #[arg(long)]
        write_full_pde: bool,

        /// Only run trunk/weight smoke test (`predict_step_trunk`); skip diffusion + writers.
        #[arg(long)]
        spike_only: bool,
    },

    /// Download model weights and static assets.
    Download {
        /// Model version: `boltz2` or `boltz1`.
        #[arg(long, default_value = "boltz2")]
        version: String,

        /// Cache directory (default: `BOLTZ_CACHE` env or `~/.cache/boltr`).
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },

    /// Structure benchmark evaluation (not implemented; see boltz-reference docs).
    Eval {
        /// Test directory label (reserved for future use).
        test_dir: String,
    },

    /// Convert an MSA file (`.a3m`, `.a3m.gz`, or Boltz `.csv`) to a Boltz `MSA` `.npz`.
    MsaToNpz {
        /// Input path (format from extension: a3m / a3m.gz / csv).
        input: PathBuf,

        /// Output `.npz` path (default: same directory, base name + `.npz`).
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Maximum number of sequences (after dedup), like Boltz preprocess.
        #[arg(long)]
        max_seqs: Option<usize>,
    },

    /// Tokenize `StructureV2` and write columnar token `.npz`.
    TokensToNpz {
        /// Boltz preprocess `StructureV2` `.npz`; mutually exclusive with DEMO.
        #[arg(short = 'i', long = "structure-npz")]
        structure_npz: Option<PathBuf>,

        /// Output token `.npz` path.
        #[arg(short, long)]
        output: PathBuf,

        /// Demo structure when `--structure-npz` is omitted (`ala` = single ALA).
        demo: Option<String>,

        /// Set `affinity_mask` for tokens on chains with this `asym_id`.
        #[arg(long)]
        affinity_asym_id: Option<i32>,
    },
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // --- tracing setup ---
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

    tracing::info!("boltr starting");

    match cli.command {
        // =======================================================================
        // predict
        // =======================================================================
        Commands::Predict {
            input,
            output,
            device,
            cache_dir,
            use_msa_server,
            msa_server_url,
            msa_pairing_strategy,
            affinity,
            use_potentials,
            recycling_steps,
            sampling_steps,
            diffusion_samples,
            max_parallel_samples,
            step_scale,
            output_format,
            max_msa_seqs,
            num_samples,
            checkpoint,
            affinity_checkpoint,
            affinity_mw_correction,
            sampling_steps_affinity,
            diffusion_samples_affinity,
            preprocessing_threads,
            r#override,
            write_full_pae,
            write_full_pde,
            spike_only,
        } => {
            let cache = resolve_cache_dir(cache_dir.as_deref());
            let out_dir = Path::new(&output).to_path_buf();
            let device_str = std::env::var("BOLTR_DEVICE").unwrap_or(device);

            predict_flow(PredictFlowArgs {
                input,
                cache,
                out_dir,
                device: device_str,
                use_msa_server,
                msa_server_url,
                msa_pairing_strategy,
                affinity,
                use_potentials,
                recycling_steps,
                sampling_steps,
                diffusion_samples,
                max_parallel_samples,
                step_scale,
                output_format,
                max_msa_seqs,
                num_samples,
                checkpoint,
                affinity_checkpoint,
                affinity_mw_correction,
                sampling_steps_affinity,
                diffusion_samples_affinity,
                preprocessing_threads,
                override_flag: r#override,
                write_full_pae,
                write_full_pde,
                spike_only,
            })
            .await?;
        }

        // =======================================================================
        // download
        // =======================================================================
        Commands::Download { version, cache_dir } => {
            let cache = resolve_cache_dir(cache_dir.as_deref());
            tokio::fs::create_dir_all(&cache).await?;
            tracing::info!(dir = %cache.display(), "downloading into cache");
            let paths = boltr_io::download_model_assets(&version, &cache).await?;
            for p in paths {
                tracing::info!(path = %p.display(), "saved");
            }
        }

        // =======================================================================
        // eval
        // =======================================================================
        Commands::Eval { test_dir } => {
            tracing::info!(test_dir = %test_dir, "boltr eval is not implemented");
            eprintln!(
                "boltr eval: native evaluation is not implemented.\n\
                 Boltz benchmark metrics (lDDT, DockQ, etc.) use OpenStructure via Docker.\n\
                 See: boltz-reference/docs/evaluation.md\n\
                 Prerequisites: Docker and a compatible OpenStructure image."
            );
            std::process::exit(2);
        }

        // =======================================================================
        // msa-to-npz
        // =======================================================================
        Commands::MsaToNpz {
            input,
            output,
            max_seqs,
        } => {
            run_msa_to_npz(&input, output.as_deref(), max_seqs)?;
        }

        // =======================================================================
        // tokens-to-npz
        // =======================================================================
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

// ---------------------------------------------------------------------------
// predict flow
// ---------------------------------------------------------------------------

/// All arguments for the predict flow, collected into a single struct for clarity.
struct PredictFlowArgs {
    input: String,
    cache: PathBuf,
    out_dir: PathBuf,
    device: String,
    use_msa_server: bool,
    msa_server_url: String,
    msa_pairing_strategy: String,
    affinity: bool,
    use_potentials: bool,
    recycling_steps: Option<i64>,
    sampling_steps: Option<i64>,
    diffusion_samples: Option<i64>,
    max_parallel_samples: Option<i64>,
    step_scale: f64,
    output_format: OutputFormat,
    max_msa_seqs: usize,
    num_samples: usize,
    checkpoint: Option<PathBuf>,
    affinity_checkpoint: Option<PathBuf>,
    affinity_mw_correction: bool,
    sampling_steps_affinity: Option<i64>,
    diffusion_samples_affinity: Option<i64>,
    preprocessing_threads: Option<usize>,
    override_flag: bool,
    write_full_pae: bool,
    write_full_pde: bool,
    spike_only: bool,
}

async fn predict_flow(args: PredictFlowArgs) -> Result<()> {
    let PredictFlowArgs {
        input,
        cache,
        out_dir,
        device,
        use_msa_server,
        ref msa_server_url,
        msa_pairing_strategy: _,
        affinity,
        use_potentials,
        recycling_steps,
        sampling_steps,
        diffusion_samples,
        max_parallel_samples,
        step_scale,
        output_format,
        max_msa_seqs,
        num_samples,
        ref checkpoint,
        ref affinity_checkpoint,
        affinity_mw_correction,
        sampling_steps_affinity,
        diffusion_samples_affinity,
        preprocessing_threads,
        override_flag,
        write_full_pae,
        write_full_pde,
        spike_only,
    } = args;

    // 1. Parse input (YAML / FASTA / directory)
    let input_path = Path::new(&input);
    let parsed = boltr_io::parse_input_path(input_path)?;
    tracing::info!(
        chains = ?parsed.summary_chain_ids(),
        "parsed input YAML"
    );

    if affinity {
        tracing::info!("--affinity: affinity inference path active");
    }
    if use_potentials {
        if affinity {
            tracing::warn!(
                "--use-potentials with --affinity: Boltz disables steering on affinity path; \
                 potentials will not be applied"
            );
        } else {
            tracing::info!("--use-potentials: inference potentials / steering enabled");
        }
    }

    // 2. Create output directory
    tokio::fs::create_dir_all(&out_dir).await?;

    // 3. Optional MSA server fetch
    if use_msa_server {
        let need = parsed.protein_sequences_for_msa();
        if !need.is_empty() {
            let proc = boltr_io::MsaProcessor::new(msa_server_url.clone());
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

    // 4. Determine backend note
    let backend_note = predict_backend_note();

    // 5. Determine boltr_predict_args.json path
    let boltr_predict_args_path = if cfg!(feature = "tch") {
        Some("boltr_predict_args.json".to_string())
    } else {
        None
    };

    // 6. Write run summary
    let summary = boltr_io::PredictionRunSummary::from_input(
        input.as_str(),
        &parsed,
        use_msa_server,
        device.as_str(),
        num_samples,
        backend_note,
        affinity,
        use_potentials,
        spike_only,
        boltr_predict_args_path.clone(),
    );
    let summary_path = out_dir.join("boltr_run_summary.json");
    summary.write_json(&summary_path)?;
    tracing::info!(path = %summary_path.display(), "wrote run summary");

    // 7. Backend-specific predict
    #[cfg(feature = "tch")]
    {
        use boltr_backend_tch::PredictArgsCliOverrides;

        let overrides = PredictArgsCliOverrides {
            recycling_steps,
            sampling_steps,
            diffusion_samples: diffusion_samples.or(Some(num_samples as i64)),
            max_parallel_samples,
        };

        predict_tch::run_predict_tch(predict_tch::PredictTchArgs {
            input_path: input_path.to_path_buf(),
            cache,
            out_dir,
            device,
            affinity,
            use_potentials,
            overrides,
            step_scale,
            output_format,
            max_msa_seqs,
            num_samples,
            checkpoint,
            affinity_checkpoint,
            affinity_mw_correction,
            sampling_steps_affinity,
            diffusion_samples_affinity,
            preprocessing_threads,
            override_flag,
            write_full_pae,
            write_full_pde,
            spike_only,
            parsed: &parsed,
        })
        .await?;
    }

    #[cfg(not(feature = "tch"))]
    {
        let _ = (
            &cache,
            recycling_steps,
            sampling_steps,
            diffusion_samples,
            max_parallel_samples,
            step_scale,
            output_format,
            max_msa_seqs,
            checkpoint,
            affinity_checkpoint,
            affinity_mw_correction,
            sampling_steps_affinity,
            diffusion_samples_affinity,
            preprocessing_threads,
            override_flag,
            write_full_pae,
            write_full_pde,
            &parsed,
        );
        if spike_only {
            tracing::warn!(
                "rebuild with `cargo build -p boltr-cli --features tch` for model execution"
            );
        }
        tracing::info!("predict pipeline complete (no tch backend linked)");
    }

    Ok(())
}

fn predict_backend_note() -> &'static str {
    #[cfg(feature = "tch")]
    {
        "tch backend linked; full predict pipeline available"
    }
    #[cfg(not(feature = "tch"))]
    {
        "rebuild with: cargo build -p boltr-cli --features tch (LibTorch required)"
    }
}

// ---------------------------------------------------------------------------
// msa-to-npz
// ---------------------------------------------------------------------------

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
    let lower = path.to_string_lossy().to_ascii_lowercase();
    if lower.ends_with(".csv") {
        boltr_io::parse_csv_path(path, max_seqs)
    } else {
        boltr_io::parse_a3m_path(path, max_seqs)
    }
}

fn default_msa_npz_path(input: &Path) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default();
    let base = Path::new(stem).file_stem().unwrap_or(stem);
    let parent = input.parent().unwrap_or(Path::new("."));
    parent.join(format!("{}.npz", base.to_string_lossy()))
}

// ---------------------------------------------------------------------------
// tokens-to-npz
// ---------------------------------------------------------------------------

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
