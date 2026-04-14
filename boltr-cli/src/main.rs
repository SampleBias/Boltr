//! `boltr` CLI — Rust-native Boltz2 inference.
//!
//! Commands: `predict`, `download`, `doctor`, `eval`, `msa-to-npz`, `tokens-to-npz`,
//! `preprocess` (Boltz Tier 1 or native Tier 2 bundle next to YAML).
//!
//! The `predict` command mirrors the upstream `boltz predict` interface and output layout
//! ([boltz-reference/docs/prediction.md](../../boltz-reference/docs/prediction.md)).

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

mod doctor;

#[cfg(feature = "tch")]
mod collate_predict_bridge;
#[cfg(feature = "tch")]
mod predict_tch;

<<<<<<< HEAD
#[cfg(feature = "tch")]
mod cuda_mem;
mod preprocess_cmd;
=======
>>>>>>> afdffbc (Refactor code for improved readability and consistency)
mod device_resolve;
mod gpu_mem;
mod preprocess_cmd;

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
    /// Write both `.pdb` and `.cif` for the same model.
    Both,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Mmcif => write!(f, "mmcif"),
            OutputFormat::Pdb => write!(f, "pdb"),
            OutputFormat::Both => write!(f, "both"),
        }
    }
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Mmcif
    }
}

/// Optional preprocess before `predict` when the bundle beside the YAML is missing or incomplete.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
enum PreprocessCli {
    /// Do not generate a preprocess bundle.
    Off,
    /// Prefer Rust native bundle for protein-only YAML; else try upstream Boltz if on `PATH`.
    #[default]
    Auto,
    /// Run upstream `boltz predict` in a staging dir and copy `manifest.json` + `.npz` next to the YAML.
    Boltz,
    /// Rust-only protein-only bundle (placeholder coordinates); see `docs/PREPROCESS_NATIVE.md`.
    Native,
}

#[derive(Parser, Debug)]
#[command(name = "boltr")]
#[command(before_help = include_str!("cli_banner.txt"))]
#[command(before_long_help = include_str!("cli_banner.txt"))]
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

        /// Compute device: `auto` (CUDA if LibTorch sees a GPU and optional VRAM check via `BOLTR_AUTO_MIN_FREE_VRAM_MB`; else CPU), `cpu`, `gpu` (CUDA required), `cuda`, or `cuda:N` (requires `--features tch` and a CUDA-capable LibTorch for GPU).
        #[arg(long, default_value = "auto")]
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

        /// Output structure format (`both` writes PDB and mmCIF).
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

        /// When the preprocess bundle beside the YAML is missing or incomplete, generate it first (recommended for structure output).
        #[arg(long, value_enum, default_value_t = PreprocessCli::Auto)]
        preprocess: PreprocessCli,

        /// Upstream Boltz executable for `--preprocess boltz` / `--preprocess auto` fallback.
        #[arg(long, default_value = "boltz")]
        bolt_command: String,

        /// Staging directory for `--preprocess boltz` (default: temp dir under `std::env::temp_dir()`).
        #[arg(long)]
        preprocess_staging: Option<PathBuf>,

        /// Keep Boltz staging directory after `--preprocess boltz`.
        #[arg(long, default_value_t = false)]
        preprocess_keep_staging: bool,

        /// Symlink instead of copy when materializing preprocess files next to the YAML.
        #[arg(long, default_value_t = false)]
        preprocess_symlink: bool,

        /// Extra arguments forwarded to `boltz predict` (repeat flag for multiple args).
        #[arg(long = "preprocess-bolt-arg")]
        preprocess_bolt_arg: Vec<String>,

        /// Record id for `--preprocess native` (`manifest.records[0].id`); default: YAML stem.
        #[arg(long)]
        preprocess_record_id: Option<String>,

        /// Directory of CCD `*.json` files for ligand/extra residue chemistry (passed to `load_input` as `extra_mols_dir`).
        #[arg(long)]
        extra_mols_dir: Option<PathBuf>,

        /// Directory containing `{record_id}.npz` residue constraint files (Boltz preprocess layout).
        #[arg(long)]
        constraints_dir: Option<PathBuf>,

        /// When set, look beside the YAML for `mols`/`extra_mols` and `constraints` if the corresponding `--*-dir` flag is omitted.
        #[arg(long, default_value_t = false)]
        preprocess_auto_extras: bool,

        /// Ensemble reference indices for atom featurization: `single` (index 0) or `multi` (up to 5 conformers).
        #[arg(long, value_enum, default_value_t = EnsembleRefCli::Single)]
        ensemble_ref: EnsembleRefCli,

        /// `CUDA_VISIBLE_DEVICES` for the Boltz preprocess subprocess only (e.g. `1` so Boltz uses another GPU than LibTorch `cuda:0`). Overrides `BOLTR_BOLTZ_CUDA_VISIBLE_DEVICES`.
        #[arg(long)]
        preprocess_cuda_visible_devices: Option<String>,

        /// Force upstream Boltz `--accelerator cpu` when LibTorch uses CUDA (OOM fallback). Env: `BOLTR_PREPROCESS_BOLTZ_CPU=1`. With `--device auto` on a single visible GPU, Boltz defaults to CPU unless `BOLTR_AUTO_PREPROCESS_BOLTZ_CPU=0`.
        #[arg(long, default_value_t = false)]
        preprocess_boltz_cpu: bool,

        /// With `--device auto` on a single visible GPU, allow Boltz preprocess to use GPU (faster; higher peak VRAM vs LibTorch). Env: `BOLTR_AUTO_BOLTZ_GPU=1`.
        #[arg(long, default_value_t = false)]
        preprocess_auto_boltz_gpu: bool,

        /// After Boltz preprocess, run `torch.cuda.empty_cache()` via a Python that can `import torch` (see `BOLTR_PYTHON`, venv next to `--bolt-command`, or `BOLTR_REPO/.venv`). On by default when `--device` resolves to CUDA; set `BOLTR_PREPROCESS_POST_BOLTZ_EMPTY_CACHE=0` to disable.
        #[arg(long, default_value_t = false)]
        preprocess_post_boltz_empty_cache: bool,
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

    /// Check LibTorch / tch linkage (CPU tensor smoke). Use `--json` for machine-readable output.
    Doctor {
        /// Print JSON (for boltr-web and scripts).
        #[arg(long)]
        json: bool,
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

    /// Generate Boltz-style preprocess bundle (`manifest.json` + `.npz`) next to a YAML file.
    Preprocess {
        /// Input `.yaml` / `.yml` (same directory receives the bundle).
        input: PathBuf,

        /// `boltz`: Tier 1 subprocess to upstream Boltz. `native`: Tier 2 Rust-only (proteins only).
        #[arg(long, value_enum, default_value_t = PreprocessBundleMode::Boltz)]
        mode: PreprocessBundleMode,

        /// Boltz executable when `mode=boltz`.
        #[arg(long, default_value = "boltz")]
        bolt_command: String,

        /// Staging directory for Boltz output (default: ephemeral temp dir).
        #[arg(long)]
        staging: Option<PathBuf>,

        /// Keep staging directory after success.
        #[arg(long, default_value_t = false)]
        keep_staging: bool,

        /// Symlink preprocess files instead of copying.
        #[arg(long, default_value_t = false)]
        symlink: bool,

        /// Forward `--use_msa_server` to Boltz when `mode=boltz`.
        #[arg(long, default_value_t = false)]
        use_msa_server: bool,

        /// Extra args for `boltz predict`.
        #[arg(long = "bolt-arg")]
        bolt_arg: Vec<String>,

        /// `CUDA_VISIBLE_DEVICES` for the Boltz subprocess only (e.g. `1`). Env: `BOLTR_BOLTZ_CUDA_VISIBLE_DEVICES`.
        #[arg(long)]
        cuda_visible_devices: Option<String>,

        /// Record id for `mode=native` (`manifest.records[0].id`).
        #[arg(long)]
        record_id: Option<String>,

        /// Max MSA sequences when parsing `.a3m` for `mode=native`.
        #[arg(long)]
        max_msa_seqs: Option<usize>,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
enum PreprocessBundleMode {
    #[default]
    Boltz,
    Native,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
enum EnsembleRefCli {
    /// Single ensemble index `0` (default; checkpoint-safe).
    #[default]
    Single,
    /// Multiple conformers (bounded by structure; experimental).
    Multi,
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
            preprocess,
            bolt_command,
            preprocess_staging,
            preprocess_keep_staging,
            preprocess_symlink,
            preprocess_bolt_arg,
            preprocess_record_id,
            extra_mols_dir,
            constraints_dir,
            preprocess_auto_extras,
            ensemble_ref,
            preprocess_cuda_visible_devices,
            preprocess_boltz_cpu,
            preprocess_auto_boltz_gpu,
            preprocess_post_boltz_empty_cache,
        } => {
            let cache = resolve_cache_dir(cache_dir.as_deref());
            let out_dir = Path::new(&output).to_path_buf();
            // `BOLTR_DEVICE` replaces CLI `--device` when set (may be `auto`, `cuda:1`, …).
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
                preprocess,
                bolt_command,
                preprocess_staging,
                preprocess_keep_staging,
                preprocess_symlink,
                preprocess_bolt_arg,
                preprocess_record_id,
                extra_mols_dir,
                constraints_dir,
                preprocess_auto_extras,
                ensemble_ref,
                preprocess_cuda_visible_devices,
                preprocess_boltz_cpu,
                preprocess_auto_boltz_gpu,
                preprocess_post_boltz_empty_cache,
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
            if version.to_lowercase() == "boltz2" || version == "2" {
                try_export_safetensors_after_download(&cache);
            }
        }

        Commands::Doctor { json } => {
            doctor::run(json)?;
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

        Commands::Preprocess {
            input,
            mode,
            bolt_command,
            staging,
            keep_staging,
            symlink,
            use_msa_server,
            bolt_arg,
            cuda_visible_devices,
            record_id,
            max_msa_seqs,
        } => match mode {
            PreprocessBundleMode::Boltz => {
                let vis = preprocess_cmd::resolve_preprocess_cuda_visible_devices(
                    cuda_visible_devices.as_deref(),
                );
                let child = preprocess_cmd::BoltzChildEnv {
                    cuda_visible_devices: vis,
                    pytorch_cuda_alloc_conf: preprocess_cmd::resolve_boltz_pytorch_alloc_conf(true),
                };
                preprocess_cmd::run_boltz_preprocess(
                    &input,
                    &bolt_command,
                    staging,
                    use_msa_server,
                    &bolt_arg,
                    symlink,
                    keep_staging,
                    &child,
                )?;
            }
            PreprocessBundleMode::Native => preprocess_cmd::run_native_preprocess(
                &input,
                record_id.as_deref(),
                max_msa_seqs,
                None,
            )?,
        },
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// predict flow
// ---------------------------------------------------------------------------

/// True if `cmd` is an existing file path, or resolves via [`which::which`].
fn command_is_runnable(cmd: &str) -> bool {
    let p = Path::new(cmd);
    if p.is_absolute() || cmd.contains('/') {
        return p.is_file();
    }
    which::which(cmd).is_ok()
}

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
    preprocess: PreprocessCli,
    bolt_command: String,
    preprocess_staging: Option<PathBuf>,
    preprocess_keep_staging: bool,
    preprocess_symlink: bool,
    preprocess_bolt_arg: Vec<String>,
    preprocess_record_id: Option<String>,
    extra_mols_dir: Option<PathBuf>,
    constraints_dir: Option<PathBuf>,
    preprocess_auto_extras: bool,
    ensemble_ref: EnsembleRefCli,
    preprocess_cuda_visible_devices: Option<String>,
    preprocess_boltz_cpu: bool,
    preprocess_auto_boltz_gpu: bool,
    preprocess_post_boltz_empty_cache: bool,
}

async fn predict_flow(args: PredictFlowArgs) -> Result<()> {
    let PredictFlowArgs {
        input,
        cache,
        out_dir,
        device: device_in,
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
        preprocess,
        bolt_command,
        preprocess_staging,
        preprocess_keep_staging,
        preprocess_symlink,
        preprocess_bolt_arg,
        preprocess_record_id,
        extra_mols_dir,
        constraints_dir,
        preprocess_auto_extras,
        ensemble_ref,
        preprocess_cuda_visible_devices,
        preprocess_boltz_cpu,
        preprocess_auto_boltz_gpu,
        preprocess_post_boltz_empty_cache,
    } = args;

    let (mut device, device_requested) = device_resolve::resolve_predict_device(&device_in)?;
    device = device_resolve::maybe_apply_auto_vram_gate(device, device_requested.as_deref());
    if let Some(ref req) = device_requested {
        tracing::info!(requested = %req, resolved = %device, "device resolution");
    } else if device_in.trim() != device {
        tracing::info!(requested = %device_in, resolved = %device, "device resolution");
    } else {
        tracing::info!(device = %device, "device");
    }

    // 1. Parse input (YAML / FASTA / directory)
    let input_path = Path::new(&input);
    let parsed = boltr_io::parse_input_path(input_path)?;
    tracing::info!(
        chains = ?parsed.summary_chain_ids(),
        "parsed input YAML"
    );

    if affinity {
        tracing::info!("--affinity: affinity inference path active");
        if matches!(preprocess, PreprocessCli::Native) {
            tracing::warn!(
                "--preprocess native does not emit pre_affinity npz; use Boltz preprocess or omit --affinity"
            );
        }
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

    // Canonical YAML parent matches preprocess writers (`canonicalize`); relative CLI paths
    // still resolve to the same `manifest.json` + `.npz` directory as the predict bridge.
    let yaml_parent = boltr_io::canonical_yaml_parent(input_path)?;
    let bundle_ready = boltr_io::preprocess_bundle_ready(input_path, affinity)?;
    let manifest_missing = !bundle_ready;
    let msa_under_yaml_for_native = manifest_missing
        && matches!(preprocess, PreprocessCli::Native | PreprocessCli::Auto)
        && use_msa_server
        && boltr_io::validate_native_eligible(&parsed).is_ok();

    let predict_cuda = preprocess_cmd::predict_device_is_cuda(&device);
<<<<<<< HEAD
    let force_boltz_cpu =
        preprocess_cmd::resolve_force_boltz_cpu(preprocess_boltz_cpu);
    let auto_boltz_gpu_opt_out =
        preprocess_cmd::resolve_auto_boltz_gpu_opt_out(preprocess_auto_boltz_gpu);
    let auto_default_boltz_cpu = preprocess_cmd::auto_default_boltz_cpu_for_memory(
        device_requested.as_deref(),
        predict_cuda,
        auto_boltz_gpu_opt_out,
    );
    if auto_default_boltz_cpu {
        tracing::info!(
            "preprocess: Boltz subprocess --accelerator cpu (--device auto, single visible GPU; use --preprocess-auto-boltz-gpu or BOLTR_AUTO_BOLTZ_GPU=1 for GPU Boltz)"
        );
    }
=======
    let force_boltz_cpu = preprocess_cmd::resolve_force_boltz_cpu(preprocess_boltz_cpu)
        || preprocess_cmd::resolve_auto_default_boltz_cpu(
            device_requested.as_deref(),
            predict_cuda,
        );
>>>>>>> afdffbc (Refactor code for improved readability and consistency)
    let bolt_preprocess_args = preprocess_cmd::bolt_preprocess_args_for_predict(
        &device,
        &preprocess_bolt_arg,
        force_boltz_cpu,
        auto_default_boltz_cpu,
    );
    let bolt_child_env = preprocess_cmd::BoltzChildEnv {
        cuda_visible_devices: preprocess_cmd::resolve_preprocess_cuda_visible_devices(
            preprocess_cuda_visible_devices.as_deref(),
        ),
        pytorch_cuda_alloc_conf: preprocess_cmd::resolve_boltz_pytorch_alloc_conf(predict_cuda),
    };

    // 3. Optional MSA server fetch
    if use_msa_server {
        let need = parsed.protein_sequences_for_msa();
        if !need.is_empty() {
            let proc = boltr_io::MsaProcessor::new(msa_server_url.clone());
            let seqs: Vec<String> = need.iter().map(|(_, s)| s.clone()).collect();
            tracing::info!(n = seqs.len(), "fetching MSAs from server");
            let msas = proc.fetch_msas(&seqs, true, true).await?;
            let msa_dir = if msa_under_yaml_for_native {
                yaml_parent.join("msa")
            } else {
                out_dir.join("msa")
            };
            tokio::fs::create_dir_all(&msa_dir).await?;
            for ((chain_id, _), a3m) in need.iter().zip(msas.iter()) {
                let path = msa_dir.join(format!("{chain_id}.a3m"));
                boltr_io::write_a3m(&path, a3m)?;
                tracing::info!(path = %path.display(), "wrote MSA");
            }
        }
    }

    // 3b. Optional preprocess bundle next to YAML (for `boltr predict` + `load_input` bridge)
    let mut preprocess_ran_boltz = false;
    if manifest_missing && preprocess != PreprocessCli::Off {
        match preprocess {
            PreprocessCli::Off => {}
            PreprocessCli::Native => {
                tracing::info!("--preprocess native: writing manifest + npz next to YAML");
                let fetched = if use_msa_server {
                    Some(yaml_parent.join("msa"))
                } else {
                    None
                };
                preprocess_cmd::run_native_preprocess(
                    input_path,
                    preprocess_record_id.as_deref(),
                    Some(max_msa_seqs),
                    fetched.as_deref(),
                )?;
            }
            PreprocessCli::Boltz => {
                tracing::info!("--preprocess boltz: running upstream Boltz");
                preprocess_ran_boltz = true;
                preprocess_cmd::run_boltz_preprocess(
                    input_path,
                    &bolt_command,
                    preprocess_staging.clone(),
                    use_msa_server,
                    &bolt_preprocess_args,
                    preprocess_symlink,
                    preprocess_keep_staging,
                    &bolt_child_env,
                )?;
            }
            PreprocessCli::Auto => {
                if boltr_io::validate_native_eligible(&parsed).is_ok() {
                    tracing::info!("--preprocess auto: using native protein-only bundle");
                    let fetched = if use_msa_server {
                        Some(yaml_parent.join("msa"))
                    } else {
                        None
                    };
                    preprocess_cmd::run_native_preprocess(
                        input_path,
                        preprocess_record_id.as_deref(),
                        Some(max_msa_seqs),
                        fetched.as_deref(),
                    )?;
                } else if command_is_runnable(&bolt_command) {
                    preprocess_ran_boltz = true;
                    preprocess_cmd::run_boltz_preprocess(
                        input_path,
                        &bolt_command,
                        preprocess_staging.clone(),
                        use_msa_server,
                        &bolt_preprocess_args,
                        preprocess_symlink,
                        preprocess_keep_staging,
                        &bolt_child_env,
                    )?;
                } else {
                    bail!(
                        "--preprocess auto: input is not eligible for native preprocess (e.g. non-empty templates/constraints, ligands, …) \
                         and `{bolt_command}` not found on PATH. Install upstream Boltz or pass --bolt-command to its executable, or use --preprocess off with an existing manifest.json beside the YAML."
                    );
                }
            }
        }
    }

    if preprocess_ran_boltz
        && predict_cuda
        && preprocess_cmd::resolve_post_boltz_empty_cache(
            preprocess_post_boltz_empty_cache,
            predict_cuda,
        )
    {
        preprocess_cmd::maybe_post_boltz_empty_cache(Some(bolt_command.as_str()))?;
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
        device_requested.clone(),
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

        let ensemble_mode = match ensemble_ref {
            EnsembleRefCli::Single => boltr_io::InferenceEnsembleMode::Single,
            EnsembleRefCli::Multi => boltr_io::InferenceEnsembleMode::Multi,
        };

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
            checkpoint: checkpoint.clone(),
            affinity_checkpoint: affinity_checkpoint.clone(),
            affinity_mw_correction,
            sampling_steps_affinity,
            diffusion_samples_affinity,
            preprocessing_threads,
            override_flag,
            write_full_pae,
            write_full_pde,
            spike_only,
            extra_mols_dir: extra_mols_dir.clone(),
            constraints_dir: constraints_dir.clone(),
            preprocess_auto_extras,
            ensemble_mode,
            device_requested,
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
            extra_mols_dir,
            constraints_dir,
            preprocess_auto_extras,
            ensemble_ref,
            diffusion_samples_affinity,
            preprocessing_threads,
            override_flag,
            write_full_pae,
            write_full_pde,
            &parsed,
        );
        // Same layout as `predict_tch` placeholder so UIs can parse `status` / `note`.
        let record_id = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("prediction")
            .to_string();
        let record_dir = out_dir.join(&record_id);
        tokio::fs::create_dir_all(&record_dir).await?;
        let marker = record_dir.join("boltr_predict_complete.txt");
        let info = serde_json::json!({
            "record_id": record_id,
            "status": "no_tch_backend",
            "affinity": affinity,
            "note": "This boltr binary was built without --features tch. Rebuild with `cargo build --release -p boltr-cli --features tch` or `cargo build-boltr` (LibTorch is fetched automatically); set `BOLTR` to that binary."
        });
        let j = serde_json::to_string_pretty(&info)?;
        tokio::fs::write(&marker, j).await?;
        tracing::info!(path = %marker.display(), "wrote boltr_predict_complete.txt (no tch backend)");

        if spike_only {
            tracing::warn!(
                "rebuild with `cargo build --release -p boltr-cli --features tch` or `cargo build-boltr` for model execution"
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
        "rebuild with: cargo build --release -p boltr-cli --features tch or cargo build-boltr (LibTorch fetched automatically)"
    }
}

/// Locate repo root containing `scripts/export_checkpoint_to_safetensors.py` (cwd, then exe ancestors).
fn boltr_repo_root() -> Option<PathBuf> {
    if let Ok(cwd) = std::env::current_dir() {
        let mut d = cwd;
        for _ in 0..12 {
            if d.join("scripts/export_checkpoint_to_safetensors.py")
                .is_file()
            {
                return Some(d);
            }
            if !d.pop() {
                break;
            }
        }
    }
    let mut d = std::env::current_exe().ok()?.parent()?.to_path_buf();
    for _ in 0..12 {
        if d.join("scripts/export_checkpoint_to_safetensors.py")
            .is_file()
        {
            return Some(d);
        }
        if !d.pop() {
            break;
        }
    }
    None
}

/// Best-effort `.ckpt` → `.safetensors` after `boltr download` (warn-only on failure).
fn try_export_safetensors_after_download(cache: &Path) {
    let Some(repo) = boltr_repo_root() else {
        tracing::warn!(
            "could not locate Boltr repo (scripts/export_checkpoint_to_safetensors.py); \
             skipping automatic safetensors export. Export manually:\n  \
             python scripts/export_checkpoint_to_safetensors.py {} {}",
            cache.join("boltz2_conf.ckpt").display(),
            cache.join("boltz2_conf.safetensors").display()
        );
        return;
    };
    let script = repo.join("scripts/export_checkpoint_to_safetensors.py");
    let py = if repo.join(".venv/bin/python").is_file() {
        repo.join(".venv/bin/python")
    } else {
        PathBuf::from("python3")
    };
    for (ckpt_name, sf_name) in [
        ("boltz2_conf.ckpt", "boltz2_conf.safetensors"),
        ("boltz2_aff.ckpt", "boltz2_aff.safetensors"),
    ] {
        let ckpt = cache.join(ckpt_name);
        let out = cache.join(sf_name);
        if !ckpt.is_file() {
            continue;
        }
        let status = std::process::Command::new(&py)
            .arg(&script)
            .arg(&ckpt)
            .arg(&out)
            .status();
        match status {
            Ok(s) if s.success() => {
                tracing::info!(path = %out.display(), "exported safetensors after download");
            }
            Ok(s) => {
                tracing::warn!(
                    code = ?s.code(),
                    ckpt = %ckpt.display(),
                    out = %out.display(),
                    "export_checkpoint_to_safetensors failed; run manually (see DEVELOPMENT.md)"
                );
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    py = %py.display(),
                    "could not run Python for safetensors export; use repo .venv or: python scripts/export_checkpoint_to_safetensors.py"
                );
                return;
            }
        }
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
