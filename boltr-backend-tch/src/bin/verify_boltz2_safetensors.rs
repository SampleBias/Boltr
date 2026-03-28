//! Compare a `.safetensors` file against [`Boltz2Model`] VarStore names (missing / unused keys).
//!
//! Model shape flags must match the export (defaults: token_s=384, token_z=128, 4 pairformer blocks).
//! Use `--strip-prefix model.` when exporting Lightning checkpoints (see `export_checkpoint_to_safetensors.py`).
//!
//! Exit code 1 if any VarStore key is missing from the file (same condition as
//! [`Boltz2Model::load_from_safetensors_require_all_vars`]).

use std::path::PathBuf;

use boltr_backend_tch::Boltz2Model;
use boltr_backend_tch::partition_safetensors_keys_for_inference;
use tch::Device;

fn usage() -> ! {
    eprintln!(
        "\
Usage: verify_boltz2_safetensors [OPTIONS] <PATH.safetensors>

Options:
  --token-s N           sequence width (default 384)
  --token-z N           pair width (default 128)
  --blocks N            pairformer depth (default 4)
  --bond-type-feature   match checkpoints with bond_type_feature=true
  --partition           print inference vs other key counts (see BOLTZ2_INFERENCE_TOP_LEVEL_KEYS)
"
    );
    std::process::exit(2);
}

fn main() {
    tch::maybe_init_cuda();
    let mut token_s = 384_i64;
    let mut token_z = 128_i64;
    let mut blocks: Option<i64> = None;
    let mut bond_type = false;
    let mut partition = false;
    let mut path: Option<PathBuf> = None;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--token-s" => {
                i += 1;
                token_s = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage());
                i += 1;
            }
            "--token-z" => {
                i += 1;
                token_z = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage());
                i += 1;
            }
            "--blocks" => {
                i += 1;
                let b: i64 = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage());
                blocks = Some(b);
                i += 1;
            }
            "--bond-type-feature" => {
                bond_type = true;
                i += 1;
            }
            "--partition" => {
                partition = true;
                i += 1;
            }
            s if s.starts_with('-') => {
                eprintln!("unknown flag: {s}");
                usage();
            }
            _ => {
                if path.is_some() {
                    usage();
                }
                path = Some(PathBuf::from(&args[i]));
                i += 1;
            }
        }
    }

    let path = path.unwrap_or_else(|| usage());
    if !path.is_file() {
        eprintln!("not a file: {}", path.display());
        std::process::exit(2);
    }

    let model = if bond_type {
        Boltz2Model::with_options_bonds(Device::Cpu, token_s, token_z, blocks, true)
    } else {
        Boltz2Model::with_options(Device::Cpu, token_s, token_z, blocks)
    };

    let missing = match model.var_store_keys_missing_in_safetensors(path.as_path()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
    };
    let extra = match model.safetensors_keys_unused_by_model(path.as_path()) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{e:#}");
            std::process::exit(1);
        }
    };

    if partition {
        let names = match boltr_backend_tch::list_safetensor_names(path.as_path()) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("{e:#}");
                std::process::exit(1);
            }
        };
        let (infer, other) = partition_safetensors_keys_for_inference(&names);
        eprintln!(
            "Partition: {} inference-related keys, {} other keys (diffusion/confidence/affinity/template/…)",
            infer.len(),
            other.len()
        );
        eprintln!("Inference top-level prefixes: {:?}", boltr_backend_tch::BOLTZ2_INFERENCE_TOP_LEVEL_KEYS);
    }

    let n_vs = model.var_store().variables().len();
    eprintln!("VarStore parameters: {n_vs}");
    eprintln!("Missing in file ({}):", missing.len());
    for k in missing.iter().take(50) {
        eprintln!("  {k}");
    }
    if missing.len() > 50 {
        eprintln!("  ... and {} more", missing.len() - 50);
    }
    eprintln!("Unused file keys ({}):", extra.len());
    for k in extra.iter().take(20) {
        eprintln!("  {k}");
    }
    if extra.len() > 20 {
        eprintln!("  ... and {} more", extra.len() - 20);
    }

    if !missing.is_empty() {
        eprintln!(
            "\nFAIL: {} VarStore keys absent from {}; fix Path segments / export --strip-prefix (see DEVELOPMENT.md).",
            missing.len(),
            path.display()
        );
        std::process::exit(1);
    }
    eprintln!("OK: every VarStore key is present in the file.");
}
