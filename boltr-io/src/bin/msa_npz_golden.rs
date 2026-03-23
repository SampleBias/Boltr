//! Golden MSA fixture for `scripts/verify_msa_npz_golden.py` — keep payloads in sync.
#![forbid(unsafe_code)]

use std::path::Path;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use boltr_io::{read_msa_npz_path, write_msa_npz_compressed, A3mMsa, A3mSequenceMeta};

fn golden_msa() -> A3mMsa {
    A3mMsa {
        residues: vec![2, 3, 4],
        deletions: vec![(1, 2)],
        sequences: vec![A3mSequenceMeta {
            seq_idx: 0,
            taxonomy_id: 9606,
            res_start: 0,
            res_end: 3,
            del_start: 0,
            del_end: 1,
        }],
    }
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  msa_npz_golden write <path.npz>\n  msa_npz_golden check <path.npz>   # must match golden fixture"
    );
    std::process::exit(2);
}

fn main() -> ExitCode {
    if let Err(e) = run() {
        eprintln!("{e:#}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| usage());
    let path = args.next();
    match cmd.as_str() {
        "write" => {
            let path = path.context("write: missing path")?;
            write_msa_npz_compressed(Path::new(&path), &golden_msa())?;
        }
        "check" => {
            let path = path.context("check: missing path")?;
            let loaded = read_msa_npz_path(Path::new(&path))?;
            if loaded != golden_msa() {
                bail!("loaded MSA does not match golden fixture");
            }
        }
        _ => usage(),
    }
    Ok(())
}
