//! Shared helpers for `boltr-cli` integration tests.

use std::fs;
use std::path::{Path, PathBuf};

/// Temp roots under `target/` (not system `/tmp`) so tests survive user quota on `/tmp`.
pub fn test_temp_root() -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("boltr-cli-test-tmp");
    fs::create_dir_all(&root).expect("create test temp root under target/");
    root
}
