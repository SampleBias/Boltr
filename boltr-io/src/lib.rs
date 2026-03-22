// Boltr - Rust Native Boltz Implementation
// I/O Operations
//
// This crate handles all input/output operations including:
// - YAML/JSON config parsing
// - File I/O for sequences, structures, MSA data
// - MSA server communication
// - Output formatting

pub mod config;
pub mod download;
pub mod format;
pub mod msa;
pub mod parser;

pub use config::BoltzInput;
/// Backward-compatible name for parsed YAML root.
pub type BoltzConfig = BoltzInput;
pub use download::download_model_assets;
pub use format::PredictionRunSummary;
pub use msa::{write_a3m, MsaProcessor};
pub use parser::parse_input_path as parse_input;
pub use parser::{parse_input_path, parse_input_str};
