// Boltr - Rust Native Boltz Implementation
// I/O Operations
//
// This crate handles all input/output operations including:
// - YAML/JSON config parsing
// - File I/O for sequences, structures, MSA data
// - MSA server communication
// - Output formatting

pub mod a3m;
pub mod config;
pub mod download;
pub mod format;
pub mod msa;
pub mod msa_csv;
pub mod msa_npz;
pub mod parser;

pub use a3m::{parse_a3m_path, parse_a3m_str, A3mMsa, A3mSequenceMeta};
pub use msa_csv::{parse_csv_path, parse_csv_str};
pub use msa_npz::{read_msa_npz_bytes, read_msa_npz_path, write_msa_npz_compressed};
pub use config::BoltzInput;
/// Backward-compatible name for parsed YAML root.
pub type BoltzConfig = BoltzInput;
pub use download::download_model_assets;
pub use format::PredictionRunSummary;
pub use msa::{write_a3m, MsaProcessor};
pub use parser::parse_input_path as parse_input;
pub use parser::{parse_input_path, parse_input_str};
