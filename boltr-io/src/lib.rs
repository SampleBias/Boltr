// Boltr - Rust Native Boltz Implementation
// I/O Operations
//
// This crate handles all input/output operations including:
// - YAML/JSON config parsing
// - File I/O for sequences, structures, MSA data
// - MSA server communication
// - Output formatting

pub mod config;
pub mod parser;
pub mod msa;
pub mod format;

pub use config::BoltzConfig;
pub use parser::parse_input;
