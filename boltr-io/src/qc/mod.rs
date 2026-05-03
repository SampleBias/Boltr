//! Protein-geometry QC and conservative coordinate relaxation for predicted structures.
//!
//! The validator works directly on [`crate::StructureV2Tables`] before serialization so every
//! writer sees the same pass/fail decision.

mod geometry;
mod protein;
mod relax;
mod report;
mod validate;

pub use relax::{relax_structure, RelaxationOutcome};
pub use report::{
    render_qc_text, BackboneBondDistance, ChainBreak, MissingBackboneAtom, PeptideBondDistance,
    QcReport, QcThresholds, StericClash, TorsionMeasure,
};
pub use validate::validate_structure_qc;
