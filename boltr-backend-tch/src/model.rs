//! Top-level model entry points.

#[cfg(feature = "tch-backend")]
pub use crate::boltz2::Boltz2Model as BoltzModel;

#[cfg(not(feature = "tch-backend"))]
use anyhow::Result;

#[cfg(not(feature = "tch-backend"))]
#[derive(Debug, Default)]
pub struct BoltzModel;

#[cfg(not(feature = "tch-backend"))]
impl BoltzModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn forward(&self) -> Result<()> {
        anyhow::bail!("build with --features tch-backend (LibTorch required)")
    }
}
