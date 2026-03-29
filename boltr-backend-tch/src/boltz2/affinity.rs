//! Affinity head: `boltz.model.modules.affinity`.
//!
//! **Roadmap (TODO.md §5.8):** port `AffinityModule`, MW correction, ensemble if required.

#[derive(Debug, Default)]
pub struct AffinityHead;

impl AffinityHead {
    pub fn new() -> Self {
        Self
    }
}

/// Alias matching Boltz `AffinityModule` naming (`boltz.model.modules.affinity`).
pub type AffinityModule = AffinityHead;
