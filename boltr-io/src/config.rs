//! Boltz YAML input types (Boltz2-oriented). See `boltz-reference/docs/prediction.md`.

use serde::{Deserialize, Serialize};

/// Root document for a Boltz job input file.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BoltzInput {
    pub sequences: Vec<SequenceEntry>,
    #[serde(default)]
    pub constraints: Option<Vec<serde_yaml::Value>>,
    #[serde(default)]
    pub templates: Option<Vec<serde_yaml::Value>>,
    #[serde(default)]
    pub properties: Option<Vec<serde_yaml::Value>>,
    #[serde(default)]
    pub version: Option<u32>,
}

/// One chain / molecule entry under `sequences:`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SequenceEntry {
    Protein { protein: PolymerEntity },
    Dna { dna: PolymerEntity },
    Rna { rna: PolymerEntity },
    Ligand { ligand: LigandEntity },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ChainIdSpec {
    Single(String),
    Many(Vec<String>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PolymerEntity {
    pub id: ChainIdSpec,
    pub sequence: String,
    #[serde(default)]
    pub msa: Option<String>,
    #[serde(default)]
    pub cyclic: Option<bool>,
    #[serde(default)]
    pub modifications: Option<Vec<serde_yaml::Value>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LigandEntity {
    pub id: ChainIdSpec,
    #[serde(default)]
    pub smiles: Option<String>,
    #[serde(default)]
    pub ccd: Option<serde_yaml::Value>,
}

impl BoltzInput {
    /// Protein chains that need an MSA path or server fetch when `msa` is absent.
    pub fn protein_sequences_for_msa(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for entry in &self.sequences {
            if let SequenceEntry::Protein { protein } = entry {
                if protein.msa.is_some() {
                    continue;
                }
                let ids = match &protein.id {
                    ChainIdSpec::Single(s) => vec![s.clone()],
                    ChainIdSpec::Many(v) => v.clone(),
                };
                for id in ids {
                    out.push((id, protein.sequence.clone()));
                }
            }
        }
        out
    }

    pub fn summary_chain_ids(&self) -> Vec<String> {
        let mut ids = Vec::new();
        for entry in &self.sequences {
            match entry {
                SequenceEntry::Protein { protein } => ids.extend(collect_ids(&protein.id)),
                SequenceEntry::Dna { dna } => ids.extend(collect_ids(&dna.id)),
                SequenceEntry::Rna { rna } => ids.extend(collect_ids(&rna.id)),
                SequenceEntry::Ligand { ligand } => ids.extend(collect_ids(&ligand.id)),
            }
        }
        ids
    }
}

fn collect_ids(spec: &ChainIdSpec) -> Vec<String> {
    match spec {
        ChainIdSpec::Single(s) => vec![s.clone()],
        ChainIdSpec::Many(v) => v.clone(),
    }
}
