use std::collections::{HashMap, HashSet};

use crate::boltz_const::{chain_type_id, TOKENS};
use crate::structure_v2::StructureV2Tables;

#[derive(Debug, Clone)]
pub(crate) struct QcAtom {
    pub index: usize,
    pub name: String,
    pub coords: [f32; 3],
}

#[derive(Debug, Clone)]
pub(crate) struct QcResidue {
    pub chain_name: String,
    pub chain_idx: usize,
    pub residue_index: i32,
    pub residue_name: String,
    pub is_standard_protein: bool,
    pub atoms: HashMap<String, QcAtom>,
}

#[derive(Debug, Clone)]
pub(crate) struct QcChain {
    pub name: String,
    pub residues: Vec<QcResidue>,
}

fn protein_mol_type() -> i8 {
    chain_type_id("PROTEIN").expect("PROTEIN chain type") as i8
}

fn standard_protein_names() -> HashSet<&'static str> {
    TOKENS[2..22].iter().copied().collect()
}

pub(crate) fn protein_chains(structure: &StructureV2Tables) -> Vec<QcChain> {
    let protein = protein_mol_type();
    let standard = standard_protein_names();
    let mut chains = Vec::new();

    for (chain_idx, chain) in structure.chains.iter().enumerate() {
        if !structure
            .chain_mask
            .get(chain_idx)
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        if chain.mol_type != protein {
            continue;
        }
        let res_start = match usize::try_from(chain.res_idx) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let res_num = match usize::try_from(chain.res_num) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let res_end = res_start.saturating_add(res_num);
        if res_start >= structure.residues.len() || res_end > structure.residues.len() {
            continue;
        }
        let mut residues = Vec::new();
        for residue in &structure.residues[res_start..res_end] {
            if !residue.is_present {
                continue;
            }
            let atom_start = match usize::try_from(residue.atom_idx) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let atom_num = match usize::try_from(residue.atom_num) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let atom_end = atom_start.saturating_add(atom_num);
            if atom_start >= structure.atoms.len() || atom_end > structure.atoms.len() {
                continue;
            }
            let mut atoms = HashMap::new();
            for (k, atom) in structure.atoms[atom_start..atom_end].iter().enumerate() {
                if !atom.is_present {
                    continue;
                }
                atoms.insert(
                    atom.name.trim().to_ascii_uppercase(),
                    QcAtom {
                        index: atom_start + k,
                        name: atom.name.trim().to_string(),
                        coords: atom.coords,
                    },
                );
            }
            let residue_name = residue.name.trim().to_ascii_uppercase();
            residues.push(QcResidue {
                chain_name: chain.name.clone(),
                chain_idx,
                residue_index: residue.res_idx + 1,
                is_standard_protein: residue.is_standard
                    && standard.contains(residue_name.as_str()),
                residue_name,
                atoms,
            });
        }
        chains.push(QcChain {
            name: chain.name.clone(),
            residues,
        });
    }
    chains
}

pub(crate) fn display_chain_name(name: &str, chain_idx: usize) -> String {
    let t = name.trim();
    if t.is_empty() {
        format!("#{}", chain_idx + 1)
    } else {
        t.to_string()
    }
}

pub(crate) fn heavy_atoms(structure: &StructureV2Tables) -> Vec<QcAtom> {
    structure
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.is_present && !a.name.trim().to_ascii_uppercase().starts_with('H'))
        .map(|(index, a)| QcAtom {
            index,
            name: a.name.trim().to_string(),
            coords: a.coords,
        })
        .collect()
}
