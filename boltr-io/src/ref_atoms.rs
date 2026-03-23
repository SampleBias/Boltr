//! Boltz `ref_atoms`, backbone atom names, and center/disto atom indices from `data/const.py`.
//!
//! Keys match Python `ref_atoms` (`PAD`, not `<pad>`). Use [`ref_atoms_key_from_token`] for tokenizer names.

/// Map tokenizer residue name to Python `ref_atoms` key (`<pad>` → `PAD`).
#[inline]
pub fn ref_atoms_key_from_token(token_name: &str) -> &str {
    match token_name {
        "<pad>" => "PAD",
        other => other,
    }
}

/// Reference atom name list per residue token (Python `ref_atoms` values).
#[must_use]
pub fn ref_atom_names(ref_key: &str) -> Option<&'static [&'static str]> {
    Some(match ref_key {
        "PAD" => &[] as &[&str],
        "UNK" => &["N", "CA", "C", "O", "CB"],
        "-" => &[],
        "ALA" => &["N", "CA", "C", "O", "CB"],
        "ARG" => &["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
        "ASN" => &["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
        "ASP" => &["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
        "CYS" => &["N", "CA", "C", "O", "CB", "SG"],
        "GLN" => &["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
        "GLU" => &["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
        "GLY" => &["N", "CA", "C", "O"],
        "HIS" => &["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "ILE" => &["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
        "LEU" => &["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
        "LYS" => &["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
        "MET" => &["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
        "PHE" => &["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "PRO" => &["N", "CA", "C", "O", "CB", "CG", "CD"],
        "SER" => &["N", "CA", "C", "O", "CB", "OG"],
        "THR" => &["N", "CA", "C", "O", "CB", "OG1", "CG2"],
        "TRP" => &[
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3",
            "CH2",
        ],
        "TYR" => &["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
        "VAL" => &["N", "CA", "C", "O", "CB", "CG1", "CG2"],
        "A" => &[
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9",
            "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4",
        ],
        "G" => &[
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9",
            "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4",
        ],
        "C" => &[
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1",
            "C2", "O2", "N3", "C4", "N4", "C5", "C6",
        ],
        "U" => &[
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1",
            "C2", "O2", "N3", "C4", "O4", "C5", "C6",
        ],
        "N" => &[
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
        ],
        "DA" => &[
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8",
            "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4",
        ],
        "DG" => &[
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8",
            "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4",
        ],
        "DC" => &[
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2",
            "O2", "N3", "C4", "N4", "C5", "C6",
        ],
        "DT" => &[
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2",
            "O2", "N3", "C4", "O4", "C5", "C7", "C6",
        ],
        "DN" => &[
            "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
        ],
        _ => return None,
    })
}

/// Lookup using a tokenizer token name (e.g. `<pad>` → empty list).
#[inline]
pub fn ref_atom_names_for_token(token_name: &str) -> Option<&'static [&'static str]> {
    ref_atom_names(ref_atoms_key_from_token(token_name))
}

pub const PROTEIN_BACKBONE_ATOM_NAMES: [&str; 4] = ["N", "CA", "C", "O"];

pub const NUCLEIC_BACKBONE_ATOM_NAMES: [&str; 12] = [
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
];

#[inline]
pub fn protein_backbone_atom_index(name: &str) -> Option<usize> {
    PROTEIN_BACKBONE_ATOM_NAMES.iter().position(|&n| n == name)
}

#[inline]
pub fn nucleic_backbone_atom_index(name: &str) -> Option<usize> {
    NUCLEIC_BACKBONE_ATOM_NAMES.iter().position(|&n| n == name)
}

/// `res_to_center_atom` → index into [`ref_atom_names`].
#[must_use]
pub fn center_atom_index(ref_key: &str) -> Option<usize> {
    let atom = center_atom_name(ref_key)?;
    let names = ref_atom_names(ref_key)?;
    names.iter().position(|&n| n == atom)
}

/// `res_to_disto_atom` → index into [`ref_atom_names`].
#[must_use]
pub fn disto_atom_index(ref_key: &str) -> Option<usize> {
    let atom = disto_atom_name(ref_key)?;
    let names = ref_atom_names(ref_key)?;
    names.iter().position(|&n| n == atom)
}

fn center_atom_name(ref_key: &str) -> Option<&'static str> {
    Some(match ref_key {
        "UNK" | "ALA" | "ARG" | "ASN" | "ASP" | "CYS" | "GLN" | "GLU" | "GLY" | "HIS" | "ILE"
        | "LEU" | "LYS" | "MET" | "PHE" | "PRO" | "SER" | "THR" | "TRP" | "TYR" | "VAL" => "CA",
        "A" | "G" | "C" | "U" | "N" | "DA" | "DG" | "DC" | "DT" | "DN" => "C1'",
        _ => return None,
    })
}

fn disto_atom_name(ref_key: &str) -> Option<&'static str> {
    Some(match ref_key {
        "UNK" | "ALA" | "ARG" | "ASN" | "ASP" | "CYS" | "GLN" | "GLU" | "HIS" | "ILE" | "LEU"
        | "LYS" | "MET" | "PHE" | "PRO" | "SER" | "THR" | "TRP" | "TYR" | "VAL" => "CB",
        "GLY" => "CA",
        "A" | "G" | "DA" | "DG" => "C4",
        "C" | "U" | "DC" | "DT" => "C2",
        "N" | "DN" => "C1'",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_and_gap_empty() {
        assert_eq!(ref_atom_names("PAD"), Some(&[] as &[&str]));
        assert_eq!(ref_atom_names("-"), Some(&[] as &[&str]));
        assert_eq!(ref_atom_names_for_token("<pad>"), Some(&[] as &[&str]));
    }

    #[test]
    fn gly_and_unk_counts() {
        assert_eq!(ref_atom_names("GLY").unwrap().len(), 4);
        assert_eq!(ref_atom_names("UNK").unwrap().len(), 5);
    }

    #[test]
    fn dt_has_c7() {
        assert!(ref_atom_names("DT").unwrap().contains(&"C7"));
    }

    #[test]
    fn center_disto_indices_match_python() {
        assert_eq!(center_atom_index("ALA"), Some(1));
        assert_eq!(disto_atom_index("GLY"), Some(1));
        assert_eq!(disto_atom_index("ALA"), Some(4));
        let c4 = ref_atom_names("A").unwrap().iter().position(|&a| a == "C4").unwrap();
        assert_eq!(disto_atom_index("A"), Some(c4));
    }
}
