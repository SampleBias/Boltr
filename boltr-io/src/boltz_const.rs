//! Boltz `data/const.py` — tokens, chain types, and one-letter ↔ token-name maps used by MSA / featurizer.
//!
//! Reference: `boltz-reference/src/boltz/data/const.py` (residues & tokens, chain_types).

/// Same order as Python `boltz.data.const.tokens` (ids = index).
pub const TOKENS: [&str; 33] = [
    "<pad>",
    "-",
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
    "A",
    "G",
    "C",
    "U",
    "N",
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",
];

pub const NUM_TOKENS: usize = TOKENS.len();

pub const CHAIN_TYPES: [&str; 4] = ["PROTEIN", "DNA", "RNA", "NONPOLYMER"];

/// Boltz `chain_type_ids`.
#[inline]
pub fn chain_type_id(name: &str) -> Option<i32> {
    CHAIN_TYPES
        .iter()
        .position(|&c| c == name)
        .map(|i| i as i32)
}

/// Resolve token string → id (`token_ids` in Python).
#[inline]
pub fn token_id(name: &str) -> Option<i32> {
    TOKENS.iter().position(|&t| t == name).map(|i| i as i32)
}

/// Inverse of [`token_id`] for valid ids.
#[inline]
pub fn token_name(id: i32) -> Option<&'static str> {
    let u = usize::try_from(id).ok()?;
    TOKENS.get(u).copied()
}

/// `unk_token_ids` from `const.py` (`PROTEIN` → UNK, `DNA` → DN, `RNA` → N).
#[inline]
pub fn unk_token_id(chain_type: &str) -> Option<i32> {
    let t = match chain_type {
        "PROTEIN" => "UNK",
        "DNA" => "DN",
        "RNA" => "N",
        _ => return None,
    };
    token_id(t)
}

/// Uppercase protein one-letter → Boltz token id (A3M / FASTA protein rows).
pub fn prot_letter_to_token_id(c: char) -> Option<i32> {
    let u = c.to_ascii_uppercase();
    let name = match u {
        'A' => "ALA",
        'R' => "ARG",
        'N' => "ASN",
        'D' => "ASP",
        'C' => "CYS",
        'E' => "GLU",
        'Q' => "GLN",
        'G' => "GLY",
        'H' => "HIS",
        'I' => "ILE",
        'L' => "LEU",
        'K' => "LYS",
        'M' => "MET",
        'F' => "PHE",
        'P' => "PRO",
        'S' => "SER",
        'T' => "THR",
        'W' => "TRP",
        'Y' => "TYR",
        'V' => "VAL",
        'X' | 'J' | 'B' | 'Z' | 'O' | 'U' => "UNK",
        '-' => "-",
        _ => return None,
    };
    token_id(name)
}

/// Uppercase DNA one-letter → token id.
pub fn dna_letter_to_token_id(c: char) -> Option<i32> {
    let u = c.to_ascii_uppercase();
    let name = match u {
        'A' => "DA",
        'G' => "DG",
        'C' => "DC",
        'T' => "DT",
        'N' => "DN",
        _ => return None,
    };
    token_id(name)
}

/// Uppercase RNA one-letter → token id.
pub fn rna_letter_to_token_id(c: char) -> Option<i32> {
    let u = c.to_ascii_uppercase();
    let name = match u {
        'A' => "A",
        'G' => "G",
        'C' => "C",
        'U' => "U",
        'N' => "N",
        _ => return None,
    };
    token_id(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_table_matches_boltz_reference_indices() {
        assert_eq!(token_id("<pad>"), Some(0));
        assert_eq!(token_id("-"), Some(1));
        assert_eq!(token_id("ALA"), Some(2));
        assert_eq!(token_id("VAL"), Some(21));
        assert_eq!(token_id("UNK"), Some(22));
        assert_eq!(token_id("A"), Some(23));
        assert_eq!(token_id("N"), Some(27));
        assert_eq!(token_id("DA"), Some(28));
        assert_eq!(token_id("DN"), Some(32));
        assert_eq!(NUM_TOKENS, 33);
    }

    #[test]
    fn unk_token_ids_match_python() {
        assert_eq!(unk_token_id("PROTEIN"), Some(22));
        assert_eq!(unk_token_id("DNA"), Some(32));
        assert_eq!(unk_token_id("RNA"), Some(27));
        assert_eq!(unk_token_id("NONPOLYMER"), None);
    }

    #[test]
    fn chain_types_match_python() {
        assert_eq!(chain_type_id("PROTEIN"), Some(0));
        assert_eq!(chain_type_id("NONPOLYMER"), Some(3));
    }

    #[test]
    fn prot_letters_match_a3m_expectations() {
        assert_eq!(prot_letter_to_token_id('a'), Some(2));
        assert_eq!(prot_letter_to_token_id('x'), Some(22));
        assert_eq!(prot_letter_to_token_id('-'), Some(1));
    }
}
