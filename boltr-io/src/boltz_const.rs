//! Boltz `data/const.py` — tokens, chain types, chemistry ids, limits; see [`crate::ref_atoms`] for `ref_atoms`.
//!
//! Reference: `boltz-reference/src/boltz/data/const.py`.

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

// --- Atoms / bonds / contacts / MSA limits (`const.py`) ---

/// Upper bound for element-type embedding slots in Boltz (distinct from [`crate::vdw_radii::VDW_RADII_LEN`]).
pub const NUM_ELEMENTS: usize = 128;

/// `min_coverage_residues` / `min_coverage_fraction` for templates (`const.py`).
pub const MIN_COVERAGE_RESIDUES: usize = 10;
pub const MIN_COVERAGE_FRACTION: f64 = 0.1;

pub const CHIRALITY_TYPES: [&str; 7] = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_SQUAREPLANAR",
    "CHI_OCTAHEDRAL",
    "CHI_TRIGONALBIPYRAMIDAL",
    "CHI_OTHER",
];

pub const UNK_CHIRALITY_TYPE: &str = "CHI_OTHER";

#[inline]
pub fn chirality_type_id(name: &str) -> Option<i32> {
    CHIRALITY_TYPES
        .iter()
        .position(|&c| c == name)
        .map(|i| i as i32)
}

pub const HYBRIDIZATION_MAP: [&str; 9] = [
    "S",
    "SP",
    "SP2",
    "SP2D",
    "SP3",
    "SP3D",
    "SP3D2",
    "OTHER",
    "UNSPECIFIED",
];

pub const UNK_HYBRIDIZATION_TYPE: &str = "UNSPECIFIED";

#[inline]
pub fn hybridization_type_id(name: &str) -> Option<i32> {
    HYBRIDIZATION_MAP
        .iter()
        .position(|&h| h == name)
        .map(|i| i as i32)
}

pub const BOND_TYPES: [&str; 6] = [
    "OTHER",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "AROMATIC",
    "COVALENT",
];

pub const UNK_BOND_TYPE: &str = "OTHER";

#[inline]
pub fn bond_type_id(name: &str) -> Option<i32> {
    BOND_TYPES
        .iter()
        .position(|&b| b == name)
        .map(|i| i as i32)
}

pub const ATOM_INTERFACE_CUTOFF: f64 = 5.0;
pub const INTERFACE_CUTOFF: f64 = 15.0;

#[inline]
pub fn pocket_contact_id(name: &str) -> Option<i32> {
    Some(match name {
        "UNSPECIFIED" => 0,
        "UNSELECTED" => 1,
        "POCKET" => 2,
        "BINDER" => 3,
        _ => return None,
    })
}

#[inline]
pub fn contact_conditioning_id(name: &str) -> Option<i32> {
    Some(match name {
        "UNSPECIFIED" => 0,
        "UNSELECTED" => 1,
        "POCKET>BINDER" => 2,
        "BINDER>POCKET" => 3,
        "CONTACT" => 4,
        _ => return None,
    })
}

pub const MAX_MSA_SEQS: usize = 16384;
pub const MAX_PAIRED_SEQS: usize = 8192;

pub const CHUNK_SIZE_THRESHOLD: usize = 384;

// --- Method conditioning (`const.py`; keys are lowercased in Python) ---

/// Distinct embedding indices used for method type (`num_method_types` in Python).
pub const NUM_METHOD_TYPES: usize = 12;

#[must_use]
pub fn method_type_id(method: &str) -> i32 {
    match method.trim().to_ascii_lowercase().as_str() {
        "md" => 0,
        "x-ray diffraction" => 1,
        "electron microscopy" => 2,
        "solution nmr" => 3,
        "solid-state nmr"
        | "neutron diffraction"
        | "electron crystallography"
        | "fiber diffraction"
        | "powder diffraction"
        | "infrared spectroscopy"
        | "fluorescence transfer"
        | "epr"
        | "theoretical model"
        | "solution scattering"
        | "other" => 4,
        "afdb" => 5,
        "boltz-1" => 6,
        "future1" => 7,
        "future2" => 8,
        "future3" => 9,
        "future4" => 10,
        "future5" => 11,
        _ => 4,
    }
}

/// Half-open bins in Kelvin: `[265,280)`, `[280,295)`, `[295,310)`; else `"other"` index.
pub const NUM_TEMP_BINS: usize = 4;

#[must_use]
pub fn temperature_bin_id(temp_k: f64) -> i32 {
    if (265.0..280.0).contains(&temp_k) {
        0
    } else if (280.0..295.0).contains(&temp_k) {
        1
    } else if (295.0..310.0).contains(&temp_k) {
        2
    } else {
        3
    }
}

/// Half-open pH bins: `[0,6)`, `[6,8)`, `[8,14)`; else `"other"` index.
pub const NUM_PH_BINS: usize = 4;

#[must_use]
pub fn ph_bin_id(ph: f64) -> i32 {
    if (0.0..6.0).contains(&ph) {
        0
    } else if (6.0..8.0).contains(&ph) {
        1
    } else if (8.0..14.0).contains(&ph) {
        2
    } else {
        3
    }
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

    #[test]
    fn chemistry_ids_match_python_order() {
        assert_eq!(chirality_type_id("CHI_UNSPECIFIED"), Some(0));
        assert_eq!(chirality_type_id("CHI_OTHER"), Some(6));
        assert_eq!(hybridization_type_id("S"), Some(0));
        assert_eq!(hybridization_type_id("UNSPECIFIED"), Some(8));
        assert_eq!(bond_type_id("OTHER"), Some(0));
        assert_eq!(bond_type_id("COVALENT"), Some(5));
        assert_eq!(pocket_contact_id("POCKET"), Some(2));
        assert_eq!(contact_conditioning_id("CONTACT"), Some(4));
        assert_eq!(MAX_MSA_SEQS, 16384);
        assert_eq!(CHUNK_SIZE_THRESHOLD, 384);
    }

    #[test]
    fn method_and_bin_ids() {
        assert_eq!(method_type_id("X-RAY DIFFRACTION"), 1);
        assert_eq!(method_type_id("electron microscopy"), 2);
        assert_eq!(method_type_id("BOLTZ-1"), 6);
        assert_eq!(method_type_id("unknown method"), 4);
        assert_eq!(NUM_METHOD_TYPES, 12);
        assert_eq!(temperature_bin_id(270.0), 0);
        assert_eq!(temperature_bin_id(310.0), 3);
        assert_eq!(ph_bin_id(7.0), 1);
        assert_eq!(ph_bin_id(99.0), 3);
    }
}
