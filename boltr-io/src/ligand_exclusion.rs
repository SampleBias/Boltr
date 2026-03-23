//! CCD codes excluded as ligand/solvent in `boltz.data.const.ligand_exclusion` (131 entries).

/// Uppercase CCD-style codes, sorted lexicographically (for maintenance / diff against Python).
pub const LIGAND_EXCLUSION_CODES: [&str; 131] = [
    "144", "15P", "1PE", "2F2", "2JC", "3HR", "3SY", "7N5", "7PE", "9JE", "AAE", "ABA", "ACE",
    "ACN", "ACT", "ACY", "AZI", "BAM", "BCN", "BCT", "BDN", "BEN", "BME", "BO3", "BTB", "BTC",
    "BU1", "C8E", "CAD", "CAQ", "CBM", "CCN", "CIT", "CL", "CLR", "CM", "CMO", "CO3", "CPT", "CXS",
    "D10", "DEP", "DIO", "DMS", "DN", "DOD", "DOX", "EDO", "EEE", "EGL", "EOH", "EOX", "EPE",
    "ETF", "FCY", "FJO", "FLC", "FMT", "FW5", "GOL", "GSH", "GTT", "GYF", "HED", "IHP", "IHS",
    "IMD", "IOD", "IPA", "IPH", "LDA", "MB3", "MEG", "MES", "MLA", "MLI", "MOH", "MPD", "MRD",
    "MSE", "MYR", "N", "NA", "NH2", "NH4", "NHE", "NO3", "O4B", "OHE", "OLA", "OLC", "OMB", "OME",
    "OXA", "P6G", "PE3", "PE4", "PEG", "PEO", "PEP", "PG0", "PG4", "PGE", "PGR", "PLM", "PO4",
    "POL", "POP", "PVO", "SAR", "SCN", "SEO", "SEP", "SIN", "SO4", "SPD", "SPM", "SR", "STE",
    "STO", "STU", "TAR", "TBU", "TME", "TPO", "TRS", "UNK", "UNL", "UNX", "UPL", "URE",
];

pub const LIGAND_EXCLUSION_COUNT: usize = LIGAND_EXCLUSION_CODES.len();

/// `true` if `code` should be excluded (ASCII case-insensitive, trimmed).
#[must_use]
pub fn is_ligand_excluded(code: &str) -> bool {
    let c = code.trim();
    if c.is_empty() {
        return false;
    }
    LIGAND_EXCLUSION_CODES
        .iter()
        .any(|&p| p.eq_ignore_ascii_case(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gol_and_case() {
        assert!(is_ligand_excluded("GOL"));
        assert!(is_ligand_excluded("gol"));
        assert!(!is_ligand_excluded("ALA"));
    }

    #[test]
    fn count_matches_python() {
        assert_eq!(LIGAND_EXCLUSION_COUNT, 131);
    }
}
