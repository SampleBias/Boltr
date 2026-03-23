//! `ambiguous_atoms` from Boltz `data/const.py`: map atom name (digits stripped) + residue
//! to an element symbol for PDB export (`boltz.data.write.pdb`, boltz2 branch).

use std::collections::HashMap;
use std::sync::LazyLock;

use serde::Deserialize;

/// Number of top-level atom keys in `ambiguous_atoms.json` (Boltz `const.ambiguous_atoms`).
pub const AMBIGUOUS_ATOMS_TOP_LEVEL_COUNT: usize = 185;

static AMBIGUOUS_ATOMS_JSON: &str = include_str!("../data/ambiguous_atoms.json");

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawRule {
    Fixed(String),
    Map(HashMap<String, String>),
}

#[derive(Debug)]
enum Rule {
    Fixed(String),
    Map(HashMap<String, String>),
}

static AMBIGUOUS_ATOMS: LazyLock<HashMap<String, Rule>> = LazyLock::new(|| {
    let raw: HashMap<String, RawRule> =
        serde_json::from_str(AMBIGUOUS_ATOMS_JSON).expect("ambiguous_atoms.json");
    raw.into_iter()
        .map(|(k, v)| {
            let rule = match v {
                RawRule::Fixed(s) => Rule::Fixed(s),
                RawRule::Map(m) => Rule::Map(m),
            };
            (k, rule)
        })
        .collect()
});

/// Strip ASCII digits from an atom name, matching Boltz PDB export: `re.sub(r"\d", "", atom_name)`.
#[must_use]
pub fn pdb_atom_key(atom_name: &str) -> String {
    atom_name.chars().filter(|c| !c.is_ascii_digit()).collect()
}

/// Resolve PDB element symbol when `atom_key` is present in the Boltz `ambiguous_atoms` table.
///
/// Mirrors `boltz.data.write.pdb.to_pdb` (boltz2): if the rule is a string, return it; if it is a
/// map, use `residue_name` when present, otherwise `"*"`.
#[must_use]
pub fn resolve_ambiguous_element(atom_key: &str, residue_name: &str) -> Option<String> {
    match AMBIGUOUS_ATOMS.get(atom_key)? {
        Rule::Fixed(s) => Some(s.clone()),
        Rule::Map(m) => m.get(residue_name).or_else(|| m.get("*")).cloned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_level_count_matches_python() {
        assert_eq!(AMBIGUOUS_ATOMS.len(), AMBIGUOUS_ATOMS_TOP_LEVEL_COUNT);
    }

    #[test]
    fn pdb_atom_key_strips_digits() {
        assert_eq!(pdb_atom_key("1HB2"), "HB");
        assert_eq!(pdb_atom_key("CA"), "CA");
    }

    #[test]
    fn fixed_rule_br() {
        assert_eq!(
            resolve_ambiguous_element("BR", "XXX").as_deref(),
            Some("BR")
        );
    }

    #[test]
    fn map_rule_ca_oex() {
        assert_eq!(
            resolve_ambiguous_element("CA", "OEX").as_deref(),
            Some("CA")
        );
    }

    #[test]
    fn map_rule_ca_fallback_star() {
        assert_eq!(resolve_ambiguous_element("CA", "ALA").as_deref(), Some("C"));
    }
}
