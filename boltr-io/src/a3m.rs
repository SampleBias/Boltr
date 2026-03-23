//! A3M MSA parsing aligned with `boltz.data.parse.a3m.parse_a3m`.
//!
//! Reference: `boltz-reference/src/boltz/data/parse/a3m.py` and `boltz.data.const` token order.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use flate2::read::GzDecoder;

/// Layout matches Boltz `MSASequence` dtype fields (indices into flat `residues` / `deletions`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct A3mSequenceMeta {
    pub seq_idx: i32,
    pub taxonomy_id: i32,
    pub res_start: usize,
    pub res_end: usize,
    pub del_start: usize,
    pub del_end: usize,
}

/// Boltz-style MSA after A3M parse (taxonomy not loaded; `taxonomy_id` is always `-1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct A3mMsa {
    pub residues: Vec<i32>,
    pub deletions: Vec<(i32, i32)>,
    pub sequences: Vec<A3mSequenceMeta>,
}

/// Boltz `const.tokens` order — must stay in sync with `boltz-reference/.../const.py`.
fn token_id_for_name(name: &str) -> Option<i32> {
    let i: i32 = match name {
        "<pad>" => 0,
        "-" => 1,
        "ALA" => 2,
        "ARG" => 3,
        "ASN" => 4,
        "ASP" => 5,
        "CYS" => 6,
        "GLN" => 7,
        "GLU" => 8,
        "GLY" => 9,
        "HIS" => 10,
        "ILE" => 11,
        "LEU" => 12,
        "LYS" => 13,
        "MET" => 14,
        "PHE" => 15,
        "PRO" => 16,
        "SER" => 17,
        "THR" => 18,
        "TRP" => 19,
        "TYR" => 20,
        "VAL" => 21,
        "UNK" => 22,
        "A" => 23,
        "G" => 24,
        "C" => 25,
        "U" => 26,
        "N" => 27,
        "DA" => 28,
        "DG" => 29,
        "DC" => 30,
        "DT" => 31,
        "DN" => 32,
        _ => return None,
    };
    Some(i)
}

fn prot_letter_to_token_name(c: char) -> Result<&'static str> {
    match c {
        'A' => Ok("ALA"),
        'R' => Ok("ARG"),
        'N' => Ok("ASN"),
        'D' => Ok("ASP"),
        'C' => Ok("CYS"),
        'E' => Ok("GLU"),
        'Q' => Ok("GLN"),
        'G' => Ok("GLY"),
        'H' => Ok("HIS"),
        'I' => Ok("ILE"),
        'L' => Ok("LEU"),
        'K' => Ok("LYS"),
        'M' => Ok("MET"),
        'F' => Ok("PHE"),
        'P' => Ok("PRO"),
        'S' => Ok("SER"),
        'T' => Ok("THR"),
        'W' => Ok("TRP"),
        'Y' => Ok("TYR"),
        'V' => Ok("VAL"),
        'X' | 'J' | 'B' | 'Z' | 'O' => Ok("UNK"),
        'U' => Ok("UNK"),
        '-' => Ok("-"),
        _ => Err(anyhow!("unknown A3M residue letter: {c:?}")),
    }
}

fn residue_char_to_id(c: char) -> Result<i32> {
    let u = c.to_ascii_uppercase();
    let name = prot_letter_to_token_name(u)?;
    token_id_for_name(name).ok_or_else(|| anyhow!("internal: token {name}"))
}

/// Boltz A3M/CSV alignment row → flat residues + deletion pairs (shared by `parse_csv*`).
pub fn parse_alignment_sequence_line(line: &str) -> Result<(Vec<i32>, Vec<(i32, i32)>)> {
    let mut residue = Vec::new();
    let mut deletion = Vec::new();
    let mut count: i32 = 0;
    let mut res_idx: i32 = 0;

    for c in line.chars() {
        if c != '-' && c.is_lowercase() {
            count += 1;
            continue;
        }
        let tid = residue_char_to_id(c)?;
        residue.push(tid);
        if count > 0 {
            deletion.push((res_idx, count));
            count = 0;
        }
        res_idx += 1;
    }

    Ok((residue, deletion))
}

/// Deduplication key: remove gaps, uppercase (Boltz `visited` set).
pub fn msa_sequence_dedup_key(line: &str) -> String {
    line.chars()
        .filter(|&c| c != '-')
        .collect::<String>()
        .to_uppercase()
}

/// Parse A3M from any buffered line iterator (no `>` line consumed before start).
pub fn parse_a3m_lines<I>(lines: I, max_seqs: Option<usize>) -> Result<A3mMsa>
where
    I: Iterator<Item = std::io::Result<String>>,
{
    let mut visited: HashSet<String> = HashSet::new();
    let mut sequences = Vec::new();
    let mut residues: Vec<i32> = Vec::new();
    let mut deletions: Vec<(i32, i32)> = Vec::new();

    let mut seq_idx: i32 = 0;
    let mut expect_seq = false;

    for line in lines {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('>') {
            expect_seq = true;
            continue;
        }
        if !expect_seq {
            continue;
        }
        expect_seq = false;

        let str_seq = msa_sequence_dedup_key(line);
        if !visited.insert(str_seq) {
            continue;
        }

        let (residue, deletion) = parse_alignment_sequence_line(line)?;

        let res_start = residues.len();
        let res_end = res_start + residue.len();
        let del_start = deletions.len();
        let del_end = del_start + deletion.len();

        sequences.push(A3mSequenceMeta {
            seq_idx,
            taxonomy_id: -1,
            res_start,
            res_end,
            del_start,
            del_end,
        });
        residues.extend(residue);
        deletions.extend(deletion);

        seq_idx += 1;
        if max_seqs.is_some_and(|m| (seq_idx as usize) >= m) {
            break;
        }
    }

    Ok(A3mMsa {
        residues,
        deletions,
        sequences,
    })
}

/// Read `.a3m` or `.a3m.gz` from disk (Boltz-compatible).
pub fn parse_a3m_path(path: &Path, max_seqs: Option<usize>) -> Result<A3mMsa> {
    let is_gz = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gz"));

    if is_gz {
        let f = File::open(path).with_context(|| path.display().to_string())?;
        let dec = GzDecoder::new(f);
        let r = BufReader::new(dec);
        parse_a3m_lines(r.lines(), max_seqs)
    } else {
        let f = File::open(path).with_context(|| path.display().to_string())?;
        let r = BufReader::new(f);
        parse_a3m_lines(r.lines(), max_seqs)
    }
}

/// Parse A3M from an in-memory string (tests / embedded fixtures).
pub fn parse_a3m_str(content: &str, max_seqs: Option<usize>) -> Result<A3mMsa> {
    parse_a3m_lines(content.lines().map(|l| Ok(l.to_string())), max_seqs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_block_and_dedupes() {
        let raw = ">q\nACDEF\n>hit1\nACDEF\n>hit2\nAC-EK\n";
        let m = parse_a3m_str(raw, None).unwrap();
        assert_eq!(m.sequences.len(), 2, "duplicate row dropped; query + AC-EK hit");
        // Gaps (`-`) are stored as token `-` (id 1), same as Python.
        assert_eq!(m.residues.len(), 10);
    }

    #[test]
    fn lowercase_inserts_deletion_annotation() {
        let raw = ">q\nAbcDE\n";
        let m = parse_a3m_str(raw, None).unwrap();
        assert_eq!(m.sequences.len(), 1);
        assert_eq!(m.residues.len(), 3);
        assert_eq!(m.deletions, vec![(1_i32, 2_i32)]);
    }
}
