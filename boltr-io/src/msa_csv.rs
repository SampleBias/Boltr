//! Paired CSV MSA (`key`, `sequence`) — `boltz.data.parse.csv.parse_csv`.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use csv::StringRecord;

use crate::a3m::{
    msa_sequence_dedup_key, parse_alignment_sequence_line, A3mMsa, A3mSequenceMeta,
};

fn validate_headers(headers: &StringRecord) -> Result<(usize, usize)> {
    let names: Vec<String> = headers.iter().map(String::from).collect();
    let mut sorted: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    sorted.sort_unstable();
    if sorted != ["key", "sequence"] {
        return Err(anyhow!(
            "Invalid CSV format, expected columns ['key', 'sequence'], got {:?}",
            names
        ));
    }
    let key_i = headers
        .iter()
        .position(|h| h == "key")
        .ok_or_else(|| anyhow!("missing 'key' column"))?;
    let seq_i = headers
        .iter()
        .position(|h| h == "sequence")
        .ok_or_else(|| anyhow!("missing 'sequence' column"))?;
    Ok((key_i, seq_i))
}

/// Boltz stores numeric taxonomy in `key` when present; otherwise `-1`.
fn taxonomy_from_key_field(field: &str) -> i32 {
    let t = field.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("nan") {
        return -1;
    }
    if let Ok(v) = t.parse::<i64>() {
        return v.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    if let Ok(f) = t.parse::<f64>() {
        return f as i32;
    }
    -1
}

fn parse_csv_records<R: std::io::Read>(
    mut rdr: csv::Reader<R>,
    max_seqs: Option<usize>,
) -> Result<A3mMsa> {
    let headers = rdr.headers()?.clone();
    let (key_i, seq_i) = validate_headers(&headers)?;

    let mut visited = HashSet::new();
    let mut sequences = Vec::new();
    let mut residues = Vec::new();
    let mut deletions = Vec::new();
    let mut seq_idx: i32 = 0;

    for rec in rdr.records() {
        let rec = rec?;
        let seq_line = rec.get(seq_i).unwrap_or("").trim();
        if seq_line.is_empty() {
            continue;
        }
        let taxonomy_id = taxonomy_from_key_field(rec.get(key_i).unwrap_or(""));

        let dedup = msa_sequence_dedup_key(seq_line);
        if !visited.insert(dedup) {
            continue;
        }

        let (residue, deletion) = parse_alignment_sequence_line(seq_line)?;
        let res_start = residues.len();
        let res_end = res_start + residue.len();
        let del_start = deletions.len();
        let del_end = del_start + deletion.len();

        sequences.push(A3mSequenceMeta {
            seq_idx,
            taxonomy_id,
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

/// Read Boltz-style CSV MSA from disk.
pub fn parse_csv_path(path: &Path, max_seqs: Option<usize>) -> Result<A3mMsa> {
    let rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| path.display().to_string())?;
    parse_csv_records(rdr, max_seqs)
}

/// Parse CSV MSA from memory (tests).
pub fn parse_csv_str(content: &str, max_seqs: Option<usize>) -> Result<A3mMsa> {
    let rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    parse_csv_records(rdr, max_seqs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_roundtrip_columns() {
        let raw = "key,sequence\n,ACDEF\n338,AC-GHK\n";
        let m = parse_csv_str(raw, None).unwrap();
        assert_eq!(m.sequences.len(), 2);
        assert_eq!(m.sequences[1].taxonomy_id, 338);
        assert!(m.residues.len() > 5);
    }

    #[test]
    fn csv_rejects_bad_header() {
        let raw = "a,b\n1,2\n";
        assert!(parse_csv_str(raw, None).is_err());
    }
}
