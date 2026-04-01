//! Boltz `MSA` as NumPy `.npz` (`savez_compressed` layout).
//!
//! Matches [boltz-reference/src/boltz/data/types.py](boltz-reference/src/boltz/data/types.py)
//! structured dtypes: `MSAResidue`, `MSADeletion`, `MSASequence`, and `NumpySerializable.dump`
//! (`np.savez_compressed(..., **asdict(msa))`).
//!
//! NPY header construction follows NumPy `lib/format.py` (sorted dict keys, growth padding, v1.0).
//!
//! **Note:** Deflate bitstreams can differ between Zip/NumPy and this writer while remaining
//! readable by `numpy.load` / `read_msa_npz_*`. For CI, compare decoded arrays or run a Python
//! roundtrip, not raw `.npz` hashes.

use std::fs::File;
use std::io::{Cursor, Read, Seek, Write};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::a3m::{A3mMsa, A3mSequenceMeta};

const MAGIC_PREFIX: &[u8] = b"\x93NUMPY";
const ARRAY_ALIGN: usize = 64;
/// NumPy `GROWTH_AXIS_MAX_DIGITS`
const GROWTH_AXIS_MAX_DIGITS: usize = 21;
/// `MAGIC_LEN` in NumPy = len(prefix) + 2 (version bytes)
const MAGIC_LEN: usize = 8;

/// `dtype.descr` string forms (little-endian) matching Boltz `types.py`.
const DESCR_RESIDUES: &str = "[('res_type', '|i1')]";
const DESCR_DELETIONS: &str = "[('res_idx', '<i2'), ('deletion', '<i2')]";
const DESCR_SEQUENCES: &str = "[('seq_idx', '<i2'), ('taxonomy', '<i4'), ('res_start', '<i4'), ('res_end', '<i4'), ('del_start', '<i4'), ('del_end', '<i4')]";

/// Aligned `itemsize` for `MSASequence` as written by this crate (`pack_sequences`: i2 + pad + 5×i4).
const SEQUENCE_RECORD_BYTES: usize = 24;
/// Packed layout from NumPy/Boltz `MSASequence` dtype (no padding after `seq_idx`).
const SEQUENCE_RECORD_BYTES_BOLTZ: usize = 22;

fn shape_repr_1d(n: usize) -> String {
    format!("({n},)")
}

fn numpy_header_dict_v1(descr_literal: &str, shape_1d: usize, fortran_order: bool) -> String {
    let shape_s = shape_repr_1d(shape_1d);
    let fo = if fortran_order { "True" } else { "False" };
    let mut inner = String::new();
    inner.push('{');
    for (k, v) in [
        ("descr", descr_literal),
        ("fortran_order", fo),
        ("shape", shape_s.as_str()),
    ] {
        inner.push('\'');
        inner.push_str(k);
        inner.push('\'');
        inner.push_str(": ");
        inner.push_str(v);
        inner.push_str(", ");
    }
    inner.push('}');
    // NumPy: pad using `len(repr(shape[0]))` for C-order 1-D arrays.
    let axis0_repr_len = format!("{shape_1d}").len();
    let growth_pad = GROWTH_AXIS_MAX_DIGITS.saturating_sub(axis0_repr_len);
    format!("{}{}", inner, " ".repeat(growth_pad))
}

fn wrap_npy_header_v1(header_dict: &str) -> Result<Vec<u8>> {
    let header = header_dict.as_bytes();
    let hlen = header.len() + 1;
    let padlen = ARRAY_ALIGN - ((MAGIC_LEN + 2 + hlen) % ARRAY_ALIGN);
    let total_header_content = hlen + padlen;
    if total_header_content > u16::MAX as usize {
        bail!("NPY v1 header too large");
    }
    let mut out = Vec::with_capacity(MAGIC_LEN + 2 + total_header_content);
    out.extend_from_slice(MAGIC_PREFIX);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(total_header_content as u16).to_le_bytes());
    out.extend_from_slice(header);
    out.extend(std::iter::repeat(b' ').take(padlen));
    out.push(b'\n');
    Ok(out)
}

fn write_npy_1d(descr_literal: &str, shape_len: usize, payload: &[u8]) -> Result<Vec<u8>> {
    let dict = numpy_header_dict_v1(descr_literal, shape_len, false);
    let mut blob = wrap_npy_header_v1(&dict)?;
    blob.extend_from_slice(payload);
    Ok(blob)
}

fn validate_msa_for_npz(msa: &A3mMsa) -> Result<()> {
    for (i, s) in msa.sequences.iter().enumerate() {
        if s.seq_idx < i16::MIN as i32 || s.seq_idx > i16::MAX as i32 {
            bail!("sequence {i}: seq_idx {} out of i16 range", s.seq_idx);
        }
        let rs = s.res_start as i64;
        let re = s.res_end as i64;
        let ds = s.del_start as i64;
        let de = s.del_end as i64;
        if rs < i32::MIN as i64
            || re > i32::MAX as i64
            || ds < i32::MIN as i64
            || de > i32::MAX as i64
        {
            bail!("sequence {i}: index range too large for i32 in npz");
        }
    }
    for &t in &msa.residues {
        if t < i8::MIN as i32 || t > i8::MAX as i32 {
            bail!("residue token {t} does not fit MSAResidue i1");
        }
    }
    for (i, &(ri, del)) in msa.deletions.iter().enumerate() {
        if ri < i16::MIN as i32 || ri > i16::MAX as i32 {
            bail!("deletion {i}: res_idx out of i16 range");
        }
        if del < i16::MIN as i32 || del > i16::MAX as i32 {
            bail!("deletion {i}: deletion count out of i16 range");
        }
    }
    Ok(())
}

fn pack_residues(msa: &A3mMsa) -> Vec<u8> {
    msa.residues.iter().map(|&t| t as u8).collect()
}

fn pack_deletions(msa: &A3mMsa) -> Vec<u8> {
    let mut v = Vec::with_capacity(msa.deletions.len() * 4);
    for &(res_idx, del) in &msa.deletions {
        v.extend_from_slice(&(res_idx as i16).to_le_bytes());
        v.extend_from_slice(&(del as i16).to_le_bytes());
    }
    v
}

fn pack_sequences(msa: &A3mMsa) -> Vec<u8> {
    let mut v = Vec::with_capacity(msa.sequences.len() * SEQUENCE_RECORD_BYTES);
    for s in &msa.sequences {
        v.extend_from_slice(&(s.seq_idx as i16).to_le_bytes());
        v.extend_from_slice(&[0u8, 0u8]); // align to i4
        v.extend_from_slice(&s.taxonomy_id.to_le_bytes());
        v.extend_from_slice(&(s.res_start as i32).to_le_bytes());
        v.extend_from_slice(&(s.res_end as i32).to_le_bytes());
        v.extend_from_slice(&(s.del_start as i32).to_le_bytes());
        v.extend_from_slice(&(s.del_end as i32).to_le_bytes());
    }
    v
}

/// Write `msa` as a NumPy-compressed `.npz` with keys `sequences`, `deletions`, `residues`
/// (Boltz `MSA.dump` / preprocess layout).
pub fn write_msa_npz_compressed(path: &Path, msa: &A3mMsa) -> Result<()> {
    validate_msa_for_npz(msa)?;
    let seq_blob = write_npy_1d(DESCR_SEQUENCES, msa.sequences.len(), &pack_sequences(msa))?;
    let del_blob = write_npy_1d(DESCR_DELETIONS, msa.deletions.len(), &pack_deletions(msa))?;
    let res_blob = write_npy_1d(DESCR_RESIDUES, msa.residues.len(), &pack_residues(msa))?;
    let mut file = File::create(path).with_context(|| path.display().to_string())?;
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let mut zw = ZipWriter::new(&mut file);
    // Same member order as Python `**asdict(msa)` field order: sequences, deletions, residues.
    zw.start_file("sequences.npy", opts)?;
    zw.write_all(&seq_blob)?;
    zw.start_file("deletions.npy", opts)?;
    zw.write_all(&del_blob)?;
    zw.start_file("residues.npy", opts)?;
    zw.write_all(&res_blob)?;
    zw.finish()?;
    Ok(())
}

#[cfg(test)]
fn write_msa_npz_compressed_to_vec(msa: &A3mMsa) -> Result<Vec<u8>> {
    validate_msa_for_npz(msa)?;
    let seq_blob = write_npy_1d(DESCR_SEQUENCES, msa.sequences.len(), &pack_sequences(msa))?;
    let del_blob = write_npy_1d(DESCR_DELETIONS, msa.deletions.len(), &pack_deletions(msa))?;
    let res_blob = write_npy_1d(DESCR_RESIDUES, msa.residues.len(), &pack_residues(msa))?;
    let mut buf = Cursor::new(Vec::new());
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let mut zw = ZipWriter::new(&mut buf);
    zw.start_file("sequences.npy", opts)?;
    zw.write_all(&seq_blob)?;
    zw.start_file("deletions.npy", opts)?;
    zw.write_all(&del_blob)?;
    zw.start_file("residues.npy", opts)?;
    zw.write_all(&res_blob)?;
    zw.finish()?;
    Ok(buf.into_inner())
}

fn parse_npy_1d_shape_and_payload(data: &[u8]) -> Result<(usize, &[u8])> {
    if data.len() < 10 {
        bail!("truncated .npy");
    }
    if &data[..6] != MAGIC_PREFIX {
        bail!("invalid NPY magic");
    }
    let ver_maj = data[6];
    let ver_min = data[7];
    if ver_maj != 1 || ver_min != 0 {
        bail!("unsupported NPY version {ver_maj}.{ver_min} (only 1.0)");
    }
    let hlen = u16::from_le_bytes([data[8], data[9]]) as usize;
    if data.len() < 10 + hlen {
        bail!("truncated NPY header");
    }
    let header_raw = &data[10..10 + hlen];
    let header_str =
        std::str::from_utf8(header_raw).map_err(|_| anyhow!("NPY header is not UTF-8/latin1"))?;
    let shape_1d = parse_shape_1d(header_str)?;
    let payload_off = 10 + hlen;
    let payload = data
        .get(payload_off..)
        .ok_or_else(|| anyhow!("missing NPY payload"))?;
    Ok((shape_1d, payload))
}

fn parse_shape_1d(header: &str) -> Result<usize> {
    // After padding, header may have leading valid dict substring; take up to first '\n' optional
    let header_trim: String = header.chars().take_while(|&c| c != '\n').collect();
    let key = "'shape': ";
    let pos = header_trim
        .find(key)
        .ok_or_else(|| anyhow!("NPY header missing 'shape'"))?;
    let rest = &header_trim[pos + key.len()..];
    if rest.starts_with("()") {
        return Ok(0);
    }
    let rest = rest
        .strip_prefix('(')
        .ok_or_else(|| anyhow!("invalid shape in NPY header"))?;
    let num = rest
        .split(',')
        .next()
        .ok_or_else(|| anyhow!("invalid shape tuple"))?
        .trim();
    Ok(num.parse::<usize>()?)
}

/// Load Boltz-format MSA npz (`MSA.load`).
pub fn read_msa_npz_path(path: &Path) -> Result<A3mMsa> {
    let mut file = File::open(path).with_context(|| path.display().to_string())?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    read_msa_npz_bytes(&buf)
}

pub fn read_msa_npz_bytes(zip_bytes: &[u8]) -> Result<A3mMsa> {
    let cursor = Cursor::new(zip_bytes);
    let mut archive = ZipArchive::new(cursor)?;
    let seq = read_zip_npy(&mut archive, "sequences")?;
    let del = read_zip_npy(&mut archive, "deletions")?;
    let res = read_zip_npy(&mut archive, "residues")?;
    decode_msa_from_npy_payloads(seq, del, res)
}

fn read_zip_npy<R: Read + Seek>(archive: &mut ZipArchive<R>, name: &str) -> Result<Vec<u8>> {
    let path = format!("{name}.npy");
    let mut v = Vec::new();
    if let Ok(mut f) = archive.by_name(&path) {
        f.read_to_end(&mut v)?;
        return Ok(v);
    }
    let mut f = archive
        .by_name(name)
        .with_context(|| format!("missing {path} in npz"))?;
    f.read_to_end(&mut v)?;
    Ok(v)
}

fn decode_msa_from_npy_payloads(
    seq_npy: Vec<u8>,
    del_npy: Vec<u8>,
    res_npy: Vec<u8>,
) -> Result<A3mMsa> {
    let (n_seq, seq_payload) = parse_npy_1d_shape_and_payload(&seq_npy)?;
    let (n_del, del_payload) = parse_npy_1d_shape_and_payload(&del_npy)?;
    let (n_res, res_payload) = parse_npy_1d_shape_and_payload(&res_npy)?;

    let seq_stride = if n_seq == 0 {
        if !seq_payload.is_empty() {
            bail!("sequences: n_seq=0 but payload non-empty");
        }
        SEQUENCE_RECORD_BYTES
    } else {
        let s = seq_payload.len() / n_seq;
        if seq_payload.len() != n_seq * s
            || (s != SEQUENCE_RECORD_BYTES && s != SEQUENCE_RECORD_BYTES_BOLTZ)
        {
            bail!(
                "sequences payload length {} incompatible with n_seq={} (expected stride {} or {})",
                seq_payload.len(),
                n_seq,
                SEQUENCE_RECORD_BYTES,
                SEQUENCE_RECORD_BYTES_BOLTZ
            );
        }
        s
    };
    if del_payload.len() != n_del * 4 {
        bail!("deletions payload length mismatch");
    }
    if res_payload.len() != n_res {
        bail!("residues payload length mismatch");
    }

    let mut sequences = Vec::with_capacity(n_seq);
    for i in 0..n_seq {
        let base = i * seq_stride;
        let chunk = &seq_payload[base..base + seq_stride];
        let (seq_idx, taxonomy, res_start, res_end, del_start, del_end) = if seq_stride
            == SEQUENCE_RECORD_BYTES_BOLTZ
        {
            (
                i16::from_le_bytes([chunk[0], chunk[1]]) as i32,
                i32::from_le_bytes([chunk[2], chunk[3], chunk[4], chunk[5]]),
                i32::from_le_bytes([chunk[6], chunk[7], chunk[8], chunk[9]]) as usize,
                i32::from_le_bytes([chunk[10], chunk[11], chunk[12], chunk[13]]) as usize,
                i32::from_le_bytes([chunk[14], chunk[15], chunk[16], chunk[17]]) as usize,
                i32::from_le_bytes([chunk[18], chunk[19], chunk[20], chunk[21]]) as usize,
            )
        } else {
            (
                i16::from_le_bytes([chunk[0], chunk[1]]) as i32,
                i32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
                i32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]) as usize,
                i32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]) as usize,
                i32::from_le_bytes([chunk[16], chunk[17], chunk[18], chunk[19]]) as usize,
                i32::from_le_bytes([chunk[20], chunk[21], chunk[22], chunk[23]]) as usize,
            )
        };
        sequences.push(A3mSequenceMeta {
            seq_idx,
            taxonomy_id: taxonomy,
            res_start,
            res_end,
            del_start,
            del_end,
        });
    }

    let mut deletions = Vec::with_capacity(n_del);
    for i in 0..n_del {
        let base = i * 4;
        let ri = i16::from_le_bytes([del_payload[base], del_payload[base + 1]]) as i32;
        let d = i16::from_le_bytes([del_payload[base + 2], del_payload[base + 3]]) as i32;
        deletions.push((ri, d));
    }

    let residues: Vec<i32> = res_payload.iter().map(|&b| b as i8 as i32).collect();

    Ok(A3mMsa {
        residues,
        deletions,
        sequences,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::a3m::parse_a3m_str;

    #[test]
    fn roundtrip_simple_a3m() {
        let raw = ">q\nACDEF\n";
        let m = parse_a3m_str(raw, None).unwrap();
        let bytes = write_msa_npz_compressed_to_vec(&m).unwrap();
        let back = read_msa_npz_bytes(&bytes).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn roundtrip_with_deletions_and_taxonomy_csv_like() {
        let m = A3mMsa {
            residues: vec![2, 3, 4],
            deletions: vec![(1, 2)],
            sequences: vec![A3mSequenceMeta {
                seq_idx: 0,
                taxonomy_id: 9606,
                res_start: 0,
                res_end: 3,
                del_start: 0,
                del_end: 1,
            }],
        };
        let bytes = write_msa_npz_compressed_to_vec(&m).unwrap();
        let back = read_msa_npz_bytes(&bytes).unwrap();
        assert_eq!(back, m);
    }
}
