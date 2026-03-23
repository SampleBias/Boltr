//! Columnar NumPy `.npz` for Boltz2 token batches (`TokenData` / token bonds).
//!
//! Layout is **not** a single structured `TokenV2` array (alignment/padding differs by NumPy version);
//! it is a zip of `.npy` files keyed under `t_*` and `bond_*` so Python can
//! `np.load(...); np.testing.assert_equal` per field or rebuild `dtype=TokenV2` for parity tests.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, Write};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::tokenize::boltz2::{TokenBondV2, TokenData};

const MAGIC_PREFIX: &[u8] = b"\x93NUMPY";
const ARRAY_ALIGN: usize = 64;
const GROWTH_AXIS_MAX_DIGITS: usize = 21;
const MAGIC_LEN: usize = 8;

const DESCR_I4: &str = "'<i4'";
const DESCR_I1: &str = "'|i1'";
const DESCR_U1: &str = "'|u1'";
const DESCR_U4: &str = "'<u4'";
const DESCR_F4: &str = "'<f4'";

fn shape_repr(dims: &[usize]) -> String {
    match dims.len() {
        0 => "()".to_string(),
        1 => format!("({},)", dims[0]),
        2 => format!("({}, {})", dims[0], dims[1]),
        _ => panic!("unsupported ndim {}", dims.len()),
    }
}

fn growth_pad_for_shape_str(shape_line: &str) -> usize {
    let inner = shape_line.trim_matches(|c| c == '(' || c == ')');
    let first = inner.split(',').next().unwrap_or("0").trim();
    let axis0_repr_len = first.len();
    GROWTH_AXIS_MAX_DIGITS.saturating_sub(axis0_repr_len)
}

fn numpy_header_dict_v1(descr_literal: &str, shape_line: &str, fortran_order: bool) -> String {
    let fo = if fortran_order { "True" } else { "False" };
    let mut inner = String::new();
    inner.push('{');
    for (k, v) in [
        ("descr", descr_literal),
        ("fortran_order", fo),
        ("shape", shape_line),
    ] {
        inner.push('\'');
        inner.push_str(k);
        inner.push('\'');
        inner.push_str(": ");
        inner.push_str(v);
        inner.push_str(", ");
    }
    inner.push('}');
    let growth_pad = growth_pad_for_shape_str(shape_line);
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
    out.extend(std::iter::repeat_n(b' ', padlen));
    out.push(b'\n');
    Ok(out)
}

fn write_npy(descr: &str, shape_dims: &[usize], payload: &[u8]) -> Result<Vec<u8>> {
    let shape_line = shape_repr(shape_dims);
    let dict = numpy_header_dict_v1(descr, &shape_line, false);
    let mut blob = wrap_npy_header_v1(&dict)?;
    blob.extend_from_slice(payload);
    Ok(blob)
}

#[inline]
fn bool_to_u1(b: bool) -> u8 {
    u8::from(b)
}

fn encode_res_name_u32(s: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, ch) in s.chars().take(8).enumerate() {
        let u = ch as u32;
        out[i * 4..i * 4 + 4].copy_from_slice(&u.to_le_bytes());
    }
    out
}

fn decode_res_name_u32(chunk: &[u8]) -> Result<String> {
    if chunk.len() != 32 {
        bail!("res_name chunk len {}", chunk.len());
    }
    let mut s = String::new();
    for i in 0..8 {
        let u = u32::from_le_bytes(
            chunk[i * 4..i * 4 + 4]
                .try_into()
                .map_err(|_| anyhow!("res_name slice"))?,
        );
        if u == 0 {
            break;
        }
        let ch = char::from_u32(u).ok_or_else(|| anyhow!("invalid res_name scalar {u}"))?;
        s.push(ch);
    }
    Ok(s)
}

fn build_token_npz_blobs(
    tokens: &[TokenData],
    bonds: &[TokenBondV2],
) -> Result<BTreeMap<String, Vec<u8>>> {
    let n = tokens.len();
    let m = bonds.len();
    let mut cols: BTreeMap<String, Vec<u8>> = BTreeMap::new();

    macro_rules! col_i4 {
        ($name:expr, $field:ident) => {{
            let mut v = Vec::with_capacity(n * 4);
            for t in tokens {
                v.extend_from_slice(&t.$field.to_le_bytes());
            }
            cols.insert($name.to_string(), write_npy(DESCR_I4, &[n], &v)?);
        }};
    }

    col_i4!("t_token_idx", token_idx);
    col_i4!("t_atom_idx", atom_idx);
    col_i4!("t_atom_num", atom_num);
    col_i4!("t_res_idx", res_idx);
    col_i4!("t_res_type", res_type);
    col_i4!("t_sym_id", sym_id);
    col_i4!("t_asym_id", asym_id);
    col_i4!("t_entity_id", entity_id);
    col_i4!("t_mol_type", mol_type);
    col_i4!("t_center_idx", center_idx);
    col_i4!("t_disto_idx", disto_idx);
    col_i4!("t_cyclic_period", cyclic_period);

    let mut frame_mask_i4 = Vec::with_capacity(n * 4);
    for t in tokens {
        frame_mask_i4.extend_from_slice(&i32::from(t.frame_mask).to_le_bytes());
    }
    cols.insert(
        "t_frame_mask.npy".to_string(),
        write_npy(DESCR_I4, &[n], &frame_mask_i4)?,
    );

    let mut res_name = Vec::with_capacity(n * 32);
    for t in tokens {
        res_name.extend_from_slice(&encode_res_name_u32(&t.res_name));
    }
    cols.insert("t_res_name.npy".to_string(), write_npy(DESCR_U4, &[n, 8], &res_name)?);

    let mut center_c = Vec::with_capacity(n * 12);
    let mut disto_c = Vec::with_capacity(n * 12);
    let mut frame_r = Vec::with_capacity(n * 36);
    let mut frame_t = Vec::with_capacity(n * 12);
    for t in tokens {
        for c in &t.center_coords {
            center_c.extend_from_slice(&c.to_le_bytes());
        }
        for c in &t.disto_coords {
            disto_c.extend_from_slice(&c.to_le_bytes());
        }
        for c in &t.frame_rot {
            frame_r.extend_from_slice(&c.to_le_bytes());
        }
        for c in &t.frame_t {
            frame_t.extend_from_slice(&c.to_le_bytes());
        }
    }
    cols.insert(
        "t_center_coords.npy".to_string(),
        write_npy(DESCR_F4, &[n, 3], &center_c)?,
    );
    cols.insert(
        "t_disto_coords.npy".to_string(),
        write_npy(DESCR_F4, &[n, 3], &disto_c)?,
    );
    cols.insert(
        "t_frame_rot.npy".to_string(),
        write_npy(DESCR_F4, &[n, 9], &frame_r)?,
    );
    cols.insert(
        "t_frame_t.npy".to_string(),
        write_npy(DESCR_F4, &[n, 3], &frame_t)?,
    );

    let mut rb = Vec::with_capacity(n);
    let mut db = Vec::with_capacity(n);
    let mut mb = Vec::with_capacity(n);
    let mut ab = Vec::with_capacity(n);
    for t in tokens {
        rb.push(bool_to_u1(t.resolved_mask));
        db.push(bool_to_u1(t.disto_mask));
        mb.push(bool_to_u1(t.modified));
        ab.push(bool_to_u1(t.affinity_mask));
    }
    cols.insert(
        "t_resolved_mask.npy".to_string(),
        write_npy(DESCR_U1, &[n], &rb)?,
    );
    cols.insert(
        "t_disto_mask.npy".to_string(),
        write_npy(DESCR_U1, &[n], &db)?,
    );
    cols.insert("t_modified.npy".to_string(), write_npy(DESCR_U1, &[n], &mb)?);
    cols.insert(
        "t_affinity_mask.npy".to_string(),
        write_npy(DESCR_U1, &[n], &ab)?,
    );

    let mut b1 = Vec::with_capacity(m * 4);
    let mut b2 = Vec::with_capacity(m * 4);
    let mut bt = Vec::with_capacity(m);
    for &(x, y, ty) in bonds {
        b1.extend_from_slice(&x.to_le_bytes());
        b2.extend_from_slice(&y.to_le_bytes());
        bt.push(ty as u8);
    }
    cols.insert("bond_token_1.npy".to_string(), write_npy(DESCR_I4, &[m], &b1)?);
    cols.insert("bond_token_2.npy".to_string(), write_npy(DESCR_I4, &[m], &b2)?);
    cols.insert("bond_type.npy".to_string(), write_npy(DESCR_I1, &[m], &bt)?);

    Ok(cols)
}

/// Write tokens and token bonds as a compressed `.npz` (zip of `.npy` columns).
pub fn write_token_batch_npz_compressed(path: &Path, tokens: &[TokenData], bonds: &[TokenBondV2]) -> Result<()> {
    let cols = build_token_npz_blobs(tokens, bonds)?;
    let mut file = File::create(path).with_context(|| path.display().to_string())?;
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let mut zw = ZipWriter::new(&mut file);
    for (name, blob) in cols {
        zw.start_file(name, opts)?;
        zw.write_all(&blob)?;
    }
    zw.finish()?;
    Ok(())
}

/// Same as [`write_token_batch_npz_compressed`] but returns raw `.npz` bytes (for tests / golden hooks).
pub fn write_token_batch_npz_to_vec(tokens: &[TokenData], bonds: &[TokenBondV2]) -> Result<Vec<u8>> {
    let cols = build_token_npz_blobs(tokens, bonds)?;
    let mut buf = Cursor::new(Vec::new());
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let mut zw = ZipWriter::new(&mut buf);
    for (name, blob) in cols {
        zw.start_file(name, opts)?;
        zw.write_all(&blob)?;
    }
    zw.finish()?;
    Ok(buf.into_inner())
}

fn parse_shape_tuple(header: &str) -> Result<Vec<usize>> {
    let key = "'shape': ";
    let pos = header
        .find(key)
        .ok_or_else(|| anyhow!("NPY header missing 'shape'"))?;
    let rest = header[pos + key.len()..]
        .chars()
        .take_while(|&c| c != '\n')
        .collect::<String>();
    let rest = rest.trim_start();
    if rest.starts_with("()") {
        return Ok(Vec::new());
    }
    let rest = rest
        .strip_prefix('(')
        .ok_or_else(|| anyhow!("invalid shape"))?;
    let end = rest
        .find(')')
        .ok_or_else(|| anyhow!("unclosed shape"))?;
    let inner = rest[..end].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for p in inner.split(',') {
        let p = p.trim();
        if p.is_empty() {
            continue;
        }
        out.push(p.parse::<usize>()?);
    }
    Ok(out)
}

fn parse_npy_shape_and_payload(data: &[u8]) -> Result<(Vec<usize>, &[u8])> {
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
    let header_str = std::str::from_utf8(header_raw)
        .map_err(|_| anyhow!("NPY header is not valid UTF-8"))?;
    let shape = parse_shape_tuple(header_str)?;
    let payload_off = 10 + hlen;
    let payload = data
        .get(payload_off..)
        .ok_or_else(|| anyhow!("missing NPY payload"))?;
    Ok((shape, payload))
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

fn read_i4_col(npy: &[u8], expect_n: usize) -> Result<Vec<i32>> {
    let (shape, payload) = parse_npy_shape_and_payload(npy)?;
    let n = *shape.first().unwrap_or(&0);
    if shape.len() != 1 || n != expect_n {
        bail!("expected 1-D len {expect_n}, got shape {shape:?}");
    }
    if payload.len() != n * 4 {
        bail!("i4 column length mismatch");
    }
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(i32::from_le_bytes(
            payload[i * 4..i * 4 + 4].try_into().unwrap(),
        ));
    }
    Ok(v)
}

fn read_u1_col(npy: &[u8], expect_n: usize) -> Result<Vec<u8>> {
    let (shape, payload) = parse_npy_shape_and_payload(npy)?;
    let n = *shape.first().unwrap_or(&0);
    if shape.len() != 1 || n != expect_n {
        bail!("expected 1-D len {expect_n}, got shape {shape:?}");
    }
    if payload.len() != n {
        bail!("u1 column length mismatch");
    }
    Ok(payload.to_vec())
}

fn read_i1_col(npy: &[u8], expect_m: usize) -> Result<Vec<i8>> {
    let (shape, payload) = parse_npy_shape_and_payload(npy)?;
    let m = *shape.first().unwrap_or(&0);
    if shape.len() != 1 || m != expect_m {
        bail!("expected 1-D len {expect_m}, got shape {shape:?}");
    }
    if payload.len() != m {
        bail!("i1 column length mismatch");
    }
    Ok(payload.iter().map(|&b| b as i8).collect())
}

fn read_f4_2d(npy: &[u8], expect_n: usize, expect_c: usize) -> Result<Vec<f32>> {
    let (shape, payload) = parse_npy_shape_and_payload(npy)?;
    if shape.len() != 2 || shape[0] != expect_n || shape[1] != expect_c {
        bail!("expected shape ({expect_n}, {expect_c}), got {shape:?}");
    }
    let need = expect_n * expect_c * 4;
    if payload.len() != need {
        bail!("f4 2d payload len mismatch");
    }
    let mut v = Vec::with_capacity(expect_n * expect_c);
    for i in 0..expect_n * expect_c {
        v.push(f32::from_le_bytes(
            payload[i * 4..i * 4 + 4].try_into().unwrap(),
        ));
    }
    Ok(v)
}

fn u1_as_bool(b: u8) -> bool {
    b != 0
}

/// Load columnar token `.npz` written by [`write_token_batch_npz_compressed`].
pub fn read_token_batch_npz_path(path: &Path) -> Result<(Vec<TokenData>, Vec<TokenBondV2>)> {
    let mut file = File::open(path).with_context(|| path.display().to_string())?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    read_token_batch_npz_bytes(&buf)
}

pub fn read_token_batch_npz_bytes(zip_bytes: &[u8]) -> Result<(Vec<TokenData>, Vec<TokenBondV2>)> {
    let cursor = Cursor::new(zip_bytes);
    let mut archive = ZipArchive::new(cursor)?;

    let bond_1 = read_zip_npy(&mut archive, "bond_token_1")?;
    let bond_2 = read_zip_npy(&mut archive, "bond_token_2")?;
    let bond_ty = read_zip_npy(&mut archive, "bond_type")?;
    let m = parse_npy_shape_and_payload(&bond_1)?
        .0
        .first()
        .copied()
        .unwrap_or(0);
    let t1 = read_i4_col(&bond_1, m)?;
    let t2 = read_i4_col(&bond_2, m)?;
    let tty = read_i1_col(&bond_ty, m)?;
    let bonds: Vec<TokenBondV2> = (0..m)
        .map(|i| (t1[i], t2[i], tty[i]))
        .collect();

    let idx_npy = read_zip_npy(&mut archive, "t_token_idx")?;
    let n = parse_npy_shape_and_payload(&idx_npy)?
        .0
        .first()
        .copied()
        .unwrap_or(0);

    let token_idx = read_i4_col(&idx_npy, n)?;
    let atom_idx = read_i4_col(&read_zip_npy(&mut archive, "t_atom_idx")?, n)?;
    let atom_num = read_i4_col(&read_zip_npy(&mut archive, "t_atom_num")?, n)?;
    let res_idx = read_i4_col(&read_zip_npy(&mut archive, "t_res_idx")?, n)?;
    let res_type = read_i4_col(&read_zip_npy(&mut archive, "t_res_type")?, n)?;
    let sym_id = read_i4_col(&read_zip_npy(&mut archive, "t_sym_id")?, n)?;
    let asym_id = read_i4_col(&read_zip_npy(&mut archive, "t_asym_id")?, n)?;
    let entity_id = read_i4_col(&read_zip_npy(&mut archive, "t_entity_id")?, n)?;
    let mol_type = read_i4_col(&read_zip_npy(&mut archive, "t_mol_type")?, n)?;
    let center_idx = read_i4_col(&read_zip_npy(&mut archive, "t_center_idx")?, n)?;
    let disto_idx = read_i4_col(&read_zip_npy(&mut archive, "t_disto_idx")?, n)?;
    let cyclic_period = read_i4_col(&read_zip_npy(&mut archive, "t_cyclic_period")?, n)?;
    let frame_mask_i = read_i4_col(&read_zip_npy(&mut archive, "t_frame_mask")?, n)?;

    let res_npy = read_zip_npy(&mut archive, "t_res_name")?;
    let (rshape, rpay) = parse_npy_shape_and_payload(&res_npy)?;
    if rshape.len() != 2 || rshape[0] != n || rshape[1] != 8 {
        bail!("t_res_name shape mismatch");
    }
    if rpay.len() != n * 32 {
        bail!("t_res_name payload size");
    }

    let center_flat = read_f4_2d(&read_zip_npy(&mut archive, "t_center_coords")?, n, 3)?;
    let disto_flat = read_f4_2d(&read_zip_npy(&mut archive, "t_disto_coords")?, n, 3)?;
    let rot_flat = read_f4_2d(&read_zip_npy(&mut archive, "t_frame_rot")?, n, 9)?;
    let ft_flat = read_f4_2d(&read_zip_npy(&mut archive, "t_frame_t")?, n, 3)?;

    let rb = read_u1_col(&read_zip_npy(&mut archive, "t_resolved_mask")?, n)?;
    let db = read_u1_col(&read_zip_npy(&mut archive, "t_disto_mask")?, n)?;
    let mb = read_u1_col(&read_zip_npy(&mut archive, "t_modified")?, n)?;
    let ab = read_u1_col(&read_zip_npy(&mut archive, "t_affinity_mask")?, n)?;

    let mut tokens = Vec::with_capacity(n);
    for i in 0..n {
        let mut center_coords = [0.0_f32; 3];
        let base_c = i * 3;
        center_coords.copy_from_slice(&center_flat[base_c..base_c + 3]);
        let mut disto_coords = [0.0_f32; 3];
        disto_coords.copy_from_slice(&disto_flat[base_c..base_c + 3]);
        let mut frame_rot = [0.0_f32; 9];
        let base_r = i * 9;
        frame_rot.copy_from_slice(&rot_flat[base_r..base_r + 9]);
        let mut frame_t = [0.0_f32; 3];
        frame_t.copy_from_slice(&ft_flat[base_c..base_c + 3]);
        let name = decode_res_name_u32(&rpay[i * 32..i * 32 + 32])?;
        tokens.push(TokenData {
            token_idx: token_idx[i],
            atom_idx: atom_idx[i],
            atom_num: atom_num[i],
            res_idx: res_idx[i],
            res_type: res_type[i],
            res_name: name,
            sym_id: sym_id[i],
            asym_id: asym_id[i],
            entity_id: entity_id[i],
            mol_type: mol_type[i],
            center_idx: center_idx[i],
            disto_idx: disto_idx[i],
            center_coords,
            disto_coords,
            resolved_mask: u1_as_bool(rb[i]),
            disto_mask: u1_as_bool(db[i]),
            modified: u1_as_bool(mb[i]),
            frame_rot,
            frame_t,
            frame_mask: frame_mask_i[i] != 0,
            cyclic_period: cyclic_period[i],
            affinity_mask: u1_as_bool(ab[i]),
        });
    }

    Ok((tokens, bonds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boltz_const::{chain_type_id, token_id};
    use crate::structure_v2::{AtomV2Row, ChainRow, ResidueRow, StructureV2Tables};
    use crate::tokenize::boltz2::tokenize_structure;
    use std::io::Cursor;

    fn ala_tables() -> StructureV2Tables {
        let p = chain_type_id("PROTEIN").unwrap() as i8;
        let coords = vec![
            [0.0_f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.5, 1.0, 0.0],
        ];
        let atoms: Vec<_> = coords
            .iter()
            .map(|&c| AtomV2Row {
                coords: c,
                is_present: true,
            })
            .collect();
        let ala_id = token_id("ALA").unwrap() as i8;
        StructureV2Tables {
            atoms,
            residues: vec![ResidueRow {
                name: "ALA".to_string(),
                res_type: ala_id,
                res_idx: 0,
                atom_idx: 0,
                atom_num: 5,
                atom_center: 1,
                atom_disto: 4,
                is_standard: true,
                is_present: true,
            }],
            chains: vec![ChainRow {
                mol_type: p,
                sym_id: 0,
                asym_id: 0,
                entity_id: 0,
                res_idx: 0,
                res_num: 1,
                cyclic_period: 0,
            }],
            chain_mask: vec![true],
            coords: coords.clone(),
            ensemble_atom_coord_idx: 0,
            bonds: vec![],
        }
    }

    #[test]
    fn roundtrip_token_npz_ala() {
        let s = ala_tables();
        let (tokens, bonds) = tokenize_structure(&s, None);
        let raw = write_token_batch_npz_to_vec(&tokens, &bonds).unwrap();
        let (back_t, back_b) = read_token_batch_npz_bytes(&raw).unwrap();
        assert_eq!(back_t, tokens);
        assert_eq!(back_b, bonds);
    }

    #[test]
    fn frame_mask_stored_as_i4() {
        let s = ala_tables();
        let (tokens, bonds) = tokenize_structure(&s, None);
        let raw = write_token_batch_npz_to_vec(&tokens, &bonds).unwrap();
        let cursor = Cursor::new(&raw);
        let mut ar = ZipArchive::new(cursor).unwrap();
        let mut buf = Vec::new();
        ar.by_name("t_frame_mask.npy")
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        let (_, pay) = parse_npy_shape_and_payload(&buf).unwrap();
        assert_eq!(pay.len(), 4);
        assert_eq!(i32::from_le_bytes(pay.try_into().unwrap()), 1);
    }
}
