//! Packed `TokenV2` layout matching Boltz [`numpy.dtype`](https://numpy.org/doc/stable/reference/arrays.dtypes.html)
//! for `boltz.data.types.TokenV2` (see upstream `types.py`).
//!
//! NumPy structured dtype with `align=False` (default) packs fields in order; padding is inserted so
//! `9f4` (`frame_rot`) starts on a 4-byte boundary after three `?` booleans (`resolved_mask`,
//! `disto_mask`, `modified`). The final `affinity_mask` bool is followed by padding so the record
//! size is a multiple of 4 bytes.
//!
//! Columnar `.npz` in [`crate::token_npz`] remains the primary interchange format; this module
//! provides **byte-exact** `|V{N}` rows for `arr.view(TokenV2)`-style checks in Python.

use anyhow::{anyhow, Result};

use crate::tokenize::boltz2::TokenData;

/// Row size in bytes for one `TokenV2` record (NumPy `align=False` packing).
pub const TOKEN_V2_NUMPY_ITEMSIZE: usize = 164;

#[inline]
fn write_i4(buf: &mut [u8], off: usize, v: i32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

#[inline]
fn write_f4(buf: &mut [u8], off: usize, v: f32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

#[inline]
fn read_i4(buf: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

#[inline]
fn read_f4(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

/// Encode `res_name` as NumPy `<U8` (8 UTF-32 LE code points, 32 bytes).
#[inline]
pub fn encode_res_name_unicode_u8(s: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, ch) in s.chars().take(8).enumerate() {
        let u = ch as u32;
        out[i * 4..i * 4 + 4].copy_from_slice(&u.to_le_bytes());
    }
    out
}

/// Decode NumPy `<U8` bytes to `String` (same rules as Boltz `np.ndarray` string fields).
#[inline]
pub fn decode_res_name_unicode_u8(chunk: &[u8; 32]) -> Result<String> {
    let mut s = String::new();
    for i in 0..8 {
        let u = u32::from_le_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        if u == 0 {
            break;
        }
        let ch = char::from_u32(u).ok_or_else(|| anyhow!("invalid res_name scalar {u}"))?;
        s.push(ch);
    }
    Ok(s)
}

/// Pack one token into a NumPy-compatible `TokenV2` row (`align=False`).
#[must_use]
pub fn pack_token_v2_row(t: &TokenData) -> [u8; TOKEN_V2_NUMPY_ITEMSIZE] {
    let mut b = [0u8; TOKEN_V2_NUMPY_ITEMSIZE];
    let mut o = 0usize;
    write_i4(&mut b, o, t.token_idx);
    o += 4;
    write_i4(&mut b, o, t.atom_idx);
    o += 4;
    write_i4(&mut b, o, t.atom_num);
    o += 4;
    write_i4(&mut b, o, t.res_idx);
    o += 4;
    write_i4(&mut b, o, t.res_type);
    o += 4;
    b[o..o + 32].copy_from_slice(&encode_res_name_unicode_u8(&t.res_name));
    o += 32;
    write_i4(&mut b, o, t.sym_id);
    o += 4;
    write_i4(&mut b, o, t.asym_id);
    o += 4;
    write_i4(&mut b, o, t.entity_id);
    o += 4;
    write_i4(&mut b, o, t.mol_type);
    o += 4;
    write_i4(&mut b, o, t.center_idx);
    o += 4;
    write_i4(&mut b, o, t.disto_idx);
    o += 4;
    for c in &t.center_coords {
        write_f4(&mut b, o, *c);
        o += 4;
    }
    for c in &t.disto_coords {
        write_f4(&mut b, o, *c);
        o += 4;
    }
    b[o] = u8::from(t.resolved_mask);
    o += 1;
    b[o] = u8::from(t.disto_mask);
    o += 1;
    b[o] = u8::from(t.modified);
    o += 1;
    // Pad so `frame_rot` (first element 4-byte aligned) starts at offset 104.
    o += 1;
    debug_assert_eq!(o, 104);
    for c in &t.frame_rot {
        write_f4(&mut b, o, *c);
        o += 4;
    }
    for c in &t.frame_t {
        write_f4(&mut b, o, *c);
        o += 4;
    }
    write_i4(&mut b, o, i32::from(t.frame_mask));
    o += 4;
    write_i4(&mut b, o, t.cyclic_period);
    o += 4;
    b[o] = u8::from(t.affinity_mask);
    o += 1;
    debug_assert_eq!(o, 161);
    // Trailing padding to multiple of 4 (struct itemsize 164).
    while o < TOKEN_V2_NUMPY_ITEMSIZE {
        b[o] = 0;
        o += 1;
    }
    b
}

/// Unpack a packed `TokenV2` row (inverse of [`pack_token_v2_row`]).
pub fn unpack_token_v2_row(buf: &[u8; TOKEN_V2_NUMPY_ITEMSIZE]) -> Result<TokenData> {
    let b = buf;
    let mut o = 0usize;
    let token_idx = read_i4(b, o);
    o += 4;
    let atom_idx = read_i4(b, o);
    o += 4;
    let atom_num = read_i4(b, o);
    o += 4;
    let res_idx = read_i4(b, o);
    o += 4;
    let res_type = read_i4(b, o);
    o += 4;
    let mut rn = [0u8; 32];
    rn.copy_from_slice(&b[o..o + 32]);
    let res_name = decode_res_name_unicode_u8(&rn)?;
    o += 32;
    let sym_id = read_i4(b, o);
    o += 4;
    let asym_id = read_i4(b, o);
    o += 4;
    let entity_id = read_i4(b, o);
    o += 4;
    let mol_type = read_i4(b, o);
    o += 4;
    let center_idx = read_i4(b, o);
    o += 4;
    let disto_idx = read_i4(b, o);
    o += 4;
    let mut center_coords = [0.0_f32; 3];
    for c in &mut center_coords {
        *c = read_f4(b, o);
        o += 4;
    }
    let mut disto_coords = [0.0_f32; 3];
    for c in &mut disto_coords {
        *c = read_f4(b, o);
        o += 4;
    }
    let resolved_mask = b[o] != 0;
    o += 1;
    let disto_mask = b[o] != 0;
    o += 1;
    let modified = b[o] != 0;
    o += 1;
    o += 1; // padding
    debug_assert_eq!(o, 104);
    let mut frame_rot = [0.0_f32; 9];
    for c in &mut frame_rot {
        *c = read_f4(b, o);
        o += 4;
    }
    let mut frame_t = [0.0_f32; 3];
    for c in &mut frame_t {
        *c = read_f4(b, o);
        o += 4;
    }
    let frame_mask = read_i4(b, o) != 0;
    o += 4;
    let cyclic_period = read_i4(b, o);
    o += 4;
    let affinity_mask = b[o] != 0;
    Ok(TokenData {
        token_idx,
        atom_idx,
        atom_num,
        res_idx,
        res_type,
        res_name,
        sym_id,
        asym_id,
        entity_id,
        mol_type,
        center_idx,
        disto_idx,
        center_coords,
        disto_coords,
        resolved_mask,
        disto_mask,
        modified,
        frame_rot,
        frame_t,
        frame_mask,
        cyclic_period,
        affinity_mask,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::structure_v2_single_ala;
    use crate::tokenize::boltz2::tokenize_structure;

    #[test]
    fn pack_unpack_roundtrip_ala() {
        let s = structure_v2_single_ala();
        let (tokens, _) = tokenize_structure(&s, None);
        assert_eq!(tokens.len(), 1);
        let t = &tokens[0];
        let packed = pack_token_v2_row(t);
        let back = unpack_token_v2_row(&packed).unwrap();
        assert_eq!(*t, back);
    }

    #[test]
    fn itemsize_matches_field_span() {
        // Last meaningful byte is affinity_mask at 160; padding 161..164.
        assert_eq!(TOKEN_V2_NUMPY_ITEMSIZE, 164);
    }
}
