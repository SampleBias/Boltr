//! Load / save Boltz `StructureV2` preprocess `.npz` (`atoms`, `bonds`, `residues`, `chains`,
//! `interfaces`, `mask`, `coords`, `ensemble`) into [`crate::structure_v2::StructureV2Tables`].
//!
//! NumPy may use **packed** or **aligned** structured layouts; both are accepted for read.
//! Writes use the **aligned** layout (common with `align=True`-style dtypes).

use std::fs::File;
use std::io::{Cursor, Read, Seek, Write};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::structure_v2::{AtomV2Row, BondV2AtomRow, ChainRow, ResidueRow, StructureV2Tables};

const MAGIC_PREFIX: &[u8] = b"\x93NUMPY";
const ARRAY_ALIGN: usize = 64;
const GROWTH_AXIS_MAX_DIGITS: usize = 21;
const MAGIC_LEN: usize = 8;

// --- Aligned record sizes (see module docs; match typical NumPy structured dtypes) ---
const ATOM_V2_AL: usize = 40;
const RESIDUE_AL: usize = 48;
const CHAIN_AL: usize = 56;
const BOND_V2_AL: usize = 28;
const COORDS_AL: usize = 12;
const ENSEMBLE_AL: usize = 8;

// Packed sizes (align=False style sequential layout)
const ATOM_V2_PK: usize = 37;
const RESIDUE_PK: usize = 43;
const CHAIN_PK: usize = 53;
const BOND_V2_PK: usize = 25;

fn shape_repr_1d(n: usize) -> String {
    format!("({n},)")
}

fn growth_pad(shape_line: &str) -> usize {
    let inner = shape_line.trim_matches(|c| c == '(' || c == ')');
    let first = inner.split(',').next().unwrap_or("0").trim();
    GROWTH_AXIS_MAX_DIGITS.saturating_sub(first.len())
}

fn numpy_header_dict_v1(descr: &str, shape_line: &str, fortran: bool) -> String {
    let fo = if fortran { "True" } else { "False" };
    let mut inner = String::new();
    inner.push('{');
    for (k, v) in [
        ("descr", descr),
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
    format!("{}{}", inner, " ".repeat(growth_pad(shape_line)))
}

fn wrap_npy_header_v1(header_dict: &str) -> Result<Vec<u8>> {
    let header = header_dict.as_bytes();
    let hlen = header.len() + 1;
    let padlen = ARRAY_ALIGN - ((MAGIC_LEN + 2 + hlen) % ARRAY_ALIGN);
    let total = hlen + padlen;
    if total > u16::MAX as usize {
        bail!("NPY header too large");
    }
    let mut out = Vec::with_capacity(MAGIC_LEN + 2 + total);
    out.extend_from_slice(MAGIC_PREFIX);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(total as u16).to_le_bytes());
    out.extend_from_slice(header);
    out.extend(std::iter::repeat_n(b' ', padlen));
    out.push(b'\n');
    Ok(out)
}

fn write_npy_1d(descr: &str, n: usize, payload: &[u8]) -> Result<Vec<u8>> {
    let shape = shape_repr_1d(n);
    let dict = numpy_header_dict_v1(descr, &shape, false);
    let mut b = wrap_npy_header_v1(&dict)?;
    b.extend_from_slice(payload);
    Ok(b)
}

fn parse_shape_tuple(header: &str) -> Result<Vec<usize>> {
    let key = "'shape': ";
    let pos = header
        .find(key)
        .ok_or_else(|| anyhow!("NPY header missing shape"))?;
    let rest: String = header[pos + key.len()..]
        .chars()
        .take_while(|&c| c != '\n')
        .collect();
    let rest = rest.trim_start();
    if rest.starts_with("()") {
        return Ok(Vec::new());
    }
    let rest = rest.strip_prefix('(').ok_or_else(|| anyhow!("bad shape"))?;
    let end = rest.find(')').ok_or_else(|| anyhow!("unclosed shape"))?;
    let inner = rest[..end].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .filter_map(|p| {
            let p = p.trim();
            if p.is_empty() {
                None
            } else {
                Some(p.parse::<usize>())
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow!("shape parse: {e}"))
}

fn parse_npy_shape_and_payload(data: &[u8]) -> Result<(Vec<usize>, &[u8])> {
    if data.len() < 10 || &data[..6] != MAGIC_PREFIX {
        bail!("invalid .npy");
    }
    if data[6] != 1 || data[7] != 0 {
        bail!("unsupported NPY version");
    }
    let hlen = u16::from_le_bytes([data[8], data[9]]) as usize;
    if data.len() < 10 + hlen {
        bail!("truncated NPY");
    }
    let header_str =
        std::str::from_utf8(&data[10..10 + hlen]).map_err(|_| anyhow!("NPY header not UTF-8"))?;
    let shape = parse_shape_tuple(header_str)?;
    let payload = data
        .get(10 + hlen..)
        .ok_or_else(|| anyhow!("missing payload"))?;
    Ok((shape, payload))
}

fn read_zip_npy<R: Read + Seek>(z: &mut ZipArchive<R>, stem: &str) -> Result<Vec<u8>> {
    let p = format!("{stem}.npy");
    let mut v = Vec::new();
    if let Ok(mut f) = z.by_name(&p) {
        f.read_to_end(&mut v)?;
        return Ok(v);
    }
    let mut f = z
        .by_name(stem)
        .with_context(|| format!("missing {p} in npz"))?;
    f.read_to_end(&mut v)?;
    Ok(v)
}

#[inline]
fn read_i32_le(s: &[u8], o: usize) -> Result<i32> {
    Ok(i32::from_le_bytes(
        s.get(o..o + 4)
            .ok_or_else(|| anyhow!("oob i32"))?
            .try_into()?,
    ))
}

#[inline]
fn read_f32_le(s: &[u8], o: usize) -> Result<f32> {
    Ok(f32::from_le_bytes(
        s.get(o..o + 4)
            .ok_or_else(|| anyhow!("oob f32"))?
            .try_into()?,
    ))
}

fn read_coords3(s: &[u8], o: usize) -> Result<[f32; 3]> {
    Ok([
        read_f32_le(s, o)?,
        read_f32_le(s, o + 4)?,
        read_f32_le(s, o + 8)?,
    ])
}

fn decode_unicode_u32(s: &[u8], code_units: usize) -> Result<String> {
    let need = code_units * 4;
    if s.len() < need {
        bail!("unicode field");
    }
    let mut out = String::new();
    for i in 0..code_units {
        let u = u32::from_le_bytes(s[i * 4..i * 4 + 4].try_into()?);
        if u == 0 {
            break;
        }
        out.push(char::from_u32(u).ok_or_else(|| anyhow!("bad scalar"))?);
    }
    Ok(out)
}

fn decode_atom_row(r: &[u8]) -> Result<AtomV2Row> {
    let (off_name, off_coords, off_flag, off_bfactor, off_plddt) = match r.len() {
        ATOM_V2_AL => (0, 16, 28, 32, 36),
        ATOM_V2_PK => (0, 16, 28, 29, 33),
        n => bail!("atoms: unexpected record size {n} (expected {ATOM_V2_AL} or {ATOM_V2_PK})"),
    };
    // name is 4 × uint32 LE (Unicode code units), occupying bytes [0..16)
    let name = decode_unicode_u32(&r[off_name..off_name + 16], 4)?;
    Ok(AtomV2Row {
        name,
        coords: read_coords3(r, off_coords)?,
        is_present: r.get(off_flag).copied().unwrap_or(0) != 0,
        bfactor: read_f32_le(r, off_bfactor)?,
        plddt: read_f32_le(r, off_plddt)?,
    })
}

fn decode_residue_row(r: &[u8]) -> Result<ResidueRow> {
    let (name_end, o_rt, o_ri, o_ai, o_an, o_ac, o_ad, o_std, o_pr) = match r.len() {
        RESIDUE_AL => (20, 20, 24, 28, 32, 36, 40, 44, 45),
        RESIDUE_PK => (20, 20, 21, 25, 29, 33, 37, 41, 42),
        n => bail!("residues: unexpected record size {n}"),
    };
    let name = decode_unicode_u32(&r[..name_end], 5)?;
    Ok(ResidueRow {
        name,
        res_type: *r.get(o_rt).ok_or_else(|| anyhow!("res_type"))? as i8,
        res_idx: read_i32_le(r, o_ri)?,
        atom_idx: read_i32_le(r, o_ai)?,
        atom_num: read_i32_le(r, o_an)?,
        atom_center: read_i32_le(r, o_ac)?,
        atom_disto: read_i32_le(r, o_ad)?,
        is_standard: r.get(o_std).copied().unwrap_or(0) != 0,
        is_present: r.get(o_pr).copied().unwrap_or(0) != 0,
    })
}

fn decode_chain_row(r: &[u8]) -> Result<ChainRow> {
    let (o_mt, o_ent, o_sym, o_asym, o_ai, o_an, o_ri, o_rn, o_cy) = match r.len() {
        CHAIN_AL => (20, 24, 28, 32, 36, 40, 44, 48, 52),
        CHAIN_PK => (20, 21, 25, 29, 33, 37, 41, 45, 49),
        n => bail!("chains: unexpected record size {n}"),
    };
    Ok(ChainRow {
        mol_type: *r.get(o_mt).ok_or_else(|| anyhow!("mol_type"))? as i8,
        sym_id: read_i32_le(r, o_sym)?,
        asym_id: read_i32_le(r, o_asym)?,
        entity_id: read_i32_le(r, o_ent)?,
        atom_idx: read_i32_le(r, o_ai)?,
        atom_num: read_i32_le(r, o_an)?,
        res_idx: read_i32_le(r, o_ri)?,
        res_num: read_i32_le(r, o_rn)?,
        cyclic_period: read_i32_le(r, o_cy)?,
    })
}

fn decode_bond_row(r: &[u8]) -> Result<BondV2AtomRow> {
    let (o_a1, o_a2, o_ty) = match r.len() {
        BOND_V2_AL | BOND_V2_PK => (16, 20, 24),
        n => bail!("bonds: unexpected record size {n}"),
    };
    Ok(BondV2AtomRow {
        atom_1: read_i32_le(r, o_a1)?,
        atom_2: read_i32_le(r, o_a2)?,
        bond_type: *r.get(o_ty).ok_or_else(|| anyhow!("bond type"))? as i8,
    })
}

/// Load Boltz `StructureV2` `.npz` into [`StructureV2Tables`].
pub fn read_structure_v2_npz_path(path: &Path) -> Result<StructureV2Tables> {
    let mut f = File::open(path).with_context(|| path.display().to_string())?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    read_structure_v2_npz_bytes(&buf)
}

pub fn read_structure_v2_npz_bytes(zip_bytes: &[u8]) -> Result<StructureV2Tables> {
    let cur = Cursor::new(zip_bytes);
    let mut z = ZipArchive::new(cur)?;

    let atoms_n = read_zip_npy(&mut z, "atoms")?;
    let bonds_n = read_zip_npy(&mut z, "bonds")?;
    let res_n = read_zip_npy(&mut z, "residues")?;
    let chains_n = read_zip_npy(&mut z, "chains")?;
    let mask_n = read_zip_npy(&mut z, "mask")?;
    let coords_n = read_zip_npy(&mut z, "coords")?;
    let ens_n = read_zip_npy(&mut z, "ensemble")?;

    let (ashape, apay) = parse_npy_shape_and_payload(&atoms_n)?;
    let n_atom = *ashape.first().unwrap_or(&0);
    if ashape.len() != 1 {
        bail!("atoms: expected 1-D");
    }
    let rec_a = if n_atom == 0 {
        0
    } else {
        if apay.len() % n_atom != 0 {
            bail!("atoms payload");
        }
        apay.len() / n_atom
    };
    let mut atoms = Vec::with_capacity(n_atom);
    for i in 0..n_atom {
        let chunk = &apay[i * rec_a..(i + 1) * rec_a];
        atoms.push(decode_atom_row(chunk)?);
    }

    let (bshape, bpay) = parse_npy_shape_and_payload(&bonds_n)?;
    let n_bond = *bshape.first().unwrap_or(&0);
    if bshape.len() != 1 {
        bail!("bonds: expected 1-D");
    }
    let rec_b = if n_bond == 0 {
        BOND_V2_AL // arbitrary
    } else {
        if bpay.len() % n_bond != 0 {
            bail!("bonds payload");
        }
        bpay.len() / n_bond
    };
    let mut bonds = Vec::with_capacity(n_bond);
    for i in 0..n_bond {
        let chunk = &bpay[i * rec_b..(i + 1) * rec_b];
        bonds.push(decode_bond_row(chunk)?);
    }

    let (rshape, rpay) = parse_npy_shape_and_payload(&res_n)?;
    let n_res = *rshape.first().unwrap_or(&0);
    if rshape.len() != 1 {
        bail!("residues: expected 1-D");
    }
    let rec_r = if n_res == 0 {
        0
    } else {
        if rpay.len() % n_res != 0 {
            bail!("residues payload");
        }
        rpay.len() / n_res
    };
    let mut residues = Vec::with_capacity(n_res);
    for i in 0..n_res {
        let chunk = &rpay[i * rec_r..(i + 1) * rec_r];
        residues.push(decode_residue_row(chunk)?);
    }

    let (cshape, cpay) = parse_npy_shape_and_payload(&chains_n)?;
    let n_ch = *cshape.first().unwrap_or(&0);
    if cshape.len() != 1 {
        bail!("chains: expected 1-D");
    }
    let rec_c = if n_ch == 0 {
        0
    } else {
        if cpay.len() % n_ch != 0 {
            bail!("chains payload");
        }
        cpay.len() / n_ch
    };
    let mut chains = Vec::with_capacity(n_ch);
    for i in 0..n_ch {
        let chunk = &cpay[i * rec_c..(i + 1) * rec_c];
        chains.push(decode_chain_row(chunk)?);
    }

    let (mshape, mpay) = parse_npy_shape_and_payload(&mask_n)?;
    let n_mask = *mshape.first().unwrap_or(&0);
    if mshape.len() != 1 || mpay.len() != n_mask {
        bail!("mask shape");
    }
    let chain_mask: Vec<bool> = mpay.iter().map(|&b| b != 0).collect();
    if chain_mask.len() != chains.len() {
        bail!(
            "mask len {} != chains len {}",
            chain_mask.len(),
            chains.len()
        );
    }

    let (xshape, xpay) = parse_npy_shape_and_payload(&coords_n)?;
    let n_coord = *xshape.first().unwrap_or(&0);
    if xshape.len() != 1 {
        bail!("coords: expected 1-D");
    }
    if n_coord > 0 && xpay.len() != n_coord * COORDS_AL {
        bail!("coords payload len");
    }
    let mut coords = Vec::with_capacity(n_coord);
    for i in 0..n_coord {
        let base = i * COORDS_AL;
        coords.push(read_coords3(xpay, base)?);
    }

    let (eshape, epay) = parse_npy_shape_and_payload(&ens_n)?;
    let n_ens = *eshape.first().unwrap_or(&0);
    if eshape.len() != 1 || n_ens < 1 {
        bail!("ensemble: need at least one row");
    }
    if epay.len() != n_ens * ENSEMBLE_AL {
        bail!("ensemble payload");
    }
    let ensemble_atom_coord_idx = read_i32_le(epay, 0)?;
    let _ensemble_atom_num = read_i32_le(epay, 4)?;

    Ok(StructureV2Tables {
        atoms,
        residues,
        chains,
        chain_mask,
        coords,
        ensemble_atom_coord_idx,
        bonds,
    })
}

// --- Writer (aligned layout only) ---

const DESCR_ATOM: &str =
    "[('name', '<U4'), ('coords', '<f4', (3,)), ('is_present', '|b1'), ('bfactor', '<f4'), ('plddt', '<f4')]";
const DESCR_RESIDUE: &str = "[('name', '<U5'), ('res_type', '|i1'), ('res_idx', '<i4'), ('atom_idx', '<i4'), ('atom_num', '<i4'), ('atom_center', '<i4'), ('atom_disto', '<i4'), ('is_standard', '|b1'), ('is_present', '|b1')]";
const DESCR_CHAIN: &str = "[('name', '<U5'), ('mol_type', '|i1'), ('entity_id', '<i4'), ('sym_id', '<i4'), ('asym_id', '<i4'), ('atom_idx', '<i4'), ('atom_num', '<i4'), ('res_idx', '<i4'), ('res_num', '<i4'), ('cyclic_period', '<i4')]";
const DESCR_BOND: &str = "[('chain_1', '<i4'), ('chain_2', '<i4'), ('res_1', '<i4'), ('res_2', '<i4'), ('atom_1', '<i4'), ('atom_2', '<i4'), ('type', '|i1')]";
const DESCR_COORDS: &str = "[('coords', '<f4', (3,))]";
const DESCR_ENSEMBLE: &str = "[('atom_coord_idx', '<i4'), ('atom_num', '<i4')]";
const DESCR_INTERFACE: &str = "[('chain_1', '<i4'), ('chain_2', '<i4')]";
const DESCR_MASK: &str = "|b1";

fn encode_u32_name(dst: &mut [u8], s: &str, code_units: usize) {
    dst[..code_units * 4].fill(0);
    for (i, ch) in s.chars().take(code_units).enumerate() {
        dst[i * 4..i * 4 + 4].copy_from_slice(&(ch as u32).to_le_bytes());
    }
}

fn pack_atom_aligned(a: &AtomV2Row) -> [u8; ATOM_V2_AL] {
    let mut r = [0u8; ATOM_V2_AL];
    encode_u32_name(&mut r[0..16], &a.name, 4);
    for (i, &c) in a.coords.iter().enumerate() {
        r[16 + i * 4..16 + i * 4 + 4].copy_from_slice(&c.to_le_bytes());
    }
    r[28] = u8::from(a.is_present);
    r[32..36].copy_from_slice(&a.bfactor.to_le_bytes());
    r[36..40].copy_from_slice(&a.plddt.to_le_bytes());
    r
}

fn pack_residue_aligned(res: &ResidueRow) -> [u8; RESIDUE_AL] {
    let mut r = [0u8; RESIDUE_AL];
    encode_u32_name(&mut r[..20], &res.name, 5);
    r[20] = res.res_type as u8;
    // pad 21-23
    r[24..28].copy_from_slice(&res.res_idx.to_le_bytes());
    r[28..32].copy_from_slice(&res.atom_idx.to_le_bytes());
    r[32..36].copy_from_slice(&res.atom_num.to_le_bytes());
    r[36..40].copy_from_slice(&res.atom_center.to_le_bytes());
    r[40..44].copy_from_slice(&res.atom_disto.to_le_bytes());
    r[44] = u8::from(res.is_standard);
    r[45] = u8::from(res.is_present);
    r
}

fn pack_chain_aligned(ch: &ChainRow) -> [u8; CHAIN_AL] {
    let mut r = [0u8; CHAIN_AL];
    encode_u32_name(&mut r[..20], "", 5);
    r[20] = ch.mol_type as u8;
    r[24..28].copy_from_slice(&ch.entity_id.to_le_bytes());
    r[28..32].copy_from_slice(&ch.sym_id.to_le_bytes());
    r[32..36].copy_from_slice(&ch.asym_id.to_le_bytes());
    r[36..40].copy_from_slice(&ch.atom_idx.to_le_bytes());
    r[40..44].copy_from_slice(&ch.atom_num.to_le_bytes());
    r[44..48].copy_from_slice(&ch.res_idx.to_le_bytes());
    r[48..52].copy_from_slice(&ch.res_num.to_le_bytes());
    r[52..56].copy_from_slice(&ch.cyclic_period.to_le_bytes());
    r
}

fn pack_bond_aligned(b: &BondV2AtomRow) -> [u8; BOND_V2_AL] {
    let mut r = [0u8; BOND_V2_AL];
    r[16..20].copy_from_slice(&b.atom_1.to_le_bytes());
    r[20..24].copy_from_slice(&b.atom_2.to_le_bytes());
    r[24] = b.bond_type as u8;
    r
}

/// Write [`StructureV2Tables`] as a Boltz-style `StructureV2` `.npz` (aligned dtypes).
pub fn write_structure_v2_npz_compressed(path: &Path, s: &StructureV2Tables) -> Result<()> {
    let mut atoms_v = Vec::with_capacity(s.atoms.len() * ATOM_V2_AL);
    for a in &s.atoms {
        atoms_v.extend_from_slice(&pack_atom_aligned(a));
    }
    let atoms_blob = write_npy_1d(DESCR_ATOM, s.atoms.len(), &atoms_v)?;

    let mut bonds_v = Vec::with_capacity(s.bonds.len() * BOND_V2_AL);
    for b in &s.bonds {
        bonds_v.extend_from_slice(&pack_bond_aligned(b));
    }
    let bonds_blob = write_npy_1d(DESCR_BOND, s.bonds.len(), &bonds_v)?;

    let mut res_v = Vec::with_capacity(s.residues.len() * RESIDUE_AL);
    for res in &s.residues {
        res_v.extend_from_slice(&pack_residue_aligned(res));
    }
    let res_blob = write_npy_1d(DESCR_RESIDUE, s.residues.len(), &res_v)?;

    let mut ch_v = Vec::with_capacity(s.chains.len() * CHAIN_AL);
    for ch in &s.chains {
        ch_v.extend_from_slice(&pack_chain_aligned(ch));
    }
    let ch_blob = write_npy_1d(DESCR_CHAIN, s.chains.len(), &ch_v)?;

    let if_blob = write_npy_1d(DESCR_INTERFACE, 0, &[])?;

    let mask: Vec<u8> = s.chain_mask.iter().map(|&b| u8::from(b)).collect();
    let mask_blob = write_npy_1d(DESCR_MASK, s.chain_mask.len(), &mask)?;

    let mut coord_v = Vec::with_capacity(s.coords.len() * COORDS_AL);
    for c in &s.coords {
        for &x in c {
            coord_v.extend_from_slice(&x.to_le_bytes());
        }
    }
    let coord_blob = write_npy_1d(DESCR_COORDS, s.coords.len(), &coord_v)?;

    let ncoord = i32::try_from(s.coords.len()).unwrap_or(i32::MAX);
    let mut ens = [0u8; ENSEMBLE_AL];
    ens[0..4].copy_from_slice(&s.ensemble_atom_coord_idx.to_le_bytes());
    ens[4..8].copy_from_slice(&ncoord.to_le_bytes());
    let ens_blob = write_npy_1d(DESCR_ENSEMBLE, 1, &ens)?;

    let mut file = File::create(path).with_context(|| path.display().to_string())?;
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let mut zw = ZipWriter::new(&mut file);
    // Match common `np.savez_compressed` key order loosely (sorted keys often used).
    for (name, blob) in [
        ("atoms.npy", atoms_blob),
        ("bonds.npy", bonds_blob),
        ("chains.npy", ch_blob),
        ("coords.npy", coord_blob),
        ("ensemble.npy", ens_blob),
        ("interfaces.npy", if_blob),
        ("mask.npy", mask_blob),
        ("residues.npy", res_blob),
    ] {
        zw.start_file(name, opts)?;
        zw.write_all(&blob)?;
    }
    zw.finish()?;
    Ok(())
}

/// Same as [`write_structure_v2_npz_compressed`] but returns bytes (tests).
pub fn write_structure_v2_npz_to_vec(s: &StructureV2Tables) -> Result<Vec<u8>> {
    let mut atoms_v = Vec::with_capacity(s.atoms.len() * ATOM_V2_AL);
    for a in &s.atoms {
        atoms_v.extend_from_slice(&pack_atom_aligned(a));
    }
    let atoms_blob = write_npy_1d(DESCR_ATOM, s.atoms.len(), &atoms_v)?;
    let mut bonds_v = Vec::with_capacity(s.bonds.len() * BOND_V2_AL);
    for b in &s.bonds {
        bonds_v.extend_from_slice(&pack_bond_aligned(b));
    }
    let bonds_blob = write_npy_1d(DESCR_BOND, s.bonds.len(), &bonds_v)?;
    let mut res_v = Vec::with_capacity(s.residues.len() * RESIDUE_AL);
    for res in &s.residues {
        res_v.extend_from_slice(&pack_residue_aligned(res));
    }
    let res_blob = write_npy_1d(DESCR_RESIDUE, s.residues.len(), &res_v)?;
    let mut ch_v = Vec::with_capacity(s.chains.len() * CHAIN_AL);
    for ch in &s.chains {
        ch_v.extend_from_slice(&pack_chain_aligned(ch));
    }
    let ch_blob = write_npy_1d(DESCR_CHAIN, s.chains.len(), &ch_v)?;
    let if_blob = write_npy_1d(DESCR_INTERFACE, 0, &[])?;
    let mask: Vec<u8> = s.chain_mask.iter().map(|&b| u8::from(b)).collect();
    let mask_blob = write_npy_1d(DESCR_MASK, s.chain_mask.len(), &mask)?;
    let mut coord_v = Vec::with_capacity(s.coords.len() * COORDS_AL);
    for c in &s.coords {
        for &x in c {
            coord_v.extend_from_slice(&x.to_le_bytes());
        }
    }
    let coord_blob = write_npy_1d(DESCR_COORDS, s.coords.len(), &coord_v)?;
    let ncoord = i32::try_from(s.coords.len()).unwrap_or(i32::MAX);
    let mut ens = [0u8; ENSEMBLE_AL];
    ens[0..4].copy_from_slice(&s.ensemble_atom_coord_idx.to_le_bytes());
    ens[4..8].copy_from_slice(&ncoord.to_le_bytes());
    let ens_blob = write_npy_1d(DESCR_ENSEMBLE, 1, &ens)?;

    let mut buf = Cursor::new(Vec::new());
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let mut zw = ZipWriter::new(&mut buf);
    for (name, blob) in [
        ("atoms.npy", atoms_blob),
        ("bonds.npy", bonds_blob),
        ("chains.npy", ch_blob),
        ("coords.npy", coord_blob),
        ("ensemble.npy", ens_blob),
        ("interfaces.npy", if_blob),
        ("mask.npy", mask_blob),
        ("residues.npy", res_blob),
    ] {
        zw.start_file(name, opts)?;
        zw.write_all(&blob)?;
    }
    zw.finish()?;
    Ok(buf.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::structure_v2_single_ala;
    use crate::tokenize::boltz2::tokenize_structure;

    #[test]
    fn roundtrip_structure_npz_matches_fixture() {
        let s = structure_v2_single_ala();
        let bytes = write_structure_v2_npz_to_vec(&s).unwrap();
        let back = read_structure_v2_npz_bytes(&bytes).unwrap();
        assert_eq!(back, s);
        let (t1, _) = tokenize_structure(&s, None);
        let (t2, _) = tokenize_structure(&back, None);
        assert_eq!(t1, t2);
    }

    /// NumPy **packed** structured dtypes (Boltz `types.py` lists, default `np.dtype` layout),
    /// optional `interfaces`, and **multi-row** `ensemble`. Regenerate:
    /// `python3 scripts/gen_structure_v2_numpy_golden.py`.
    #[test]
    fn golden_numpy_packed_structure_v2_matches_ala_fixture() {
        const BYTES: &[u8] = include_bytes!("../tests/fixtures/structure_v2_numpy_packed_ala.npz");
        let s = structure_v2_single_ala();
        let got = read_structure_v2_npz_bytes(BYTES).expect("golden npz");
        assert_eq!(got, s);
        let (t1, _) = tokenize_structure(&s, None);
        let (t2, _) = tokenize_structure(&got, None);
        assert_eq!(t1, t2);
    }
}
