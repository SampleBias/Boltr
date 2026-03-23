//! Write `tests/fixtures/collate_golden/trunk_smoke_collate.safetensors` (no PyTorch).
//!
//! Run from repo root: `cargo run -p boltr-io --bin write_collate_golden`

use std::borrow::Cow;
use std::env;
use std::path::PathBuf;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use safetensors::tensor::{serialize_to_file, Dtype, SafeTensorError, View};

struct OwnedTensor {
    name: &'static str,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &OwnedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn f32_tensor(name: &'static str, shape: &[usize], rng: &mut StdRng) -> OwnedTensor {
    let n: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n * 4);
    for _ in 0..n {
        let v: f32 = rng.gen();
        data.extend_from_slice(&v.to_le_bytes());
    }
    OwnedTensor {
        name,
        dtype: Dtype::F32,
        shape: shape.to_vec(),
        data,
    }
}

fn f32_zeros(name: &'static str, shape: &[usize]) -> OwnedTensor {
    let n: usize = shape.iter().product();
    let data = vec![0u8; n * 4];
    OwnedTensor {
        name,
        dtype: Dtype::F32,
        shape: shape.to_vec(),
        data,
    }
}

fn i64_tensor(name: &'static str, shape: &[usize], fill: i64) -> OwnedTensor {
    let n: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n * 8);
    for _ in 0..n {
        data.extend_from_slice(&fill.to_le_bytes());
    }
    OwnedTensor {
        name,
        dtype: Dtype::I64,
        shape: shape.to_vec(),
        data,
    }
}

fn i64_ones(name: &'static str, shape: &[usize]) -> OwnedTensor {
    i64_tensor(name, shape, 1)
}

fn i64_zeros(name: &'static str, shape: &[usize]) -> OwnedTensor {
    i64_tensor(name, shape, 0)
}

fn main() -> Result<(), SafeTensorError> {
    let mut rng = StdRng::seed_from_u64(42);
    let b = 1usize;
    let n = 4usize;
    let s_msa = 8usize;
    let num_tokens = 33usize;
    let tdim = 1usize;
    let token_s = 384usize;

    let tensors = vec![
        f32_tensor("s_inputs", &[b, n, token_s], &mut rng),
        f32_zeros("token_pad_mask", &[b, n]),
        i64_zeros("msa", &[b, s_msa, n]),
        i64_zeros("msa_paired", &[b, s_msa, n]),
        i64_ones("msa_mask", &[b, s_msa, n]),
        i64_zeros("has_deletion", &[b, s_msa, n]),
        f32_zeros("deletion_value", &[b, s_msa, n]),
        f32_zeros("deletion_mean", &[b, n]),
        f32_tensor("profile", &[b, n, num_tokens], &mut rng),
        f32_zeros("template_restype", &[b, tdim, n, num_tokens]),
        f32_zeros("template_mask", &[b, tdim, n]),
    ];

    let out: PathBuf = env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("tests/fixtures/collate_golden/trunk_smoke_collate.safetensors");

    let iter = tensors
        .iter()
        .map(|t| (t.name.to_string(), t))
        .collect::<Vec<_>>();
    serialize_to_file(iter, &None, &out)?;
    eprintln!("Wrote {} ({} tensors)", out.display(), tensors.len());
    Ok(())
}
