//! Write [`Boltz2Model::with_options`] (64 / 32 / 1 pairformer block, no bond-type embedding)
//! to `tests/fixtures/boltz2_smoke/boltz2_smoke.safetensors` for pinned strict-load tests.

use std::path::Path;

use boltr_backend_tch::Boltz2Model;
use tch::Device;

fn main() {
    tch::maybe_init_cuda();
    let device = Device::Cpu;
    let m = Boltz2Model::with_options(device, 64, 32, Some(1));
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/boltz2_smoke/boltz2_smoke.safetensors");
    std::fs::create_dir_all(out.parent().unwrap()).expect("create fixture dir");
    m.var_store().save(&out).expect("VarStore::save");
    eprintln!(
        "Wrote {} ({} parameters)",
        out.display(),
        m.var_store().variables().len()
    );
}
