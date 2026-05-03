use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct QcThresholds {
    pub peptide_cn_min: f32,
    pub peptide_cn_max: f32,
    pub ca_ca_min: f32,
    pub ca_ca_max: f32,
    pub n_ca_min: f32,
    pub n_ca_max: f32,
    pub ca_c_min: f32,
    pub ca_c_max: f32,
    pub c_o_min: f32,
    pub c_o_max: f32,
    pub hard_chain_break_cn: f32,
    pub hard_overlap: f32,
    /// Fail when `abs(omega)` is smaller than this value; trans peptide bonds sit near 180 deg.
    pub omega_abs_min_degrees: f32,
}

impl Default for QcThresholds {
    fn default() -> Self {
        Self {
            peptide_cn_min: 1.15,
            peptide_cn_max: 1.55,
            ca_ca_min: 2.9,
            ca_ca_max: 4.5,
            n_ca_min: 1.25,
            n_ca_max: 1.70,
            ca_c_min: 1.30,
            ca_c_max: 1.80,
            c_o_min: 1.05,
            c_o_max: 1.45,
            hard_chain_break_cn: 2.0,
            hard_overlap: 1.0,
            omega_abs_min_degrees: 90.0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MissingBackboneAtom {
    pub chain: String,
    pub residue_index: i32,
    pub residue_name: String,
    pub atom: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackboneBondDistance {
    pub chain: String,
    pub residue_index: i32,
    pub residue_name: String,
    pub bond: String,
    pub distance_angstrom: f32,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PeptideBondDistance {
    pub chain: String,
    pub from_residue_index: i32,
    pub to_residue_index: i32,
    pub distance_angstrom: f32,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChainBreak {
    pub chain: String,
    pub from_residue_index: i32,
    pub to_residue_index: i32,
    pub c_to_n_distance_angstrom: Option<f32>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StericClash {
    pub atom_1_index: usize,
    pub atom_2_index: usize,
    pub atom_1: String,
    pub atom_2: String,
    pub distance_angstrom: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TorsionMeasure {
    pub chain: String,
    pub from_residue_index: i32,
    pub to_residue_index: i32,
    pub omega_degrees: f32,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct QcReport {
    pub model_filename: String,
    pub chain_count: usize,
    pub residue_count: usize,
    pub missing_backbone_atoms: Vec<MissingBackboneAtom>,
    pub backbone_bond_distances: Vec<BackboneBondDistance>,
    pub peptide_bond_distances: Vec<PeptideBondDistance>,
    pub ca_ca_distances: Vec<PeptideBondDistance>,
    pub detected_chain_breaks: Vec<ChainBreak>,
    pub steric_clashes: Vec<StericClash>,
    pub omega_torsions: Vec<TorsionMeasure>,
    pub radius_of_gyration: Option<f32>,
    pub passed: bool,
    pub fail_reasons: Vec<String>,
    pub relaxation_attempted: bool,
    pub relaxation_fixed: bool,
}

pub fn render_qc_text(report: &QcReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("Model: {}\n", report.model_filename));
    out.push_str(&format!(
        "Status: {}\n",
        if report.passed { "PASS" } else { "FAIL" }
    ));
    out.push_str(&format!("Chains: {}\n", report.chain_count));
    out.push_str(&format!("Residues: {}\n", report.residue_count));
    if let Some(rg) = report.radius_of_gyration {
        out.push_str(&format!("Radius of gyration: {rg:.3} A\n"));
    } else {
        out.push_str("Radius of gyration: n/a\n");
    }
    out.push_str(&format!(
        "Relaxation attempted: {}\n",
        report.relaxation_attempted
    ));
    out.push_str(&format!("Relaxation fixed: {}\n", report.relaxation_fixed));
    if !report.fail_reasons.is_empty() {
        out.push_str("\nFail reasons:\n");
        for r in &report.fail_reasons {
            out.push_str(&format!("- {r}\n"));
        }
    }
    out.push_str(&format!(
        "\nMissing backbone atoms: {}\n",
        report.missing_backbone_atoms.len()
    ));
    for m in &report.missing_backbone_atoms {
        out.push_str(&format!(
            "- chain {} residue {} {} missing {}\n",
            m.chain, m.residue_index, m.residue_name, m.atom
        ));
    }
    out.push_str(&format!(
        "\nChain breaks: {}\n",
        report.detected_chain_breaks.len()
    ));
    for b in &report.detected_chain_breaks {
        let d = b
            .c_to_n_distance_angstrom
            .map(|v| format!("{v:.3} A"))
            .unwrap_or_else(|| "n/a".to_string());
        out.push_str(&format!(
            "- chain {} {} -> {}: {} ({})\n",
            b.chain, b.from_residue_index, b.to_residue_index, d, b.reason
        ));
    }
    out.push_str(&format!(
        "\nSteric clashes: {}\n",
        report.steric_clashes.len()
    ));
    for c in report.steric_clashes.iter().take(50) {
        out.push_str(&format!(
            "- atom {} {} vs atom {} {}: {:.3} A\n",
            c.atom_1_index, c.atom_1, c.atom_2_index, c.atom_2, c.distance_angstrom
        ));
    }
    if report.steric_clashes.len() > 50 {
        out.push_str(&format!(
            "- ... {} more clashes omitted from text report\n",
            report.steric_clashes.len() - 50
        ));
    }
    out
}
