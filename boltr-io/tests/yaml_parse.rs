//! Comprehensive YAML parsing tests for the Boltz schema.
//!
//! Tests every aspect of [`BoltzInput`] deserialization:
//! - Entity types: protein, dna, rna, ligand (SMILES & CCD)
//! - Multi-chain entities (`id: [A, B]` format)
//! - Ligand type dispatch (SMILES vs CCD vs multi-CCD)
//! - Modifications, cyclic peptides, MSA paths
//! - Constraints: bond, pocket, contact, and mixed
//! - Templates: CIF and PDB with chain mapping
//! - Properties: affinity
//! - Version field
//! - Helper methods on BoltzInput
//! - Full schema integration

use std::path::PathBuf;

use boltr_io::config::*;
use boltr_io::parse_input_path;

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/yaml");
    p.push(name);
    p
}

fn parse_fixture(name: &str) -> BoltzInput {
    parse_input_path(&fixture(name)).unwrap_or_else(|e| panic!("failed to parse {name}: {e}"))
}

fn parse_yaml_str(yaml: &str) -> BoltzInput {
    boltr_io::parse_input_str(yaml).expect("parse inline YAML")
}

// ─── Version field ────────────────────────────────────────────────────────────

#[test]
fn version_field_parses() {
    let input = parse_fixture("version_field.yaml");
    assert_eq!(input.version, Some(1));
}

#[test]
fn version_field_defaults_to_none_when_absent() {
    let input = parse_fixture("minimal_protein_only.yaml");
    assert_eq!(input.version, None);
}

// ─── Protein entity ──────────────────────────────────────────────────────────

#[test]
fn minimal_protein_entity() {
    let input = parse_fixture("minimal_protein_only.yaml");
    assert_eq!(input.summary_chain_ids(), vec!["A"]);
    assert!(input.proteins().len() == 1);
    assert!(input.ligands().is_empty());
    assert!(input.dnas().is_empty());
    assert!(input.rnas().is_empty());
    assert!(input.constraints.is_none());
    assert!(input.templates.is_none());
    assert!(input.properties.is_none());
}

#[test]
fn protein_sequence_preserved() {
    let input = parse_fixture("minimal_protein_only.yaml");
    let protein = &input.proteins()[0];
    assert_eq!(protein.sequence, "ACDEFGHIKLMNPQRSTVWY");
}

#[test]
fn protein_msa_path() {
    let input = parse_fixture("protein_msa.yaml");
    let protein = &input.proteins()[0];
    assert_eq!(protein.msa.as_deref(), Some("./msa.a3m"));
}

#[test]
fn protein_msa_paths_helper() {
    let input = parse_fixture("protein_msa.yaml");
    let paths = input.protein_msa_paths();
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0], ("A".to_string(), "./msa.a3m".to_string()));
}

#[test]
fn protein_sequences_for_msa_helper() {
    // protein_msa.yaml has msa set, so it should be excluded from "needs MSA" list
    let input = parse_fixture("protein_msa.yaml");
    let needs_msa = input.protein_sequences_for_msa();
    assert!(needs_msa.is_empty());

    // minimal_protein_only.yaml has no msa, so it should need one
    let input2 = parse_fixture("minimal_protein_only.yaml");
    let needs_msa2 = input2.protein_sequences_for_msa();
    assert_eq!(needs_msa2.len(), 1);
    assert_eq!(needs_msa2[0].0, "A");
}

// ─── DNA entity ──────────────────────────────────────────────────────────────

#[test]
fn dna_entity_parses() {
    let input = parse_fixture("dna_entity.yaml");
    assert!(input.dnas().len() == 1);
    assert!(input.proteins().is_empty());
    let dna = &input.dnas()[0];
    assert_eq!(dna.sequence, "ATCGATCGATCG");
    assert_eq!(dna.id.to_vec(), vec!["C"]);
}

// ─── RNA entity ──────────────────────────────────────────────────────────────

#[test]
fn rna_entity_parses() {
    let input = parse_fixture("rna_entity.yaml");
    assert!(input.rnas().len() == 1);
    assert!(input.proteins().is_empty());
    let rna = &input.rnas()[0];
    assert_eq!(rna.sequence, "AUCGAUCGAUCG");
    assert_eq!(rna.id.to_vec(), vec!["D"]);
}

// ─── Multi-entity ────────────────────────────────────────────────────────────

#[test]
fn multi_entity_all_types() {
    let input = parse_fixture("multi_entity.yaml");
    assert_eq!(input.summary_chain_ids(), vec!["A", "B", "C", "D", "E"]);
    assert_eq!(input.proteins().len(), 2);
    assert_eq!(input.dnas().len(), 1);
    assert_eq!(input.rnas().len(), 1);
    assert_eq!(input.ligands().len(), 1);
}

// ─── Multi-chain entity ──────────────────────────────────────────────────────

#[test]
fn multi_chain_entity_id_list() {
    let input = parse_fixture("multi_chain_entity.yaml");
    assert_eq!(input.summary_chain_ids(), vec!["A", "B"]);

    let protein = &input.proteins()[0];
    match &protein.id {
        ChainIdSpec::Many(v) => assert_eq!(v, &vec!["A".to_string(), "B".to_string()]),
        other => panic!("expected ChainIdSpec::Many, got {other:?}"),
    }
    assert_eq!(protein.id.len(), 2);
    assert!(!protein.id.is_empty());
}

#[test]
fn single_chain_id_spec() {
    let input = parse_fixture("minimal_protein_only.yaml");
    let protein = &input.proteins()[0];
    match &protein.id {
        ChainIdSpec::Single(s) => assert_eq!(s, "A"),
        other => panic!("expected ChainIdSpec::Single, got {other:?}"),
    }
    assert_eq!(protein.id.len(), 1);
    assert!(!protein.id.is_empty());
}

// ─── Ligand types ────────────────────────────────────────────────────────────

#[test]
fn ligand_smiles_type() {
    let input = parse_fixture("ligand_smiles.yaml");
    let lig = &input.ligands()[0];
    assert!(lig.is_smiles());
    assert!(!lig.is_ccd());
    assert_eq!(lig.ligand_type(), LigandType::Smiles);
    assert_eq!(
        lig.smiles.as_deref(),
        Some("CC(=O)OC1=CC=CC=C1C(=O)O")
    );
    assert!(lig.ccd.is_none());
}

#[test]
fn ligand_ccd_single() {
    let input = parse_fixture("ligand_ccd_single.yaml");
    let lig = &input.ligands()[0];
    assert!(lig.is_ccd());
    assert!(!lig.is_smiles());
    assert_eq!(lig.ligand_type(), LigandType::Ccd);

    match lig.ccd.as_ref() {
        Some(LigandCcdCode::Single(s)) => assert_eq!(s, "HEM"),
        other => panic!("expected Single CCD code, got {other:?}"),
    }
    assert_eq!(lig.ccd.as_ref().unwrap().primary(), "HEM");
    assert_eq!(lig.ccd.as_ref().unwrap().to_vec(), vec!["HEM"]);
}

#[test]
fn ligand_ccd_multi() {
    let input = parse_fixture("ligand_ccd_multi.yaml");
    let lig = &input.ligands()[0];
    assert!(lig.is_ccd());

    match lig.ccd.as_ref() {
        Some(LigandCcdCode::Many(v)) => {
            assert_eq!(v, &vec!["HEM".to_string(), "HEC".to_string()]);
        }
        other => panic!("expected Many CCD codes, got {other:?}"),
    }
    assert_eq!(lig.ccd.as_ref().unwrap().primary(), "HEM");
    assert_eq!(
        lig.ccd.as_ref().unwrap().to_vec(),
        vec!["HEM".to_string(), "HEC".to_string()]
    );
}

#[test]
fn ligand_type_display() {
    assert_eq!(LigandType::Smiles.to_string(), "SMILES");
    assert_eq!(LigandType::Ccd.to_string(), "CCD");
    assert_eq!(LigandType::Unspecified.to_string(), "unspecified");
}

#[test]
fn ligand_unspecified_when_neither_smiles_nor_ccd() {
    let yaml = r#"
sequences:
  - ligand:
      id: Z
"#;
    let input = parse_yaml_str(yaml);
    let lig = &input.ligands()[0];
    assert_eq!(lig.ligand_type(), LigandType::Unspecified);
    assert!(!lig.is_smiles());
    assert!(!lig.is_ccd());
}

// ─── Modifications ───────────────────────────────────────────────────────────

#[test]
fn protein_modifications() {
    let input = parse_fixture("modifications.yaml");
    let protein = &input.proteins()[0];
    let mods = protein.modifications.as_ref().expect("modifications");
    assert_eq!(mods.len(), 2);
    assert_eq!(mods[0].position, 5);
    assert_eq!(mods[0].ccd, "SEP");
    assert_eq!(mods[1].position, 10);
    assert_eq!(mods[1].ccd, "TPO");
    assert!(input.has_modifications());
}

#[test]
fn no_modifications_when_absent() {
    let input = parse_fixture("minimal_protein_only.yaml");
    let protein = &input.proteins()[0];
    assert!(protein.modifications.is_none());
    assert!(!input.has_modifications());
}

// ─── Cyclic peptide ──────────────────────────────────────────────────────────

#[test]
fn cyclic_protein_flag() {
    let input = parse_fixture("cyclic_protein.yaml");
    let protein = &input.proteins()[0];
    assert_eq!(protein.cyclic, Some(true));
    assert!(input.has_cyclic());
}

#[test]
fn non_cyclic_protein() {
    let input = parse_fixture("minimal_protein_only.yaml");
    let protein = &input.proteins()[0];
    assert!(protein.cyclic.is_none() || protein.cyclic == Some(false));
    assert!(!input.has_cyclic());
}

// ─── Bond constraint ─────────────────────────────────────────────────────────

#[test]
fn bond_constraint_parses() {
    let input = parse_fixture("constraints_bond.yaml");
    let bonds = input.bond_constraints();
    assert_eq!(bonds.len(), 1);

    let bond = &bonds[0];
    match &bond.bond.atom1 {
        ConstraintAtomRef::List(v) => {
            assert_eq!(v.len(), 3);
            assert_eq!(v[0].as_str().unwrap(), "A");
            assert_eq!(v[1].as_i64().unwrap(), 1);
            assert_eq!(v[2].as_str().unwrap(), "N");
        }
    }
    match &bond.bond.atom2 {
        ConstraintAtomRef::List(v) => {
            assert_eq!(v.len(), 3);
            assert_eq!(v[0].as_str().unwrap(), "B");
            assert_eq!(v[1].as_i64().unwrap(), 1);
            assert_eq!(v[2].as_str().unwrap(), "C1");
        }
    }
}

#[test]
fn no_constraints_when_absent() {
    let input = parse_fixture("minimal_protein_only.yaml");
    assert!(input.bond_constraints().is_empty());
    assert!(input.pocket_constraints().is_empty());
    assert!(input.contact_constraints().is_empty());
}

// ─── Pocket constraint ───────────────────────────────────────────────────────

#[test]
fn pocket_constraint_parses() {
    let input = parse_fixture("constraints_pocket.yaml");
    let pockets = input.pocket_constraints();
    assert_eq!(pockets.len(), 1);

    let pocket = &pockets[0];
    assert_eq!(pocket.pocket.binder, "B");
    assert_eq!(pocket.pocket.contacts.len(), 3);
    assert_eq!(pocket.pocket.max_distance, 8.0);
    assert!(pocket.pocket.force);
}

#[test]
fn pocket_default_max_distance() {
    // Pocket with no max_distance should default to 6.0
    let yaml = r#"
sequences:
  - protein:
      id: A
      sequence: "ACDEFGHIKLMNPQRSTVWY"
  - ligand:
      id: B
      ccd: HEM
constraints:
  - pocket:
      binder: B
      contacts:
        - [A, 10]
"#;
    let input = parse_yaml_str(yaml);
    let pockets = input.pocket_constraints();
    assert_eq!(pockets[0].pocket.max_distance, 6.0);
    assert!(!pockets[0].pocket.force);
}

#[test]
fn pocket_contact_refs() {
    let input = parse_fixture("constraints_pocket.yaml");
    let pocket = &input.pocket_constraints()[0];
    for contact in &pocket.pocket.contacts {
        match contact {
            PocketContactRef::List(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[0].as_str().unwrap(), "A");
            }
        }
    }
}

// ─── Contact constraint ──────────────────────────────────────────────────────

#[test]
fn contact_constraint_parses() {
    let input = parse_fixture("constraints_contact.yaml");
    let contacts = input.contact_constraints();
    assert_eq!(contacts.len(), 1);

    let contact = &contacts[0];
    assert_eq!(contact.contact.max_distance, 10.0);
    assert!(!contact.contact.force);

    match &contact.contact.token1 {
        ConstraintTokenRef::List(v) => {
            assert_eq!(v[0].as_str().unwrap(), "A");
            assert_eq!(v[1].as_i64().unwrap(), 5);
        }
    }
    match &contact.contact.token2 {
        ConstraintTokenRef::List(v) => {
            assert_eq!(v[0].as_str().unwrap(), "B");
            assert_eq!(v[1].as_i64().unwrap(), 10);
        }
    }
}

#[test]
fn contact_default_max_distance() {
    let yaml = r#"
sequences:
  - protein:
      id: A
      sequence: "ACDEFGHIKLMNPQRSTVWY"
  - protein:
      id: B
      sequence: "ACDEFGHIKLMNPQRSTVWY"
constraints:
  - contact:
      token1: [A, 1]
      token2: [B, 2]
"#;
    let input = parse_yaml_str(yaml);
    let contacts = input.contact_constraints();
    assert_eq!(contacts[0].contact.max_distance, 6.0);
    assert!(!contacts[0].contact.force);
}

// ─── Mixed constraints ───────────────────────────────────────────────────────

#[test]
fn mixed_constraints_all_three_types() {
    let input = parse_fixture("constraints_mixed.yaml");
    assert_eq!(input.bond_constraints().len(), 1);
    assert_eq!(input.pocket_constraints().len(), 1);
    assert_eq!(input.contact_constraints().len(), 1);
}

// ─── Template CIF ────────────────────────���───────────────────────────────────

#[test]
fn template_cif_parses() {
    let input = parse_fixture("template_cif.yaml");
    let templates = input.template_entries();
    assert_eq!(templates.len(), 1);

    let tmpl = &templates[0];
    assert!(tmpl.is_cif());
    assert!(!tmpl.is_pdb());
    assert_eq!(tmpl.path(), Some("template.cif"));
    assert!(!tmpl.force);
    assert!(tmpl.threshold.is_none());

    match tmpl.chain_id.as_ref() {
        Some(TemplateChainId::Single(s)) => assert_eq!(s, "A"),
        other => panic!("expected Single chain id, got {other:?}"),
    }
}

// ─── Template PDB ────────────────────────────────────────────────────────────

#[test]
fn template_pdb_parses() {
    let input = parse_fixture("template_pdb.yaml");
    let templates = input.template_entries();
    assert_eq!(templates.len(), 1);

    let tmpl = &templates[0];
    assert!(tmpl.is_pdb());
    assert!(!tmpl.is_cif());
    assert_eq!(tmpl.path(), Some("template.pdb"));
    assert!(tmpl.force);
    assert_eq!(tmpl.threshold, Some(5.0));

    // Multi-chain template chain_id
    match tmpl.chain_id.as_ref() {
        Some(TemplateChainId::Many(v)) => {
            assert_eq!(v, &vec!["A".to_string(), "B".to_string()]);
        }
        other => panic!("expected Many chain ids, got {other:?}"),
    }

    // Multi-chain template_id
    match tmpl.template_id.as_ref() {
        Some(TemplateChainId::Many(v)) => {
            assert_eq!(v, &vec!["X".to_string(), "Y".to_string()]);
        }
        other => panic!("expected Many template ids, got {other:?}"),
    }
}

#[test]
fn template_chain_id_to_vec() {
    let single = TemplateChainId::Single("A".to_string());
    assert_eq!(single.to_vec(), vec!["A"]);
    let many = TemplateChainId::Many(vec!["A".to_string(), "B".to_string()]);
    assert_eq!(many.to_vec(), vec!["A", "B"]);
}

// ─── Properties / Affinity ───────────────────────────────────────────────────

#[test]
fn affinity_property_parses() {
    let input = parse_fixture("properties_affinity.yaml");
    assert_eq!(input.affinity_binder(), Some("B".to_string()));
}

#[test]
fn no_affinity_when_absent() {
    let input = parse_fixture("minimal_protein_only.yaml");
    assert_eq!(input.affinity_binder(), None);
}

// ─── Full schema integration ─────────────────────────────────────────────────

#[test]
fn full_schema_integration() {
    let input = parse_fixture("full_schema.yaml");

    // Version
    assert_eq!(input.version, Some(1));

    // Entities — 5 entries: protein(multi-chain), dna, rna, ligand(CCD), ligand(SMILES)
    assert_eq!(input.summary_chain_ids(), vec!["A", "B", "C", "D", "E", "F"]);
    assert_eq!(input.proteins().len(), 1); // one protein entity with 2 chains
    assert_eq!(input.dnas().len(), 1);
    assert_eq!(input.rnas().len(), 1);
    assert_eq!(input.ligands().len(), 2);

    // Protein details
    let protein = &input.proteins()[0];
    match &protein.id {
        ChainIdSpec::Many(v) => assert_eq!(v, &vec!["A".to_string(), "B".to_string()]),
        _ => panic!("expected multi-chain protein"),
    }
    assert_eq!(protein.msa.as_deref(), Some("./msa.a3m"));
    assert_eq!(protein.cyclic, Some(false));
    let mods = protein.modifications.as_ref().unwrap();
    assert_eq!(mods.len(), 1);
    assert_eq!(mods[0].position, 5);
    assert_eq!(mods[0].ccd, "SEP");

    // DNA
    let dna = &input.dnas()[0];
    assert_eq!(dna.id.to_vec(), vec!["C"]);
    assert_eq!(dna.sequence, "ATCGATCG");

    // RNA
    let rna = &input.rnas()[0];
    assert_eq!(rna.id.to_vec(), vec!["D"]);
    assert_eq!(rna.sequence, "AUCGAUCG");

    // Ligand CCD (multi-code)
    let lig_ccd = &input.ligands()[0];
    assert_eq!(lig_ccd.ligand_type(), LigandType::Ccd);
    match lig_ccd.ccd.as_ref() {
        Some(LigandCcdCode::Many(v)) => assert_eq!(v, &vec!["HEM".to_string(), "HEC".to_string()]),
        other => panic!("expected multi CCD, got {other:?}"),
    }

    // Ligand SMILES
    let lig_smiles = &input.ligands()[1];
    assert_eq!(lig_smiles.ligand_type(), LigandType::Smiles);
    assert_eq!(lig_smiles.smiles.as_deref(), Some("CC(=O)O"));

    // Constraints — all three types
    assert_eq!(input.bond_constraints().len(), 1);
    assert_eq!(input.pocket_constraints().len(), 1);
    assert_eq!(input.contact_constraints().len(), 1);

    // Bond detail
    let bond = &input.bond_constraints()[0];
    match &bond.bond.atom1 {
        ConstraintAtomRef::List(v) => {
            assert_eq!(v[0].as_str().unwrap(), "A");
            assert_eq!(v[2].as_str().unwrap(), "N");
        }
    }
    match &bond.bond.atom2 {
        ConstraintAtomRef::List(v) => {
            assert_eq!(v[0].as_str().unwrap(), "E");
            assert_eq!(v[2].as_str().unwrap(), "FE");
        }
    }

    // Pocket detail
    let pocket = &input.pocket_constraints()[0];
    assert_eq!(pocket.pocket.binder, "E");
    assert_eq!(pocket.pocket.contacts.len(), 2);
    assert_eq!(pocket.pocket.max_distance, 6.0);
    assert!(pocket.pocket.force);

    // Contact detail
    let contact = &input.contact_constraints()[0];
    match &contact.contact.token1 {
        ConstraintTokenRef::List(v) => {
            assert_eq!(v[0].as_str().unwrap(), "A");
            assert_eq!(v[1].as_i64().unwrap(), 5);
        }
    }
    match &contact.contact.token2 {
        ConstraintTokenRef::List(v) => {
            assert_eq!(v[0].as_str().unwrap(), "B");
            assert_eq!(v[1].as_i64().unwrap(), 15);
        }
    }
    assert_eq!(contact.contact.max_distance, 8.0);
    assert!(!contact.contact.force);

    // Templates — two entries
    let templates = input.template_entries();
    assert_eq!(templates.len(), 2);

    // Template 1: CIF
    assert!(templates[0].is_cif());
    assert_eq!(templates[0].path(), Some("template.cif"));
    assert!(!templates[0].force);

    // Template 2: PDB
    assert!(templates[1].is_pdb());
    assert_eq!(templates[1].path(), Some("template.pdb"));
    assert!(templates[1].force);
    assert_eq!(templates[1].threshold, Some(5.0));
    match templates[1].chain_id.as_ref() {
        Some(TemplateChainId::Many(v)) => {
            assert_eq!(v, &vec!["A".to_string(), "B".to_string()]);
        }
        other => panic!("expected Many chain ids, got {other:?}"),
    }
    match templates[1].template_id.as_ref() {
        Some(TemplateChainId::Single(s)) => assert_eq!(s, "X"),
        other => panic!("expected Single template id, got {other:?}"),
    }

    // Properties
    assert_eq!(input.affinity_binder(), Some("E".to_string()));

    // Helper flags
    assert!(input.has_modifications());
    assert!(!input.has_cyclic()); // cyclic is false, not true
}

// ─── Serialization round-trip ────────────────────────────────────────────────

#[test]
fn roundtrip_serialize_deserialize() {
    let input = parse_fixture("full_schema.yaml");
    let yaml_str = serde_yaml::to_string(&input).expect("serialize");
    let back: BoltzInput = serde_yaml::from_str(&yaml_str).expect("deserialize");
    assert_eq!(back.summary_chain_ids(), input.summary_chain_ids());
    assert_eq!(back.version, input.version);
    assert_eq!(back.proteins().len(), input.proteins().len());
    assert_eq!(back.ligands().len(), input.ligands().len());
    assert_eq!(back.bond_constraints().len(), input.bond_constraints().len());
    assert_eq!(back.template_entries().len(), input.template_entries().len());
    assert_eq!(back.affinity_binder(), input.affinity_binder());
}

// ─── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn empty_constraints_list() {
    let yaml = r#"
sequences:
  - protein:
      id: A
      sequence: "ACDEFGHIKLMNPQRSTVWY"
constraints: []
"#;
    let input = parse_yaml_str(yaml);
    assert!(input.constraints.is_some());
    assert!(input.bond_constraints().is_empty());
    assert!(input.pocket_constraints().is_empty());
    assert!(input.contact_constraints().is_empty());
}

#[test]
fn empty_templates_list() {
    let yaml = r#"
sequences:
  - protein:
      id: A
      sequence: "ACDEFGHIKLMNPQRSTVWY"
templates: []
"#;
    let input = parse_yaml_str(yaml);
    assert!(input.templates.is_some());
    assert!(input.template_entries().is_empty());
}

#[test]
fn empty_properties_list() {
    let yaml = r#"
sequences:
  - protein:
      id: A
      sequence: "ACDEFGHIKLMNPQRSTVWY"
properties: []
"#;
    let input = parse_yaml_str(yaml);
    assert!(input.properties.is_some());
    assert_eq!(input.affinity_binder(), None);
}

#[test]
fn multiple_proteins_each_with_different_options() {
    let yaml = r#"
sequences:
  - protein:
      id: A
      sequence: "ACDEFGHIKLMNPQRSTVWY"
      msa: ./msa_a.a3m
  - protein:
      id: B
      sequence: "ACDEFGHIKLMNPQRSTVWY"
      cyclic: true
      modifications:
        - position: 1
          ccd: MSE
  - protein:
      id: C
      sequence: "ACDEFGHIKLMNPQRSTVWY"
"#;
    let input = parse_yaml_str(yaml);
    assert_eq!(input.proteins().len(), 3);
    assert_eq!(input.summary_chain_ids(), vec!["A", "B", "C"]);

    // First protein has MSA
    assert_eq!(input.proteins()[0].msa.as_deref(), Some("./msa_a.a3m"));

    // Second protein is cyclic with modifications
    assert_eq!(input.proteins()[1].cyclic, Some(true));
    let mods = input.proteins()[1].modifications.as_ref().unwrap();
    assert_eq!(mods[0].position, 1);
    assert_eq!(mods[0].ccd, "MSE");

    // Third protein is plain
    assert!(input.proteins()[2].msa.is_none());
    assert!(input.proteins()[2].cyclic.is_none());
    assert!(input.proteins()[2].modifications.is_none());
}

#[test]
fn ligand_smiles_and_ccd_exclusive() {
    // When both are provided (invalid per schema), SMILES should take precedence
    // in the enum dispatch since smiles is listed first
    let yaml = r#"
sequences:
  - ligand:
      id: Z
      smiles: "CCO"
      ccd: HEM
"#;
    let input = parse_yaml_str(yaml);
    let lig = &input.ligands()[0];
    // Both are present; ligand_type dispatches based on first match
    assert!(lig.smiles.is_some());
    assert!(lig.ccd.is_some());
}

#[test]
fn chain_id_spec_edge_cases() {
    // Empty list (technically valid YAML)
    let single = ChainIdSpec::Single("A".to_string());
    assert!(!single.is_empty());
    assert_eq!(single.len(), 1);

    let many = ChainIdSpec::Many(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    assert!(!many.is_empty());
    assert_eq!(many.len(), 3);
    assert_eq!(many.to_vec(), vec!["A", "B", "C"]);

    let empty = ChainIdSpec::Many(vec![]);
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
}

#[test]
fn ligand_ccd_code_edge_cases() {
    let single = LigandCcdCode::Single("ATP".to_string());
    assert_eq!(single.primary(), "ATP");
    assert_eq!(single.to_vec(), vec!["ATP"]);

    let empty_many = LigandCcdCode::Many(vec![]);
    assert_eq!(empty_many.primary(), "");
    assert!(empty_many.to_vec().is_empty());
}

// ─── Existing doc-example test preserved ──────────────────────────────────────

#[test]
fn boltz_input_deserializes_doc_example_yaml() {
    let yaml = r#"
sequences:
  - protein:
      id: A
      sequence: "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSQ"
      msa: ./msa.a3m
  - ligand:
      id: B
      ccd: HEM
constraints:
  - bond:
      atom1: [A, 1, N]
      atom2: [B, 1, C1]
templates:
  - cif: template.cif
    chain_id: A
"#;
    let input = parse_yaml_str(yaml);
    assert_eq!(input.summary_chain_ids(), vec!["A", "B"]);
}

#[test]
fn yaml_roundtrip_minimal_protein_only() {
    let input = parse_fixture("minimal_protein_only.yaml");
    let yaml = serde_yaml::to_string(&input).expect("serialize BoltzInput");
    let parsed: BoltzInput = serde_yaml::from_str(&yaml).expect("roundtrip parse");
    assert_eq!(parsed.version, input.version);
    assert_eq!(parsed.summary_chain_ids(), input.summary_chain_ids());
}

// ─── Existing fixture test preserved ──────────────────────────────────────────

#[test]
fn parse_minimal_protein_fixture() {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/minimal_protein.yaml");
    let input = parse_input_path(&p).expect("parse fixture");
    assert_eq!(input.summary_chain_ids(), vec!["A".to_string()]);
}
