#!/usr/bin/env python3
"""
Dump full post-collate batch from Boltz2InferenceDataModule for Rust parity testing.

This script runs Python Boltz inference and saves the fully collated batch
from Boltz2InferenceDataModule.collate() as a safetensors file.

Usage:
    python3 scripts/dump_full_collate_golden.py \
        --manifest tests/fixtures/collate_golden/manifest.json \
        --target-dir tests/fixtures/collate_golden/ \
        --msa-dir tests/fixtures/collate_golden/ \
        --output tests/fixtures/collate_golden/full_collate_golden.safetensors
"""

import argparse
import sys
from pathlib import Path

# Add boltz to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "boltz-reference" / "src"))

try:
    import numpy as np
    import torch
    from safetensors.numpy import save_file
    from boltz.data.module.inferencev2 import (
        Boltz2InferenceDataModule,
        load_input,
    )
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.mol import load_canonicals
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data import types
    from boltz.data.types import Manifest
except ImportError as e:
    print(f"Error importing Boltz modules: {e}")
    print("Make sure boltz-reference/src is in PYTHONPATH")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump full collate batch from Boltz2InferenceDataModule"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest.json file",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Path to target directory with preprocessed structures",
    )
    parser.add_argument(
        "--msa-dir",
        type=Path,
        required=True,
        help="Path to MSA directory",
    )
    parser.add_argument(
        "--mol-dir",
        type=Path,
        required=False,
        help="Path to molecules directory (for canonicals)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output safetensors file",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=1,
        help="Number of examples to collate (default: 1)",
    )
    parser.add_argument(
        "--record-ids",
        type=str,
        nargs='+',
        help="Specific record IDs to process (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading manifest from {args.manifest}")
    manifest = Manifest.load(args.manifest)

    # Filter records if specific IDs provided
    if args.record_ids:
        manifest = types.Manifest(
            records=[r for r in manifest.records if r.id in args.record_ids]
        )
        print(f"Processing {len(manifest.records)} specified records: {args.record_ids}")

    # Get mol dir (default to target-dir if not specified)
    mol_dir = args.mol_dir if args.mol_dir else args.target_dir

    print(f"Manifest contains {len(manifest.records)} records")
    print(f"Processing up to {args.num_examples} examples")

    # Create data module
    data_module = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=args.target_dir,
        msa_dir=args.msa_dir,
        mol_dir=mol_dir,
        num_workers=0,  # Use single-process for deterministic output
        constraints_dir=None,  # Not needed for basic golden
        template_dir=None,  # Not needed for basic golden
        extra_mols_dir=None,
        override_method=None,
        affinity=False,
    )

    # Get dataloader
    dataloader = data_module.predict_dataloader()

    # Process examples
    examples_to_process = min(args.num_examples, len(dataloader.dataset))
    print(f"Processing {examples_to_process} examples...")

    collated = None
    for i in range(examples_to_process):
        print(f"  Processing example {i+1}/{examples_to_process}: {dataloader.dataset[i].id}")
        batch = dataloader.dataset[i]
        if collated is None:
            # First example - initialize collated dict
            collated = {k: v.unsqueeze(0) for k, v in batch.items()}
        else:
            # Subsequent examples - collate with existing
            collated = dataloader.collate_fn([collated, batch])

    if collated is None:
        print("Error: No examples processed!")
        sys.exit(1)

    # Remove batch dimension for single-example collation (to match expected format)
    # This is the format that collate() produces: dict[str, Tensor]
    # where each tensor already has batch dim prepended
    # But for multi-example collation, collate() adds the batch dim
    if args.num_examples == 1:
        # For single example, we need to remove the extra batch dim we added
        collated = {k: v.squeeze(0) for k, v in collated.items()}

    print(f"\nCollated batch contains {len(collated)} keys")
    print("Keys and shapes:")
    for key in sorted(collated.keys()):
        shape = collated[key].shape
        dtype = collated[key].dtype
        print(f"  {key:30s} {str(shape):25s} {str(dtype):10s}")

    # Save to safetensors
    print(f"\nSaving to {args.output}")
    save_file(collated, args.output)

    print("\nSuccessfully saved full collate golden!")
    print(f"Output file: {args.output}")
    print(f"Total keys: {len(collated)}")

    # Show excluded keys if any
    excluded = [
        "all_coords",
        "all_resolved_mask",
        "crop_to_all_atom_map",
        "chain_symmetries",
        "amino_acids_symmetries",
        "ligand_symmetries",
        "record",
        "affinity_mw",
    ]
    excluded_present = [k for k in excluded if k in collated]
    if excluded_present:
        print(f"\nNote: Excluded keys present (not stacked): {excluded_present}")


if __name__ == "__main__":
    main()
