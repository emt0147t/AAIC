"""Dataset Mixture Audit and Provenance Inspector CLI.

Usage:
    python scripts/audit_dataset.py --dataset vivos
    python scripts/audit_dataset.py --all
"""

import argparse
import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.registry import DATASET_REGISTRY, list_all_provenance


def main():
    parser = argparse.ArgumentParser(description="Audit Vietnamese Speech Datasets")
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset key to audit")
    parser.add_argument("--all", action="store_true", help="Audit all candidate datasets in registry")
    args = parser.parse_args()

    print("=" * 70)
    print("VIETNAMESE ASR DATASET AUDIT & PROVENANCE REGISTRY")
    print("=" * 70)

    items = list_all_provenance() if (args.all or not args.dataset) else [DATASET_REGISTRY.get(args.dataset.lower())]

    for prov in items:
        if not prov:
            print(f"Dataset '{args.dataset}' not recognized!")
            continue

        print(f"\n[{prov.name}] ({prov.dataset_key})")
        print(f"  HF Repo:         {prov.hf_repo_id}")
        print(f"  Pinned Revision: {prov.pinned_revision}")
        print(f"  Quality Tier:    {prov.quality_tier.value.upper()}")
        print(f"  Domain:          {prov.domain.value}")
        print(f"  Origin Team:     {prov.provenance_origin}")
        print(f"  License:         {prov.license}")
        print(f"  Commercial OK:   {prov.commercial_use_allowed}")
        print(f"  Redistribution:  {prov.redistribution_allowed}")
        print(f"  Train Weight:    {prov.recommended_train_weight:.2f}")
        print(f"  Transcript Col:  {prov.transcript_column}")
        print(f"  Speaker ID Col:  {prov.speaker_id_column}")
        print(f"  Description:     {prov.description}")
        if prov.notes:
            print(f"  Notes:           {prov.notes}")

    print("\n" + "=" * 70)
    print("Audit completed.")


if __name__ == "__main__":
    main()
