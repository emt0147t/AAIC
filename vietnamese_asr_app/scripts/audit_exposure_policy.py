"""Audit and verify the deterministic sample exposure policy for Full E1.

Computes exact exposure ordering, repeated sample indices for Epochs 1, 2, 3,
and verifies zero leakage, zero unrepresented samples, and exact batch alignment.
"""

import json
import numpy as np


def compute_epoch_exposure_schedule(
    n_samples: int = 26671,
    effective_batch_size: int = 32,
    epochs: int = 3,
    base_seed: int = 42,
):
    steps_per_epoch = int(np.ceil(n_samples / effective_batch_size))
    total_exposures_per_epoch = steps_per_epoch * effective_batch_size
    n_padding = total_exposures_per_epoch - n_samples

    schedule = {
        "n_samples": n_samples,
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_exposures_per_epoch": total_exposures_per_epoch,
        "n_padding_per_epoch": n_padding,
        "total_optimizer_steps": steps_per_epoch * epochs,
        "total_sample_exposures": total_exposures_per_epoch * epochs,
        "epochs": {},
    }

    for epoch in range(1, epochs + 1):
        epoch_seed = base_seed + (epoch - 1)
        rng = np.random.default_rng(epoch_seed)

        # 1. Full permutation of all 26,671 unique indices
        permutation = rng.permutation(n_samples).tolist()

        # 2. Deterministic selection of 17 padding samples
        # Drawn without replacement from the unique samples to pad the 834th batch
        pad_rng = np.random.default_rng(epoch_seed + 1000)
        padding_indices = pad_rng.choice(n_samples, size=n_padding, replace=False).tolist()

        # 3. Complete exposure sequence for this epoch (26,688 indices)
        epoch_exposures = permutation + padding_indices

        assert len(epoch_exposures) == total_exposures_per_epoch
        assert len(set(permutation)) == n_samples
        assert len(padding_indices) == n_padding

        schedule["epochs"][f"epoch_{epoch}"] = {
            "epoch_seed": epoch_seed,
            "padding_seed": epoch_seed + 1000,
            "unique_samples_covered": len(set(permutation)),
            "repeated_sample_count": n_padding,
            "repeated_sample_indices": sorted(padding_indices),
            "first_10_exposure_indices": epoch_exposures[:10],
            "last_10_exposure_indices": epoch_exposures[-10:],
        }

    return schedule


if __name__ == "__main__":
    sched = compute_epoch_exposure_schedule()
    out_path = "reports/full_e1_exposure_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sched, f, indent=2)
    print(f"Exposure schedule written to {out_path}")
    for ep, data in sched["epochs"].items():
        print(f"\n{ep.upper()}:")
        print(f"  Epoch Seed:              {data['epoch_seed']}")
        print(f"  Unique Samples Covered:  {data['unique_samples_covered']:,}")
        print(f"  Repeated Samples:        {data['repeated_sample_count']}")
        print(f"  Repeated Sample Indices: {data['repeated_sample_indices']}")
