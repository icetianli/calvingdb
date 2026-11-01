"""
Build the training sample index from the zarr stores produced by
02_process_observations.py. Each valid (glacier, reference date, horizon) sample is
saved as pre-resolved zarr indices in the train, val, and test split files.
"""

import json
import zarr
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from typing import Optional
from collections import deque

ZARR_DIR  = Path(
    "/path/to/calvdb/TrainingDataset_processed/zarr"
)
OUT_DIR   = Path(
    "/path/to/calvdb/TrainingDataset_processed/splits"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANCHOR_OFFSETS = [-365*4, -365*3, -365*2, -365, -30]
HORIZONS       = [365]
TOLERANCE      = 60
MIN_GAP        = 60
REFERENCE_MIN_GAP = 30

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20
SPLIT_SEED  = 42


def _backtrack_assign(
    obs: np.ndarray,
    candidates: list[list[int]],
    min_gap: int,
    slot: int,
    chosen: list[int],
) -> Optional[list[int]]:
    """
    Recursively assign one candidate observation per slot so that all chosen
    observations satisfy the minimum gap constraint.
    """
    if slot == len(candidates):
        return chosen

    for idx in candidates[slot]:
        day = obs[idx]
        if all(abs(day - obs[c]) >= min_gap for c in chosen):
            result = _backtrack_assign(
                obs, candidates, min_gap, slot + 1, chosen + [idx]
            )
            if result is not None:
                return result

    return None


def select_input_observations(
    obs_days: list[float],
    anchor_offsets: list[int],
    horizon: int,
    tolerance: int,
    min_gap: int,
) -> Optional[dict]:
    """
    Select one observation per anchor slot plus one target observation, subject to
    anchor-proximity and minimum-spacing constraints.
    """
    obs     = np.array(obs_days, dtype=float)
    anchors = sorted(anchor_offsets)

    candidates = []
    for anchor in anchors:
        lo   = anchor - tolerance
        hi   = min(anchor + tolerance, -1)
        mask = (obs >= lo) & (obs <= hi)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return None
        idxs_sorted = idxs[np.argsort(np.abs(obs[idxs] - anchor))]
        candidates.append(idxs_sorted.tolist())

    assignment = _backtrack_assign(obs, candidates, min_gap, slot=0, chosen=[])
    if assignment is None:
        return None

    target_lo   = horizon - tolerance
    target_hi   = horizon + tolerance
    target_mask = (obs >= target_lo) & (obs <= target_hi)
    target_idxs = np.where(target_mask)[0]
    if len(target_idxs) == 0:
        return None

    target_idx = target_idxs[int(np.argmin(np.abs(obs[target_idxs] - horizon)))]

    return {
        "input_indices": assignment,
        "input_days":    obs[assignment].tolist(),
        "target_index":  int(target_idx),
        "target_day":    float(obs[target_idx]),
        "day_gaps":      [float(obs[i] - a)
                          for i, a in zip(assignment, anchors)],
    }


def generate_samples_for_glacier(zarr_path: Path) -> list[dict]:
    """
    Open one glacier's zarr store and enumerate all valid training samples.
    """
    try:
        store = zarr.open_group(str(zarr_path), mode="r")
    except Exception as e:
        print(f"  WARNING: could not open {zarr_path}: {e}")
        return []

    glacier_id = store.attrs.get("glacier_id", zarr_path.stem)

    dates_raw = store["dates"][:]
    valid_sdt = store["valid_sdt"][:]

    dates_parsed = []
    for d in dates_raw:
        try:
            dates_parsed.append(date.fromisoformat(str(d)))
        except (ValueError, TypeError):
            dates_parsed.append(None)

    valid_zarr_indices = [
        i for i, (v, d) in enumerate(zip(valid_sdt, dates_parsed))
        if v and d is not None
    ]
    valid_dates = [dates_parsed[i] for i in valid_zarr_indices]

    n_valid = len(valid_zarr_indices)
    if n_valid < len(ANCHOR_OFFSETS) + 1:
        return []

    reference_candidates = []
    last_reference_date = None
    for ref_zarr_idx, ref_date in zip(valid_zarr_indices, valid_dates):
        if last_reference_date is not None:
            if (ref_date - last_reference_date).days < REFERENCE_MIN_GAP:
                continue
        reference_candidates.append((ref_zarr_idx, ref_date))
        last_reference_date = ref_date

    samples   = []
    seen      = set()

    for ref_pos, (ref_zarr_idx, ref_date) in enumerate(reference_candidates):
        rel_days = [
            (d - ref_date).days for d in valid_dates
        ]

        for horizon in HORIZONS:
            result = select_input_observations(
                obs_days=rel_days,
                anchor_offsets=ANCHOR_OFFSETS,
                horizon=horizon,
                tolerance=TOLERANCE,
                min_gap=MIN_GAP,
            )
            if result is None:
                continue

            input_zarr_indices = [
                valid_zarr_indices[i] for i in result["input_indices"]
            ]
            target_zarr_index  = valid_zarr_indices[result["target_index"]]

            fingerprint = (
                glacier_id,
                tuple(input_zarr_indices),
                target_zarr_index,
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

            samples.append({
                "glacier_id":          glacier_id,
                "zarr_path":           str(zarr_path),
                "reference_date":      ref_date.isoformat(),
                "horizon_days":        horizon,

                "input_zarr_indices":  input_zarr_indices,
                "target_zarr_index":   target_zarr_index,

                "input_days":          result["input_days"],
                "target_day":          result["target_day"],
                "day_gaps":            result["day_gaps"],
                "anchor_offsets":      ANCHOR_OFFSETS,
            })

    return samples


def main():
    zarr_paths  = sorted(ZARR_DIR.glob("*.zarr"))

    if not zarr_paths:
        print(f"ERROR: no zarr stores found under {ZARR_DIR}")
        return

    print(f"Found {len(zarr_paths)} zarr stores\n")
    print(f"Sampling parameters")
    print(f"  anchor_offsets : {ANCHOR_OFFSETS}")
    print(f"  horizons       : {HORIZONS} days")
    print(f"  tolerance      : ±{TOLERANCE} days")
    print(f"  min_gap        : {MIN_GAP} days\n")
    print(f"  ref_min_gap    : {REFERENCE_MIN_GAP} days\n")

    glacier_ids = [p.stem for p in zarr_paths]
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled_ids = glacier_ids.copy()
    rng.shuffle(shuffled_ids)

    n_glaciers = len(shuffled_ids)
    n_train = int(np.floor(n_glaciers * TRAIN_RATIO))
    n_val = int(np.floor(n_glaciers * VAL_RATIO))
    n_test = n_glaciers - n_train - n_val

    train_glacier_ids = set(shuffled_ids[:n_train])
    val_glacier_ids = set(shuffled_ids[n_train:n_train + n_val])
    test_glacier_ids = set(shuffled_ids[n_train + n_val:])

    print("Glacier split")
    print(f"  total glaciers : {n_glaciers}")
    print(f"  train glaciers : {len(train_glacier_ids)} ({TRAIN_RATIO:.0%})")
    print(f"  val glaciers   : {len(val_glacier_ids)} ({VAL_RATIO:.0%})")
    print(f"  test glaciers  : {len(test_glacier_ids)} ({TEST_RATIO:.0%})")
    print(f"  split seed     : {SPLIT_SEED}\n")

    all_samples   = []
    per_glacier   = []

    for zarr_path in zarr_paths:
        glacier_id = zarr_path.stem
        samples    = generate_samples_for_glacier(zarr_path)
        all_samples.extend(samples)
        per_glacier.append({
            "glacier_id": glacier_id,
            "n_samples":  len(samples),
        })
        print(f"  {glacier_id:40s}  {len(samples):6d} samples")

    print(f"\nTotal samples before split : {len(all_samples)}")

    train = [s for s in all_samples if s["glacier_id"] in train_glacier_ids]
    val   = [s for s in all_samples if s["glacier_id"] in val_glacier_ids]
    test  = [s for s in all_samples if s["glacier_id"] in test_glacier_ids]

    train_ids = {s["glacier_id"] for s in train}
    val_ids   = {s["glacier_id"] for s in val}
    test_ids  = {s["glacier_id"] for s in test}

    assert train_ids.isdisjoint(val_ids),  "Glacier leakage: train ∩ val"
    assert train_ids.isdisjoint(test_ids), "Glacier leakage: train ∩ test"
    assert val_ids.isdisjoint(test_ids),   "Glacier leakage: val ∩ test"

    json.dump(train, open(OUT_DIR / "train.json", "w"), indent=2)
    json.dump(val,   open(OUT_DIR / "val.json",   "w"), indent=2)
    json.dump(test,  open(OUT_DIR / "test.json",  "w"), indent=2)

    pd.DataFrame(per_glacier).to_parquet(
        OUT_DIR / "per_glacier_counts.parquet", index=False
    )

    params = {
        "anchor_offsets":  ANCHOR_OFFSETS,
        "horizons":        HORIZONS,
        "tolerance_days":  TOLERANCE,
        "min_gap_days":    MIN_GAP,
        "reference_min_gap_days": REFERENCE_MIN_GAP,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
            "seed": SPLIT_SEED,
        },
        "split": {
            "test_glaciers":  sorted(test_ids),
            "val_glaciers":   sorted(val_ids),
            "train_glaciers": sorted(train_ids),
        },
        "counts": {
            "train": len(train),
            "val":   len(val),
            "test":  len(test),
            "total": len(all_samples),
        },
    }
    json.dump(params, open(OUT_DIR / "sampling_params.json", "w"), indent=2)

    print(f"\n{'='*50}")
    print(f"SPLIT SUMMARY")
    print(f"{'='*50}")
    print(f"  Train : {len(train):7d} samples  ({len(train_ids)} glaciers)")
    print(f"  Val   : {len(val):7d} samples  ({len(val_ids)} glaciers)")
    print(f"  Test  : {len(test):7d} samples  ({len(test_ids)} glaciers)")
    print(f"  Total : {len(all_samples):7d} samples")
    print(f"{'='*50}")
    print(f"\nOutputs written to {OUT_DIR}")
    print(f"  train.json  val.json  test.json")
    print(f"  sampling_params.json")
    print(f"  per_glacier_counts.parquet")


if __name__ == "__main__":
    main()
