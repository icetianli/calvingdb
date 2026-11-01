"""
Compute per-modality normalisation statistics from the training split only.
Run it after 03_build_index.py and before training; results are saved to
splits/normalisation_stats.json.
"""

import json
import argparse
import numpy as np
import zarr
from pathlib import Path
from typing import Optional

SPLITS_DIR = Path(
    "/path/to/calvdb/TrainingDataset_processed/splits"
)
TRAIN_INDEX_PATH = SPLITS_DIR / "train.json"
OUT_PATH         = SPLITS_DIR / "normalisation_stats.json"

N_IMG_CHANNELS = 5
D_GEOPHYS      = 6
D_CLIMATE      = 2

_stores: dict[str, zarr.Group] = {}

def get_store(zarr_path: str) -> zarr.Group:
    if zarr_path not in _stores:
        _stores[zarr_path] = zarr.open_group(zarr_path, mode="r")
    return _stores[zarr_path]


class WelfordAccumulator:
    """
    Online mean and variance via Welford's algorithm.
    """

    def __init__(self):
        self.n     = 0
        self.mean  = 0.0
        self.M2    = 0.0

    def update(self, values: np.ndarray) -> None:
        """Add a batch of values, flattening them before processing."""
        flat = np.asarray(values, dtype=np.float64).ravel()
        for x in flat:
            if not np.isfinite(x):
                continue
            self.n    += 1
            delta      = x - self.mean
            self.mean += delta / self.n
            delta2     = x - self.mean
            self.M2   += delta * delta2

    def result(self) -> tuple[float, float]:
        """Return (mean, std), or (0.0, 1.0) if no data has been seen."""
        if self.n < 2:
            return float(self.mean), 1.0
        variance = self.M2 / (self.n - 1)
        return float(self.mean), float(np.sqrt(variance))


class PerChannelWelfordAccumulator:
    """
    Independent Welford accumulator per channel.
    """

    def __init__(self, n_channels: int):
        self.accumulators = [WelfordAccumulator() for _ in range(n_channels)]
        self.n_channels   = n_channels

    def update(self, arr: np.ndarray) -> None:
        """
        Update each channel's statistics from a (C,), (K, C), (C, H, W) or (K, C, H, W)
        array.
        """
        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 1:
            for c in range(self.n_channels):
                self.accumulators[c].update(np.array([arr[c]]))
        else:
            channel_axis = 0 if arr.ndim == 3 else 1
            for c in range(self.n_channels):
                if arr.ndim == 2:
                    self.accumulators[c].update(arr[:, c])
                elif arr.ndim == 3:
                    self.accumulators[c].update(arr[c])
                elif arr.ndim == 4:
                    self.accumulators[c].update(arr[:, c])

    def result(self) -> tuple[list[float], list[float]]:
        """Return ([means], [stds]) across channels."""
        means, stds = [], []
        for acc in self.accumulators:
            m, s = acc.result()
            means.append(m)
            stds.append(s)
        return means, stds


def compute_stats(
    train_index_path: Path,
    out_path:         Path,
    n_samples:        int  = 2000,
    seed:             int  = 0,
    verbose:          bool = True,
) -> dict:
    """
    Compute per-modality normalisation statistics from a random subset of the training
    split.
    """
    with open(train_index_path) as f:
        all_samples = json.load(f)

    n_total  = len(all_samples)
    n_use    = min(n_samples, n_total)
    rng      = np.random.default_rng(seed)
    chosen   = rng.choice(n_total, size=n_use, replace=False)

    if verbose:
        print(f"Computing normalisation statistics")
        print(f"  Training samples total : {n_total}")
        print(f"  Samples used           : {n_use}")
        print(f"  Random seed            : {seed}\n")

    sdt_acc     = WelfordAccumulator()
    disp_acc    = WelfordAccumulator()
    imagery_acc = PerChannelWelfordAccumulator(N_IMG_CHANNELS)
    geophys_acc = PerChannelWelfordAccumulator(D_GEOPHYS)
    climate_acc = PerChannelWelfordAccumulator(D_CLIMATE)

    n_sdt_seen = n_img_seen = n_geo_seen = n_clm_seen = 0

    for progress_i, sample_idx in enumerate(chosen):
        s     = all_samples[int(sample_idx)]
        store = get_store(s["zarr_path"])
        ii    = s["input_zarr_indices"]
        ti    = s["target_zarr_index"]

        try:
            sdt_inputs = store["sdt"][ii].astype(np.float32)
            sdt_target = store["sdt"][[ti]].astype(np.float32)
            sdt_all    = np.concatenate([sdt_inputs, sdt_target], axis=0)

            valid_sdt_flags = store["valid_sdt"][:]
            for k, zarr_idx in enumerate(ii):
                if valid_sdt_flags[zarr_idx]:
                    sdt_acc.update(sdt_inputs[k])
            if valid_sdt_flags[ti]:
                sdt_acc.update(sdt_target[0])

            n_sdt_seen += 1

            li = ii[-1]
            if valid_sdt_flags[ti] and valid_sdt_flags[li]:
                trace_flags = store["valid_trace"][:]
                if trace_flags[ti]:
                    trace_px = store["trace"][ti].astype(bool)
                    if trace_px.any():
                        disp = (store["sdt"][ti].astype(np.float32)
                                - store["sdt"][li].astype(np.float32))[trace_px]
                        disp_acc.update(disp[np.isfinite(disp)])

        except Exception as e:
            if verbose:
                print(f"  WARNING: SDT read failed for sample {sample_idx}: {e}")

        try:
            valid_img_flags = store["valid_img"][:]
            img_stack = store["imagery"][ii].astype(np.float32)
            for k, zarr_idx in enumerate(ii):
                if valid_img_flags[zarr_idx]:
                    imagery_acc.update(img_stack[k])
            n_img_seen += 1
        except Exception as e:
            if verbose:
                print(f"  WARNING: imagery read failed for sample {sample_idx}: {e}")

        try:
            geo_stack = store["geophys"][ii].astype(np.float32)
            geophys_acc.update(geo_stack)
            n_geo_seen += 1
        except Exception as e:
            if verbose:
                print(f"  WARNING: geophys read failed for sample {sample_idx}: {e}")

        try:
            valid_clm_flags = store["valid_clm"][:]
            clm_stack = store["climate"][ii].astype(np.float32)
            for k, zarr_idx in enumerate(ii):
                if valid_clm_flags[zarr_idx]:
                    climate_acc.update(clm_stack[k])
            n_clm_seen += 1
        except Exception as e:
            if verbose:
                print(f"  WARNING: climate read failed for sample {sample_idx}: {e}")

        if verbose and (progress_i + 1) % 200 == 0:
            print(f"  {progress_i + 1:5d} / {n_use}")

    sdt_mean, sdt_std        = sdt_acc.result()
    _,        disp_std_val   = disp_acc.result()
    img_means, img_stds      = imagery_acc.result()
    geo_means, geo_stds      = geophys_acc.result()
    clm_means, clm_stds      = climate_acc.result()

    stats = {
        "sdt": {
            "mean": sdt_mean,
            "std":  sdt_std,
        },
        "displacement": {
            "std": disp_std_val,
        },
        "imagery": {
            "mean": img_means,
            "std":  img_stds,
        },
        "geophys": {
            "mean": geo_means,
            "std":  geo_stds,
        },
        "climate": {
            "mean": clm_means,
            "std":  clm_stds,
        },
        "meta": {
            "n_samples_requested": n_samples,
            "n_samples_used":      n_use,
            "n_total_train":       n_total,
            "seed":                seed,
            "n_sdt_seen":          n_sdt_seen,
            "n_img_seen":          n_img_seen,
            "n_geo_seen":          n_geo_seen,
            "n_clm_seen":          n_clm_seen,
            "img_channel_order":   ["blue","green","red","nir","swir1"],
            "geophys_channel_order": [
                "bed_elevation","surface_elevation","thickness",
                "velocity","strain_rate","ice_fjord_mask"
            ],
            "climate_channel_order": ["mar_runoff","ocean_temp"],
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    if verbose:
        print(f"\n{'='*55}")
        print(f"NORMALISATION STATISTICS")
        print(f"{'='*55}")

        print(f"\nSDT (metres, clipped ±2000)")
        print(f"  mean : {sdt_mean:10.3f}")
        print(f"  std  : {sdt_std:10.3f}")

        print(f"\nDisplacement SDT(t+365)−SDT(t−30) at trace pixels")
        print(f"  std  : {disp_std_val:10.3f}  (scale = sdt_std/disp_std = {sdt_std/max(disp_std_val,1e-6):.2f}×)")

        img_names = ["blue","green","red","nir","swir1"]
        print(f"\nImagery (surface reflectance, 5 channels)")
        for name, m, s in zip(img_names, img_means, img_stds):
            print(f"  {name:8s}  mean={m:.5f}  std={s:.5f}")

        geo_names = ["bed_elev","surf_elev","thickness",
                     "velocity","strain_rate","fjord_mask"]
        print(f"\nGeophysical (6 channels)")
        for name, m, s in zip(geo_names, geo_means, geo_stds):
            print(f"  {name:12s}  mean={m:10.3f}  std={s:10.3f}")

        clm_names = ["mar_runoff","ocean_temp"]
        print(f"\nClimate (2 scalars)")
        for name, m, s in zip(clm_names, clm_means, clm_stds):
            print(f"  {name:12s}  mean={m:10.4f}  std={s:10.4f}")

        print(f"\nWritten to {out_path}")
        print(f"{'='*55}")

    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute normalisation statistics from the training split."
    )
    parser.add_argument(
        "--train_index", type=Path, default=TRAIN_INDEX_PATH,
        help="Path to train.json (default: %(default)s)"
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_PATH,
        help="Output path for normalisation_stats.json (default: %(default)s)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=2000,
        help="Number of training samples to use (default: %(default)d)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducible subsampling (default: %(default)d)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_stats(
        train_index_path = args.train_index,
        out_path         = args.out,
        n_samples        = args.n_samples,
        seed             = args.seed,
        verbose          = True,
    )
