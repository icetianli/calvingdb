"""
PyTorch Dataset for the CalvingDB calving front prediction benchmark.
It loads samples from the preprocessed split index and zarr stores, applying
normalisation and building temporal features for each input sequence.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset

DEFAULT_STATS_PATH = Path(
    "/path/to/calvdb/TrainingDataset_processed"
    "/splits/normalisation_stats.json"
)

VALID_MODALITIES = {"sdt", "imagery", "geophys", "climate"}


class CalvingFrontDataset(Dataset):
    """
    Dataset for multi-modal calving front prediction.
    """

    def __init__(
        self,
        index_path: str | Path,
        stats_path: str | Path | None = DEFAULT_STATS_PATH,
        modalities: list[str] | None = None,
        augment: bool = False,
    ):
        self.index_path = Path(index_path)
        self.augment    = augment

        if modalities is None:
            self.modalities = list(VALID_MODALITIES)
        else:
            unknown = set(modalities) - VALID_MODALITIES
            if unknown:
                raise ValueError(
                    f"Unknown modalities: {unknown}. "
                    f"Valid choices: {VALID_MODALITIES}"
                )
            self.modalities = list(modalities)

        if "sdt" not in self.modalities:
            raise ValueError("'sdt' must always be included in modalities.")

        with open(self.index_path) as f:
            self.samples: list[dict] = json.load(f)

        if len(self.samples) == 0:
            raise RuntimeError(f"Index file {index_path} contains no samples.")

        self.stats: dict | None = None
        self._norm_params: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if stats_path is not None:
            stats_path = Path(stats_path)
            if stats_path.exists():
                with open(stats_path) as f:
                    self.stats = json.load(f)
                self._norm_params = self._prepare_norm_params(self.stats)
            else:
                print(
                    f"WARNING: stats_path not found ({stats_path}). "
                    "Proceeding without normalisation."
                )

        self._disp_scale: float = 1.0
        if self.stats is not None and "displacement" in self.stats:
            sdt_std_v  = float(self.stats["sdt"]["std"])
            disp_std_v = float(self.stats["displacement"]["std"])
            self._disp_scale = sdt_std_v / max(disp_std_v, 1e-6)

        self._stores: dict[str, zarr.Group] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_store(self, zarr_path: str) -> zarr.Group:
        """Return a cached zarr group, opening it on first access."""
        if zarr_path not in self._stores:
            self._stores[zarr_path] = zarr.open_group(zarr_path, mode="r")
        return self._stores[zarr_path]

    @staticmethod
    def _prepare_norm_params(
        stats: dict,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Convert JSON stats into broadcast-friendly numpy arrays.
        """
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key in ("sdt", "imagery", "geophys", "climate"):
            if key not in stats:
                continue
            item = stats[key]
            if "mean" not in item or "std" not in item:
                continue

            mean = np.asarray(item["mean"], dtype=np.float32)
            std = np.asarray(item["std"], dtype=np.float32)

            std = np.maximum(std, np.float32(1e-6))
            out[key] = (mean, std)

        return out

    @staticmethod
    def _reshape_channel_stats(arr: np.ndarray, stats_vec: np.ndarray) -> np.ndarray:
        """Reshape 1D channel stats to broadcast over known tensor layouts."""
        if stats_vec.ndim != 1:
            return stats_vec

        if arr.ndim == 4 and arr.shape[1] == stats_vec.shape[0]:
            return stats_vec.reshape(1, -1, 1, 1)
        if arr.ndim == 3 and arr.shape[0] == stats_vec.shape[0]:
            return stats_vec.reshape(-1, 1, 1)
        if arr.ndim == 2 and arr.shape[1] == stats_vec.shape[0]:
            return stats_vec.reshape(1, -1)

        return stats_vec

    def _normalise(self, arr: np.ndarray, key: str) -> np.ndarray:
        """
        Z-score normalise using pre-computed training-split statistics.
        """
        if key not in self._norm_params:
            return arr

        mean, std = self._norm_params[key]
        mean = self._reshape_channel_stats(arr, mean)
        std = self._reshape_channel_stats(arr, std)
        return (arr - mean) / std

    @staticmethod
    def _build_tau(
        input_days:     list[float],
        day_gaps:       list[float],
        anchor_offsets: list[int],
        reference_date: str,
    ) -> np.ndarray:
        """
        Build the per-slot temporal feature matrix τ ∈ ℝ^(K, 4).

        τ[k, 0] = d* / max(|anchor_offsets|)
        τ[k, 1] = doy / 365
        τ[k, 2] = gap / max_anchor_spacing
        τ[k, 3] = (d* - anchor) / tolerance
        """
        K             = len(input_days)
        ref           = date.fromisoformat(reference_date)
        max_offset    = float(max(abs(a) for a in anchor_offsets))
        max_spacing   = float(max(abs(anchor_offsets[i] - anchor_offsets[i-1])
                               for i in range(1, len(anchor_offsets))))

        tau = np.zeros((K, 4), dtype=np.float32)

        for k in range(K):
            d_star  = float(input_days[k])
            anchor  = float(anchor_offsets[k])
            abs_day = ref + __import__("datetime").timedelta(days=int(d_star))
            doy     = abs_day.timetuple().tm_yday

            if k < K - 1:
                gap = float(input_days[k + 1]) - d_star
            else:
                gap = 0.0 - d_star

            tau[k, 0] = d_star  / max_offset
            tau[k, 1] = doy     / 365.0
            tau[k, 2] = gap     / max(max_spacing, 1)
            tau[k, 3] = (d_star - anchor) / 10.0

        return tau

    def __getitem__(self, idx: int) -> dict:
        s     = self.samples[idx]
        store = self._get_store(s["zarr_path"])

        ii    = s["input_zarr_indices"]
        ti    = s["target_zarr_index"]
        K     = len(ii)

        valid_img_flags = store["valid_img"][:]
        valid_clm_flags = store["valid_clm"][:]

        valid_img = np.array([valid_img_flags[i] for i in ii], dtype=bool)
        valid_clm = np.array([valid_clm_flags[i] for i in ii], dtype=bool)

        spatial_channels = []

        sdt_stack = store["sdt"][ii].astype(np.float32)
        sdt_stack = np.nan_to_num(sdt_stack, nan=0.0, posinf=0.0, neginf=0.0)
        sdt_stack = self._normalise(sdt_stack, "sdt")
        spatial_channels.append(sdt_stack[:, np.newaxis])

        sdt_diff = np.diff(sdt_stack, axis=0, prepend=sdt_stack[[0]])
        spatial_channels.append(sdt_diff[:, np.newaxis])

        last_input_sdt_norm = sdt_stack[-1].copy()

        if "imagery" in self.modalities:
            imagery_stack = store["imagery"][ii].astype(np.float32)
            imagery_stack = np.nan_to_num(imagery_stack, nan=0.0, posinf=0.0, neginf=0.0)
            imagery_stack = self._normalise(imagery_stack, "imagery")
            for k in range(K):
                if not valid_img[k]:
                    imagery_stack[k] = 0.0
            spatial_channels.append(imagery_stack)

        if "geophys" in self.modalities:
            geophys_stack = store["geophys"][ii].astype(np.float32)
            geophys_stack = np.nan_to_num(geophys_stack, nan=0.0, posinf=0.0, neginf=0.0)
            geophys_stack = self._normalise(geophys_stack, "geophys")
            geophys_stack = geophys_stack[:, :5]
            spatial_channels.append(geophys_stack)

        inputs = np.concatenate(spatial_channels, axis=1)

        target_sdt = store["sdt"][[ti]].astype(np.float32)
        target_sdt = np.nan_to_num(target_sdt, nan=0.0, posinf=0.0, neginf=0.0)
        target_sdt = self._normalise(target_sdt, "sdt")

        target_sdt_norm_2d       = target_sdt[0]
        target_displacement_norm = (target_sdt_norm_2d - last_input_sdt_norm) * self._disp_scale

        if "climate" in self.modalities:
            climate = store["climate"][ii].astype(np.float32)
            for k in range(K):
                if not valid_clm[k]:
                    climate[k] = 0.0
            climate = self._normalise(climate, "climate")
        else:
            D_c     = store["climate"].shape[1]
            climate = np.zeros((K, D_c), dtype=np.float32)

        tau = self._build_tau(
            input_days     = s["input_days"],
            day_gaps       = s["day_gaps"],
            anchor_offsets = s["anchor_offsets"],
            reference_date = s["reference_date"],
        )

        last_input_zarr_idx = ii[-1]
        if (
            "trace" in store
            and "valid_trace" in store
            and bool(store["valid_trace"][last_input_zarr_idx])
        ):
            input_trace = store["trace"][last_input_zarr_idx].astype(np.float32)
        else:
            raw_norm = sdt_stack[-1]
            if self.stats is not None:
                sdt_mean = float(self.stats["sdt"]["mean"])
                sdt_std  = float(self.stats["sdt"]["std"])
                sdt_m    = raw_norm * sdt_std + sdt_mean
            else:
                sdt_m = raw_norm
            input_trace = (np.abs(sdt_m) < 45.0).astype(np.float32)
            import warnings
            warnings.warn(
                f"Trace unavailable for last input slot (zarr_idx={last_input_zarr_idx}). "
                "Using SDT zero-crossing approximation for loss weighting.",
                stacklevel=2,
            )
        input_trace = input_trace[np.newaxis]

        if (
            "trace" in store
            and "valid_trace" in store
            and bool(store["valid_trace"][ti])
        ):
            target_trace = store["trace"][ti].astype(np.float32)[np.newaxis]
        else:
            H_t, W_t = store["sdt"][ti].shape
            target_trace = np.zeros((1, H_t, W_t), dtype=np.float32)

        if self.augment and torch.rand(1).item() > 0.5:
            inputs                   = np.flip(inputs,                   axis=-1).copy()
            target_sdt               = np.flip(target_sdt,               axis=-1).copy()
            input_trace              = np.flip(input_trace,              axis=-1).copy()
            target_trace             = np.flip(target_trace,             axis=-1).copy()
            last_input_sdt_norm      = np.flip(last_input_sdt_norm,      axis=-1).copy()
            target_displacement_norm = np.flip(target_displacement_norm, axis=-1).copy()
            sdt_diff                 = np.flip(sdt_diff,                 axis=-1).copy()

        if self.augment and torch.rand(1).item() > 0.5:
            inputs                   = np.flip(inputs,                   axis=-2).copy()
            target_sdt               = np.flip(target_sdt,               axis=-2).copy()
            input_trace              = np.flip(input_trace,              axis=-2).copy()
            target_trace             = np.flip(target_trace,             axis=-2).copy()
            last_input_sdt_norm      = np.flip(last_input_sdt_norm,      axis=-2).copy()
            target_displacement_norm = np.flip(target_displacement_norm, axis=-2).copy()
            sdt_diff                 = np.flip(sdt_diff,                 axis=-2).copy()

        if self.augment and K > 1 and torch.rand(1).item() > 0.5:
            k_drop = np.random.randint(0, K - 1)
            inputs[k_drop] = 0.0

        if self.augment:
            noise_std = 0.06
            inputs[:, 0:2] += np.random.randn(*inputs[:, 0:2].shape).astype(np.float32) * noise_std

        return {
            "inputs":         torch.from_numpy(inputs),

            "target":         torch.from_numpy(target_sdt),

            "input_trace":    torch.from_numpy(input_trace),
            "target_trace":   torch.from_numpy(target_trace),

            "climate":        torch.from_numpy(climate),

            "tau":            torch.from_numpy(tau),

            "valid_img":      torch.from_numpy(valid_img),
            "valid_clm":      torch.from_numpy(valid_clm),

            "glacier_id":           s["glacier_id"],
            "reference_date":       s["reference_date"],
            "horizon_days":         s["horizon_days"],
            "zarr_path":            s["zarr_path"],

            "input_zarr_indices":   ii,
            "target_zarr_index":    ti,

            "last_input_sdt":       torch.from_numpy(last_input_sdt_norm[None]),
            "target_displacement":  torch.from_numpy(target_displacement_norm[None]),

            "last_input_zarr_index": int(ii[-1]),
        }


def calving_collate_fn(batch: list[dict]) -> dict:
    """
    Collate function that supports variable spatial sizes across glaciers.
    """
    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch.")

    max_h = max(int(s["inputs"].shape[-2]) for s in batch)
    max_w = max(int(s["inputs"].shape[-1]) for s in batch)

    padded_inputs        = []
    padded_targets       = []
    padded_input_traces  = []
    padded_target_traces = []
    padded_last_input_sdts = []
    padded_target_disps    = []
    spatial_masks          = []

    for s in batch:
        h = int(s["inputs"].shape[-2])
        w = int(s["inputs"].shape[-1])
        pad_h = max_h - h
        pad_w = max_w - w

        padded_inputs.append(F.pad(s["inputs"], (0, pad_w, 0, pad_h)))
        padded_targets.append(F.pad(s["target"], (0, pad_w, 0, pad_h)))
        padded_input_traces.append(F.pad(s["input_trace"],  (0, pad_w, 0, pad_h)))
        padded_target_traces.append(F.pad(s["target_trace"], (0, pad_w, 0, pad_h)))
        padded_last_input_sdts.append(F.pad(s["last_input_sdt"],      (0, pad_w, 0, pad_h)))
        padded_target_disps.append(   F.pad(s["target_displacement"], (0, pad_w, 0, pad_h)))

        mask = torch.zeros((1, max_h, max_w), dtype=torch.bool)
        mask[:, :h, :w] = True
        spatial_masks.append(mask)

    return {
        "inputs":        torch.stack(padded_inputs,        dim=0),
        "target":        torch.stack(padded_targets,       dim=0),
        "input_trace":   torch.stack(padded_input_traces,  dim=0),
        "target_trace":  torch.stack(padded_target_traces, dim=0),
        "spatial_valid_mask": torch.stack(spatial_masks,   dim=0),
        "last_input_sdt":      torch.stack(padded_last_input_sdts, dim=0),
        "target_displacement": torch.stack(padded_target_disps,    dim=0),
        "last_input_zarr_index": [s["last_input_zarr_index"] for s in batch],
        "climate": torch.stack([s["climate"] for s in batch], dim=0),
        "tau": torch.stack([s["tau"] for s in batch], dim=0),
        "valid_img": torch.stack([s["valid_img"] for s in batch], dim=0),
        "valid_clm": torch.stack([s["valid_clm"] for s in batch], dim=0),
        "glacier_id": [s["glacier_id"] for s in batch],
        "reference_date": [s["reference_date"] for s in batch],
        "horizon_days": torch.tensor([s["horizon_days"] for s in batch], dtype=torch.long),
        "zarr_path": [s["zarr_path"] for s in batch],
        "original_hw": [(int(s["inputs"].shape[-2]), int(s["inputs"].shape[-1])) for s in batch],
        "input_zarr_indices": [s["input_zarr_indices"] for s in batch],
        "target_zarr_index":  [s["target_zarr_index"]  for s in batch],
    }


def compute_and_save_stats(
    train_index_path: str | Path,
    out_path: str | Path,
    n_samples: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Compute per-modality normalisation statistics from a random subset of the training
    split and write them to a JSON file.
    """
    with open(train_index_path) as f:
        samples = json.load(f)

    rng      = np.random.default_rng(seed)
    n        = min(n_samples, len(samples))
    chosen   = rng.choice(len(samples), size=n, replace=False)
    stores   = {}

    def get_store(p: str) -> zarr.Group:
        if p not in stores:
            stores[p] = zarr.open_group(p, mode="r")
        return stores[p]

    sdt_vals      = []
    img_vals      = []
    geo_vals      = []
    clm_vals      = []
    disp_vals     = []

    print(f"Computing normalisation stats from {n} training samples...")

    for idx in chosen:
        s     = samples[int(idx)]
        store = get_store(s["zarr_path"])
        ii    = s["input_zarr_indices"]
        ti    = s["target_zarr_index"]

        sdt_in  = store["sdt"][ii].astype(np.float32).ravel()
        sdt_tgt = store["sdt"][[ti]].astype(np.float32).ravel()
        sdt_all_pix = np.concatenate([sdt_in, sdt_tgt])
        sdt_vals.append(sdt_all_pix[np.isfinite(sdt_all_pix)])

        img = store["imagery"][ii].astype(np.float32)
        img_vals.append(np.array([np.nanmean(img[:, c]) for c in range(img.shape[1])]))

        geo = store["geophys"][ii].astype(np.float32)
        geo_vals.append(np.array([np.nanmean(geo[:, c]) for c in range(geo.shape[1])]))

        clm = store["climate"][ii].astype(np.float32)
        clm_vals.append(clm.mean(axis=0))

        li = ii[-1]
        if bool(store["valid_trace"][ti]):
            trace_px = store["trace"][ti].astype(bool)
            if trace_px.any():
                disp = (store["sdt"][ti].astype(np.float32)
                        - store["sdt"][li].astype(np.float32))[trace_px]
                disp_vals.extend(disp[np.isfinite(disp)].tolist())

    sdt_all  = np.concatenate(sdt_vals)
    img_all  = np.stack(img_vals,  axis=0)
    geo_all  = np.stack(geo_vals,  axis=0)
    clm_all  = np.stack(clm_vals,  axis=0)

    disp_all = np.array(disp_vals, dtype=np.float32) if disp_vals else np.array([0.0])
    stats = {
        "sdt": {
            "mean": float(np.nanmean(sdt_all)),
            "std":  float(np.nanstd(sdt_all)),
        },
        "displacement": {
            "std": float(np.nanstd(disp_all)),
        },
        "imagery": {
            "mean": img_all.mean(axis=0).tolist(),
            "std":  img_all.std(axis=0).tolist(),
        },
        "geophys": {
            "mean": geo_all.mean(axis=0).tolist(),
            "std":  geo_all.std(axis=0).tolist(),
        },
        "climate": {
            "mean": clm_all.mean(axis=0).tolist(),
            "std":  clm_all.std(axis=0).tolist(),
        },
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Stats written to {out_path}")
    print(f"  SDT          mean={stats['sdt']['mean']:.2f}  std={stats['sdt']['std']:.2f}")
    print(f"  Displacement std={stats['displacement']['std']:.2f}  (disp_scale={stats['sdt']['std']/stats['displacement']['std']:.2f})")
    print(f"  Imagery (per channel mean): {[f'{v:.4f}' for v in stats['imagery']['mean']]}")
    print(f"  Geophys (per channel mean): {[f'{v:.4f}' for v in stats['geophys']['mean']]}")
    print(f"  Climate (per variable mean): {[f'{v:.4f}' for v in stats['climate']['mean']]}")

    return stats


def make_dataloaders(
    splits_dir:  str | Path,
    stats_path:  str | Path | None = DEFAULT_STATS_PATH,
    modalities:  list[str] | None = None,
    batch_size:  int = 8,
    num_workers: int = 4,
    pin_memory:  bool = True,
) -> tuple[torch.utils.data.DataLoader, ...]:
    """
    Build train, val, and test DataLoaders from the split JSON files in splits_dir.
    """
    from torch.utils.data import DataLoader

    splits_dir = Path(splits_dir)

    train_ds = CalvingFrontDataset(
        splits_dir / "train.json",
        stats_path=stats_path,
        modalities=modalities,
        augment=True,
    )
    val_ds = CalvingFrontDataset(
        splits_dir / "val.json",
        stats_path=stats_path,
        modalities=modalities,
        augment=False,
    )
    test_ds = CalvingFrontDataset(
        splits_dir / "test.json",
        stats_path=stats_path,
        modalities=modalities,
        augment=False,
    )

    shared = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        collate_fn  = calving_collate_fn,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **shared)
    val_loader   = DataLoader(val_ds,   shuffle=False, **shared)
    test_loader  = DataLoader(test_ds,  shuffle=False, **shared)

    print(f"DataLoaders ready")
    print(f"  Train : {len(train_ds):7d} samples  {len(train_loader):5d} batches")
    print(f"  Val   : {len(val_ds):7d} samples  {len(val_loader):5d} batches")
    print(f"  Test  : {len(test_ds):7d} samples  {len(test_loader):5d} batches")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import sys

    splits_dir = Path(
        "/path/to/calvdb/TrainingDataset_processed/splits"
    )

    stats_path = splits_dir / "normalisation_stats.json"
    if not stats_path.exists():
        print("normalisation_stats.json not found — computing from training split...")
        compute_and_save_stats(
            train_index_path = splits_dir / "train.json",
            out_path         = stats_path,
            n_samples        = 500,
        )

    ds = CalvingFrontDataset(
        index_path = splits_dir / "train.json",
        stats_path = stats_path,
        modalities = None,
        augment    = False,
    )

    print(f"\nDataset length: {len(ds)}")

    sample = ds[0]
    print("\nSample keys and shapes / types:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s}  {tuple(v.shape)}  {v.dtype}")
        else:
            print(f"  {k:20s}  {v}")

    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds,
        batch_size=4,
        num_workers=0,
        shuffle=False,
        collate_fn=calving_collate_fn,
    )
    batch  = next(iter(loader))
    print("\nFirst batch:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s}  {tuple(v.shape)}")
        else:
            print(f"  {k:20s}  {v}")
