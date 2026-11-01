"""
Post-training evaluation script for the CalvingDB benchmark.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import zarr
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CalvingFrontDataset, calving_collate_fn
from metrics import (
    compute_metrics,
    compute_skill_score,
    aggregate_metrics,
    denormalise_sdt,
    DEFAULT_RESOLUTION_M,
    DEFAULT_FRONT_BAND_M,
)


SDT_VAR_THRESHOLD_M2: float = 100.0

SPLITS_DIR = Path(
    "/path/to/calvdb/TrainingDataset_processed/splits"
)
LOG_DIR = Path("logs/")

SENSOR_CODE_MAP = {0: "S2", 1: "LE07", 2: "LC08", 3: "LC09", 255: "unknown"}


_zarr_stores: dict[str, zarr.Group] = {}


def get_zarr_store(zarr_path: str) -> zarr.Group:
    if zarr_path not in _zarr_stores:
        _zarr_stores[zarr_path] = zarr.open_group(zarr_path, mode="r")
    return _zarr_stores[zarr_path]


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device:          torch.device,
) -> tuple[torch.nn.Module, dict]:
    """
    Load a model from a training checkpoint, dispatching on cfg["model"]["name"].
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "config" not in ckpt:
        raise KeyError(
            f"Checkpoint {checkpoint_path} has no 'config' key. "
            "Ensure train.py saves the config alongside the model weights."
        )

    cfg = ckpt["config"]
    m   = cfg["model"]

    from models.convlstm    import CalvingUNet
    from models.predrnn     import CalvingPredRNN
    from models.simvp       import CalvingSimVP
    from models.phydnet     import CalvingPhyDNet
    from models.stgcn       import CalvingSTGCN
    from models.earthformer import CalvingEarthFormer

    base_kwargs = dict(
        in_channels   = m["K"] * m["C_total"],
        base_channels = m["base_channels"],
        climate_dim   = m["climate_dim"],
        tau_dim       = m["tau_dim"],
        num_layers    = m.get("num_layers", 3),
    )

    name = m.get("name", "CalvingUNet")
    if name == "CalvingUNet":
        model = CalvingUNet(**base_kwargs)
    elif name == "CalvingPredRNN":
        model = CalvingPredRNN(**base_kwargs)
    elif name == "CalvingSimVP":
        model = CalvingSimVP(
            **{k: v for k, v in base_kwargs.items() if k != "num_layers"},
            num_layers         = m.get("num_layers", 4),
            translator_kernels = tuple(m.get("translator_kernels", [3, 5, 7])),
        )
    elif name == "CalvingPhyDNet":
        model = CalvingPhyDNet(
            **{k: v for k, v in base_kwargs.items() if k != "num_layers"},
            num_layers    = m.get("num_layers", 1),
            n_moments     = m.get("n_moments", 4),
            moment_kernel = m.get("moment_kernel", 7),
        )
    elif name == "CalvingSTGCN":
        model = CalvingSTGCN(
            **{k: v for k, v in base_kwargs.items() if k != "num_layers"},
            num_layers = m.get("num_layers", 2),
            kt         = m.get("kt", 3),
        )
    elif name == "CalvingEarthFormer":
        model = CalvingEarthFormer(
            **{k: v for k, v in base_kwargs.items() if k != "num_layers"},
            num_layers  = m.get("num_layers", 4),
            num_heads   = m.get("num_heads", 4),
            window_size = tuple(m.get("window_size", [5, 2, 2])),
            mlp_ratio   = m.get("mlp_ratio", 4.0),
            drop        = m.get("drop", 0.0),
            attn_drop   = m.get("attn_drop", 0.0),
        )
    else:
        raise ValueError(f"Unknown model name in checkpoint: {name!r}")

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    epoch    = ckpt.get("epoch", "unknown")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"Loaded checkpoint from epoch {epoch}  (val_loss={val_loss:.4f})")

    return model, cfg


def load_trace(
    zarr_path:         str,
    target_zarr_index: int,
) -> Optional[np.ndarray]:
    """
    Load the calving front trace for a single sample.
    """
    store = get_zarr_store(zarr_path)
    if "trace" in store and "valid_trace" in store:
        if bool(store["valid_trace"][target_zarr_index]):
            return store["trace"][target_zarr_index].astype(bool)
    return None


def load_persistence_sdt(
    zarr_path:        str,
    last_input_index: int,
) -> Optional[np.ndarray]:
    """
    Load the SDT at the last input time step as the persistence baseline (metres).
    """
    store = get_zarr_store(zarr_path)
    if "sdt" not in store:
        return None
    return store["sdt"][last_input_index].astype(np.float32)


def get_sensor_name(zarr_path: str, input_zarr_indices: list[int]) -> str:
    """
    Return the sensor name for the last input time step from zarr["sensors"].
    """
    store = get_zarr_store(zarr_path)
    if "sensors" not in store:
        return "unknown"
    code = int(store["sensors"][input_zarr_indices[-1]])
    return SENSOR_CODE_MAP.get(code, "unknown")


def date_to_season(iso_date: str) -> str:
    """
    Map an ISO date string to a meteorological season (Svalbard conventions).
    """
    try:
        month = date.fromisoformat(iso_date).month
    except (ValueError, TypeError):
        return "unknown"

    if month in (12, 1, 2):  return "DJF"
    if month in (3,  4, 5):  return "MAM"
    if month in (6,  7, 8):  return "JJA"
    return "SON"


@torch.no_grad()
def run_evaluation(
    model:        torch.nn.Module,
    dataloader:   DataLoader,
    stats:        dict,
    device:       torch.device,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    front_band_m: float = DEFAULT_FRONT_BAND_M,
    use_amp:      bool  = True,
) -> list[dict]:
    """
    Run the full evaluation loop over a DataLoader.
    """
    records = []

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        model_input = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with autocast(enabled=use_amp):
            pred_disp_norm = model(model_input)

        pred_disp_norm = pred_disp_norm.cpu().float().numpy()
        tgt_norm       = batch["target"].float().numpy()

        sdt_std  = float(stats["sdt"]["std"])
        sdt_mean = float(stats["sdt"]["mean"])
        disp_std = float(stats.get("displacement", {}).get("std", sdt_std))

        svm         = batch["spatial_valid_mask"].numpy()
        original_hw = batch["original_hw"]

        B = pred_disp_norm.shape[0]

        for b in range(B):

            glacier_id     = batch["glacier_id"][b]
            reference_date = batch["reference_date"][b]
            horizon_days   = int(batch["horizon_days"][b])
            zarr_path      = batch["zarr_path"][b]
            ii             = batch["input_zarr_indices"][b]
            ti             = int(batch["target_zarr_index"][b])
            last_input_idx = int(batch["last_input_zarr_index"][b])

            h, w = original_hw[b]

            pred_disp_m = pred_disp_norm[b, 0] * disp_std

            store            = get_zarr_store(zarr_path)
            last_input_sdt_m = store["sdt"][last_input_idx].astype(np.float32)

            pred_m = np.clip(
                pred_disp_m[:h, :w] + last_input_sdt_m[:h, :w],
                -2000.0, 2000.0,
            ).astype(np.float32)

            target_m = (tgt_norm[b, 0][:h, :w] * sdt_std + sdt_mean).astype(np.float32)

            trace = load_trace(zarr_path, target_zarr_index=ti)

            persist_m = load_persistence_sdt(zarr_path, last_input_idx)
            if persist_m is not None:
                persist_m = persist_m[:h, :w]

            sdt_input_stack = np.stack(
                [store["sdt"][i].astype(np.float32) for i in ii],
                axis=0,
            )
            sdt_var    = np.var(sdt_input_stack, axis=0)
            active_mask = (sdt_var[:h, :w] > SDT_VAR_THRESHOLD_M2)

            sensor = get_sensor_name(zarr_path, ii)

            model_metrics = compute_metrics(
                pred_sdt     = pred_m,
                target_sdt   = target_m,
                target_trace = trace,
                active_mask  = active_mask,
                resolution_m = resolution_m,
                front_band_m = front_band_m,
            )

            if persist_m is not None:
                persist_metrics = compute_metrics(
                    pred_sdt     = persist_m,
                    target_sdt   = target_m,
                    target_trace = trace,
                    active_mask  = active_mask,
                    resolution_m = resolution_m,
                    front_band_m = front_band_m,
                )
            else:
                persist_metrics = {k: float("nan") for k in model_metrics}

            skill = compute_skill_score(
                model_metrics["mde"],
                persist_metrics["mde"],
            )

            if trace is not None and trace.any():
                front_px = trace.astype(bool)
                mean_pred_disp_m    = float(np.abs(pred_disp_m[:h, :w][front_px]).mean())
                mean_true_disp_m    = float(np.abs((target_m - persist_m)[front_px]).mean()
                                            if persist_m is not None else float("nan"))
                mean_pred_disp_sign = float(pred_disp_m[:h, :w][front_px].mean())
            else:
                mean_pred_disp_m    = float("nan")
                mean_true_disp_m    = float("nan")
                mean_pred_disp_sign = float("nan")

            records.append({
                "glacier_id":     glacier_id,
                "reference_date": reference_date,
                "horizon_days":   horizon_days,
                "sensor":         sensor,
                "season":         date_to_season(reference_date),
                "zarr_path":      zarr_path,

                "has_trace":      trace is not None,

                **{f"model_{k}":   v for k, v in model_metrics.items()},

                **{f"persist_{k}": v for k, v in persist_metrics.items()},

                "skill_score":    skill,

                "mean_pred_displacement_m":    mean_pred_disp_m,
                "mean_true_displacement_m":    mean_true_disp_m,
                "mean_pred_displacement_sign": mean_pred_disp_sign,
            })

    return records


def compute_stratified_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean metrics stratified by sensor, season, and horizon.
    """
    metric_cols = [c for c in df.columns
                   if c.startswith("model_") or c == "skill_score"]
    strata = []

    for groupby_col in ["sensor", "season", "horizon_days"]:
        if groupby_col not in df.columns:
            continue
        for group_val, group_df in df.groupby(groupby_col):
            row = {
                "stratum_col": groupby_col,
                "stratum_val": str(group_val),
                "n_samples":   len(group_df),
            }
            for col in metric_cols:
                valid = group_df[col].dropna()
                row[f"{col}_mean"]   = float(valid.mean())   if len(valid) else float("nan")
                row[f"{col}_median"] = float(valid.median()) if len(valid) else float("nan")
            strata.append(row)

    return pd.DataFrame(strata)


def compute_per_glacier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics per glacier.
    """
    metric_cols = [c for c in df.columns
                   if c.startswith("model_") or c == "skill_score"]
    rows = []

    for glacier_id, gdf in df.groupby("glacier_id"):
        row = {
            "glacier_id": glacier_id,
            "n_samples":  len(gdf),
            "date_min":   gdf["reference_date"].min(),
            "date_max":   gdf["reference_date"].max(),
            "sensors":    ",".join(sorted(gdf["sensor"].unique())),
        }
        for col in metric_cols:
            valid = gdf[col].dropna()
            row[f"{col}_mean"]   = float(valid.mean())   if len(valid) else float("nan")
            row[f"{col}_median"] = float(valid.median()) if len(valid) else float("nan")
            row[f"{col}_std"]    = float(valid.std())    if len(valid) else float("nan")
        rows.append(row)

    return pd.DataFrame(rows).sort_values("glacier_id")


def build_summary(
    df:        pd.DataFrame,
    split:     str,
    ckpt_path: str,
) -> dict:
    """
    Build a concise summary dict with headline numbers for the paper's results table.
    """
    def _mean(col):   return float(df[col].dropna().mean())   if col in df.columns else float("nan")
    def _std(col):    return float(df[col].dropna().std())    if col in df.columns else float("nan")
    def _median(col): return float(df[col].dropna().median()) if col in df.columns else float("nan")

    headline_metrics = [
        "mae_front", "rmse_front", "mde", "hausdorff",
        "advance_retreat_bias", "area_error_km2", "front_iou",
    ]

    summary = {
        "split":       split,
        "checkpoint":  str(ckpt_path),
        "n_samples":   len(df),
        "n_glaciers":  int(df["glacier_id"].nunique()),
        "model":       {},
        "persistence": {},
        "skill":       {},
    }

    for m in headline_metrics:
        summary["model"][m] = {
            "mean":   _mean(f"model_{m}"),
            "std":    _std(f"model_{m}"),
            "median": _median(f"model_{m}"),
        }
        summary["persistence"][m] = {
            "mean":   _mean(f"persist_{m}"),
            "std":    _std(f"persist_{m}"),
            "median": _median(f"persist_{m}"),
        }

    summary["skill"]["mean"]   = _mean("skill_score")
    summary["skill"]["std"]    = _std("skill_score")
    summary["skill"]["median"] = _median("skill_score")

    if "skill_score" in df.columns:
        valid_skill = df["skill_score"].dropna()
        summary["skill"]["frac_positive"] = float(
            (valid_skill > 0).mean() if len(valid_skill) else float("nan")
        )

    return summary


def print_summary(summary: dict) -> None:
    """Print a formatted summary table to stdout."""
    print(f"\n{'='*65}")
    print(f"EVALUATION RESULTS — {summary['split'].upper()} SPLIT")
    print(f"{'='*65}")
    print(f"  Checkpoint : {summary['checkpoint']}")
    print(f"  Samples    : {summary['n_samples']}")
    print(f"  Glaciers   : {summary['n_glaciers']}")
    print(f"\n{'Metric':<28} {'Model':>12} {'Persistence':>14} {'Skill':>10}")
    print(f"  {'-'*60}")

    metric_units = {
        "mae_front":            "m",
        "rmse_front":           "m",
        "mde":                  "m",
        "hausdorff":            "m",
        "advance_retreat_bias": "m",
        "area_error_km2":       "km²",
        "front_iou":            "",
    }

    for m, unit in metric_units.items():
        model_val   = summary["model"].get(m, {}).get("mean", float("nan"))
        persist_val = summary["persistence"].get(m, {}).get("mean", float("nan"))
        label       = f"{m} ({unit})" if unit else m

        if m in ("mde", "mae_front", "rmse_front", "hausdorff"):
            sk = 1.0 - model_val / persist_val \
                 if persist_val > 0 and not np.isnan(persist_val) \
                 else float("nan")
            skill_str = f"{sk:>10.3f}"
        else:
            skill_str = f"{'—':>10}"

        mv = f"{model_val:>12.2f}"   if not np.isnan(model_val)   else f"{'nan':>12}"
        pv = f"{persist_val:>14.2f}" if not np.isnan(persist_val) else f"{'nan':>14}"
        print(f"  {label:<28} {mv} {pv} {skill_str}")

    print(f"\n  Overall skill score  : {summary['skill']['mean']:.3f} "
          f"± {summary['skill']['std']:.3f}")
    print(f"  Fraction > 0 (beats persistence): "
          f"{summary['skill'].get('frac_positive', float('nan')):.1%}")
    print(f"{'='*65}\n")


def evaluate(
    checkpoint_path: Path,
    split:           str   = "test",
    splits_dir:      Path  = SPLITS_DIR,
    log_dir:         Path  = LOG_DIR,
    batch_size:      int   = 4,
    num_workers:     int   = 4,
    resolution_m:    float = DEFAULT_RESOLUTION_M,
    front_band_m:    float = DEFAULT_FRONT_BAND_M,
    use_amp:         bool  = True,
    device_str:      str   = "cuda",
) -> pd.DataFrame:
    """
    Run the full evaluation pipeline for a trained checkpoint on the given split.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        device_str if torch.cuda.is_available() or device_str == "cpu"
        else "cpu"
    )
    print(f"Device        : {device}")
    print(f"Resolution    : {resolution_m} m  (EPSG:3995)")

    model, cfg = load_model_from_checkpoint(checkpoint_path, device)

    stats_path = splits_dir / "normalisation_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"normalisation_stats.json not found at {stats_path}. "
            "Run compute_stats.py before evaluate.py."
        )
    with open(stats_path) as f:
        stats = json.load(f)

    index_path = splits_dir / f"{split}.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"{split}.json not found at {index_path}. "
            "Run 03_build_index.py before evaluate.py."
        )

    d  = cfg["data"]
    ds = CalvingFrontDataset(
        index_path = index_path,
        stats_path = str(stats_path),
        modalities = d.get("modalities"),
        augment    = False,
    )

    loader = DataLoader(
        ds,
        batch_size  = batch_size,
        num_workers = num_workers,
        shuffle     = False,
        pin_memory  = device.type == "cuda",
        collate_fn  = calving_collate_fn,
    )
    print(f"Evaluating {len(ds)} samples from '{split}' split "
          f"({len(loader)} batches)\n")

    records = run_evaluation(
        model        = model,
        dataloader   = loader,
        stats        = stats,
        device       = device,
        resolution_m = resolution_m,
        front_band_m = front_band_m,
        use_amp      = use_amp,
    )

    if not records:
        warnings.warn("No records produced — check that the DataLoader is non-empty.")
        return pd.DataFrame()

    df           = pd.DataFrame(records)
    per_glacier  = compute_per_glacier(df)
    stratified   = compute_stratified_breakdown(df)
    summary      = build_summary(df, split, checkpoint_path)

    csv_path     = log_dir / f"{split}_results.csv"
    glacier_path = log_dir / f"{split}_per_glacier.csv"
    strat_path   = log_dir / f"{split}_stratified.csv"
    summary_path = log_dir / f"{split}_summary.json"

    df.to_csv(csv_path,          index=False)
    per_glacier.to_csv(glacier_path, index=False)
    stratified.to_csv(strat_path,    index=False)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)

    print(f"Outputs written to {log_dir}")
    print(f"  {csv_path.name:<40}  per-sample results")
    print(f"  {glacier_path.name:<40}  per-glacier aggregates")
    print(f"  {strat_path.name:<40}  stratified breakdown")
    print(f"  {summary_path.name:<40}  headline numbers (JSON)")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CalvingUNet checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/path/to/calvdb/experiments/exp19_phydnet/checkpoints/best.pt"),
        help="Path to checkpoint file (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="Which split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--splits_dir", type=Path, default=SPLITS_DIR,
        help="Directory containing split JSON files and stats",
    )
    parser.add_argument(
        "--log_dir", type=Path, default=LOG_DIR,
        help="Directory to write output files (default: logs/)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for evaluation (default: 4)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader worker processes (default: 4)",
    )
    parser.add_argument(
        "--resolution_m", type=float, default=DEFAULT_RESOLUTION_M,
        help=f"Spatial resolution in metres per pixel (default: {DEFAULT_RESOLUTION_M})",
    )
    parser.add_argument(
        "--front_band_m", type=float, default=DEFAULT_FRONT_BAND_M,
        help=f"Evaluation band half-width in metres (default: {DEFAULT_FRONT_BAND_M})",
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device: 'cuda', 'cpu', or 'cuda:N' (default: cuda)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.split == "test":
        val_summary = args.log_dir / "val_summary.json"
        if not val_summary.exists():
            raise RuntimeError(
                "val_summary.json not found in log_dir. "
                "Run evaluation on the val split first:\n"
                "  python evaluate.py --split val\n"
                "Only run on the test split when all modelling decisions "
                "are finalised."
            )

    evaluate(
        checkpoint_path = args.checkpoint,
        split           = args.split,
        splits_dir      = args.splits_dir,
        log_dir         = args.log_dir,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        resolution_m    = args.resolution_m,
        front_band_m    = args.front_band_m,
        use_amp         = not args.no_amp,
        device_str      = args.device,
    )
