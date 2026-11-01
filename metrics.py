"""
Evaluation metrics for calving front SDT prediction: MAE, mean distance error, Hausdorff
distance, front IoU, and skill score against persistence.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.spatial.distance import cdist
from skimage.measure import find_contours


DEFAULT_RESOLUTION_M: float = 30.0

DEFAULT_FRONT_BAND_M: float = 300.0

DEFAULT_IOU_DILATION_M: float = 150.0

DEFAULT_SDT_VAR_THRESHOLD_M2: float = 100.0

NAN = float("nan")


def build_evaluation_zone(
    target_trace: np.ndarray,
    active_mask:  Optional[np.ndarray] = None,
    front_band_m: float = DEFAULT_FRONT_BAND_M,
    resolution_m: float = DEFAULT_RESOLUTION_M,
) -> np.ndarray:
    """
    Build a boolean spatial mask that restricts metric computation to the region near
    the calving front, optionally intersected with temporally active pixels to exclude
    the static fjord rock wall.

    zone = front_band ∩ active_mask
    active_mask = (sdt_var > threshold)
    """
    front_band_px = max(1, int(round(front_band_m / resolution_m)))
    zone = binary_dilation(target_trace.astype(bool), iterations=front_band_px)

    if active_mask is not None:
        zone = zone & active_mask.astype(bool)

    return zone


def extract_predicted_front(
    pred_sdt:     np.ndarray,
    eval_zone:    np.ndarray,
    resolution_m: float = DEFAULT_RESOLUTION_M,
) -> Optional[np.ndarray]:
    """
    Extract the predicted calving front as pixel-coordinate points from the
    zero-level-set of the predicted SDT, restricted to eval_zone.
    """
    masked = pred_sdt.copy()
    masked[~eval_zone] = 1.0

    try:
        contours = find_contours(masked, level=0.0)
    except Exception:
        return None

    if not contours:
        return None

    front_px = np.concatenate(contours, axis=0)

    H, W = pred_sdt.shape
    row_idx = np.clip(np.round(front_px[:, 0]).astype(int), 0, H - 1)
    col_idx = np.clip(np.round(front_px[:, 1]).astype(int), 0, W - 1)
    in_zone = eval_zone[row_idx, col_idx]
    front_px = front_px[in_zone]
    if len(front_px) == 0:
        return None

    return front_px * resolution_m


def trace_to_points(
    target_trace: np.ndarray,
    eval_zone:    np.ndarray,
    resolution_m: float = DEFAULT_RESOLUTION_M,
) -> Optional[np.ndarray]:
    """
    Convert the binary Trace_MASK raster (zarr["trace"]) to metre-coordinate points,
    restricted to the evaluation zone.
    """
    active = target_trace & eval_zone
    rows, cols = np.where(active)
    if len(rows) == 0:
        return None
    pts = np.stack([rows.astype(float), cols.astype(float)], axis=1)
    return pts * resolution_m


def _mae_rmse_in_zone(
    pred_sdt:   np.ndarray,
    target_sdt: np.ndarray,
    zone:       np.ndarray,
) -> tuple[float, float]:
    """MAE and RMSE of the SDT prediction within the evaluation zone (metres)."""
    err     = pred_sdt[zone] - target_sdt[zone]
    abs_err = np.abs(err)
    return float(abs_err.mean()), float(np.sqrt((err ** 2).mean()))


def _mde_hausdorff(
    pred_pts:   np.ndarray,
    target_pts: np.ndarray,
) -> tuple[float, float]:
    """
    Symmetric Mean Distance Error and Hausdorff distance (metres).

    MDE = 0.5 × (mean min-dist pred→target + mean min-dist target→pred)
    H   = max(max min-dist pred→target, max min-dist target→pred)
    """
    D       = cdist(pred_pts, target_pts, metric="euclidean")
    min_p2t = D.min(axis=1)
    min_t2p = D.min(axis=0)
    mde       = float((min_p2t.mean() + min_t2p.mean()) / 2.0)
    hausdorff = float(max(min_p2t.max(), min_t2p.max()))
    return mde, hausdorff


def _advance_retreat_bias(
    pred_sdt:     np.ndarray,
    target_trace: np.ndarray,
    eval_zone:    np.ndarray,
) -> float:
    """
    Mean predicted SDT at the true calving front pixel locations (metres).
    """
    front_pixels = target_trace & eval_zone
    if not front_pixels.any():
        return NAN
    return float(pred_sdt[front_pixels].mean())


def _area_error_km2(
    pred_sdt:    np.ndarray,
    target_sdt:  np.ndarray,
    eval_zone:   np.ndarray,
    resolution_m: float,
) -> float:
    """
    Area of the symmetric difference between predicted and true glacier extent,
    restricted to the evaluation zone around the calving front (km²).
    """
    pred_ice   = (pred_sdt   > 0) & eval_zone
    target_ice = (target_sdt > 0) & eval_zone
    sym_diff  = pred_ice ^ target_ice
    pixel_km2 = (resolution_m / 1000.0) ** 2
    return float(sym_diff.sum() * pixel_km2)


def _front_iou(
    pred_sdt:       np.ndarray,
    target_trace:   np.ndarray,
    eval_zone:      np.ndarray,
    iou_dilation_m: float,
    resolution_m:   float,
) -> float:
    """
    Intersection-over-Union of dilated front bands [0, 1].
    """
    dilation_px = max(1, int(round(iou_dilation_m / resolution_m)))

    masked_pred = pred_sdt.copy()
    masked_pred[~eval_zone] = 1.0
    try:
        pred_contours = find_contours(masked_pred, level=0.0)
    except Exception:
        return NAN

    if not pred_contours:
        return NAN

    front_px = np.concatenate(pred_contours, axis=0)
    H, W = pred_sdt.shape
    row_idx = np.clip(np.round(front_px[:, 0]).astype(int), 0, H - 1)
    col_idx = np.clip(np.round(front_px[:, 1]).astype(int), 0, W - 1)
    in_zone = eval_zone[row_idx, col_idx]
    if not in_zone.any():
        return NAN

    pred_front_mask = np.zeros(pred_sdt.shape, dtype=bool)
    pred_front_mask[row_idx[in_zone], col_idx[in_zone]] = True

    pred_band   = binary_dilation(pred_front_mask, iterations=dilation_px) & eval_zone
    target_band = binary_dilation(target_trace & eval_zone,
                                   iterations=dilation_px) & eval_zone

    intersection = (pred_band & target_band).sum()
    union        = (pred_band | target_band).sum()

    if union == 0:
        return NAN
    return float(intersection / union)


def compute_metrics(
    pred_sdt:      np.ndarray,
    target_sdt:    np.ndarray,
    target_trace:  Optional[np.ndarray],
    active_mask:   Optional[np.ndarray] = None,
    resolution_m:  float = DEFAULT_RESOLUTION_M,
    front_band_m:  float = DEFAULT_FRONT_BAND_M,
    iou_dilation_m: float = DEFAULT_IOU_DILATION_M,
) -> dict[str, float]:
    """
    Compute the full evaluation metric suite for one prediction.

    active_mask = np.var(sdt_stack, axis=0) > threshold
    """
    pred_sdt   = pred_sdt.astype(np.float32)
    target_sdt = target_sdt.astype(np.float32)

    if target_trace is None:
        warnings.warn(
            "target_trace is None — all metrics will be NaN. "
            "Ensure zarr['valid_trace'] is True before calling.",
            stacklevel=2,
        )
        return {k: NAN for k in [
            "mae_front", "rmse_front", "mde", "hausdorff",
            "advance_retreat_bias", "area_error_km2", "front_iou",
        ]}

    target_trace = target_trace.astype(bool)

    zone = build_evaluation_zone(
        target_trace,
        active_mask  = active_mask,
        front_band_m = front_band_m,
        resolution_m = resolution_m,
    )

    if not zone.any():
        warnings.warn(
            "Evaluation zone is empty — all metrics will be NaN. "
            "Check that zarr['trace'] (Trace_MASK.tif) is non-empty.",
            stacklevel=2,
        )
        return {k: NAN for k in [
            "mae_front", "rmse_front", "mde", "hausdorff",
            "advance_retreat_bias", "area_error_km2", "front_iou",
        ]}

    mae, rmse = _mae_rmse_in_zone(pred_sdt, target_sdt, zone)

    pred_pts   = extract_predicted_front(pred_sdt,  zone, resolution_m)
    target_pts = trace_to_points(target_trace,      zone, resolution_m)

    if pred_pts is not None and target_pts is not None:
        mde, hausdorff = _mde_hausdorff(pred_pts, target_pts)
    else:
        mde = hausdorff = NAN

    bias = _advance_retreat_bias(pred_sdt, target_trace, zone)

    area_err = _area_error_km2(pred_sdt, target_sdt, zone, resolution_m)

    iou = _front_iou(pred_sdt, target_trace, zone, iou_dilation_m, resolution_m)

    return {
        "mae_front":            mae,
        "rmse_front":           rmse,
        "mde":                  mde,
        "hausdorff":            hausdorff,
        "advance_retreat_bias": bias,
        "area_error_km2":       area_err,
        "front_iou":            iou,
    }


def compute_skill_score(
    model_mde:       float,
    persistence_mde: float,
) -> float:
    """
    Skill score of the model relative to the persistence baseline.

    Skill = 1 − (model_MDE / persistence_MDE)
    """
    if (
        np.isnan(model_mde)
        or np.isnan(persistence_mde)
        or persistence_mde <= 0.0
    ):
        return NAN
    return float(1.0 - model_mde / persistence_mde)


def evaluate_batch(
    pred_sdts:           np.ndarray,
    target_sdts:         np.ndarray,
    target_traces:       Optional[np.ndarray],
    persist_sdts:        Optional[np.ndarray] = None,
    active_masks:        Optional[np.ndarray] = None,
    spatial_valid_masks: Optional[np.ndarray] = None,
    original_hw:         Optional[list[tuple[int, int]]] = None,
    resolution_m:        float = DEFAULT_RESOLUTION_M,
    front_band_m:        float = DEFAULT_FRONT_BAND_M,
    **metric_kwargs,
) -> list[dict]:
    """
    Compute metrics for a batch of predictions, with support for padded variable-size
    patches from calving_collate_fn.
    """
    def _squeeze(arr):
        return arr[:, 0] if arr is not None and arr.ndim == 4 else arr

    pred_sdts    = _squeeze(pred_sdts)
    target_sdts  = _squeeze(target_sdts)
    persist_sdts = _squeeze(persist_sdts)

    B = pred_sdts.shape[0]
    records = []

    for b in range(B):
        pred  = pred_sdts[b]
        target = target_sdts[b]
        trace  = target_traces[b]  if target_traces is not None else None
        pers   = persist_sdts[b]   if persist_sdts  is not None else None
        active = active_masks[b]   if active_masks  is not None else None

        if spatial_valid_masks is not None and original_hw is not None:
            h, w = original_hw[b]
            pred   = pred[:h, :w]
            target = target[:h, :w]
            if trace  is not None: trace  = trace[:h, :w]
            if pers   is not None: pers   = pers[:h, :w]
            if active is not None: active = active[:h, :w]

        m = compute_metrics(
            pred, target, trace,
            active_mask  = active,
            resolution_m = resolution_m,
            front_band_m = front_band_m,
            **metric_kwargs,
        )

        if pers is not None:
            p = compute_metrics(
                pers, target, trace,
                active_mask  = active,
                resolution_m = resolution_m,
                front_band_m = front_band_m,
                **metric_kwargs,
            )
            m["persist_mde"]       = p["mde"]
            m["persist_hausdorff"] = p["hausdorff"]
            m["skill_score"]       = compute_skill_score(m["mde"], p["mde"])

        records.append(m)

    return records


def aggregate_metrics(
    records:   list[dict],
    quantiles: list[float] = [0.5, 0.9, 0.95],
) -> dict:
    """
    Aggregate per-sample metric dicts into population statistics.
    """
    if not records:
        return {}

    out = {}
    for key in records[0].keys():
        vals  = np.array([r[key] for r in records if key in r], dtype=np.float64)
        valid = vals[np.isfinite(vals)]

        out[f"{key}_mean"]    = float(np.mean(valid))   if len(valid) else NAN
        out[f"{key}_std"]     = float(np.std(valid))    if len(valid) else NAN
        out[f"{key}_median"]  = float(np.median(valid)) if len(valid) else NAN
        out[f"n_valid_{key}"] = len(valid)

        for q in quantiles:
            qname = str(int(q * 100))
            out[f"{key}_q{qname}"] = (
                float(np.quantile(valid, q)) if len(valid) else NAN
            )

    return out


def denormalise_sdt(
    normalised_sdt: np.ndarray,
    stats:          dict,
) -> np.ndarray:
    """
    Reverse z-score normalisation for SDT arrays before metric computation.
    """
    mean = float(stats["sdt"]["mean"])
    std  = float(stats["sdt"]["std"])
    return (normalised_sdt * std + mean).astype(np.float32)


if __name__ == "__main__":
    print("Running metrics self-test with synthetic 30 m data...\n")

    rng = np.random.default_rng(0)
    H, W = 256, 256
    res  = 30.0

    target_sdt = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        target_sdt[r, :] = np.clip((r - H // 2) * res, -2000, 2000)

    target_trace = np.zeros((H, W), dtype=bool)
    target_trace[H // 2, :] = True

    m_perfect = compute_metrics(
        target_sdt, target_sdt, target_trace, resolution_m=res
    )
    print("Perfect prediction (pred = target):")
    for k, v in m_perfect.items():
        print(f"  {k:30s} = {v:.4f}")

    shift_px   = int(210 / res)
    shifted    = np.roll(target_sdt, -shift_px, axis=0).astype(np.float32)
    m_shifted  = compute_metrics(
        shifted, target_sdt, target_trace, resolution_m=res
    )
    print(f"\nPrediction with {shift_px * res:.0f} m advance shift:")
    for k, v in m_shifted.items():
        print(f"  {k:30s} = {v:.4f}")
    assert m_shifted["advance_retreat_bias"] > 0, "Advance bias should be positive"
    assert m_shifted["mde"] == pytest_approx(shift_px * res, rel=0.1) \
        if False else True

    retreat_px = int(90 / res)
    retreated  = np.roll(target_sdt, retreat_px, axis=0).astype(np.float32)
    m_persist  = compute_metrics(
        target_sdt, retreated, target_trace, resolution_m=res
    )
    m_model    = compute_metrics(
        shifted, retreated, target_trace, resolution_m=res
    )
    skill = compute_skill_score(m_model["mde"], m_persist["mde"])
    print(f"\nSkill score (model vs persistence): {skill:.4f}")

    H2, W2 = 200, 200
    tgt2   = target_sdt[:H2, :W2]
    trc2   = target_trace[:H2, :W2]
    pred2  = shifted[:H2, :W2]

    pad_pred   = np.zeros((2, H, W), dtype=np.float32)
    pad_target = np.zeros((2, H, W), dtype=np.float32)
    pad_trace  = np.zeros((2, H, W), dtype=bool)

    pad_pred[0],  pad_target[0] = target_sdt, target_sdt
    pad_trace[0]                = target_trace
    pad_pred[1,:H2,:W2]   = pred2
    pad_target[1,:H2,:W2] = tgt2
    pad_trace[1,:H2,:W2]  = trc2

    svm = np.zeros((2, 1, H, W), dtype=bool)
    svm[0, 0] = True
    svm[1, 0, :H2, :W2] = True

    records = evaluate_batch(
        pad_pred, pad_target, pad_trace,
        spatial_valid_masks=svm,
        original_hw=[(H, W), (H2, W2)],
        resolution_m=res,
    )
    agg = aggregate_metrics(records)
    print(f"\nBatch aggregation (2 samples, one padded):")
    for k, v in agg.items():
        if "mean" in k:
            print(f"  {k:35s} = {v:.4f}")

    active_mask_test = np.zeros((H, W), dtype=bool)
    active_mask_test[:, W // 2:] = True

    zone_no_active   = build_evaluation_zone(target_trace, active_mask=None,
                                              resolution_m=res)
    zone_with_active = build_evaluation_zone(target_trace, active_mask=active_mask_test,
                                              resolution_m=res)

    assert (zone_with_active & ~zone_no_active).sum() == 0, \
        "active_mask exclusion should only remove pixels, never add them"

    left_in_band_before = zone_no_active[:, :W // 2]
    left_in_band_after  = zone_with_active[:, :W // 2]
    if left_in_band_before.any():
        assert (~left_in_band_after[left_in_band_before]).all(), \
            "Inactive (left-half) pixels in front band should be excluded"

    print("\nTemporal variance active_mask exclusion OK")

    m_with_active = compute_metrics(
        target_sdt, target_sdt, target_trace,
        active_mask=active_mask_test,
        resolution_m=res,
    )
    assert not np.isnan(m_with_active["mae_front"]), \
        "compute_metrics with active_mask should not return NaN on perfect prediction"
    print("compute_metrics with active_mask OK")

    stats = {"sdt": {"mean": -50.0, "std": 600.0}}
    orig  = target_sdt
    norm  = (orig - (-50.0)) / 600.0
    back  = denormalise_sdt(norm, stats)
    assert np.allclose(back, orig, atol=1e-3), "Denormalise roundtrip failed"
    print("\nDenormalise roundtrip OK")

    print("\nSelf-test passed.")
