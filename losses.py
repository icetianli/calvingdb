"""
Loss function for calving front SDT displacement prediction.
It implements an L1 loss weighted by distance to the calving front trace.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def trace_proximity_weight(
    trace:        torch.Tensor,
    decay_pixels: float = 10.0,
    min_weight:   float = 0.0,
    mask:         Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute a spatial weight map that is highest at the calving front and decays
    exponentially with pixel distance from the trace.

    w(x) = (1 - min_weight) × exp(-dist(x) / decay_pixels) + min_weight
    """
    trace = trace.float()
    H, W  = trace.shape[-2], trace.shape[-1]

    max_dist_px = min(
        math.ceil(-math.log(min_weight + 1e-8) * decay_pixels),
        round(4 * decay_pixels),
        max(H, W),
    )

    dist_map = torch.where(
        trace > 0.5,
        torch.zeros_like(trace),
        torch.full_like(trace, float(max_dist_px)),
    )

    for step in range(1, max_dist_px + 1):
        reachable     = (dist_map < step).float()
        dilated       = F.max_pool2d(reachable, kernel_size=3, stride=1, padding=1)
        newly_reached = (dilated > 0) & (dist_map >= step)
        dist_map      = torch.where(
            newly_reached,
            torch.full_like(dist_map, float(step)),
            dist_map,
        )
        if mask is not None:
            remaining = (dist_map >= max_dist_px) & mask
        else:
            remaining = dist_map >= max_dist_px
        if not remaining.any():
            break

    w = (1.0 - min_weight) * torch.exp(-dist_map / (decay_pixels + 1e-8)) + min_weight

    if mask is not None:
        m     = mask.float()
        denom = (w * m).sum(dim=(-2, -1), keepdim=True) + 1e-8
        count = m.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
        w     = w * count / denom
    else:
        w = w / (w.mean(dim=(-2, -1), keepdim=True) + 1e-8)

    return w


def _masked_mean(
    tensor: torch.Tensor,
    mask:   Optional[torch.Tensor],
) -> torch.Tensor:
    """Mean over valid pixels only; falls back to .mean() when mask is None."""
    if mask is None:
        return tensor.mean()
    m = mask.float()
    return (tensor * m).sum() / (m.sum() + 1e-8)


class CalvingLoss(nn.Module):
    """
    Front-proximity-weighted L1 loss on the SDT displacement field.

    L = mean_valid[ w(x) × |pred_disp(x) − target_disp(x)| ]
    displacement_metres = displacement_norm × disp_std
    """

    def __init__(
        self,
        decay_pixels: float = 10.0,
        min_weight:   float = 0.0,
    ):
        super().__init__()
        self.decay_pixels = decay_pixels
        self.min_weight   = min_weight

    def forward(
        self,
        pred_displacement:   torch.Tensor,
        target_displacement: torch.Tensor,
        input_trace:         Optional[torch.Tensor] = None,
        mask:                Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute front-proximity-weighted L1 loss on displacement.
        """
        if input_trace is not None and input_trace.any():
            w = trace_proximity_weight(
                input_trace,
                decay_pixels = self.decay_pixels,
                min_weight   = self.min_weight,
                mask         = mask,
            )
        else:
            warnings.warn(
                "input_trace is None or empty — using uniform weights. "
                "Loss will not be focused on the calving front.",
                stacklevel=2,
            )
            w = torch.ones_like(pred_displacement)
            if mask is not None:
                w = w * mask.float()

        loss = _masked_mean(
            w * torch.abs(pred_displacement - target_displacement), mask
        )

        return {"loss": loss}


def build_loss(cfg: dict) -> CalvingLoss:
    """
    Construct CalvingLoss from a config dict (from configs/*.yaml).
    """
    return CalvingLoss(
        decay_pixels = cfg.get("decay_pixels", 10.0),
        min_weight   = cfg.get("min_weight",   0.0),
    )


if __name__ == "__main__":
    import warnings as _warnings

    print("Running losses self-test (displacement formulation)...\n")
    torch.manual_seed(0)

    B, H, W = 2, 64, 64

    true_disp = torch.zeros(B, 1, H, W)
    true_disp[:, :, H // 2 :, :] = -0.25

    trace = torch.zeros(B, 1, H, W)
    trace[:, :, H // 2, :] = 1.0

    mask = torch.ones(B, 1, H, W, dtype=torch.bool)
    mask[:, :, :, -4:] = False

    criterion = CalvingLoss(decay_pixels=4.0)

    def check(name: str, cond: bool, val: float | None = None) -> None:
        tag = "PASS" if cond else "FAIL"
        msg = f"[{tag}]  {name}"
        if val is not None:
            msg += f"  (value={val:.6f})"
        print(msg)

    out = criterion(true_disp.clone(), true_disp, input_trace=trace, mask=mask)
    check("1. Perfect pred → loss near 0",
          out["loss"].item() < 1e-6, out["loss"].item())

    persist_pred = torch.zeros_like(true_disp)
    out_persist  = criterion(persist_pred, true_disp, input_trace=trace, mask=mask)
    check("2. Persistence (zero pred) with nonzero disp → loss > 0",
          out_persist["loss"].item() > 0, out_persist["loss"].item())

    near_correct = true_disp + 0.02 * torch.randn_like(true_disp)
    out_model    = criterion(near_correct, true_disp, input_trace=trace, mask=mask)
    check("3. Near-correct model loss < persistence loss",
          out_model["loss"].item() < out_persist["loss"].item(),
          out_model["loss"].item())

    out_with = criterion(near_correct, true_disp, input_trace=trace)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        out_none = criterion(near_correct, true_disp, input_trace=None)
    check("4. With trace → valid positive loss",
          not torch.isnan(out_with["loss"]) and out_with["loss"].item() > 0,
          out_with["loss"].item())
    check("4. None trace → valid loss + warning containing 'uniform'",
          len(caught) >= 1
          and "uniform" in str(caught[0].message).lower()
          and not torch.isnan(out_none["loss"])
          and out_none["loss"].item() > 0,
          out_none["loss"].item())

    out_nomask = criterion(persist_pred, true_disp, input_trace=trace)
    out_masked = criterion(persist_pred, true_disp, input_trace=trace, mask=mask)
    check("5. No mask → positive loss",  out_nomask["loss"].item() > 0, out_nomask["loss"].item())
    check("5. With mask → positive loss", out_masked["loss"].item() > 0, out_masked["loss"].item())

    pred_grad = true_disp + 0.3 * torch.randn_like(true_disp)
    pred_grad.requires_grad_(True)
    out = criterion(pred_grad, true_disp, input_trace=trace, mask=mask)
    out["loss"].backward()
    grad_ok = (
        pred_grad.grad is not None
        and not torch.isnan(pred_grad.grad).any()
        and not torch.isinf(pred_grad.grad).any()
    )
    check("6. Backward → non-None, non-NaN, non-Inf grads", grad_ok,
          pred_grad.grad.norm().item() if pred_grad.grad is not None else float("nan"))

    cfg_dict = {"decay_pixels": 8.0, "min_weight": 0.1}
    loss2    = build_loss(cfg_dict)
    out2     = loss2(pred_grad.detach(), true_disp, input_trace=trace)
    check("7. build_loss from dict → valid output",
          "loss" in out2 and not torch.isnan(out2["loss"]))
    check("7. build_loss → only 'loss' key in return dict",
          set(out2.keys()) == {"loss"})

    legacy_cfg = {
        "decay_pixels": 6.0, "min_weight": 0.05,
        "lambda_recon": 1.0, "lambda_focal": 0.3,
        "lambda_sign": 0.2, "decay_factor": 0.2,
    }
    loss_legacy = build_loss(legacy_cfg)
    out_legacy  = loss_legacy(pred_grad.detach(), true_disp, input_trace=trace)
    check("   build_loss ignores legacy config keys",
          "loss" in out_legacy and not torch.isnan(out_legacy["loss"]))

    print("\nSelf-test complete.")
