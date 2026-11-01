"""
Training script for the CalvingDB calving front prediction benchmark.
It reads a YAML config, builds the model and data loaders, and runs training with
validation, checkpointing, and early stopping.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import CalvingFrontDataset, calving_collate_fn
from losses import CalvingLoss, build_loss, trace_proximity_weight, _masked_mean

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def set_seed(seed: int) -> None:
    """
    Fix all random seeds for reproducible training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_config(path: str | Path) -> dict:
    """
    Load a YAML config file and return it as a nested dict.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    required_sections = {"data", "model", "loss", "optimiser",
                         "scheduler", "training", "paths"}
    missing = required_sections - set(cfg.keys())
    if missing:
        raise KeyError(
            f"Config file {path} is missing required sections: {missing}"
        )
    return cfg


def build_model(cfg: dict) -> nn.Module:
    """
    Construct the model from the config.
    """
    from models.convlstm import CalvingUNet
    from models.predrnn import CalvingPredRNN
    from models.simvp import CalvingSimVP
    from models.phydnet import CalvingPhyDNet
    from models.stgcn import CalvingSTGCN
    from models.earthformer import CalvingEarthFormer

    m = cfg["model"]

    kwargs = dict(
        in_channels   = m["K"] * m["C_total"],
        base_channels = m["base_channels"],
        climate_dim   = m["climate_dim"],
        tau_dim       = m["tau_dim"],
        num_layers    = m.get("num_layers", 3),
    )

    if m["name"] == "CalvingUNet":
        return CalvingUNet(**kwargs)

    if m["name"] == "CalvingPredRNN":
        return CalvingPredRNN(**kwargs)

    if m["name"] == "CalvingSimVP":
        return CalvingSimVP(
            **{k: v for k, v in kwargs.items() if k != "num_layers"},
            num_layers         = m.get("num_layers", 4),
            translator_kernels = tuple(m.get("translator_kernels", [3, 5, 7])),
        )

    if m["name"] == "CalvingPhyDNet":
        return CalvingPhyDNet(
            **{k: v for k, v in kwargs.items() if k != "num_layers"},
            num_layers    = m.get("num_layers", 1),
            n_moments     = m.get("n_moments", 4),
            moment_kernel = m.get("moment_kernel", 7),
        )

    if m["name"] == "CalvingSTGCN":
        return CalvingSTGCN(
            **{k: v for k, v in kwargs.items() if k != "num_layers"},
            num_layers = m.get("num_layers", 2),
            kt         = m.get("kt", 3),
        )

    if m["name"] == "CalvingEarthFormer":
        return CalvingEarthFormer(
            **{k: v for k, v in kwargs.items() if k != "num_layers"},
            num_layers  = m.get("num_layers", 4),
            num_heads   = m.get("num_heads", 4),
            window_size = tuple(m.get("window_size", [5, 8, 8])),
            mlp_ratio   = m.get("mlp_ratio", 4.0),
            drop        = m.get("drop", 0.0),
            attn_drop   = m.get("attn_drop", 0.0),
        )

    raise ValueError(f"Unknown model name: {m['name']}")


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from the config.
    """
    d          = cfg["data"]
    splits_dir = Path(d["splits_dir"])
    shared     = dict(
        stats_path = d["stats_path"],
        modalities = d.get("modalities"),
    )

    train_ds = CalvingFrontDataset(
        index_path = splits_dir / "train.json",
        augment    = d.get("augment_train", True),
        **shared,
    )
    val_ds = CalvingFrontDataset(
        index_path = splits_dir / "val.json",
        augment    = False,
        **shared,
    )

    n_workers = d["num_workers"]
    loader_kwargs = dict(
        batch_size         = d["batch_size"],
        num_workers        = n_workers,
        pin_memory         = d.get("pin_memory", True),
        drop_last          = False,
        persistent_workers = n_workers > 0,
        prefetch_factor    = 4 if n_workers > 0 else None,
        collate_fn         = calving_collate_fn,
    )

    train_loader = DataLoader(
        train_ds, shuffle=True,  **loader_kwargs
    )
    val_loader   = DataLoader(
        val_ds,   shuffle=False, **loader_kwargs
    )

    return train_loader, val_loader


def build_optimiser(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    o = cfg["optimiser"]
    name = o["name"]

    if name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr           = o["lr"],
            weight_decay = o.get("weight_decay", 1e-4),
            betas        = tuple(o.get("betas", [0.9, 0.999])),
        )
    if name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr           = o["lr"],
            weight_decay = o.get("weight_decay", 0.0),
            betas        = tuple(o.get("betas", [0.9, 0.999])),
        )
    if name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr       = o["lr"],
            momentum = o.get("momentum", 0.9),
            weight_decay = o.get("weight_decay", 1e-4),
        )
    raise ValueError(f"Unknown optimiser: {name}")


def build_scheduler(
    optimiser: torch.optim.Optimizer,
    cfg:       dict,
    steps_per_epoch: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    s    = cfg.get("scheduler", {})
    name = s.get("name", "none")

    if name == "none" or name is None:
        return None

    if name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max   = s.get("T_max", cfg["training"]["n_epochs"]),
            eta_min = s.get("eta_min", 1e-6),
        )
    if name == "OneCycleLR":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimiser,
            max_lr        = s.get("max_lr", cfg["optimiser"]["lr"] * 10),
            steps_per_epoch = steps_per_epoch,
            epochs        = cfg["training"]["n_epochs"],
            pct_start     = s.get("pct_start", 0.3),
        )
    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode     = "min",
            factor   = s.get("factor", 0.5),
            patience = s.get("patience", 10),
            min_lr   = s.get("min_lr", 1e-6),
        )
    if name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimiser,
            step_size = s.get("step_size", 20),
            gamma     = s.get("gamma", 0.5),
        )

    if name == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimiser,
            T_0     = s.get("T_0", 50),
            T_mult  = s.get("T_mult", 1),
            eta_min = s.get("eta_min", 1e-6),
        )

    raise ValueError(f"Unknown scheduler: {name}")


def save_checkpoint(
    path:      Path,
    epoch:     int,
    model:     nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler,
    scaler:    GradScaler,
    val_loss:  float,
    cfg:       dict,
    is_best:   bool = False,
) -> None:
    """
    Save a full training checkpoint.
    """
    state = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "scaler":    scaler.state_dict(),
        "val_loss":  val_loss,
        "config":    cfg,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    torch.save(state, path)

    if is_best:
        shutil.copyfile(path, path.parent / "best.pt")


def load_checkpoint(
    path:      Path,
    model:     nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler,
    scaler:    GradScaler,
    device:    torch.device,
) -> tuple[int, float]:
    """
    Resume training from a checkpoint.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimiser.load_state_dict(ckpt["optimiser"])
    scaler.load_state_dict(ckpt["scaler"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("val_loss", float("inf"))
    print(f"Resumed from epoch {ckpt['epoch']}  (val_loss={best_val_loss:.4f})")
    return start_epoch, best_val_loss


def train_one_epoch(
    model:         nn.Module,
    loader:        DataLoader,
    optimiser:     torch.optim.Optimizer,
    criterion:     CalvingLoss,
    scaler:        GradScaler,
    device:        torch.device,
    grad_clip:     float,
    use_amp:       bool,
    scheduler_step_every_batch: bool = False,
    scheduler     = None,
    epoch:         int = 0,
    writer:        Optional[SummaryWriter] = None,
    global_step:   int = 0,
    disp_scale:    float = 1.0,
) -> tuple[dict, int]:
    """
    Run one full pass over the training DataLoader.
    """
    model.train()

    totals   = {}
    n_batches = 0
    t0        = time.time()

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", unit="batch", leave=False)

    for batch in pbar:
        batch = {
            k: v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        target_displacement = batch["target_displacement"]
        last_input_sdt      = batch["last_input_sdt"]
        target_abs          = batch["target"]
        mask   = batch["spatial_valid_mask"]

        optimiser.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            pred_displacement = model(batch)
            losses = criterion(
                pred_displacement,
                target_displacement,
                input_trace = batch.get("input_trace"),
                mask        = batch.get("spatial_valid_mask"),
            )

        scaler.scale(losses["loss"]).backward()

        scaler.unscale_(optimiser)
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=grad_clip
        )

        scaler.step(optimiser)
        scaler.update()

        if scheduler_step_every_batch and scheduler is not None:
            scheduler.step()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n_batches += 1
        global_step += 1

        pbar.set_postfix({
            "loss": f"{losses['loss'].item():.4f}",
            "grad": f"{grad_norm:.3f}",
        })

        if global_step % 50 == 0:
            if writer is not None:
                for k, v in losses.items():
                    writer.add_scalar(f"train_batch/{k}", v.item(), global_step)
                writer.add_scalar("train_batch/grad_norm", grad_norm, global_step)
                writer.add_scalar(
                    "train_batch/lr",
                    optimiser.param_groups[0]["lr"],
                    global_step,
                )

            wandb_log_batch(
                {k: v.item() for k, v in losses.items()},
                float(grad_norm),
                optimiser.param_groups[0]["lr"],
                global_step,
            )

            with torch.no_grad():
                trace = batch.get("input_trace")
                if trace is not None and trace.any():
                    pred_abs    = pred_displacement.detach() * disp_scale + last_input_sdt
                    persist_abs = last_input_sdt
                    w_skill = trace_proximity_weight(
                        trace.to(device), decay_pixels=10.0, min_weight=0.0,
                        mask=batch.get("spatial_valid_mask"),
                    )
                    vmask       = batch.get("spatial_valid_mask")
                    model_err   = _masked_mean(
                        w_skill * torch.abs(pred_abs - target_abs), vmask
                    )
                    persist_err = _masked_mean(
                        w_skill * torch.abs(persist_abs - target_abs), vmask
                    )
                    proxy_skill = 1.0 - model_err / persist_err.clamp(min=1e-3)
                    if writer is not None:
                        writer.add_scalar(
                            "train_batch/proxy_skill_vs_persistence",
                            proxy_skill.item(), global_step,
                        )
                        writer.add_scalar(
                            "train_batch/model_front_err",
                            model_err.item(), global_step,
                        )
                        writer.add_scalar(
                            "train_batch/persist_front_err",
                            persist_err.item(), global_step,
                        )
                    if _WANDB_AVAILABLE and wandb.run:
                        wandb.log({
                            "batch/proxy_skill_vs_persistence": proxy_skill.item(),
                            "batch/model_front_err":            model_err.item(),
                            "batch/persist_front_err":          persist_err.item(),
                        }, step=global_step)

    elapsed = time.time() - t0
    means   = {k: v / max(n_batches, 1) for k, v in totals.items()}
    means["epoch_time_s"] = elapsed

    return means, global_step


@torch.no_grad()
def validate(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  CalvingLoss,
    device:     torch.device,
    use_amp:    bool,
    epoch:      int = 0,
    disp_scale: float = 1.0,
) -> dict:
    """
    Run one full pass over the validation DataLoader.
    """
    model.eval()

    totals       = {}
    n_batches    = 0
    skill_sum    = 0.0
    skill_count  = 0

    pbar = tqdm(loader, desc=f"Val   epoch {epoch}", unit="batch", leave=False)

    for batch in pbar:
        batch = {
            k: v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        target_displacement = batch["target_displacement"]
        last_input_sdt      = batch["last_input_sdt"]
        target_abs          = batch["target"]

        with autocast("cuda", enabled=use_amp):
            pred_displacement = model(batch)
            losses = criterion(
                pred_displacement,
                target_displacement,
                input_trace = batch.get("input_trace"),
                mask        = batch.get("spatial_valid_mask"),
            )

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n_batches += 1

        trace = batch.get("input_trace")
        if trace is not None and trace.any():
            pred_abs    = pred_displacement * disp_scale + last_input_sdt
            persist_abs = last_input_sdt
            w_skill = trace_proximity_weight(
                trace, decay_pixels=10.0, min_weight=0.0,
                mask=batch.get("spatial_valid_mask"),
            )
            vmask       = batch.get("spatial_valid_mask")
            model_err   = _masked_mean(
                w_skill * torch.abs(pred_abs - target_abs), vmask
            )
            persist_err = _masked_mean(
                w_skill * torch.abs(persist_abs - target_abs), vmask
            )
            skill_sum   += (1.0 - model_err / persist_err.clamp(min=1e-3)).item()
            skill_count += 1

        pbar.set_postfix({"loss": f"{losses['loss'].item():.4f}"})

    result = {k: v / max(n_batches, 1) for k, v in totals.items()}
    if skill_count > 0:
        result["val_proxy_skill_vs_persistence"] = skill_sum / skill_count
    return result


def find_lr(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: CalvingLoss,
    device:    torch.device,
    start_lr:  float = 1e-7,
    end_lr:    float = 1e-1,
    n_steps:   int   = 100,
) -> tuple[list[float], list[float]]:
    """
    Run a learning rate range test (Smith 2017).
    """
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimiser, gamma=(end_lr / start_lr) ** (1 / n_steps)
    )
    scaler    = GradScaler("cuda")

    lrs, losses = [], []
    best_loss   = float("inf")

    for i, batch in enumerate(loader):
        if i >= n_steps:
            break
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimiser.zero_grad(set_to_none=True)
        with autocast("cuda"):
            pred_displacement = model(batch)
            loss_d = criterion(
                pred_displacement,
                batch["target_displacement"],
                input_trace = batch.get("input_trace"),
                mask        = batch.get("spatial_valid_mask"),
            )
        scaler.scale(loss_d["loss"]).backward()
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()

        lr   = optimiser.param_groups[0]["lr"]
        loss = loss_d["loss"].item()

        smooth_loss = loss if not losses else 0.98 * losses[-1] + 0.02 * loss

        lrs.append(lr)
        losses.append(smooth_loss)

        if smooth_loss < best_loss:
            best_loss = smooth_loss
        if smooth_loss > 4 * best_loss:
            break

    return lrs, losses


class TrainingState:
    """
    Tracks training history, best validation loss, early stopping counter, and global
    step count.
    """

    def __init__(self, patience: int):
        self.best_val_loss  = float("inf")
        self.no_improve     = 0
        self.patience       = patience
        self.global_step    = 0
        self.history: list[dict] = []

    def update(self, epoch: int, train_m: dict, val_m: dict) -> bool:
        """
        Update state with the metrics from one epoch.
        """
        val_loss = val_m["loss"]
        improved = val_loss < self.best_val_loss

        if improved:
            self.best_val_loss = val_loss
            self.no_improve    = 0
        else:
            self.no_improve   += 1

        entry = {
            "epoch":    epoch,
            "improved": improved,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}":   v for k, v in val_m.items()},
        }
        self.history.append(entry)
        return improved

    @property
    def should_stop(self) -> bool:
        return self.no_improve >= self.patience

    def save_history(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


def setup_wandb(cfg: dict, resume_from: Optional[str]) -> bool:
    """
    Initialise a wandb run from the config.
    """
    wb = cfg.get("wandb", {})
    if not wb.get("enabled", False) or not _WANDB_AVAILABLE:
        return False

    wandb.init(
        project  = wb.get("project", "calvingdb"),
        entity   = wb.get("entity") or None,
        name     = wb.get("run_name") or None,
        tags     = wb.get("tags", []),
        notes    = wb.get("notes", ""),
        config   = cfg,
        resume   = "allow" if resume_from else None,
    )
    return True


def wandb_log_epoch(
    train_m:     dict,
    val_m:       dict,
    lr:          float,
    epoch:       int,
    global_step: int,
) -> None:
    """Log epoch-level metrics to wandb using global_step as the x-axis."""
    if not (_WANDB_AVAILABLE and wandb.run):
        return
    log = {"epoch": epoch, "lr": lr}
    for k, v in train_m.items():
        if k != "epoch_time_s":
            log[f"train/{k}"] = v
    for k, v in val_m.items():
        log[f"val/{k}"] = v
    wandb.log(log, step=global_step)


def wandb_log_batch(losses: dict, grad_norm: float, lr: float, step: int) -> None:
    """Log per-batch metrics to wandb (called every 50 steps)."""
    if not (_WANDB_AVAILABLE and wandb.run):
        return
    log = {"batch/grad_norm": grad_norm, "batch/lr": lr}
    for k, v in losses.items():
        log[f"batch/{k}"] = v
    wandb.log(log, step=step)


def log_epoch_metrics(
    writer:   SummaryWriter,
    train_m:  dict,
    val_m:    dict,
    lr:       float,
    epoch:    int,
) -> None:
    """Log all epoch-level metrics to TensorBoard."""
    for k, v in train_m.items():
        if k != "epoch_time_s":
            writer.add_scalar(f"train_epoch/{k}", v, epoch)

    for k, v in val_m.items():
        writer.add_scalar(f"val_epoch/{k}", v, epoch)

    writer.add_scalar("train_epoch/lr", lr, epoch)


def log_model_graph(
    writer: SummaryWriter,
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """
    Attempt to log the model graph to TensorBoard.
    """
    try:
        batch = next(iter(loader))
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        pass
    except Exception:
        pass


def print_epoch(
    epoch:    int,
    n_epochs: int,
    train_m:  dict,
    val_m:    dict,
    state:    TrainingState,
    lr:       float,
) -> None:
    improved_marker = "★" if state.history[-1]["improved"] else " "
    print(
        f"Epoch {epoch:4d}/{n_epochs}  "
        f"train={train_m['loss']:.4f}  "
        f"val={val_m['loss']:.4f}  "
        f"best={state.best_val_loss:.4f}  "
        f"lr={lr:.2e}  "
        f"patience={state.no_improve}/{state.patience}  "
        f"{improved_marker}"
    )

    if state.history[-1]["improved"]:
        components = [k for k in val_m if k.startswith("loss_")]
        if components:
            parts = "  ".join(f"{k[5:]}={val_m[k]:.4f}" for k in components)
            print(f"  ↳ {parts}")

    proxy = val_m.get("val_proxy_skill_vs_persistence", float("nan"))
    if proxy == proxy:
        print(f"  ↳ skill_vs_persist={proxy:.3f}")


def train(
    config_path:  str | Path = "configs/default.yaml",
    resume_from:  Optional[str | Path] = None,
) -> None:
    """
    Run the full training pipeline from a YAML config.
    """
    cfg = load_config(config_path)
    t   = cfg["training"]

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir  = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(config_path, ckpt_dir / "config.yaml")

    seed = t.get("seed", 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {cfg['model']['name']}")
    print(f"  Total parameters     : {n_params:,}")
    print(f"  Trainable parameters : {n_train:,}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"\nDataset")
    print(f"  Train : {len(train_loader.dataset):7d} samples  "
          f"{len(train_loader):5d} batches")
    print(f"  Val   : {len(val_loader.dataset):7d} samples  "
          f"{len(val_loader):5d} batches")

    _stats_path = Path(cfg["data"]["stats_path"])
    disp_scale  = 1.0
    if _stats_path.exists():
        with open(_stats_path) as _f:
            _stats = json.load(_f)
        _sdt_std  = float(_stats["sdt"]["std"])
        _disp_std = float(_stats.get("displacement", {}).get("std", _sdt_std))
        disp_scale = _disp_std / max(_sdt_std, 1e-6)

    criterion = build_loss(cfg["loss"])
    optimiser = build_optimiser(model, cfg)
    use_amp   = t.get("mixed_precision", True)
    scaler    = GradScaler("cuda", enabled=use_amp)

    scheduler_name             = cfg.get("scheduler", {}).get("name", "none")
    scheduler_step_every_batch = scheduler_name == "OneCycleLR"
    scheduler                  = build_scheduler(
        optimiser, cfg, steps_per_epoch=len(train_loader)
    )

    print(f"\nOptimiser : {cfg['optimiser']['name']}  "
          f"lr={cfg['optimiser']['lr']:.2e}")
    print(f"Scheduler : {scheduler_name}")
    print(f"Loss      : CalvingLoss  "
          f"decay_pixels={cfg['loss'].get('decay_pixels', 10.0)}  "
          f"min_weight={cfg['loss'].get('min_weight', 0.05)}")
    print(f"AMP       : {use_amp}")
    print(f"Grad clip : {t.get('grad_clip_norm', 1.0)}")

    writer = SummaryWriter(log_dir=str(log_dir))

    use_wandb = setup_wandb(cfg, resume_from)
    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)
        print(f"wandb    : run '{wandb.run.name}'  "
              f"project '{wandb.run.project}'")
    else:
        print("wandb    : disabled")

    state      = TrainingState(patience=t.get("early_stopping_patience", 20))
    start_epoch = 1

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        start_epoch, state.best_val_loss = load_checkpoint(
            resume_path, model, optimiser, scheduler, scaler, device
        )
        print(f"\nResuming from epoch {start_epoch}")

    print(f"\n{'='*65}")
    print(f"TRAINING  —  {t['n_epochs']} epochs  "
          f"(start={start_epoch}, "
          f"val_every={t.get('val_every', 1)}, "
          f"save_every={t.get('save_every', 10)})")
    print(f"{'='*65}\n")

    val_m    = {}
    t0_total = time.time()

    for epoch in range(start_epoch, t["n_epochs"] + 1):
        train_m, state.global_step = train_one_epoch(
            model        = model,
            loader       = train_loader,
            optimiser    = optimiser,
            criterion    = criterion,
            scaler       = scaler,
            device       = device,
            grad_clip    = t.get("grad_clip_norm", 1.0),
            use_amp      = use_amp,
            scheduler_step_every_batch = scheduler_step_every_batch,
            scheduler    = scheduler if scheduler_step_every_batch else None,
            epoch        = epoch,
            writer       = writer,
            global_step  = state.global_step,
            disp_scale   = disp_scale,
        )

        if scheduler is not None and not scheduler_step_every_batch:
            if isinstance(
                scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(val_m.get("loss", float("inf")))
            else:
                scheduler.step()

        current_lr = optimiser.param_groups[0]["lr"]

        should_validate = (
            epoch % t.get("val_every", 1) == 0
            or epoch == t["n_epochs"]
        )

        if should_validate:
            val_m    = validate(model, val_loader, criterion, device,
                                use_amp, epoch, disp_scale=disp_scale)
            improved = state.update(epoch, train_m, val_m)

            log_epoch_metrics(writer, train_m, val_m, current_lr, epoch)
            wandb_log_epoch(train_m, val_m, current_lr, epoch, state.global_step)

            print_epoch(epoch, t["n_epochs"], train_m, val_m, state, current_lr)

            if improved:
                save_checkpoint(
                    path      = ckpt_dir / f"epoch_{epoch:04d}.pt",
                    epoch     = epoch,
                    model     = model,
                    optimiser = optimiser,
                    scheduler = scheduler,
                    scaler    = scaler,
                    val_loss  = state.best_val_loss,
                    cfg       = cfg,
                    is_best   = True,
                )

        else:
            print(
                f"Epoch {epoch:4d}/{t['n_epochs']}  "
                f"train={train_m['loss']:.4f}  "
                f"lr={current_lr:.2e}"
            )

        save_every = t.get("save_every", 10)
        if epoch % save_every == 0:
            save_checkpoint(
                path      = ckpt_dir / f"epoch_{epoch:04d}.pt",
                epoch     = epoch,
                model     = model,
                optimiser = optimiser,
                scheduler = scheduler,
                scaler    = scaler,
                val_loss  = val_m.get("loss", float("nan")),
                cfg       = cfg,
                is_best   = False,
            )

        state.save_history(log_dir / "history.json")

        if state.should_stop:
            print(
                f"\nEarly stopping at epoch {epoch} — "
                f"no improvement for {state.patience} epochs."
            )
            break

    save_checkpoint(
        path      = ckpt_dir / "final.pt",
        epoch     = epoch,
        model     = model,
        optimiser = optimiser,
        scheduler = scheduler,
        scaler    = scaler,
        val_loss  = val_m.get("loss", float("nan")),
        cfg       = cfg,
        is_best   = False,
    )

    writer.close()
    if use_wandb:
        wandb.finish()

    total_time = time.time() - t0_total
    print(f"\n{'='*65}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"  Total time     : {total_time/3600:.2f} h")
    print(f"  Epochs run     : {epoch - start_epoch + 1}")
    print(f"  Best val loss  : {state.best_val_loss:.4f}")
    print(f"  Best checkpoint: {ckpt_dir / 'best.pt'}")
    print(f"  History        : {log_dir / 'history.json'}")
    print(f"{'='*65}\n")
    print("Next step:")
    print("  python evaluate.py --split val   # tune on val")
    print("  python evaluate.py --split test  # final numbers for paper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CalvingUNet for calving front SDT prediction."
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--resume", type=str,
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    parser.add_argument(
        "--find_lr", action="store_true",
        help="Run learning rate range test and exit without training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.find_lr:
        cfg    = load_config(args.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = build_model(cfg).to(device)
        set_seed(cfg["training"].get("seed", 42))

        train_loader, _ = build_dataloaders(cfg)
        criterion       = build_loss(cfg["loss"])

        print("Running learning rate range test...")
        lrs, losses = find_lr(model, train_loader, criterion, device)

        import csv
        out = Path(cfg["paths"]["log_dir"]) / "lr_finder.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lr", "loss"])
            writer.writerows(zip(lrs, losses))
        print(f"LR finder results written to {out}")
        print(f"Suggested LR range: {lrs[np.argmin(losses)]:.2e} "
              f"(minimum loss at index {np.argmin(losses)})")

    else:
        train(
            config_path = args.config,
            resume_from = args.resume,
        )
