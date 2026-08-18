"""Common training scaffold for unet_ag.

Each propuesta defines:
- ``build_model(args, geometry, grid) -> nn.Module``
- ``build_inputs(sample, geometry, grid, device) -> dict``
- ``forward_and_loss(model, inputs, target, args) -> (loss, breakdown, x_pred)``

and calls ``run_training`` with those callables plus the CLI args.

Logging:
- CSV per epoch with all loss components + ``x_pred.std()``.
- Checkpoint every ``save_every`` epochs + ``best.pt`` on val improvement.
- Resume from ``--resume``.

Validation:
- Computed every ``val_every`` epochs on the val split.
- Best metric = total val loss (lowest).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class TrainArgs:
    data_root: Path
    output_dir: Path
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1                     # phantoms per step (forward is per-phantom)
    save_every: int = 5
    val_every: int = 1
    grad_clip: float = 1.0
    device: str = "cuda"
    seed: int = 42
    max_phantoms_train: Optional[int] = None
    max_phantoms_val: Optional[int] = None
    smoke_steps: Optional[int] = None
    resume: bool = False


def parse_common_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_phantoms_train", type=int, default=None)
    parser.add_argument("--max_phantoms_val", type=int, default=None)
    parser.add_argument("--smoke_steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _write_csv_row(path: Path, row: dict) -> None:
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


def save_ckpt(model: nn.Module, opt: torch.optim.Optimizer, epoch: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
        },
        path,
    )


def load_ckpt(model: nn.Module, opt: torch.optim.Optimizer, path: Path) -> int:
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd["model"])
    opt.load_state_dict(sd["opt"])
    return int(sd.get("epoch", 0))


def run_training(
    args,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_iter_fn: Callable,        # () -> Iterable of (inputs, target) tuples
    val_iter_fn: Callable,          # () -> Iterable of (inputs, target) tuples (may be None)
    forward_loss_fn: Callable,      # (model, inputs, target, args) -> (loss, breakdown, x_pred)
) -> None:
    out = Path(args.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.csv"
    val_log_path = out / "val_log.csv"

    model.train()
    start_epoch = 0
    if args.resume:
        last = sorted((out / "checkpoints").glob("epoch_*.pt"))
        if last:
            start_epoch = load_ckpt(model, optimizer, last[-1]) + 1
            print(f"resumed from {last[-1]} at epoch {start_epoch}")

    best_val = float("inf")
    total_steps = 0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()
        running: dict = {}
        n_steps = 0
        for inputs, target in train_iter_fn():
            optimizer.zero_grad()
            loss, br, x_pred = forward_loss_fn(model, inputs, target, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            for k, v in br.items():
                running[k] = running.get(k, 0.0) + float(v)
            running["x_pred_std"] = running.get("x_pred_std", 0.0) + float(x_pred.detach().std().item())
            n_steps += 1
            total_steps += 1
            if args.smoke_steps and total_steps >= args.smoke_steps:
                break

        for k in running:
            running[k] /= max(n_steps, 1)
        row = {"epoch": epoch, "n_steps": n_steps, "epoch_time_s": time.perf_counter() - t0, **running}
        _write_csv_row(log_path, row)
        print(f"[ep {epoch:3d}] " + " ".join(f"{k}={v:.4g}" for k, v in row.items() if k not in ("epoch",)))

        if val_iter_fn is not None and (epoch % args.val_every == 0):
            model.eval()
            v_running: dict = {}
            v_steps = 0
            with torch.no_grad():
                for inputs, target in val_iter_fn():
                    _, vbr, _ = forward_loss_fn(model, inputs, target, args)
                    for k, v in vbr.items():
                        v_running[k] = v_running.get(k, 0.0) + float(v)
                    v_steps += 1
            for k in v_running:
                v_running[k] /= max(v_steps, 1)
            _write_csv_row(val_log_path, {"epoch": epoch, **{f"val_{k}": v for k, v in v_running.items()}})
            val_loss = v_running.get("total", v_running.get("mse", float("nan")))
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(model, optimizer, epoch, out / "checkpoints" / "best.pt")
            model.train()

        if (epoch + 1) % args.save_every == 0:
            save_ckpt(model, optimizer, epoch, out / "checkpoints" / f"epoch_{epoch:04d}.pt")

        if args.smoke_steps and total_steps >= args.smoke_steps:
            print(f"smoke_steps={args.smoke_steps} reached, exiting.")
            break

    save_ckpt(model, optimizer, args.epochs - 1, out / "checkpoints" / "final.pt")
