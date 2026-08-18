"""Supervised image-domain losses.

Three flavours:

- ``mse_l1_loss`` — MSE + L1 (PWUS literature standard).
- ``mse_l1_ssim_loss`` — F3 fix added after first B1 smoke plateau'd: combines
  MSE (low-freq fidelity), L1 (sparsity), and (1 − SSIM) (local structure).
  Defaults: 0.7 / 0.2 / 0.1.
- ``multi_target_loss`` — for propuesta B2.

SSIM is implemented from scratch (no external dep) via 11×11 Gaussian-windowed
local statistics + standard constants.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────
# SSIM (custom — no torchmetrics / pytorch_msssim dependency)
# ──────────────────────────────────────────────────────────────────


def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5,
                     device: torch.device | str = "cpu",
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    k2d = g[:, None] * g[None, :]
    return k2d.unsqueeze(0).unsqueeze(0)              # (1, 1, K, K)


def ssim_index(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Mean SSIM over the batch. Inputs ``(B, 1, H, W)``."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    k = _gaussian_kernel(kernel_size, sigma, device=pred.device, dtype=pred.dtype)
    pad = kernel_size // 2
    mu1 = F.conv2d(pred, k, padding=pad)
    mu2 = F.conv2d(target, k, padding=pad)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, k, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(target * target, k, padding=pad) - mu2_sq
    sigma12 = F.conv2d(pred * target, k, padding=pad) - mu12
    numer = (2 * mu12 + C1) * (2 * sigma12 + C2)
    denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (numer / denom).mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, **kw) -> torch.Tensor:
    """``1 − SSIM`` (a loss to minimise)."""
    return 1.0 - ssim_index(pred, target, **kw)


# ──────────────────────────────────────────────────────────────────
# Combined supervised losses
# ──────────────────────────────────────────────────────────────────


def mse_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """``MSE(pred, target) + l1_weight · L1(pred, target)``.

    Returns
    -------
    loss : scalar Tensor
    breakdown : dict with float components ``mse``, ``l1``, ``total``
    """
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    total = mse + l1_weight * l1
    return total, {
        "mse": float(mse.item()),
        "l1": float(l1.item()),
        "total": float(total.item()),
    }


def mse_l1_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 0.7,
    l1_weight: float = 0.2,
    ssim_weight: float = 0.1,
    data_range: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Combined loss: ``α·MSE + β·L1 + γ·(1 − SSIM)``.

    F3 fix for propuesta B1: SSIM penalises local structural mismatch even when
    the global MSE is satisfied by predicting near-constant images — exactly
    the failure mode observed in run 184971 (B1 collapsed to x_pred_std~0.001
    while MSE plateau'd at ~0.0025).

    ``data_range`` should be the target's dynamic range (≈ 1.0 for the
    envelope+log-compressed targets produced by F1).
    """
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    ssim_l = ssim_loss(pred, target, data_range=data_range)
    total = mse_weight * mse + l1_weight * l1 + ssim_weight * ssim_l
    return total, {
        "mse": float(mse.item()),
        "l1": float(l1.item()),
        "ssim_loss": float(ssim_l.item()),
        "total": float(total.item()),
    }


def multi_target_loss(
    pred: torch.Tensor,
    targets: dict,                       # {"schiffner": Tensor, "das75": Tensor, "mv": Tensor, ...}
    weights: dict,                       # {"schiffner": 0.6, "das75": 0.3, "mv": 0.1, ...}
    base_loss: str = "mse",              # "mse" | "l1" | "mse_l1"
    l1_weight: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Weighted sum of supervised losses against multiple targets (B2)."""
    if set(targets) != set(weights):
        raise ValueError(f"targets keys {set(targets)} != weights keys {set(weights)}")

    total = 0.0 if isinstance(0.0, torch.Tensor) else None
    breakdown: dict = {}
    accum = None
    for k, w in weights.items():
        t = targets[k]
        if base_loss == "mse":
            l = F.mse_loss(pred, t)
        elif base_loss == "l1":
            l = F.l1_loss(pred, t)
        elif base_loss == "mse_l1":
            l = F.mse_loss(pred, t) + l1_weight * F.l1_loss(pred, t)
        else:
            raise ValueError(f"unknown base_loss {base_loss!r}")
        breakdown[f"loss_{k}"] = float(l.item())
        weighted = float(w) * l
        accum = weighted if accum is None else (accum + weighted)
    breakdown["total"] = float(accum.item())
    return accum, breakdown
