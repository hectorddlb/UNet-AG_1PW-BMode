"""
Visualization utilities for B-mode ultrasound images.

All display functions use the log-compressed convention:
    ``20 * log10( |image| / max(|image|) )``
clipped to a given dynamic range.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def to_bmode_db(image: Tensor, dynamic_range_dB: float = 60.0) -> np.ndarray:
    """Convert a complex image to log-compressed B-mode (dB).

    Parameters
    ----------
    image : Tensor
        Complex-valued image (N_z, N_x).
    dynamic_range_dB : float
        Dynamic range in dB.

    Returns
    -------
    np.ndarray
        Real-valued image in dB, clipped to ``[-dynamic_range_dB, 0]``.
    """
    env = image.abs()
    env_max = env.max()
    if env_max == 0:
        return np.full(image.shape, -dynamic_range_dB)
    db = 20.0 * torch.log10(env / env_max + 1e-30)
    db = torch.clamp(db, min=-dynamic_range_dB, max=0.0)
    return db.numpy()


def plot_bmode(
    image: Tensor,
    positions_x: Tensor,
    positions_z: Tensor,
    *,
    dynamic_range_dB: float = 60.0,
    title: str = "",
    ax: plt.Axes | None = None,
    cmap: str = "gray",
) -> plt.Axes:
    """Display a single B-mode image.

    Parameters
    ----------
    image : Tensor
        Complex-valued image (N_z, N_x).
    positions_x, positions_z : Tensor
        Grid coordinates in metres (converted to mm for display).
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 7))

    db = to_bmode_db(image, dynamic_range_dB)

    x_mm = positions_x.numpy() * 1e3
    z_mm = positions_z.numpy() * 1e3

    im = ax.imshow(
        db,
        extent=[x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]],
        aspect="auto",
        cmap=cmap,
        vmin=-dynamic_range_dB,
        vmax=0,
    )
    ax.set_xlabel("Lateral position (mm)")
    ax.set_ylabel("Axial position (mm)")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, label="dB")
    return ax


def plot_comparison(
    images: list[Tensor],
    labels: list[str],
    positions_x: Tensor,
    positions_z: Tensor,
    *,
    dynamic_range_dB: float = 60.0,
    suptitle: str = "",
    save_path: str | Path | None = None,
):
    """Side-by-side comparison of multiple B-mode images."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 7), squeeze=False)
    axes = axes[0]

    x_mm = positions_x.numpy() * 1e3
    z_mm = positions_z.numpy() * 1e3

    for i, (img, label) in enumerate(zip(images, labels)):
        db = to_bmode_db(img, dynamic_range_dB)
        im = axes[i].imshow(
            db,
            extent=[x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]],
            aspect="auto",
            cmap="gray",
            vmin=-dynamic_range_dB,
            vmax=0,
        )
        axes[i].set_xlabel("Lateral position (mm)")
        if i == 0:
            axes[i].set_ylabel("Axial position (mm)")
        axes[i].set_title(label)
        plt.colorbar(im, ax=axes[i], label="dB")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  → Saved: {save_path}")

    return fig
