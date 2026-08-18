"""Soft-DAS front-end with learnable apodization (B1-v2).

Beamformer that maps a single plane-wave RF acquisition ``y[d, k]`` to an
image-domain field ``I[z, x]`` via delay-and-sum, with a *learnable* F-number
aperture and a soft (sigmoid-edge) apodization window. The DAS output is
RF-domain (signed, oscillating) — phase information is preserved, in
contrast to the envelope+log compression that B1/A2 apply to ``H_Fdf^T y``.

Math
----
For 1 PW at normal incidence, the round-trip time for grid point ``(z, x)``
and receiver element at ``x_k`` is::

    τ(z, x, x_k) = z/c + √(z² + (x − x_k)²) / c

The DAS image is::

    I(z, x) = gain · Σ_k w(z, x, x_k) · y(s(z, x, x_k), k)

where ``s = (τ − t_start) · fs`` is the (fractional) sample index and the
soft-edge F-number apodization is::

    a(z) = z / (2 · F)              # half-aperture width allowed at depth z
    u    = (a(z) − |x − x_k|) / (a(z) · edge)
    w    = σ(u)                     # ≈ 1 inside aperture, ≈ 0 outside, smooth

Learnable γ parameters (≤ 3, per the project's α/γ contract on H):
    - ``log_F``   : log(F#), init log(1.5)
    - ``log_edge``: log of edge softness (fraction of half-aperture), init log(0.1)
    - ``log_gain``: log(gain), init log(1.0) = 0

Out-of-bound sample indices are clamped to 0; this also zeros the contribution
from receivers that are physically too lateral for a given depth.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def _channel_positions(geometry: dict) -> np.ndarray:
    """Centred linear-array element positions, x_k for k=0..K-1."""
    K = int(geometry["n_elements"])
    pitch = float(geometry["pitch_m"])
    return (np.arange(K, dtype=np.float64) - (K - 1) / 2.0) * pitch


def _build_delay_lut(
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    geometry: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute geometric LUTs that don't depend on learnable γ.

    Returns
    -------
    sample_idx : (Nz, Nx, K) float32 — fractional sample index into RF[d, k]
    dx_abs     : (Nz, Nx, K) float32 — |x − x_k|, for apodization
    z_grid_b   : (Nz, 1, 1)  float32 — broadcastable z for aperture width
    """
    c = float(geometry["speed_of_sound"])
    fs = float(geometry["sampling_freq_hz"])
    t_start = float(geometry.get("t_start", 0.0))
    x_k = _channel_positions(geometry)                                     # (K,)
    z = np.asarray(grid_z, dtype=np.float64)[:, None, None]               # (Nz, 1, 1)
    x = np.asarray(grid_x, dtype=np.float64)[None, :, None]               # (1, Nx, 1)
    xk = x_k[None, None, :]                                               # (1, 1, K)
    dx = np.abs(x - xk)                                                   # (Nz, Nx, K) bcast
    dx = np.broadcast_to(dx, (z.shape[0], x.shape[1], xk.shape[2])).copy()
    # Round-trip time for 1-PW normal incidence.
    tau = z / c + np.sqrt(z * z + (x - xk) ** 2) / c                       # (Nz, Nx, K)
    s = (tau - t_start) * fs                                              # (Nz, Nx, K)
    return s.astype(np.float32), dx.astype(np.float32), z.astype(np.float32)


class SoftDASFrontEnd(nn.Module):
    """Differentiable DAS layer with learnable F-number aperture.

    Parameters
    ----------
    grid_z, grid_x : 1-D numpy arrays
        Image-domain sampling grids (m), of size Nz, Nx.
    geometry : dict
        Acquisition geometry. Needs keys ``speed_of_sound``, ``sampling_freq_hz``,
        ``t_start``, ``pitch_m``, ``n_elements``.
    init_F : float, default 1.5
        Initial F-number (same as Schiffner's F_ub baseline).
    init_edge : float, default 0.1
        Initial apodization edge softness, as a fraction of the half-aperture.
    init_gain : float, default 1.0
        Initial output gain.
    train_F, train_edge, train_gain : bool
        Whether each γ-param is learnable.
    """

    def __init__(
        self,
        grid_z: np.ndarray,
        grid_x: np.ndarray,
        geometry: dict,
        init_F: float = 1.5,
        init_edge: float = 0.1,
        init_gain: float = 1.0,
        train_F: bool = True,
        train_edge: bool = True,
        train_gain: bool = True,
    ):
        super().__init__()

        sample_idx, dx_abs, z_grid = _build_delay_lut(grid_z, grid_x, geometry)
        # Register as buffers (move with .to(device), but no gradients).
        self.register_buffer("sample_idx", torch.from_numpy(sample_idx))   # (Nz, Nx, K)
        self.register_buffer("dx_abs",     torch.from_numpy(dx_abs))       # (Nz, Nx, K)
        self.register_buffer("z_half",     torch.from_numpy(z_grid))       # (Nz, 1, 1)
        self.Nz, self.Nx, self.K = sample_idx.shape

        # γ-parameters in log-space for positivity + smooth gradients.
        log_F = math.log(init_F)
        log_edge = math.log(init_edge)
        log_gain = math.log(init_gain)
        self.log_F   = nn.Parameter(torch.tensor(log_F,    dtype=torch.float32), requires_grad=train_F)
        self.log_edge = nn.Parameter(torch.tensor(log_edge, dtype=torch.float32), requires_grad=train_edge)
        self.log_gain = nn.Parameter(torch.tensor(log_gain, dtype=torch.float32), requires_grad=train_gain)

    @property
    def F_number(self) -> Tensor:
        return torch.exp(self.log_F)

    @property
    def edge(self) -> Tensor:
        return torch.exp(self.log_edge)

    @property
    def gain(self) -> Tensor:
        return torch.exp(self.log_gain)

    def apodization(self) -> Tensor:
        """Soft-edge F-number apodization, shape (Nz, Nx, K), values ~[0,1]."""
        a = self.z_half / (2.0 * self.F_number)                            # (Nz, 1, 1)
        edge = self.edge.clamp(min=1e-3, max=1.0)
        # u > 0 means well inside the aperture → w → 1; u < 0 outside → w → 0.
        u = (a - self.dx_abs) / (a * edge + 1e-12)
        return torch.sigmoid(u)

    def forward(self, rf: Tensor) -> Tensor:
        """Soft-DAS beamform.

        Parameters
        ----------
        rf : Tensor
            Plane-wave RF, shape ``(D, K)`` or ``(B, D, K)``. ``D = n_samples``.

        Returns
        -------
        img : Tensor
            Image-domain signal, shape ``(Nz, Nx)`` or ``(B, Nz, Nx)``.
        """
        if rf.dim() == 2:
            rf_b = rf.unsqueeze(0)
            squeeze = True
        else:
            rf_b = rf
            squeeze = False
        B, D, K = rf_b.shape
        if K != self.K:
            raise ValueError(f"RF has K={K} but front-end built for K={self.K}")

        s = self.sample_idx                                                # (Nz, Nx, K)
        idx_floor = torch.clamp(s.floor().long(), 0, D - 1)                # (Nz, Nx, K)
        idx_ceil = torch.clamp(idx_floor + 1, 0, D - 1)
        frac = (s - idx_floor.float()).clamp(0.0, 1.0)                     # (Nz, Nx, K)

        # Mask samples whose nominal index falls outside the RF buffer.
        in_bounds = ((s >= 0.0) & (s <= float(D - 1))).float()             # (Nz, Nx, K)

        # Gather RF values along the time axis per receiver.
        # rf_b: (B, D, K). For each receiver k, we need RF[:, idx, k]; vectorise.
        # Use torch.gather with shape (B, Nz*Nx, K) over dim=1.
        flat_idx_floor = idx_floor.view(self.Nz * self.Nx, K).unsqueeze(0).expand(B, -1, -1)  # (B, Nz*Nx, K)
        flat_idx_ceil  = idx_ceil.view(self.Nz * self.Nx, K).unsqueeze(0).expand(B, -1, -1)
        # rf_b: (B, D, K) → gather along dim=1
        rf_floor = torch.gather(rf_b, dim=1, index=flat_idx_floor)         # (B, Nz*Nx, K)
        rf_ceil  = torch.gather(rf_b, dim=1, index=flat_idx_ceil)
        rf_floor = rf_floor.view(B, self.Nz, self.Nx, K)
        rf_ceil  = rf_ceil.view(B, self.Nz, self.Nx, K)

        # Linear interpolation along time.
        rf_at = rf_floor * (1.0 - frac) + rf_ceil * frac                   # (B, Nz, Nx, K)
        rf_at = rf_at * in_bounds                                          # zero OOB samples

        # Apodization (Nz, Nx, K) → broadcast over batch.
        w = self.apodization()                                             # (Nz, Nx, K)
        img = (rf_at * w).sum(dim=-1)                                      # (B, Nz, Nx)
        img = img * self.gain
        if squeeze:
            img = img.squeeze(0)
        return img

    def extra_repr(self) -> str:
        with torch.no_grad():
            return (f"Nz={self.Nz}, Nx={self.Nx}, K={self.K}, "
                    f"F={float(self.F_number):.3f}, "
                    f"edge={float(self.edge):.3f}, "
                    f"gain={float(self.gain):.3f}")
