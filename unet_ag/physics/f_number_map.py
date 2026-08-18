"""F-number maps used as conditioning (B1) and as spectral weight (A2).

Schiffner ``GratingAngleLB(angle_lb_deg, F_ub)`` defines a frequency-dependent
F-number:

    F(p/λ) = max(F_lb_from_grating(p/λ), F_ub)

where ``p/λ = pitch · f / c``. There is also an additional depth cap that
Schiffner DAS applies per pixel:

    F_eff(f, z) = min(F(f), sqrt(f · z / (2.88 · c)))

This module produces:

- ``compute_f_number_grid(f_hz, z_m, geometry)`` → ``(Nf, Nz)`` array of
  effective F-numbers.
- ``compute_f_number_map_at_fc(grid_z, grid_x, geometry)`` → ``(Nz, Nx)`` map
  using ``f = fc`` (centre frequency), broadcast across x. Used by B1.
- ``compute_spectral_weight(f_hz, geometry, mode)`` → ``(Nf,)`` 1-D weight
  used by A2 (e.g. ``W(f) = 1 / F(f)²``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# Schiffner FNumber lives under schiffner_fd-f/ (hyphen). Reuse via the same
# pattern as physics/operator_fdf.py.
_SCHIFFNER_ROOT = Path(__file__).resolve().parents[2] / "schiffner_fd-f"
if str(_SCHIFFNER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCHIFFNER_ROOT))

from schiffner_das.f_numbers import FNumber, GratingAngleLB  # noqa: E402


def default_f_number() -> FNumber:
    return GratingAngleLB(60.0, 1.5)


def compute_f_number_grid(
    f_hz: np.ndarray,
    z_m: np.ndarray,
    geometry: dict,
    f_number: FNumber | None = None,
) -> np.ndarray:
    """Effective F-number on a (frequency, depth) grid.

    Returns
    -------
    F_eff : (Nf, Nz) float64
    """
    if f_number is None:
        f_number = default_f_number()
    pitch = float(geometry["pitch_m"])
    c = float(geometry["speed_of_sound"])

    f_hz = np.asarray(f_hz, dtype=np.float64).reshape(-1)
    z_m = np.asarray(z_m, dtype=np.float64).reshape(-1)

    # Schiffner's compute_values takes p/λ as torch tensor.
    p_over_lambda = torch.from_numpy(pitch * f_hz / c)
    F_base = f_number.compute_values(p_over_lambda).cpu().numpy()  # (Nf,)

    # Depth cap: F_z(f, z) = sqrt(f * z / (2.88 * c))
    fz = np.sqrt(np.outer(f_hz, z_m) / (2.88 * c))                 # (Nf, Nz)
    F_eff = np.minimum(F_base[:, None], fz)
    # Where F_eff == 0 (z=0 edge) set to F_base to avoid divide-by-zero downstream.
    F_eff = np.where(F_eff <= 0, F_base[:, None], F_eff)
    return F_eff


def compute_f_number_map_band_avg(
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    geometry: dict,
    f_number: FNumber | None = None,
    f_band_hz: tuple[float, float] | None = None,
    n_freq_bins: int = 64,
) -> np.ndarray:
    """Band-averaged Schiffner depth-cap term ``sqrt(f·z / (2.88·c))`` (F2 fix).

    Why this instead of ``min(F_base, F_depth_cap)`` averaged across the band:
    in the PICMUS depth range (5–50 mm) the depth cap is ≫ F_ub at all freqs
    of interest, so ``min(F_base, F_cap)`` is uniformly clipped to F_ub ≈ 1.5
    and provides no spatial signal. The unclipped depth-cap term
    ``sqrt(f·z / 2.88c)`` *is* the depth-dependent part of Schiffner's F-number
    formula and grows as √z — that's what we want the Conditional Attention
    Gates to see.

    Averaged across the spectral band [0.5·fc, 1.5·fc] for robustness.
    """
    c = float(geometry["speed_of_sound"])
    if f_band_hz is None:
        fc = float(geometry.get("fc", geometry["sampling_freq_hz"] / 4.0))
        f_band_hz = (0.5 * fc, 1.5 * fc)
    f_hz = np.linspace(f_band_hz[0], f_band_hz[1], n_freq_bins)
    # Depth cap matrix: F_cap(f, z) = sqrt(f · z / (2.88 · c))
    F_cap = np.sqrt(np.outer(f_hz, np.asarray(grid_z, dtype=np.float64)) / (2.88 * c))
    F_cap_avg = F_cap.mean(axis=0)                                            # (Nz,)
    return np.broadcast_to(F_cap_avg[:, None], (F_cap_avg.size, grid_x.size)).astype(np.float32).copy()


def compute_f_number_map_at_fc(
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    geometry: dict,
    f_number: FNumber | None = None,
    fc_hz: float | None = None,
) -> np.ndarray:
    """F-number conditioning map evaluated at centre frequency.

    Used by B1 ConditionalAttentionGate. Returns ``(Nz, Nx)`` broadcastable
    over x. The pattern is depth-dependent (Schiffner's depth cap), uniform
    across lateral position.
    """
    if fc_hz is None:
        fc_hz = float(geometry.get("fc", geometry["sampling_freq_hz"] / 4.0))
    F = compute_f_number_grid(np.array([fc_hz]), grid_z, geometry, f_number)[0]  # (Nz,)
    return np.broadcast_to(F[:, None], (F.size, grid_x.size)).astype(np.float32).copy()


def compute_spectral_weight(
    f_hz: np.ndarray,
    geometry: dict,
    mode: str = "inv_F_squared",
    f_number: FNumber | None = None,
) -> np.ndarray:
    """1-D weight ``W(f)`` for the Fourier-loss in propuesta A2.

    Modes
    -----
    inv_F_squared
        ``W(f) = 1 / F(f)²`` — gives more weight to frequency bands with a
        small (more permissive) F-number, where Schiffner aperture passes
        more elements and the PSF is sharper.
    inv_F
        ``W(f) = 1 / F(f)``
    flat
        ``W(f) = 1`` — sanity baseline.
    """
    if mode == "flat":
        return np.ones_like(np.asarray(f_hz, dtype=np.float32))
    if f_number is None:
        f_number = default_f_number()
    pitch = float(geometry["pitch_m"])
    c = float(geometry["speed_of_sound"])
    p_over_lambda = torch.from_numpy(pitch * np.asarray(f_hz, dtype=np.float64) / c)
    F = f_number.compute_values(p_over_lambda).cpu().numpy()
    F_safe = np.where(F > 0, F, 1e-6)
    if mode == "inv_F_squared":
        return (1.0 / (F_safe * F_safe)).astype(np.float32)
    if mode == "inv_F":
        return (1.0 / F_safe).astype(np.float32)
    raise ValueError(f"unknown mode {mode!r}")
