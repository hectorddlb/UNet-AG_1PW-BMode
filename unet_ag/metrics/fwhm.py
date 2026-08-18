"""FWHM (Full Width at Half Maximum) lateral and axial.

Pipeline per peak:
  1. Take 1-D profile through the peak (envelope, normalised to peak in dB).
  2. Find left/right crossings at −6 dB (== half-maximum on linear scale).
  3. Linear-interpolate to sub-pixel precision.
  4. Convert pixel distance to mm.

For PICMUS resolution_distortion the 7 wires are aggregated as median + IQR.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


HALF_MAX_DB = -6.0206  # 20·log10(0.5)


def _envelope_safe(image: np.ndarray) -> np.ndarray:
    """Envelope along axial axis; if image already non-negative, treat as envelope."""
    if image.min() >= 0.0:
        return image
    return np.abs(hilbert(image, axis=0))


def _profile_to_db(profile: np.ndarray) -> np.ndarray:
    peak = float(profile.max())
    if peak <= 0.0:
        return np.full_like(profile, -np.inf)
    return 20.0 * np.log10(profile / peak + 1e-30)


def _fwhm_from_profile(profile_db: np.ndarray, peak_idx_local: int, d_mm: float) -> float:
    """Linear-interpolated FWHM in mm from a 1-D dB profile.

    Searches outward from ``peak_idx_local`` until the profile drops below
    ``HALF_MAX_DB`` on each side, then linearly interpolates the crossing.

    Returns ``NaN`` if the profile never crosses ``HALF_MAX_DB`` on at least
    one side — that means the wire is not resolved (PSF wider than the image
    extent at this depth). Previously this returned the grid width, which
    silently saturated the reported FWHM at the bounding box and obscured
    that B1 produced near-uniform fields (diagnose 2026-05-21).
    """
    n = profile_db.size
    # Left crossing
    li = peak_idx_local
    while li > 0 and profile_db[li] >= HALF_MAX_DB:
        li -= 1
    left_crossed = profile_db[li] < HALF_MAX_DB
    if not left_crossed or li + 1 > n - 1:
        # ``li + 1 > n - 1`` only happens when the located peak sits on the last
        # sample AND is itself below half-max (the while loop never steps), so
        # there is no bracket to interpolate: the wire is not resolved.
        return float("nan")
    x0, x1 = float(li), float(li + 1)
    y0, y1 = profile_db[li], profile_db[li + 1]
    left = x0 + (HALF_MAX_DB - y0) / (y1 - y0) * (x1 - x0) if y1 != y0 else x0

    # Right crossing
    ri = peak_idx_local
    while ri < n - 1 and profile_db[ri] >= HALF_MAX_DB:
        ri += 1
    right_crossed = profile_db[ri] < HALF_MAX_DB
    if not right_crossed or ri - 1 < 0:
        # Mirror of the left guard: peak on sample 0 and already below half-max.
        return float("nan")
    x0, x1 = float(ri - 1), float(ri)
    y0, y1 = profile_db[ri - 1], profile_db[ri]
    right = x0 + (HALF_MAX_DB - y0) / (y1 - y0) * (x1 - x0) if y1 != y0 else x1

    return float((right - left) * d_mm)


def compute_fwhm_lateral(
    image: np.ndarray,
    peak_idx: tuple[int, int],
    dx_mm: float,
) -> float:
    env = _envelope_safe(image)
    z0, x0 = peak_idx
    profile = env[z0, :]
    profile_db = _profile_to_db(profile)
    return _fwhm_from_profile(profile_db, int(x0), dx_mm)


def compute_fwhm_axial(
    image: np.ndarray,
    peak_idx: tuple[int, int],
    dz_mm: float,
) -> float:
    env = _envelope_safe(image)
    z0, x0 = peak_idx
    profile = env[:, x0]
    profile_db = _profile_to_db(profile)
    return _fwhm_from_profile(profile_db, int(z0), dz_mm)


def aggregate_fwhms_per_wire(
    image: np.ndarray,
    wires_expected_mm: list[tuple[float, float]],
    dx_mm: float,
    dz_mm: float,
    grid_z_mm: np.ndarray,
    grid_x_mm: np.ndarray,
    search_radius_mm: float = 1.0,
) -> dict:
    """For each wire (z_mm, x_mm), locate peak within ``search_radius_mm`` and
    compute lateral + axial FWHM. Return per-wire arrays + median.
    """
    env = _envelope_safe(image)
    lats: list[float] = []
    axs: list[float] = []
    for z_mm, x_mm in wires_expected_mm:
        rz = int(round(search_radius_mm / dz_mm))
        rx = int(round(search_radius_mm / dx_mm))
        iz0 = int(np.argmin(np.abs(grid_z_mm - z_mm)))
        ix0 = int(np.argmin(np.abs(grid_x_mm - x_mm)))
        z_lo, z_hi = max(0, iz0 - rz), min(env.shape[0], iz0 + rz + 1)
        x_lo, x_hi = max(0, ix0 - rx), min(env.shape[1], ix0 + rx + 1)
        sub = env[z_lo:z_hi, x_lo:x_hi]
        if sub.size == 0:
            continue
        local = np.unravel_index(int(np.argmax(sub)), sub.shape)
        peak = (z_lo + local[0], x_lo + local[1])
        lats.append(compute_fwhm_lateral(env, peak, dx_mm))
        axs.append(compute_fwhm_axial(env, peak, dz_mm))

    lats_arr = np.asarray(lats, dtype=np.float32)
    axs_arr = np.asarray(axs, dtype=np.float32)
    # Resolved = wires with a valid (non-NaN) FWHM on at least the lateral axis.
    # If the lateral PSF never crosses -6 dB the wire is not resolved at all.
    resolved_lat = ~np.isnan(lats_arr) if lats_arr.size else np.array([], dtype=bool)
    resolved_ax = ~np.isnan(axs_arr) if axs_arr.size else np.array([], dtype=bool)
    return {
        "lat": lats_arr,
        "ax": axs_arr,
        "lat_median": float(np.nanmedian(lats_arr)) if resolved_lat.any() else float("nan"),
        "ax_median": float(np.nanmedian(axs_arr)) if resolved_ax.any() else float("nan"),
        "n_resolved_lat": int(resolved_lat.sum()),
        "n_resolved_ax": int(resolved_ax.sum()),
        "n_total": int(lats_arr.size),
    }
