"""Phantom-aware mask generation for PICMUS eval.

Unlike Field II, PICMUS phantom HDF5 stores scatterer positions and analytic
cyst geometry directly (no density inference needed):

  US/US_DATASET0000/
    scatterers_positions    (3, N)     m   x,y,z
    scatterers_amplitude    (N,)
    phantom_occlusionCenterX (M,)      m
    phantom_occlusionCenterZ (M,)      m
    phantom_occlusionDiameter(M,)      m

For ``resolution_distortion`` the 20 scatterers are the wires. For
``contrast_speckle`` there are 9 analytic occlusions (anechoic cysts).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_dilation


def _open(phantom_path: Path):
    return h5py.File(phantom_path, "r")["US/US_DATASET0000"]


def points_wire_targets(phantom_path: Path) -> list[tuple[float, float]]:
    """Return scatterer (z_mm, x_mm) tuples for the PICMUS resolution phantom."""
    with h5py.File(phantom_path, "r") as f:
        g = f["US/US_DATASET0000"]
        pos = np.asarray(g["scatterers_positions"])    # (3, N)
    # pos[0]=x, pos[1]=y, pos[2]=z (m)
    return [(float(pos[2, i] * 1e3), float(pos[0, i] * 1e3)) for i in range(pos.shape[1])]


def cyst_masks(
    phantom_path: Path,
    grid_z_m: np.ndarray,
    grid_x_m: np.ndarray,
    bg_ring_pixels: int = 6,
    shrink_inside_frac: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build analytic disc masks for PICMUS contrast_speckle.

    Parameters
    ----------
    shrink_inside_frac : float
        Shrink the disc radius by this factor for the "inside" mask, to keep
        pixels away from the cyst edge where the speckle envelope is partly
        contaminated (standard PICMUS convention — typically 0.7–0.8).
    bg_ring_pixels : int
        Width (pixels) of the annular background ring around each cyst.

    Returns
    -------
    mask_inside : (Nz, Nx) bool   union of shrunk disc interiors
    mask_outside : (Nz, Nx) bool  union of annular tissue rings (excludes cyst interiors)
    cysts : list of dicts         per-cyst metadata
    """
    with h5py.File(phantom_path, "r") as f:
        g = f["US/US_DATASET0000"]
        cx_m = np.asarray(g["phantom_occlusionCenterX"]).reshape(-1).astype(np.float64)
        cz_m = np.asarray(g["phantom_occlusionCenterZ"]).reshape(-1).astype(np.float64)
        diam_m = np.asarray(g["phantom_occlusionDiameter"]).reshape(-1).astype(np.float64)

    Nz, Nx = int(grid_z_m.size), int(grid_x_m.size)
    zz, xx = np.meshgrid(grid_z_m, grid_x_m, indexing="ij")

    inside_full = np.zeros((Nz, Nx), dtype=bool)
    full_disc = np.zeros((Nz, Nx), dtype=bool)
    cysts: list[dict] = []
    for k in range(cx_m.size):
        r_m = 0.5 * diam_m[k]
        r_in = r_m * float(shrink_inside_frac)
        d2 = (zz - cz_m[k]) ** 2 + (xx - cx_m[k]) ** 2
        m_in = d2 <= r_in ** 2
        m_full = d2 <= r_m ** 2
        if m_in.sum() == 0:
            continue
        inside_full |= m_in
        full_disc |= m_full
        cysts.append(dict(
            mask=m_in,
            center_z_m=float(cz_m[k]),
            center_x_m=float(cx_m[k]),
            radius_m=float(r_m),
            n_pixels=int(m_in.sum()),
        ))

    if inside_full.sum() == 0:
        empty = np.zeros((Nz, Nx), dtype=bool)
        return empty, empty, []

    # Background = dilation of full discs minus the discs themselves.
    outside = binary_dilation(full_disc, iterations=int(bg_ring_pixels)) & ~full_disc
    return inside_full, outside, cysts
