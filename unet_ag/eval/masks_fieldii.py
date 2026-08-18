"""Phantom-aware mask generation for Field II eval.

- ``points_wire_targets``: returns the (z_mm, x_mm) of each scatterer for
  FWHM aggregation via ``metrics.fwhm.aggregate_fwhms_per_wire``.
- ``cyst_masks``: detects anechoic cyst regions by scatterer-density
  thresholding, returns (mask_inside, mask_outside) on the eval grid for
  CNR / gCNR.

Both helpers operate on the ``positions`` and ``amplitudes`` arrays stored
inside Field II ``.mat`` files; no synthetic ground truth is invented.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.ndimage import binary_dilation, binary_erosion, label


def _load_scatterers(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    m = sio.loadmat(str(mat_path))
    pos = np.asarray(m["positions"], dtype=np.float64)        # (N, 3) [x,y,z]
    amp = np.asarray(m["amplitudes"], dtype=np.float64).ravel()
    return pos, amp


def points_wire_targets(mat_path: Path) -> list[tuple[float, float]]:
    """Return scatterer (z_mm, x_mm) tuples for a points phantom."""
    pos, _ = _load_scatterers(mat_path)
    return [(float(p[2] * 1e3), float(p[0] * 1e3)) for p in pos]


def cyst_masks(
    mat_path: Path,
    grid_z_m: np.ndarray,
    grid_x_m: np.ndarray,
    density_quantile: float = 0.10,
    min_cyst_pixels: int = 80,
    bg_ring_pixels: int = 6,
    edge_exclude_pixels: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Detect cyst regions from scatterer density and return masks.

    Returns
    -------
    mask_inside : (Nz, Nx) bool
        Union of detected anechoic regions (all cysts merged).
    mask_outside : (Nz, Nx) bool
        Annulus around the inside mask (in tissue, not in cyst interiors).
    cysts : list of dicts
        Per-detected-cyst metadata: ``{"mask", "center_z_m", "center_x_m",
        "radius_m", "n_pixels"}``.
    """
    pos, _ = _load_scatterers(mat_path)
    Nz, Nx = int(grid_z_m.size), int(grid_x_m.size)

    # Bin scatterers onto the eval grid; the density map shows cysts as
    # low-count regions because the phantom generator thins scatterers there.
    z_edges = np.concatenate([
        [grid_z_m[0] - 0.5 * (grid_z_m[1] - grid_z_m[0])],
        0.5 * (grid_z_m[:-1] + grid_z_m[1:]),
        [grid_z_m[-1] + 0.5 * (grid_z_m[-1] - grid_z_m[-2])],
    ])
    x_edges = np.concatenate([
        [grid_x_m[0] - 0.5 * (grid_x_m[1] - grid_x_m[0])],
        0.5 * (grid_x_m[:-1] + grid_x_m[1:]),
        [grid_x_m[-1] + 0.5 * (grid_x_m[-1] - grid_x_m[-2])],
    ])
    H, _, _ = np.histogram2d(pos[:, 2], pos[:, 0], bins=[z_edges, x_edges])  # (Nz, Nx)

    # Exclude image borders AND any pixels beyond the actual phantom volume
    # (Field II places no scatterers past pos[:,2].max(); the empty bottom
    # band would otherwise look like a giant "cyst").
    interior = np.zeros((Nz, Nx), dtype=bool)
    e = int(edge_exclude_pixels)
    interior[e:Nz - e, e:Nx - e] = True
    z_max_phantom = float(pos[:, 2].max())
    z_min_phantom = float(pos[:, 2].min())
    interior &= (grid_z_m[:, None] <= z_max_phantom - 1e-3)
    interior &= (grid_z_m[:, None] >= z_min_phantom + 1e-3)
    x_max_phantom = float(pos[:, 0].max())
    x_min_phantom = float(pos[:, 0].min())
    interior &= (grid_x_m[None, :] <= x_max_phantom - 1e-3)
    interior &= (grid_x_m[None, :] >= x_min_phantom + 1e-3)

    threshold = float(np.quantile(H[interior], density_quantile))
    low = (H <= threshold) & interior

    # Remove thin sparse strips by eroding then dilating.
    low = binary_erosion(low, iterations=2)
    low = binary_dilation(low, iterations=2)

    labels, n = label(low)
    cysts: list[dict] = []
    inside_full = np.zeros((Nz, Nx), dtype=bool)
    zz, xx = np.meshgrid(grid_z_m, grid_x_m, indexing="ij")
    for k in range(1, n + 1):
        m = labels == k
        if m.sum() < min_cyst_pixels:
            continue
        # Skip if blob touches the excluded border (likely edge artefact).
        if m[:e].any() or m[Nz - e:].any() or m[:, :e].any() or m[:, Nx - e:].any():
            continue
        zi, xi = np.where(m)
        cz_m = float(grid_z_m[int(round(zi.mean()))])
        cx_m = float(grid_x_m[int(round(xi.mean()))])
        # Equivalent-area circle radius: r = sqrt(area / π).
        cell_area = float((grid_z_m[1] - grid_z_m[0]) * (grid_x_m[1] - grid_x_m[0]))
        r_m = float(np.sqrt(m.sum() * cell_area / np.pi))
        cysts.append(dict(mask=m, center_z_m=cz_m, center_x_m=cx_m,
                          radius_m=r_m, n_pixels=int(m.sum())))
        inside_full |= m

    if inside_full.sum() == 0:
        empty = np.zeros((Nz, Nx), dtype=bool)
        return empty, empty, []

    outside = binary_dilation(inside_full, iterations=bg_ring_pixels) & ~inside_full
    return inside_full, outside, cysts
