"""Generalized Contrast-to-Noise Ratio (Rodriguez-Molares et al. 2020).

    gCNR = 1 − ∫ min(p_inside(x), p_outside(x)) dx

Histogram-overlap formulation, normalised so totals integrate to 1. Range
[0, 1]; 1 = perfectly separated, 0 = identical distributions. Robust to
monotonic transforms (log compression, dynamic-range squashing).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def _envelope(image: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(image, axis=0))


def compute_gcnr(
    image: np.ndarray,
    mask_inside: np.ndarray,
    mask_outside: np.ndarray,
    n_bins: int = 256,
    on_envelope: bool = True,
) -> float:
    img = _envelope(image) if on_envelope else image
    inside = img[mask_inside.astype(bool)].ravel()
    outside = img[mask_outside.astype(bool)].ravel()
    if inside.size == 0 or outside.size == 0:
        return float("nan")

    lo = float(min(inside.min(), outside.min()))
    hi = float(max(inside.max(), outside.max()))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)

    h_in, _ = np.histogram(inside, bins=edges)
    h_out, _ = np.histogram(outside, bins=edges)
    p_in = h_in / max(h_in.sum(), 1)
    p_out = h_out / max(h_out.sum(), 1)
    overlap = float(np.minimum(p_in, p_out).sum())
    return float(1.0 - overlap)
