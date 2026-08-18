"""Contrast-to-Noise Ratio (dB).

    CNR(dB) = 20 · log10(|μ_inside − μ_outside| / sqrt(σ²_inside + σ²_outside))

Computed on the envelope (Hilbert magnitude) by default — set
``on_envelope=False`` to operate directly on the input (useful for testing
with synthetic gaussian intensities).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def _envelope(image: np.ndarray) -> np.ndarray:
    """Hilbert envelope along the axial (first) axis."""
    return np.abs(hilbert(image, axis=0))


def compute_cnr(
    image: np.ndarray,
    mask_inside: np.ndarray,
    mask_outside: np.ndarray,
    on_envelope: bool = True,
) -> float:
    img = _envelope(image) if on_envelope else image
    inside = img[mask_inside.astype(bool)]
    outside = img[mask_outside.astype(bool)]
    if inside.size == 0 or outside.size == 0:
        return float("nan")
    num = abs(float(inside.mean()) - float(outside.mean()))
    den = float(np.sqrt(inside.var() + outside.var()))
    if den < 1e-30:
        return float("nan")
    return float(20.0 * np.log10(num / den))
