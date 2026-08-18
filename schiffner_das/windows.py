"""
Window functions for receive apodization.

All windows operate on the *normalised* position ``|x| / half_width`` and
return 0 outside the unit interval.  The input is expected to be the
**absolute value** of the normalised position (non-negative).

Designed to work element-wise on arbitrary-shaped :class:`torch.Tensor`
inputs so that they can be used inside vectorised beamforming kernels and,
later, inside a differentiable forward model.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Window(ABC):
    """Abstract base class for all window functions."""

    @abstractmethod
    def compute_samples(self, positions_over_halfwidth_abs: Tensor) -> Tensor:
        """Evaluate the window at the given normalised positions.

        Parameters
        ----------
        positions_over_halfwidth_abs : Tensor
            Non-negative tensor  ``|x| / half_width``.

        Returns
        -------
        Tensor
            Window values in [0, 1].  Zero where input >= 1.
        """

    @abstractmethod
    def __repr__(self) -> str: ...


class Boxcar(Window):
    """Rectangular (Dirichlet) window: 1 inside, 0 outside."""

    def compute_samples(self, positions_over_halfwidth_abs: Tensor) -> Tensor:
        return (positions_over_halfwidth_abs < 1.0).to(positions_over_halfwidth_abs.dtype)

    def __repr__(self) -> str:
        return "Boxcar()"


class Hann(Window):
    """Hann (raised cosine) window."""

    def compute_samples(self, positions_over_halfwidth_abs: Tensor) -> Tensor:
        inside = positions_over_halfwidth_abs < 1.0
        samples = torch.zeros_like(positions_over_halfwidth_abs)
        samples[inside] = (
            1.0 + torch.cos(math.pi * positions_over_halfwidth_abs[inside])
        ) / 2.0
        return samples

    def __repr__(self) -> str:
        return "Hann()"


class Triangular(Window):
    """Triangular (Bartlett / Fejér) window."""

    def compute_samples(self, positions_over_halfwidth_abs: Tensor) -> Tensor:
        inside = positions_over_halfwidth_abs < 1.0
        samples = torch.zeros_like(positions_over_halfwidth_abs)
        samples[inside] = 1.0 - positions_over_halfwidth_abs[inside]
        return samples

    def __repr__(self) -> str:
        return "Triangular()"


class Tukey(Window):
    """Tukey (cosine-tapered) window.

    Parameters
    ----------
    fraction_cosine : float
        Fraction of the window in the cosine taper (0 < r < 1).
        ``r → 0`` approaches a boxcar, ``r → 1`` approaches Hann.
    """

    def __init__(self, fraction_cosine: float = 0.5):
        if not (0 < fraction_cosine < 1):
            raise ValueError("fraction_cosine must be in (0, 1).")
        self.fraction_cosine = float(fraction_cosine)
        self.fraction_rectangle = 1.0 - self.fraction_cosine

    def compute_samples(self, positions_over_halfwidth_abs: Tensor) -> Tensor:
        p = positions_over_halfwidth_abs
        inside = p < 1.0
        samples = inside.to(p.dtype)

        # Taper region: fraction_rectangle < |x|/hw < 1
        taper = inside & (p > self.fraction_rectangle)
        if taper.any():
            rel = (p[taper] - self.fraction_rectangle) / self.fraction_cosine
            samples[taper] = (1.0 + torch.cos(math.pi * rel)) / 2.0

        return samples

    def __repr__(self) -> str:
        return f"Tukey(fraction_cosine={self.fraction_cosine:.2f})"
