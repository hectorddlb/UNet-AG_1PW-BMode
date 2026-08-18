"""
F-number models for frequency-dependent receive aperture control.

Implements the F-number hierarchy from Schiffner's MATLAB repository:
  - ConstantFNumber: fixed F-number for all frequencies
  - GratingAngleLB: frequency-dependent F-number enforcing a minimum angular
    distance of the first-order grating lobe (Eq. 3 in [1])

These classes are designed to be reusable in a PyTorch-based forward model
(H_{Fd-F}) for integration into a self-supervised DL loss function.

References
----------
[1] M. F. Schiffner and G. Schmitz, "Frequency-dependent F-number increases
    the contrast and the spatial resolution in fast pulse-echo ultrasound
    imaging," IEEE IUS 2021.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class FNumber(ABC):
    """Abstract base class for all F-number models."""

    @abstractmethod
    def compute_values(self, element_pitch_over_lambda: Tensor) -> Tensor:
        """Compute F-number values for given element-pitch-to-wavelength ratios.

        Parameters
        ----------
        element_pitch_over_lambda : Tensor
            Ratio p/λ for each frequency bin.  Shape: ``(N_freq,)`` or
            ``(N_freq, 1)`` — any broadcastable 1-D shape is fine.

        Returns
        -------
        Tensor
            F-number values with the same shape as *element_pitch_over_lambda*.
        """

    @abstractmethod
    def __repr__(self) -> str: ...


class ConstantFNumber(FNumber):
    """Fixed F-number independent of frequency.

    Parameters
    ----------
    value : float
        Constant F-number value.  ``0`` means full aperture.
    """

    def __init__(self, value: float = 1.5):
        if value < 0:
            raise ValueError("F-number value must be non-negative.")
        self.value = float(value)

    def compute_values(self, element_pitch_over_lambda: Tensor) -> Tensor:
        return torch.full_like(element_pitch_over_lambda, self.value)

    def __repr__(self) -> str:
        return f"ConstantFNumber(value={self.value:.2f})"


class GratingAngleLB(FNumber):
    r"""Frequency-dependent F-number from grating-lobe angular distance.

    Enforces a lower bound ``χ_lb`` on the angular distance of the first-order
    grating lobe.  The closed-form expression (Eq. 3 in [1]) reads

    .. math::
        F_{lb}(\chi_{lb}, \lambda) = \frac{1}{2}
        \sqrt{\frac{1}{\bigl(\lambda/p - \sin\chi_{lb}\bigr)^2} - 1}

    valid for  ``1/(1+sin χ_lb) < p/λ < 1/sin χ_lb``.

    Parameters
    ----------
    angle_lb_deg : float
        Minimum angular distance of the first-order grating lobe (degrees).
    F_number_ub : float
        Upper bound on the F-number (used when p/λ exceeds the valid range).
    distance_deg : float, optional
        Anti-aliasing angular distance (degrees).  Negative means inactive.
    """

    def __init__(
        self,
        angle_lb_deg: float = 60.0,
        F_number_ub: float = 1.5,
        distance_deg: float = -1.0,
    ):
        if not (0 < angle_lb_deg <= 90):
            raise ValueError("angle_lb_deg must be in (0, 90].")
        if F_number_ub < 0:
            raise ValueError("F_number_ub must be non-negative.")

        self.angle_lb_deg = float(angle_lb_deg)
        self.F_number_ub = float(F_number_ub)
        self.distance_deg = float(distance_deg)

        # Derived constants
        self.thresh = math.sin(math.radians(self.angle_lb_deg))
        self.lower_bound = 1.0 / (1.0 + self.thresh)
        self.upper_bound = 1.0 / (
            1.0 / math.sqrt(1.0 + (2.0 * self.F_number_ub) ** 2) + self.thresh
        )
        self.critical = 1.0 / self.thresh

    def compute_values(self, element_pitch_over_lambda: Tensor) -> Tensor:
        pol = element_pitch_over_lambda
        values = torch.zeros_like(pol)

        # Region where the closed-form F-number is valid
        mask_relevant = (pol > self.lower_bound) & (pol < self.upper_bound)
        if mask_relevant.any():
            pol_rel = pol[mask_relevant]
            values[mask_relevant] = (
                torch.sqrt(1.0 / (1.0 / pol_rel - self.thresh) ** 2 - 1.0) / 2.0
            )

        # Region where grating-lobe condition cannot be met → use upper bound
        mask_impossible = pol >= self.upper_bound
        values[mask_impossible] = self.F_number_ub

        # Optional anti-aliasing via angular-distance model
        if self.distance_deg >= 0:
            aa = _DistanceConstant(self.distance_deg)
            values_aa = aa.compute_values(pol)
            values = torch.max(values, values_aa)

        return values

    def __repr__(self) -> str:
        s = (
            f"GratingAngleLB(angle_lb_deg={self.angle_lb_deg:.2f}, "
            f"F_ub={self.F_number_ub:.2f})"
        )
        return s


# ---------------------------------------------------------------------------
# Internal helper for anti-aliasing (distance.constant in MATLAB)
# ---------------------------------------------------------------------------
class _DistanceConstant(FNumber):
    """Fixed angular-distance F-number model (internal helper).

    Computes  ``F = sqrt(1/(1/pol - sin(distance_deg))^2 - 1) / 2``
    capped by F_ub = 1 / (2 * sin(distance_deg))  for the 90° regime.
    """

    def __init__(self, distance_deg: float):
        self.distance_deg = float(distance_deg)
        self.thresh = math.sin(math.radians(self.distance_deg))

    def compute_values(self, element_pitch_over_lambda: Tensor) -> Tensor:
        pol = element_pitch_over_lambda
        values = torch.zeros_like(pol)
        if self.thresh > 0:
            mask = pol > 0.5  # rough guard; follows MATLAB distance.constant logic
            if mask.any():
                denom = 1.0 / pol[mask] - self.thresh
                valid = denom > 0
                if valid.any():
                    sub = torch.zeros_like(pol[mask])
                    sub[valid] = torch.sqrt(1.0 / denom[valid] ** 2 - 1.0) / 2.0
                    values[mask] = sub
        return values

    def __repr__(self) -> str:
        return f"_DistanceConstant(distance_deg={self.distance_deg:.2f})"
