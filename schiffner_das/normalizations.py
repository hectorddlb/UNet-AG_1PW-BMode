"""
Normalization policies for apodization weights.

- ``NormalizationOff``: identity — weights are returned unchanged.
- ``NormalizationOn``: each row (per-frequency, per-voxel) is divided by the
  sum across the element (last) dimension.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Normalization(ABC):
    """Abstract base class for all normalization policies."""

    @abstractmethod
    def apply(self, samples: Tensor) -> Tensor:
        """Apply normalization to apodization weights.

        Parameters
        ----------
        samples : Tensor
            Apodization weights.  The **last** dimension corresponds to
            array elements.

        Returns
        -------
        Tensor
            Normalised weights (same shape).
        """

    @abstractmethod
    def __repr__(self) -> str: ...


class NormalizationOff(Normalization):
    """No normalization — return weights unchanged."""

    def apply(self, samples: Tensor) -> Tensor:
        return samples

    def __repr__(self) -> str:
        return "NormalizationOff()"


class NormalizationOn(Normalization):
    """Row-wise normalization: divide each row by its element-sum.

    Matches the MATLAB ``normalizations.on`` class which computes
    ``samples ./ sum(samples, 2)`` (sum across the element dimension).
    """

    def apply(self, samples: Tensor) -> Tensor:
        s = samples.sum(dim=-1, keepdim=True)
        # Avoid division by zero for empty apertures
        s = torch.where(s.abs() > 0, s, torch.ones_like(s))
        return samples / s

    def __repr__(self) -> str:
        return "NormalizationOn()"
