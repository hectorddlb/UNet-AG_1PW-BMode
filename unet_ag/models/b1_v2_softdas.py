"""B1-v2 wrapper — Soft-DAS front-end + B1 UNet+AG refinement.

The previous B1/A2 family uses ``x0 = env_log(H_Fdf^T y)`` as input: the
envelope+log compression destroys phase, which is exactly the information
that carries narrow lateral localisation (zero-crossings) in RF imaging.

B1-v2 replaces the fixed envelope+log input by a *learnable*, *signed*
soft-DAS image:

    rf → SoftDASFrontEnd (F#, edge, gain learnable) → I_signed → B1UNet+AG → ŷ ∈ [0,1]

The UNet then learns the env+log mapping from the signed RF-domain image,
i.e. it has access to phase information for lateral resolution.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .b1_unet_attention_fnum import B1UNetAttentionFNum
from ..physics.soft_das import SoftDASFrontEnd


class B1V2SoftDAS(nn.Module):
    """Soft-DAS + UNet+AG end-to-end model.

    Parameters
    ----------
    grid_z, grid_x : 1-D numpy arrays
        Image-domain grids used by the soft-DAS front-end.
    geometry : dict
        Acquisition geometry.
    base_ch, embed_ch : int
        UNet capacity. Default ``base_ch=48`` (~7.9M params) — matches A2-v6.
    train_F, train_edge, train_gain : bool
        Whether each γ-param of the front-end is trainable.
    init_F, init_edge, init_gain : float
        Initial γ values.

    Forward signature mirrors ``B1UNetAttentionFNum`` so the rest of the
    pipeline (trainer / evaluator) can swap models with minimal changes.
    """

    def __init__(
        self,
        grid_z: np.ndarray,
        grid_x: np.ndarray,
        geometry: dict,
        base_ch: int = 48,
        embed_ch: int = 16,
        init_F: float = 1.5,
        init_edge: float = 0.1,
        init_gain: float = 1.0,
        train_F: bool = True,
        train_edge: bool = True,
        train_gain: bool = True,
    ):
        super().__init__()
        self.frontend = SoftDASFrontEnd(
            grid_z, grid_x, geometry,
            init_F=init_F, init_edge=init_edge, init_gain=init_gain,
            train_F=train_F, train_edge=train_edge, train_gain=train_gain,
        )
        self.unet = B1UNetAttentionFNum(
            in_ch=1, base_ch=base_ch, embed_ch=embed_ch, output_activation="sigmoid"
        )

    def forward(self, rf: Tensor, f_map: Tensor) -> Tensor:
        """Beamform then refine.

        Parameters
        ----------
        rf : Tensor
            Plane-wave RF, shape ``(B, D, K)`` or ``(D, K)``.
        f_map : Tensor
            F-number conditioning map, ``(B, 1, Nz, Nx)``.
        """
        img = self.frontend(rf)
        if img.dim() == 2:
            img = img.unsqueeze(0)                                         # (1, Nz, Nx)
        img = img.unsqueeze(1)                                             # (B, 1, Nz, Nx)
        return self.unet(img, f_map)

    @property
    def gamma(self) -> dict[str, float]:
        with torch.no_grad():
            return {
                "F":    float(self.frontend.F_number),
                "edge": float(self.frontend.edge),
                "gain": float(self.frontend.gain),
            }
