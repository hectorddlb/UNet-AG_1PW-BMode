"""Propuesta A2 — RF encoder + ResU-Net with sigmoid head (env+log target).

**Vertiente**: con H (α, *opcional*). H_Fdf no aparece en el grafo autograd;
sólo se puede usar como término extra en la loss si se requiere fidelidad
espacial dura. La variante por defecto de A2 **no usa H**: la "fidelidad"
viene de la loss Fourier-weighted en dominio imagen, que es el aporte
físicamente motivado por el F-number de Schiffner.

**Input declarado**: RF crudo 1 PW, ``(B, 1, D, K)``, z-score + ``/max(|.|)``.

**Output**: ``(B, 1, Nz, Nx)`` con ``sigmoid`` final → ``[0, 1]``. El target
es **envelope+log compress (60 dB)** del Schiffner Fd-F 75 PW, igual que B1.
La razón: A1 colapsó en ``predict-zero`` con target lineal signed (run 185174,
2026-05-21); env+log destruye esa cuenca y B1 ya validó la receta.

**Diferencias vs A1**:
- A1 output lineal (signed); A2 output sigmoid [0,1].
- A1 supervisa contra target Schiffner crudo signed; A2 contra env+log [0,1].
- A1 loss = MSE+L1 + ``λ·‖y − H_Fdf x'‖²``; A2 loss = MSE+L1 + ``γ·Fourier_W``
  donde W(f) = 1/F²_Schiffner(f). Ver
  ``unet_ag.losses.fourier_weighted.mse_l1_fourier_loss``.

**Diferencias vs B1**:
- B1 input = ``H_Fdf^T y`` (adjoint image, fuera del grafo) + ``f_map`` para
  conditioning de Attention Gates. A2 input = RF crudo, sin AG, sin
  conditioning espacial.
- B1 tercer término = ``1 − SSIM`` (estructura local). A2 tercer término =
  Fourier-weighted (estructura espectral con prior físico F-number).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .shared_blocks import (
    ConvBlock,
    DownBlock,
    ResBlock,
    RFEncoder1DTime,
    UpBlock,
)


class A2UNetFourier(nn.Module):
    """RF encoder + 3-level ResU-Net; sigmoid output for env+log target."""

    def __init__(
        self,
        n_samples: int,
        n_elements: int,
        nz: int,
        nx: int,
        base_ch: int = 32,
        norm: str = "bn",
    ):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.nz, self.nx = int(nz), int(nx)

        self.rf_encoder = RFEncoder1DTime(
            n_samples=n_samples,
            n_elements=n_elements,
            nz=nz,
            nx=nx,
            out_ch=c1,
        )

        self.d1 = DownBlock(c1, c2, norm=norm)
        self.d2 = DownBlock(c2, c3, norm=norm)
        self.d3 = DownBlock(c3, c4, norm=norm)

        self.bot = nn.Sequential(
            ResBlock(c4, norm=norm),
            ResBlock(c4, norm=norm),
        )

        self.u3 = UpBlock(c4, c4, c3, norm=norm)
        self.u2 = UpBlock(c3, c3, c2, norm=norm)
        self.u1 = UpBlock(c2, c2, c1, norm=norm)

        self.head = nn.Sequential(
            ConvBlock(c1, c1, 3, norm=norm),
            nn.Conv2d(c1, 1, 1),
        )

    def forward(self, rf: torch.Tensor) -> torch.Tensor:
        """rf : (B, 1, D, K)  →  sigmoid output (B, 1, Nz, Nx) in [0, 1]."""
        feat = self.rf_encoder(rf)
        s1, x = self.d1(feat)
        s2, x = self.d2(x)
        s3, x = self.d3(x)
        x = self.bot(x)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        return torch.sigmoid(self.head(x))
