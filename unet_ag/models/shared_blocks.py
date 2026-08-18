"""Shared building blocks for the four propuestas.

All four backbones reuse these. The *novel* component of each propuesta sits in
its own module (B1 = ConditionalAttentionGate; A1 = RFEncoder + H loss;
B2 = multi-head decoder; A2 = Fourier-weighted loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm: str, ch: int) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(ch)
    if norm == "gn":
        return nn.GroupNorm(min(8, ch), ch)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm {norm!r}")


def _make_act(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"unknown activation {activation!r}")


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        norm: str = "bn",
        activation: str = "relu",
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=(norm == "none"))
        self.norm = _make_norm(norm, out_ch)
        self.act = _make_act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch: int, norm: str = "bn"):
        super().__init__()
        self.c1 = ConvBlock(ch, ch, 3, norm=norm)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=(norm == "none"))
        self.n2 = _make_norm(norm, ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n2(self.c2(self.c1(x)))
        return self.act(h + x)


class DownBlock(nn.Module):
    """Two ConvBlocks then 2× max-pool."""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "bn"):
        super().__init__()
        self.body = nn.Sequential(
            ConvBlock(in_ch, out_ch, 3, norm=norm),
            ConvBlock(out_ch, out_ch, 3, norm=norm),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (skip, downsampled)."""
        skip = self.body(x)
        return skip, self.pool(skip)


class UpBlock(nn.Module):
    """Bilinear up + ConvBlock × 2 (post-concat)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str = "bn"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.reduce = ConvBlock(in_ch, skip_ch, 1, norm=norm)
        self.body = nn.Sequential(
            ConvBlock(2 * skip_ch, out_ch, 3, norm=norm),
            ConvBlock(out_ch, out_ch, 3, norm=norm),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce(x)
        # Pad if needed (asymmetric grids — PICMUS 612×388 etc.).
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        return self.body(torch.cat([x, skip], dim=1))


class AttentionGate(nn.Module):
    """Standard Oktay et al. 2018 Attention Gate (no conditioning).

    α = σ(ψ(ReLU(W_x · x + W_g · g)))
    y = α ⊙ x

    Parameters
    ----------
    x_ch : channels of the encoder skip ``x``
    g_ch : channels of the decoder gating signal ``g``
    inter_ch : channels of the intermediate space
    """

    def __init__(self, x_ch: int, g_ch: int, inter_ch: int):
        super().__init__()
        self.W_x = nn.Conv2d(x_ch, inter_ch, 1, bias=False)
        self.W_g = nn.Conv2d(g_ch, inter_ch, 1, bias=True)
        self.psi = nn.Conv2d(inter_ch, 1, 1, bias=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # g may be on a different spatial resolution; align to x.
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        z = F.relu(self.W_x(x) + self.W_g(g))
        alpha = torch.sigmoid(self.psi(z))
        return x * alpha


class RFEncoder1DTime(nn.Module):
    """Encodes (D, K) RF into a (C, Nz, Nx) feature map.

    Pipeline:
      1. 1-D conv along time dim D with kernel ~ pulse length.
      2. Per-element mixing via grouped convs to fuse channel context.
      3. Bilinear resize (D', K) → (Nz, Nx).

    For B1 we DON'T use this — B1 takes the H_Fdf adjoint as input (cheaper,
    fewer params). A1 uses this as the dedicated RF encoder.
    """

    def __init__(
        self,
        n_samples: int,
        n_elements: int,
        nz: int,
        nx: int,
        ch_time: int = 32,
        ch_mix: int = 64,
        out_ch: int = 32,
    ):
        super().__init__()
        self.nz, self.nx = int(nz), int(nx)
        # 1-D conv along time. Input shape (B, 1, D, K). Stride 4 over D.
        self.time_convs = nn.Sequential(
            nn.Conv2d(1, ch_time, kernel_size=(11, 1), stride=(4, 1), padding=(5, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_time, ch_time, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0)),
            nn.ReLU(inplace=True),
        )
        # Element mixing
        self.mix = nn.Sequential(
            nn.Conv2d(ch_time, ch_mix, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_mix, out_ch, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

    def forward(self, rf: torch.Tensor) -> torch.Tensor:
        if rf.dim() == 3:  # (B, D, K)
            rf = rf.unsqueeze(1)
        if rf.dim() != 4:
            raise ValueError(f"rf must be (B, 1, D, K) or (B, D, K); got {rf.shape}")
        h = self.time_convs(rf)
        h = self.mix(h)
        h = F.interpolate(h, size=(self.nz, self.nx), mode="bilinear", align_corners=False)
        return h
