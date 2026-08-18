"""Propuesta B1 — U-Net with Conditional Attention Gates by F-number.

**Vertiente**: sin H (la H aparece sólo indirectamente porque la entrada al
modelo es la imagen ``H_Fdf^T y``, calculada offline o en el dataloader; la
red en sí no contiene H).

**Input declarado**: imagen adjunta ``x0 = H_Fdf^T y`` con ``y = RF 1 PW``,
shape ``(B, 1, Nz, Nx)``. Computar fuera del grafo autograd (contrato α
heredado del wrapper).

**F-number conditioning**: mapa ``F(z, fc)`` shape ``(B, 1, Nz, Nx)`` calculado
en ``unet_ag.physics.f_number_map.compute_f_number_map_at_fc``. Se inyecta
en cada Attention Gate vía un pequeño MLP que produce un bias per-channel y
por-spatial sobre el camino de gating.

**Target**: Schiffner Fd-F 75 PW (Nz, Nx), supervisado.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared_blocks import ConvBlock, DownBlock, UpBlock


class FNumberEmbedding(nn.Module):
    """1×1 conv MLP that maps the F-number conditioning map to per-AG features.

    Input  : (B, 1, Nz, Nx) raw F-number values (in [F_ub, max_depth_cap])
    Output : (B, embed_ch, Nz, Nx) feature map normalised by tanh
    """

    def __init__(self, embed_ch: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, embed_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_ch, embed_ch, 1),
            nn.Tanh(),
        )

    def forward(self, f_map: torch.Tensor) -> torch.Tensor:
        # Normalise input to a unit-ish range to avoid saturating early.
        # F values are typically in [1.5, ~10]; log keeps them well-conditioned.
        f_in = torch.log1p(f_map.clamp(min=1e-6))
        return self.body(f_in)


class ConditionalAttentionGate(nn.Module):
    """Oktay 2018 Attention Gate + F-number conditioning.

        α = σ(ψ(ReLU(W_x · x + W_g · g + W_f · f_embed)))
        y = α ⊙ x

    f_embed is broadcast/resampled to the spatial size of ``x``.
    """

    def __init__(self, x_ch: int, g_ch: int, f_ch: int, inter_ch: int):
        super().__init__()
        self.W_x = nn.Conv2d(x_ch, inter_ch, 1, bias=False)
        self.W_g = nn.Conv2d(g_ch, inter_ch, 1, bias=True)
        self.W_f = nn.Conv2d(f_ch, inter_ch, 1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, 1, bias=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor, f_embed: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if f_embed.shape[-2:] != x.shape[-2:]:
            f_embed = F.interpolate(f_embed, size=x.shape[-2:], mode="bilinear", align_corners=False)
        z = F.relu(self.W_x(x) + self.W_g(g) + self.W_f(f_embed))
        alpha = torch.sigmoid(self.psi(z))
        return x * alpha


class B1UNetAttentionFNum(nn.Module):
    """B1 backbone.

    Architecture
    ------------
    Encoder: 4 DownBlocks (1 → 32 → 64 → 128 → 256).
    Bottleneck: ResBlock × 2 @ 256.
    Decoder: 4 UpBlocks (256 → 128 → 64 → 32 → 16) with skip + Conditional AG.
    Output head: 1×1 conv to 1 channel; no activation (regressing image-domain
    real values; output can be negative).

    The F-number conditioning is computed *once* per phantom (Schiffner depth
    cap at fc); embeddings are reused at every AG, downsampled to match the
    AG's spatial resolution.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        embed_ch: int = 16,
        norm: str = "bn",
        output_activation: str = "sigmoid",   # "sigmoid" | "none"
    ):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.output_activation = output_activation

        self.f_embed = FNumberEmbedding(embed_ch)

        self.d1 = DownBlock(in_ch, c1, norm=norm)
        self.d2 = DownBlock(c1, c2, norm=norm)
        self.d3 = DownBlock(c2, c3, norm=norm)
        self.d4 = DownBlock(c3, c4, norm=norm)

        self.bot = nn.Sequential(
            ConvBlock(c4, c4, 3, norm=norm),
            ConvBlock(c4, c4, 3, norm=norm),
        )

        self.ag4 = ConditionalAttentionGate(c4, c4, embed_ch, c4 // 2)
        self.u4 = UpBlock(c4, c4, c3, norm=norm)
        self.ag3 = ConditionalAttentionGate(c3, c3, embed_ch, c3 // 2)
        self.u3 = UpBlock(c3, c3, c2, norm=norm)
        self.ag2 = ConditionalAttentionGate(c2, c2, embed_ch, c2 // 2)
        self.u2 = UpBlock(c2, c2, c1, norm=norm)
        self.ag1 = ConditionalAttentionGate(c1, c1, embed_ch, c1 // 2)
        self.u1 = UpBlock(c1, c1, c1, norm=norm)

        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor, f_map: torch.Tensor) -> torch.Tensor:
        """x : (B, 1, Nz, Nx) — H_Fdf^T y image
        f_map : (B, 1, Nz, Nx) — F-number conditioning map
        Returns image (B, 1, Nz, Nx).
        """
        f_emb = self.f_embed(f_map)

        s1, x = self.d1(x)
        s2, x = self.d2(x)
        s3, x = self.d3(x)
        s4, x = self.d4(x)
        x = self.bot(x)

        s4 = self.ag4(s4, x, f_emb)
        x = self.u4(x, s4)
        s3 = self.ag3(s3, x, f_emb)
        x = self.u3(x, s3)
        s2 = self.ag2(s2, x, f_emb)
        x = self.u2(x, s2)
        s1 = self.ag1(s1, x, f_emb)
        x = self.u1(x, s1)

        y = self.out(x)
        if self.output_activation == "sigmoid":
            y = torch.sigmoid(y)
        return y
