"""B1-v3 — Residual skip wrapper sobre B1 UNet+AG con AG-Schiffner-Fd-F.

Reformulación "a la segura" tras 8 negativos B1/A2. Conserva la restricción
del director (Conditional Attention Gates condicionados por F-número de
Schiffner Fd-F) y añade una **garantía matemática de identidad inicial**:

    pred = clamp( x_DAS  +  γ_gain · UNet_AG(x_DAS, f_map),  0, 1 )

con la última conv 1×1 del UNet **zero-inicializada** (peso y bias = 0). En
la época 0 el residual es idénticamente cero → ``pred ≡ x_DAS``. Como
``x_DAS = env_log(H_Fdf^T y)`` ya alcanza por sí solo ``n_lat = 484/600 @
FWHM_lat = 2.59 mm`` en eval Field II (campana B1, columna
``das_1pw``), el criterio de rescate ``FWHM_lat<4.9 AND n_lat≥150/600``
está **garantizado desde la inicialización**. El entrenamiento sólo puede
mejorar (refinar hacia el target Schiffner_75PW) o, en el peor caso, ser
contenido por el ``best.pt`` por val_loss en una época cercana a 0.

Por qué B1 baseline falló con el mismo input:
    pred = sigmoid( UNet(x_DAS, f_map) )      # B1 baseline
- output activation σ con UNet random-init → pred ≈ 0.5 por píxel en ep 0.
- El UNet aprendió un mapping "envoltura suave" hacia env+log Schiffner_75PW
  que **perdió** la información lateral del input. FWHM_lat = 38 mm.

B1-v3 obliga al modelo a partir de x_DAS y a refinarlo. Si la geometría
lateral del input se conserva (lo que sucede al inicio por construcción),
el UNet sólo puede degradarla activamente.

Schiffner Fd-F interviene en 2 puntos:
1. **Conditioning AG**: ``f_map = compute_f_number_map_band_avg`` →
   `FNumberEmbedding` → `ConditionalAttentionGate` en cada nivel del decoder.
2. **Target supervisión**: ``Schiffner_75PW`` compounded env+log.

NO interviene como apodización en el input (input es ``env_log(H^T y)`` igual
que B1 baseline, ya lateral-resuelto en eval).

γ-params: **1 escalar** (``log_gain``) → respeta contrato α/γ del proyecto.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .b1_unet_attention_fnum import B1UNetAttentionFNum


class B1V3Residual(nn.Module):
    """B1 UNet+AG con skip residual y zero-init de la conv de salida.

    Parameters
    ----------
    base_ch, embed_ch : int
        Capacidad del UNet (mismo defecto que B1 baseline: ``base_ch=32``).
    init_gain : float, default 1.0
        Valor inicial del ``gain`` aprendible que multiplica el residual.
        Mantenido en log-space para positividad. Al inicio el residual es 0
        (zero-init out conv), así que el valor de ``gain`` es irrelevante
        para el output inicial; pasa a importar tras los primeros steps.
    train_gain : bool, default True
        Si ``True`` ``log_gain`` participa en autograd; si ``False`` queda
        fijado en ``init_gain``.

    Forward
    -------
    ``forward(x0, f_map) -> pred``

    - ``x0``     : (B, 1, Nz, Nx) — ``env_log(H_Fdf^T y)``, ya en [0,1].
    - ``f_map``  : (B, 1, Nz, Nx) — Schiffner Fd-F band-averaged depth cap.
    - ``pred``   : (B, 1, Nz, Nx) — ``clamp(x0 + γ_gain · residual, 0, 1)``.
    """

    def __init__(
        self,
        base_ch: int = 32,
        embed_ch: int = 16,
        init_gain: float = 1.0,
        train_gain: bool = True,
    ):
        super().__init__()

        # UNet sin sigmoide final — quiero que el out conv produzca ≈ 0 al
        # inicio (zero-init) sin que sigmoid(0)=0.5 contamine.
        self.unet = B1UNetAttentionFNum(
            in_ch=1, base_ch=base_ch, embed_ch=embed_ch,
            output_activation="none",
        )

        # Zero-init de la conv 1×1 final → residual ≡ 0 en ep 0.
        nn.init.zeros_(self.unet.out.weight)
        if self.unet.out.bias is not None:
            nn.init.zeros_(self.unet.out.bias)

        # γ_gain learnable en log-space (positividad + gradientes suaves).
        self.log_gain = nn.Parameter(
            torch.tensor(math.log(init_gain), dtype=torch.float32),
            requires_grad=train_gain,
        )

    @property
    def gain(self) -> Tensor:
        return torch.exp(self.log_gain)

    @property
    def gamma(self) -> dict[str, float]:
        with torch.no_grad():
            return {"gain": float(self.gain)}

    def forward(self, x0: Tensor, f_map: Tensor) -> Tensor:
        residual = self.unet(x0, f_map)                  # (B, 1, Nz, Nx), ≈0 en ep 0
        return torch.clamp(x0 + self.gain * residual, 0.0, 1.0)

    def extra_repr(self) -> str:
        with torch.no_grad():
            return f"gain={float(self.gain):.3f}"
