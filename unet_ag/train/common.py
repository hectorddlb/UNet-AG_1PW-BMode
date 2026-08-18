"""Piezas compartidas por los entrenadores y por la evaluacion.

Cuatro funciones que definen como se construye la entrada del modelo y como se
lee su supervision. Viven aparte porque las usan tanto `b1_v4_anchored` /
`b1_v5_pulse_aug` como el codigo de evaluacion, y tienen que coincidir
exactamente entre entrenamiento y test:

  _build_grid            rejilla de reconstruccion (Nz, Nx) a partir de la
                         geometria de la sonda. z_max = (ns-1)*c/(2*fs).
  _adjoint_image         x0 = H^T y para una sola onda plana a 0 grados,
                         calculado bajo no_grad (no hay autograd a traves de H).
  _envelope_log_compress envolvente por Hilbert + compresion logaritmica a
                         60 dB de rango dinamico, normalizada a [0, 1].
  _load_target           imagen objetivo Schiffner Fd-F 75 PW desde
                         <target_root>/<ptype>/<phantom_id>/compounded.npy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from scipy.signal import hilbert

from ..physics.h_fdf_wrapper import make_op_for_angle


DYN_RANGE_DB = 60.0


def _envelope_log_compress(img: Tensor, dyn_range_db: float = DYN_RANGE_DB) -> Tensor:
    """Hilbert envelope along axial → log compress to [0, 1] over ``dyn_range_db``.

    F1 fix: smoke 184971 plateau'd at MSE≈var(target_raw) because the raw
    Schiffner target is sparse (mostly zero, peaks at scatterer positions),
    so the MSE-optimal constant predictor wins. Envelope+log compress yields
    a smooth target with full dynamic range, removing the zero attractor.
    """
    arr = img.detach().cpu().numpy()
    env = np.abs(hilbert(arr, axis=0))
    # Relative normalisation: an absolute epsilon (old ``+ 1e-12``) swamps
    # Field II native amplitudes (~1e-23), flattening the target to constant
    # black — the collapse observed in run 206197 (mixed das75_fixed targets).
    env_max = float(env.max())
    if env_max <= 0.0:
        return torch.zeros(arr.shape, dtype=torch.float32)
    db = 20.0 * np.log10(env / env_max + 1e-30)
    db = np.clip(db, -dyn_range_db, 0.0)
    norm = (db + dyn_range_db) / dyn_range_db                # [0, 1]
    return torch.from_numpy(norm.astype(np.float32))


def _build_grid(geometry: dict, nz: int, nx: int) -> tuple[Tensor, Tensor]:
    """PICMUS-like default grid sized to the requested (Nz, Nx).

    Axial range from 5 mm to (n_samples · c / (2 · fs)) ≈ depth covered.
    Lateral range from ± (K · pitch / 2) — full aperture span.
    """
    c = float(geometry["speed_of_sound"])
    fs = float(geometry["sampling_freq_hz"])
    ns = int(geometry["n_samples"])
    K = int(geometry["n_elements"])
    pitch = float(geometry["pitch_m"])
    z_max = (ns - 1) * c / (2.0 * fs)
    grid_z = torch.linspace(5e-3, z_max, nz, dtype=torch.float32)
    grid_x = torch.linspace(-K * pitch / 2.0, +K * pitch / 2.0, nx, dtype=torch.float32)
    return grid_z, grid_x


def _adjoint_image(rf_1pw: Tensor, geometry: dict, grid_z: Tensor, grid_x: Tensor, device: str) -> Tensor:
    """``H_Fdf^T y`` for the 1 PW angle. NO autograd (α contract)."""
    with torch.no_grad():
        op = make_op_for_angle(0.0, geometry, grid_z, grid_x, device=device, nf_chunk=None)
        y = rf_1pw.to(device).to(op.float_dtype)
        x0 = op.adjoint(y)
        if x0.is_complex():
            x0 = x0.real
    return x0.float().detach()


def _load_target(target_root: Path, phantom_id: str, ptype: str, nz: int, nx: int) -> Optional[Tensor]:
    candidates = [
        target_root / ptype / phantom_id / "compounded.npy",
        target_root / phantom_id / "compounded.npy",
    ]
    for p in candidates:
        if p.exists():
            arr = np.load(p).astype(np.float32)
            t = torch.from_numpy(arr)
            if t.shape != (nz, nx):
                # Resize via bilinear — same grid family, different sampling.
                t = torch.nn.functional.interpolate(
                    t.unsqueeze(0).unsqueeze(0), size=(nz, nx), mode="bilinear", align_corners=False
                )[0, 0]
            return t
    return None
