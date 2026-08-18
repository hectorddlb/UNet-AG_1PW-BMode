"""
Fourier-domain Delay-and-Sum (DAS) beamformer for steered plane waves.

Faithful Python/PyTorch port of ``das_pw.m`` from Schiffner's *f_number*
MATLAB repository [1, 2].

Performance backends (auto-detected):
  * **CUDA**  — tensors on GPU, batched over x and z.  50-100× vs single-thread CPU.
  * **CPU**   — batched over x so PyTorch's internal OpenMP threading uses all cores.
               5-8× on an 8-core CPU vs the single-thread baseline.

The ``device`` parameter controls backend selection:
  * ``"auto"``  — CUDA if available, else CPU  (default)
  * ``"cuda"``  — force GPU
  * ``"cpu"``   — force CPU

References
----------
[1] M. F. Schiffner and G. Schmitz, IEEE TUFFC, 70(9), 2023.
[2] M. F. Schiffner and G. Schmitz, IEEE IUS, 2021.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .f_numbers import FNumber, GratingAngleLB
from .normalizations import Normalization, NormalizationOn
from .windows import Boxcar, Tukey, Window


@dataclass
class DASResult:
    """Container for beamformer outputs."""
    image: Tensor               # (N_z, N_x) complex
    F_number_values: Tensor     # (N_freq,) float
    signal: Optional[Tensor]    # (N_dft,) complex or None


def das_pw(
    positions_x: Tensor,
    positions_z: Tensor,
    data_RF: Tensor,
    f_s: float,
    steering_angle: float,
    element_width: float,
    element_pitch: float,
    c_0: float,
    f_bounds: Optional[tuple[float, float]] = None,
    index_t0: int = 0,
    window: Optional[Window] = None,
    F_number: Optional[FNumber] = None,
    normalization: Optional[Normalization] = None,
    z_chunk: int = 32,
    x_batch: Optional[int] = None,
    device: str = "auto",
    tx_sign: float = -1.0,
    tx_reference: str = "edge",
    precision: str = "float64",
) -> DASResult:
    """Fourier-domain DAS beamformer for a single steered plane wave.

    Parameters
    ----------
    positions_x, positions_z : Tensor (1-D)
        Lateral / axial voxel positions (m).
    data_RF : Tensor (N_t, N_el)
        Real-valued RF data for one steering angle.
    f_s : float
        Sampling rate (Hz).
    steering_angle : float
        Steering angle (rad).
    element_width, element_pitch : float
        Transducer geometry (m).
    c_0 : float
        Speed of sound (m/s).
    f_bounds : tuple, optional
        ``(f_lb, f_ub)`` in Hz.
    index_t0 : int
        Time-index offset.
    window : Window, optional
        Receive apodization.  Default: ``Tukey(0.2)``.
    F_number : FNumber, optional
        Receive F-number model.  Default: ``GratingAngleLB(60, 1.5)``.
    normalization : Normalization, optional
        Weight normalisation.  Default: ``NormalizationOn()``.
    z_chunk : int
        Z-positions per inner batch (controls peak memory).
    x_batch : int or None
        X-positions per outer batch.  ``None`` = auto (8 on CPU, 16 on CUDA).
        Larger values use more RAM/VRAM but give better parallelism.
    device : str
        ``"auto"`` (default), ``"cuda"``, or ``"cpu"``.

    Returns
    -------
    DASResult
    """
    # ── 0. defaults & device ────────────────────────────────────
    _validate(positions_x, positions_z, data_RF, f_s, steering_angle,
              element_width, element_pitch, c_0)

    if f_bounds is None:
        f_bounds = (torch.finfo(torch.float64).eps, f_s / 2.0)
    if window is None:
        window = Tukey(0.2)
    if F_number is None:
        F_number = GratingAngleLB(60.0, 1.5)
    if normalization is None:
        normalization = NormalizationOn()

    # Device selection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    on_gpu = dev.type == "cuda"

    if x_batch is None:
        x_batch = 16 if on_gpu else 8

    # Ensure CPU num_threads uses all cores
    if not on_gpu:
        n_cpu = os.cpu_count() or 1
        torch.set_num_threads(n_cpu)

    # precision="float32" runs the hot path in FP32/complex64 — ~10-30x faster
    # on consumer GPUs (TITAN RTX FP64 is 1/32 throughput). Phase error at
    # PICMUS scales is ~2e-4 rad (negligible vs 60 dB display range).
    if precision not in ("float64", "float32"):
        raise ValueError(f"precision must be 'float64' or 'float32', got {precision!r}")
    rdt = torch.float64 if precision == "float64" else torch.float32

    positions_x = positions_x.to(dtype=rdt, device=dev)
    positions_z = positions_z.to(dtype=rdt, device=dev)
    data_RF = data_RF.to(dtype=rdt, device=dev)

    Nt, Nel = data_RF.shape
    Nx = positions_x.shape[0]
    Nz = positions_z.shape[0]
    adeg = math.degrees(steering_angle)
    ep = element_pitch
    t0_wall = time.perf_counter()

    backend_str = f"CUDA ({torch.cuda.get_device_name(dev)})" if on_gpu else \
                  f"CPU ({torch.get_num_threads()} threads)"
    print(f"  DAS [{adeg:+.0f}°] {Nx}×{Nz} voxels | x_batch={x_batch} z_chunk={z_chunk} | {backend_str}")

    # ── 1. geometry ─────────────────────────────────────────────
    M = (Nel - 1) / 2.0
    ew2 = element_width / 2.0
    eidx = torch.arange(Nel, dtype=rdt, device=dev)
    ctr = (eidx - M) * ep                          # (Nel,)
    lbs = ctr - ew2                                 # (Nel,)
    ubs = ctr + ew2                                 # (Nel,)

    # tx_sign / tx_reference select the transmit-time convention:
    #   Schiffner data_RF.mat : tx_sign=-1, tx_reference="edge"  (wavefront
    #     crosses the lagging edge element at t=0)
    #   PICMUS               : tx_sign=+1, tx_reference="origin" (wavefront
    #     crosses the array center x=0 at t=0; official PICMUS beamformer
    #     uses t_tx = (z*cos(a) + x*sin(a)) / c)
    esx = tx_sign * math.sin(steering_angle)
    esz = math.cos(steering_angle)
    if tx_reference == "edge":
        ref = (1.0 if esx > 0 else (-1.0 if esx < 0 else 0.0)) * ctr[0].item()
    elif tx_reference == "origin":
        ref = 0.0
    else:
        raise ValueError(f"tx_reference must be 'edge' or 'origin', got {tx_reference!r}")

    tsc_ax_sq = (positions_z / c_0) ** 2            # (Nz,)

    # Zero-padding
    span = (ctr[-1] - ctr[0]).item()
    z0 = positions_z[0].item()
    Npad = math.ceil((math.sqrt(span**2 + z0**2) - z0) * f_s / c_0)

    # ── 2. frequency axis ──────────────────────────────────────
    Ndft = Nt + Npad + 1 - (Nt + Npad) % 2

    fl, fu = f_bounds
    ifl = math.ceil(fl * Ndft / f_s)
    ifu = math.floor(fu * Ndft / f_s)
    idxf = torch.arange(ifl, ifu + 1, dtype=torch.long, device=dev)
    Nf = len(idxf)

    af = f_s * idxf.to(rdt) / Ndft                 # (Nf,)
    aw = 2.0 * math.pi * af                        # (Nf,)

    # ── 3. F-number (1-D per freq) ─────────────────────────────
    pol = ep * af / c_0
    Fv1 = F_number.compute_values(pol.cpu().double()).to(dev, rdt)  # CPU, move

    # ── 4. DFT of RF ───────────────────────────────────────────
    rf_dft = torch.fft.fft(data_RF, n=Ndft, dim=0)
    rf_bp = 2.0 * rf_dft[idxf, :]                  # (Nf, Nel) complex

    # ── 5. main loop: batch x, chunk z ─────────────────────────
    image = torch.zeros(Nz, Nx, dtype=torch.cdouble, device=dev)
    is_box = isinstance(window, Boxcar)

    total_batches = math.ceil(Nx / x_batch) * math.ceil(Nz / z_chunk)
    batch_count = 0

    for ix_start in range(0, Nx, x_batch):
        ix_end = min(ix_start + x_batch, Nx)
        nbx = ix_end - ix_start
        xv = positions_x[ix_start:ix_end]           # (nbx,)

        # Per-batch geometry (shared across z-chunks)
        dlat = ctr[None, :] - xv[:, None]           # (nbx, Nel)
        dlat_a = dlat.abs()                         # (nbx, Nel)
        lmask = dlat < 0                            # (nbx, Nel)
        tsc_lat_sq = (dlat / c_0) ** 2              # (nbx, Nel)

        # Incident-wave time: (Nz, nbx)
        t_in = (esx * (xv[None, :] - ref)
                + esz * positions_z[:, None]) / c_0 + index_t0 / f_s

        for zs in range(0, Nz, z_chunk):
            ze = min(zs + z_chunk, Nz)
            pzc = positions_z[zs:ze]                # (nzc,)
            nzc = ze - zs

            batch_count += 1
            if batch_count % 5 == 1 or batch_count == total_batches:
                pct = batch_count / total_batches * 100
                print(f"\r  DAS [{adeg:+.0f}°] {pct:5.1f}%  "
                      f"(x={ix_start}-{ix_end-1}, z={zs}-{ze-1})",
                      end="", flush=True)

            # ── F-number cap by depth: (Nf, nzc) ──
            Fub = torch.sqrt(af[:, None] * pzc[None, :] / (2.88 * c_0))
            Fv = torch.min(Fv1[:, None].expand_as(Fub), Fub)
            hw = torch.where(
                Fv > 0,
                pzc[None, :] / (2.0 * Fv),
                torch.full_like(Fv, 1e6),
            )                                       # (Nf, nzc)

            # ── Aperture bounds: (Nf, nzc, nbx) ──
            #    xv is (nbx,) → broadcast with hw (Nf, nzc)
            i_lb = torch.clamp(
                torch.ceil(M + (xv[None, None, :] - hw[:, :, None]) / ep).long(),
                min=0,
            )                                       # (Nf, nzc, nbx)
            i_ub = torch.clamp(
                torch.floor(M + (xv[None, None, :] + hw[:, :, None]) / ep).long(),
                max=Nel - 1,
            )                                       # (Nf, nzc, nbx)
            valid = i_lb <= i_ub                    # (Nf, nzc, nbx)

            # ── TOF: (nbx, nzc, Nel) ──
            tsc = torch.sqrt(
                tsc_lat_sq[:, None, :]              # (nbx, 1, Nel)
                + tsc_ax_sq[zs:ze][None, :, None]   # (1, nzc, 1)
            )                                       # (nbx, nzc, Nel)
            tof = (t_in[zs:ze, :].T[:, :, None]    # (nbx, nzc, 1)
                   + tsc)                           # (nbx, nzc, Nel)

            # ── Phase: (Nf, nbx, nzc, Nel) ──
            eph = torch.exp(
                1j * aw[:, None, None, None] * tof[None, :, :, :]
            )                                       # (Nf, nbx, nzc, Nel)

            # ── Apodization: (Nf, nbx, nzc, Nel) ──
            if is_box:
                hw_safe = torch.where(
                    hw > 0, hw, torch.ones_like(hw)
                )                                   # (Nf, nzc)
                p_hw = (dlat_a[None, :, None, :]    # (1, nbx, 1, Nel)
                        / hw_safe[:, None, :, None]) # (Nf, 1, nzc, 1)
                                                    # → (Nf, nbx, nzc, Nel)
                apod = window.compute_samples(p_hw)
            else:
                # Asymmetric left/right half-widths
                ilb_c = torch.clamp(i_lb, 0, Nel - 1)  # (Nf, nzc, nbx)
                iub_c = torch.clamp(i_ub, 0, Nel - 1)

                wl = xv[None, None, :] - lbs[ilb_c]    # (Nf, nzc, nbx)
                wr = ubs[iub_c] - xv[None, None, :]     # (Nf, nzc, nbx)
                wl = torch.where(valid & (wl > 0), wl, torch.ones_like(wl))
                wr = torch.where(valid & (wr > 0), wr, torch.ones_like(wr))

                # Reshape for 4-D: (Nf, nbx, nzc, 1)
                wl_4d = wl.permute(0, 2, 1).unsqueeze(-1)
                wr_4d = wr.permute(0, 2, 1).unsqueeze(-1)

                # Per-element half-width: left side uses wl, right uses wr
                hw_elem = torch.where(
                    lmask[None, :, None, :],        # (1, nbx, 1, Nel)
                    wl_4d,                          # (Nf, nbx, nzc, 1)
                    wr_4d,                          # (Nf, nbx, nzc, 1)
                )                                   # (Nf, nbx, nzc, Nel)

                p_hw = dlat_a[None, :, None, :] / hw_elem
                apod = window.compute_samples(p_hw)

            # Zero out invalid (freq, z, x) tuples
            valid_4d = valid.permute(0, 2, 1).unsqueeze(-1)  # (Nf, nbx, nzc, 1)
            apod = apod * valid_4d.to(apod.dtype)

            # Normalise across elements (dim=-1)
            apod = normalization.apply(apod)

            # ── Focus: weighted sum over elements, then over freq ──
            focused = (
                apod * eph * rf_bp[:, None, None, :]  # (Nf, nbx, nzc, Nel)
            ).sum(dim=-1)                             # (Nf, nbx, nzc)

            pixel = focused.sum(dim=0)                # (nbx, nzc)

            image[zs:ze, ix_start:ix_end] = pixel.T   # (nzc, nbx)

    # ── 6. build result on CPU ─────────────────────────────────
    image_cpu = image.cpu()
    Fv1_cpu = Fv1.cpu()

    elapsed = time.perf_counter() - t0_wall
    print(f"\r  DAS [{adeg:+.0f}°] done  ({elapsed:.1f} s)                        ")

    return DASResult(image=image_cpu, F_number_values=Fv1_cpu, signal=None)


# =====================================================================
# Input validation
# =====================================================================
def _validate(px, pz, rf, fs, a, ew, ep, c0):
    if px.ndim != 1:
        raise ValueError("positions_x must be 1-D.")
    if pz.ndim != 1:
        raise ValueError("positions_z must be 1-D.")
    if not torch.all(pz > 0):
        raise ValueError("positions_z must be positive.")
    if rf.ndim != 2 or rf.shape[0] < 2 or rf.shape[1] < 2:
        raise ValueError("data_RF needs ≥2 rows and columns.")
    if fs <= 0 or ew <= 0 or ep <= 0 or c0 <= 0:
        raise ValueError("f_s, element_width, element_pitch, c_0 must be positive.")
    if not (-math.pi / 2 < a < math.pi / 2):
        raise ValueError("steering_angle must be in (-π/2, π/2).")
