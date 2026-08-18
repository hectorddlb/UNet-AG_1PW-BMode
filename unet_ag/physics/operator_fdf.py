"""
Fourier-domain forward operator for PWUS with Schiffner's frequency-dependent
F-number (Phase 4 Option C).

This module replaces the time-domain `build_measurement_matrix` (Hanning
apodization, scipy.sparse CSR) with a PyTorch-based operator that applies
the same physics as `schiffner_das.das_pw` but in the forward direction.

The operator is mathematically the adjoint of Schiffner DAS: given an image
`x` (Nz, Nx), it produces the RF data `y` (D, K) that, when fed back through
DAS with the same F-number, recovers approximately `x`.

Design contract (see `docs/drafts/PHASE4_SPRINT0_FOURIER.md`):
- One operator per steering angle. Multi-angle wraps a list of these.
- All math in PyTorch complex64; gradients flow via autograd.
- Adjoint computed analytically (transpose of the forward), validated by
  `<op(x), y> ≈ <x, op.adjoint(y)>` (test T2).
- scipy-LSQR interop via `to_scipy_linear_operator()`.

References
----------
[1] Schiffner & Schmitz, IEEE TUFFC 70(9), 2023.
[2] schiffner_fd-f/schiffner_das/beamforming.py (the adjoint we mirror).
[3] Zhang et al., Med. Image Anal. 70, 2021 (the H_Zhang we replace).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse.linalg as splinalg
import torch
import torch.nn as nn
from torch import Tensor

# `schiffner_das` vive en la raiz del repo; anadimos esa raiz a sys.path para
# que el modulo sea importable aunque se ejecute desde otro directorio.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from schiffner_das.f_numbers import FNumber, GratingAngleLB  # noqa: E402
from schiffner_das.normalizations import Normalization, NormalizationOn  # noqa: E402
from schiffner_das.windows import Tukey, Window  # noqa: E402


class OpHFdf(nn.Module):
    """Fourier-domain forward PWUS operator with Schiffner Fd-F apodization.

    Maps a beamformed image ``x`` of shape ``(Nz, Nx)`` to raw RF channel
    data ``y`` of shape ``(D, K)``. The mapping is linear and the adjoint
    ``op.adjoint(y)`` produces an image-domain reconstruction equivalent (up
    to normalization) to a single-angle Schiffner DAS.

    Construction is per steering angle. Use ``MultiAngleOpHFdf`` to handle
    a sweep of angles.

    Parameters
    ----------
    grid_z, grid_x : Tensor (1-D)
        Axial / lateral grid positions (m). ``grid_z`` must be strictly positive.
    n_elements : int
        K, number of transducer elements (e.g. 128 for PICMUS).
    pitch_m, element_width_m : float
        Transducer geometry.
    sampling_freq_hz : float
        fs.
    speed_of_sound : float
        c.
    n_samples : int
        D, number of RF time samples to produce in the forward pass.
    angle_rad : float
        Steering angle of the transmitted PW.
    f_bounds_hz : (float, float), optional
        Spectral band ``(f_lb, f_ub)``. Defaults to ``(0.5 fc, 1.5 fc)`` with
        ``fc = sampling_freq_hz / 4`` (PICMUS-style).
    index_t0 : int
        Sample-index offset corresponding to ``t = 0``. For Field II this is
        ``-round(t_start * fs)`` (the DFT time axis is ``t - t_start``, so the
        extra delay is negative — verified against ``das_1pw`` by cross-
        correlation). For PICMUS this is 0.
    f_number : FNumber, optional
        Receive F-number model. Default ``GratingAngleLB(60, 1.5)``.
    window : Window, optional
        Receive apodization shape. Default ``Tukey(0.2)``.
    normalization : Normalization, optional
        Per-pixel apodization normalization. Default ``NormalizationOn``
        (matches Schiffner DAS).
    device : str
        ``'cuda'`` or ``'cpu'``.
    dtype : torch.dtype
        Complex dtype. Default ``torch.complex64`` (memory-efficient).
    nf_chunk : int, optional
        Number of frequency bands processed per inner step. Larger values
        amortize per-iteration Python/CUDA-launch overhead at the cost of
        ``O(nf_chunk · K · Nz · Nx)`` extra working memory. Defaults to 8
        on CUDA and 2 on CPU. On a 24 GB GPU at full PICMUS grid the safe
        ceiling is roughly 32; on an A100 (40 GB) up to 64.
    """

    def __init__(
        self,
        grid_z: Tensor,
        grid_x: Tensor,
        n_elements: int,
        pitch_m: float,
        element_width_m: float,
        sampling_freq_hz: float,
        speed_of_sound: float,
        n_samples: int,
        angle_rad: float = 0.0,
        f_bounds_hz: Optional[tuple[float, float]] = None,
        index_t0: int = 0,
        f_number: Optional[FNumber] = None,
        window: Optional[Window] = None,
        normalization: Optional[Normalization] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.complex64,
        nf_chunk: Optional[int] = None,
    ):
        super().__init__()
        # ── Defaults ───────────────────────────────────────────────────
        if f_number is None:
            f_number = GratingAngleLB(60.0, 1.5)
        if window is None:
            window = Tukey(0.2)
        if normalization is None:
            normalization = NormalizationOn()

        self.K = int(n_elements)
        self.D = int(n_samples)
        self.Nz = int(len(grid_z))
        self.Nx = int(len(grid_x))
        self.pitch = float(pitch_m)
        self.element_width = float(element_width_m)
        self.fs = float(sampling_freq_hz)
        self.c = float(speed_of_sound)
        self.angle_rad = float(angle_rad)
        self.index_t0 = int(index_t0)
        self.f_number = f_number
        self.window = window
        self.normalization = normalization
        self.device = torch.device(device)
        self.dtype = dtype  # complex
        self.float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        # nf_chunk: number of frequency bands to process per inner step.
        # Larger = fewer Python iters, more memory per step. Auto default:
        # 8 on CUDA, 2 on CPU. Override via constructor.
        if nf_chunk is None:
            nf_chunk = 8 if self.device.type == "cuda" else 2
        self.nf_chunk = int(nf_chunk)

        # ── f_bounds default ──────────────────────────────────────────
        # NOTE: this default mirrors the PICMUS-style band. For Field II we
        # may want to derive from pulse spectrum. Verify in Sprint 0.
        if f_bounds_hz is None:
            fc_est = sampling_freq_hz / 4.0  # PICMUS-style: fs = 4 * fc
            f_bounds_hz = (0.5 * fc_est, 1.5 * fc_est)
        self.f_lb, self.f_ub = f_bounds_hz

        # ── Precompute (Sprint 0 D2) ──────────────────────────────────
        # Stored as buffers (non-trainable) so they move with .to(device).
        # See docs/drafts/PHASE4_SPRINT0_FOURIER.md §3.3.
        self._build_geometry(grid_z, grid_x)
        self._build_frequency_axis()
        self._build_tau_tof()
        self._build_t0_phase()

        self._norm_op_cached: Optional[float] = None

    # ──────────────────────────────────────────────────────────────────
    # Internal precompute
    # ──────────────────────────────────────────────────────────────────
    def _build_geometry(self, grid_z: Tensor, grid_x: Tensor) -> None:
        """Cache element positions and pixel grid."""
        grid_z = torch.as_tensor(grid_z, dtype=self.float_dtype, device=self.device)
        grid_x = torch.as_tensor(grid_x, dtype=self.float_dtype, device=self.device)
        eidx = torch.arange(self.K, dtype=self.float_dtype, device=self.device)
        x_elem = (eidx - (self.K - 1) / 2.0) * self.pitch  # (K,)

        self.register_buffer("grid_z", grid_z)
        self.register_buffer("grid_x", grid_x)
        self.register_buffer("x_elem", x_elem)
        # Element edges for asymmetric window (matches Schiffner)
        self.register_buffer("elem_lb", x_elem - self.element_width / 2.0)
        self.register_buffer("elem_ub", x_elem + self.element_width / 2.0)

    def _build_frequency_axis(self) -> None:
        """Build the DFT band indices and angular frequencies.

        Following ``schiffner_das.das_pw`` lines 159–174: zero-padding so the
        DFT length covers the maximum TOF, then take positive bands inside
        ``[f_lb, f_ub]``. We rely on conjugate symmetry of the real RF spectrum
        and apply a factor of 2 on positive bands in the adjoint (and forward,
        for transpose consistency).
        """
        # Zero-padding so DFT covers the maximum incident-wave / receive TOF.
        aperture_span = float(self.x_elem[-1] - self.x_elem[0])
        z0 = float(self.grid_z[0])
        Npad = math.ceil((math.sqrt(aperture_span ** 2 + z0 ** 2) - z0) * self.fs / self.c)
        # Force odd Ndft (matches Schiffner; avoids ambiguity at Nyquist)
        Ndft = self.D + Npad + 1 - (self.D + Npad) % 2
        self.Ndft = int(Ndft)

        # Band indices in (0, Ndft//2]
        ifl = math.ceil(self.f_lb * Ndft / self.fs)
        ifu = math.floor(self.f_ub * Ndft / self.fs)
        idxf = torch.arange(ifl, ifu + 1, dtype=torch.long, device=self.device)
        Nf = int(len(idxf))
        if Nf == 0:
            raise ValueError(
                f"Empty frequency band: f_bounds={self.f_lb:.2e}..{self.f_ub:.2e} Hz "
                f"yields no DFT bins at Ndft={Ndft}, fs={self.fs:.2e}."
            )

        af = self.fs * idxf.to(self.float_dtype) / Ndft  # actual freqs in Hz, shape (Nf,)
        aw = 2.0 * math.pi * af                          # angular freqs, shape (Nf,)

        self.register_buffer("idxf", idxf)
        self.register_buffer("af", af)
        self.register_buffer("aw", aw)
        self.Nf = Nf

        # F-number per frequency (compute on CPU as Schiffner does, then move)
        pol = (self.pitch * af / self.c).cpu()
        Fv1 = self.f_number.compute_values(pol).to(self.device, self.float_dtype)
        self.register_buffer("Fv1", Fv1)

    def _build_tau_tof(self) -> None:
        """Precompute geometric time-of-flight τ_tof(k, z, x) = τ_transmit + τ_receive.

        Uses Schiffner's reference-point convention for the incident wave so
        the adjoint can be compared bitwise to ``das_pw``. The ``index_t0``
        offset is applied as a phase shift downstream — NOT folded into
        ``tau_tof`` — so this buffer is purely geometric.

        Memory: ``K · Nz · Nx · 4`` bytes (float32). Typical ≈ 120 MB.
        """
        K = self.K
        # Schiffner convention (das_pw line 153–155)
        esx = -math.sin(self.angle_rad)
        esz = math.cos(self.angle_rad)
        if esx > 0:
            ref = 1.0 * float(self.x_elem[0])
        elif esx < 0:
            ref = -1.0 * float(self.x_elem[0])
        else:
            ref = 0.0
        self.esx = float(esx)
        self.esz = float(esz)
        self.ref = float(ref)

        # Incident wave time, shape (Nz, Nx)
        t_in = (esx * (self.grid_x[None, :] - ref)
                + esz * self.grid_z[:, None]) / self.c

        # Receive time per element, shape (K, Nz, Nx)
        dz_sq = self.grid_z[None, :, None] ** 2                       # (1, Nz, 1)
        dx_lat = self.grid_x[None, None, :] - self.x_elem[:, None, None]  # (K, 1, Nx)
        tsc = torch.sqrt(dz_sq + dx_lat ** 2) / self.c                # (K, Nz, Nx)

        # Total geometric TOF — broadcast (1, Nz, Nx) + (K, Nz, Nx)
        tau_tof = t_in[None, :, :] + tsc                              # (K, Nz, Nx)
        self.register_buffer("tau_tof", tau_tof)

        # Helpers for asymmetric Tukey apodization (see _apod_for_freq_batch):
        # dlat_signed[k, x] = x_elem[k] - grid_x[x]  (positive if element right of pixel)
        dlat_signed = self.x_elem[:, None] - self.grid_x[None, :]     # (K, Nx)
        self.register_buffer("dlat_signed", dlat_signed)
        self.register_buffer("dlat_abs", dlat_signed.abs())
        self.register_buffer("lmask", dlat_signed < 0)                # element is LEFT of pixel

    def _build_t0_phase(self) -> None:
        """``exp(-i ω · index_t0 / fs)`` per frequency band, shape ``(Nf,)`` complex.

        Applied multiplicatively to ``Y[f, k]`` in forward; conjugate in adjoint.
        Captures the Field II ``t_start`` offset (PICMUS uses 0).
        """
        phase = -self.aw * (self.index_t0 / self.fs)  # (Nf,) real
        # Complex exp via real cos/sin (autograd-friendly for any future learnable t0)
        re = torch.cos(phase)
        im = torch.sin(phase)
        t0_phase = torch.complex(re, im).to(self.dtype)
        self.register_buffer("t0_phase", t0_phase)

    def _apod_for_freq_batch(self, fi_start: int, fi_end: int) -> Tensor:
        """Apodization weights for a chunk of frequency bands, vectorized.

        Replicates ``schiffner_das.das_pw`` lines 218–290 (per-z depth cap on
        F, aperture half-width, asymmetric Tukey using active aperture edges,
        NormalizationOn), with a leading frequency batch dim so all bands in
        ``[fi_start, fi_end)`` are computed in one fused pass.

        Returns
        -------
        Tensor of shape ``(B, K, Nz, Nx)`` real, where ``B = fi_end - fi_start``.
        """
        f_b = self.af[fi_start:fi_end]                                # (B,)
        F_b = self.Fv1[fi_start:fi_end]                               # (B,)
        B = f_b.shape[0]

        # Per-z depth cap (B, Nz)
        Fub_z = torch.sqrt(f_b[:, None] * self.grid_z[None, :] / (2.88 * self.c))
        F_eff = torch.minimum(F_b[:, None].expand(B, self.Nz), Fub_z)
        # Where F == 0, sentinel half-width so no element passes the aperture test.
        hw_z = torch.where(
            F_eff > 0,
            self.grid_z[None, :] / (2.0 * F_eff),
            torch.full_like(F_eff, 1e6),
        )                                                              # (B, Nz)

        M_half = (self.K - 1) / 2.0
        x_minus = self.grid_x[None, None, :] - hw_z[:, :, None]        # (B, Nz, Nx)
        x_plus = self.grid_x[None, None, :] + hw_z[:, :, None]         # (B, Nz, Nx)
        i_lb = torch.clamp(torch.ceil(M_half + x_minus / self.pitch).long(), min=0)
        i_ub = torch.clamp(torch.floor(M_half + x_plus / self.pitch).long(), max=self.K - 1)
        valid = i_lb <= i_ub                                           # (B, Nz, Nx)

        ilb_c = torch.clamp(i_lb, 0, self.K - 1)
        iub_c = torch.clamp(i_ub, 0, self.K - 1)
        # Fancy index: self.elem_lb is (K,); ilb_c is (B, Nz, Nx) long → (B, Nz, Nx) float.
        wl = self.grid_x[None, None, :] - self.elem_lb[ilb_c]          # (B, Nz, Nx)
        wr = self.elem_ub[iub_c] - self.grid_x[None, None, :]          # (B, Nz, Nx)
        wl = torch.where(valid & (wl > 0), wl, torch.ones_like(wl))
        wr = torch.where(valid & (wr > 0), wr, torch.ones_like(wr))

        # Per-(B, K, Nz, Nx) half-width: lmask picks wl vs wr based on element side.
        hw_elem = torch.where(
            self.lmask[None, :, None, :],     # (1, K, 1, Nx)
            wl[:, None, :, :],                # (B, 1, Nz, Nx)
            wr[:, None, :, :],                # (B, 1, Nz, Nx)
        )                                     # (B, K, Nz, Nx)

        p_over_hw = self.dlat_abs[None, :, None, :] / hw_elem          # (B, K, Nz, Nx)
        apod = self.window.compute_samples(p_over_hw)                  # (B, K, Nz, Nx)

        # Aperture-index mask, broadcast to (B, K, Nz, Nx)
        elem_idx = torch.arange(self.K, device=self.device).view(1, -1, 1, 1)
        in_aper = (elem_idx >= i_lb[:, None, :, :]) & (elem_idx <= i_ub[:, None, :, :])
        apod = apod * (valid[:, None, :, :] & in_aper).to(apod.dtype)

        # NormalizationOn along element axis (now dim=1).
        s = apod.sum(dim=1, keepdim=True)                              # (B, 1, Nz, Nx)
        s = torch.where(s.abs() > 0, s, torch.ones_like(s))
        apod = apod / s

        return apod                                                    # (B, K, Nz, Nx)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    def forward(self, x: Tensor) -> Tensor:
        """Forward: image ``x`` of shape ``(Nz, Nx)`` → RF data ``y`` of shape ``(D, K)``.

        Algorithm (see ``docs/drafts/PHASE4_SPRINT0_FOURIER.md`` §3.1):
          1. For each positive-band frequency ``f_b``, compute
             ``V[f_b, k] = 2 · Σ_n  apod(f_b, k, n) · exp(-i ω_b τ_geom(k, n)) · x[n]``
             then ``V[f_b, k] *= exp(-i ω_b · t0/fs)``.
          2. Mirror conjugate to negative bands so the IFFT yields a real signal.
          3. ``y = IFFT(V_full).real[:D]``.

        The factor 2 on positive bands is the transpose of the factor 2 in the adjoint
        (which itself doubles positive-band contributions because conjugate-symmetric
        negative bands carry the same information).
        """
        if x.shape != (self.Nz, self.Nx):
            raise ValueError(f"x.shape {x.shape} != expected ({self.Nz}, {self.Nx})")

        x_c = x.to(self.dtype) if not x.is_complex() else x  # (Nz, Nx) complex

        # Build positive-band spectrum (K, Nf) in chunks of nf_chunk frequencies.
        # Each chunk computes apod, phase, kernel and contracts with x in a single
        # fused pass — replaces an Nf-long Python/CUDA-launch loop.
        V_pos = torch.zeros(self.K, self.Nf, dtype=self.dtype, device=self.device)
        for fi_start in range(0, self.Nf, self.nf_chunk):
            fi_end = min(fi_start + self.nf_chunk, self.Nf)
            apod_b = self._apod_for_freq_batch(fi_start, fi_end)       # (B, K, Nz, Nx) real
            aw_b = self.aw[fi_start:fi_end]                            # (B,)
            phase_b = torch.exp(
                -1j * aw_b[:, None, None, None] * self.tau_tof[None, :, :, :]
            )                                                          # (B, K, Nz, Nx) complex
            kernel_b = apod_b.to(self.dtype) * phase_b                 # (B, K, Nz, Nx) complex
            # Contract over (Nz, Nx) with x: (B, K, Nz, Nx) · (Nz, Nx) → (B, K)
            V_chunk = (kernel_b * x_c[None, None, :, :]).sum(dim=(2, 3))
            V_pos[:, fi_start:fi_end] = V_chunk.transpose(0, 1)        # (K, B)

        # Apply t0 phase. NO factor 2 here: the conjugate-mirror to negative bands
        # below explicitly populates both halves of the spectrum, so the IFFT
        # naturally accounts for the bilateral contribution. The Schiffner factor 2
        # lives only in the adjoint where it compensates for "summing positive
        # bands only" (test T2 verifies that this choice yields a true adjoint pair).
        V_pos = V_pos * self.t0_phase[None, :]                         # (K, Nf)

        # Assemble full DFT spectrum of length Ndft with conjugate symmetry
        V_full = torch.zeros(self.K, self.Ndft, dtype=self.dtype, device=self.device)
        V_full[:, self.idxf] = V_pos
        # Mirror conjugate to negative bands (idx_neg = Ndft - idxf_pos, modulo)
        idx_neg = (self.Ndft - self.idxf) % self.Ndft
        V_full[:, idx_neg] = V_pos.conj()

        # IFFT along DFT axis, take first D real samples, return (D, K)
        y_full = torch.fft.ifft(V_full, dim=-1).real                   # (K, Ndft)
        # IFFT in torch has 1/Ndft scaling; Schiffner's DFT/IFFT convention matches.
        y = y_full[:, : self.D].transpose(0, 1).contiguous()           # (D, K)
        return y

    def adjoint(self, y: Tensor) -> Tensor:
        """Adjoint: RF data ``y`` of shape ``(D, K)`` → image ``x_hat`` of shape ``(Nz, Nx)``.

        Equivalent to ``schiffner_das.das_pw(y).real`` for a single angle, with our
        ``NormalizationOn`` choice. Algorithm:
          1. Zero-pad ``y`` along time axis to length ``Ndft``, then ``FFT``.
          2. Take positive bands and multiply by 2 · conj(t0_phase).
          3. Focus: ``x_hat[n] = Re Σ_f Σ_k apod(f, k, n) · exp(+i ω τ_geom(k, n)) · Y[f, k]``.
        """
        if y.shape != (self.D, self.K):
            raise ValueError(f"y.shape {y.shape} != expected ({self.D}, {self.K})")

        # Zero-pad along time axis, FFT
        y_t = y.transpose(0, 1)                                        # (K, D)
        # FFT of length Ndft — pad implicitly via `n=` argument
        Y_full = torch.fft.fft(y_t.to(self.dtype), n=self.Ndft, dim=-1)  # (K, Ndft) complex

        # Positive bands with t0 correction (conj because adjoint).
        # Factor (2/Ndft): the "2" accounts for conjugate-symmetric negative bands
        # carried implicitly; the "1/Ndft" compensates for torch.fft.ifft's internal
        # 1/Ndft scaling so that <Hx, y> = <x, H^T y> holds exactly (test T2).
        # Note: this makes adjoint(y) smaller than Schiffner DAS by a factor of Ndft;
        # R1 regression test allows per-image scale match.
        Y_pos = (2.0 / self.Ndft) * Y_full[:, self.idxf] * self.t0_phase.conj()[None, :]  # (K, Nf)

        # Focus: accumulate per band, chunked over frequency.
        x_hat = torch.zeros(self.Nz, self.Nx, dtype=self.dtype, device=self.device)
        for fi_start in range(0, self.Nf, self.nf_chunk):
            fi_end = min(fi_start + self.nf_chunk, self.Nf)
            apod_b = self._apod_for_freq_batch(fi_start, fi_end)       # (B, K, Nz, Nx)
            aw_b = self.aw[fi_start:fi_end]                            # (B,)
            phase_b = torch.exp(
                +1j * aw_b[:, None, None, None] * self.tau_tof[None, :, :, :]
            )                                                          # (B, K, Nz, Nx)
            kernel_b = apod_b.to(self.dtype) * phase_b                 # (B, K, Nz, Nx)
            # Y_pos chunk: (K, B) → (B, K, 1, 1) for broadcast contraction
            Y_chunk = Y_pos[:, fi_start:fi_end].transpose(0, 1)        # (B, K)
            x_hat = x_hat + (
                kernel_b * Y_chunk[:, :, None, None]
            ).sum(dim=(0, 1))                                          # (Nz, Nx)

        return x_hat.real

    @property
    def norm_op(self) -> float:
        """Estimate ‖H‖_op via power iteration on H^T H. Cached after first call."""
        if self._norm_op_cached is not None:
            return self._norm_op_cached
        n_iter = 30
        x = torch.randn(self.Nz, self.Nx, dtype=self.float_dtype, device=self.device)
        x = x / x.norm()
        with torch.no_grad():
            for _ in range(n_iter):
                y = self.forward(x)
                x = self.adjoint(y)
                nrm = float(x.norm())
                x = x / (nrm + 1e-30)
        self._norm_op_cached = math.sqrt(nrm)
        return self._norm_op_cached

    # ──────────────────────────────────────────────────────────────────
    # Interop
    # ──────────────────────────────────────────────────────────────────
    def to_scipy_linear_operator(self) -> splinalg.LinearOperator:
        """Wrap as scipy.sparse.linalg.LinearOperator for LSQR.

        Both directions transfer to CPU/numpy. Slow for high iter counts;
        use only for occasional baseline computation.
        """
        Nz, Nx, D, K = self.Nz, self.Nx, self.D, self.K
        device = self.device
        dtype = self.float_dtype

        def matvec(x_flat):
            x = torch.from_numpy(x_flat.reshape(Nz, Nx).astype(np.float32)).to(device, dtype)
            with torch.no_grad():
                y = self.forward(x)
            return y.cpu().numpy().reshape(-1).astype(np.float32)

        def rmatvec(y_flat):
            y = torch.from_numpy(y_flat.reshape(D, K).astype(np.float32)).to(device, dtype)
            with torch.no_grad():
                xh = self.adjoint(y)
            return xh.cpu().numpy().reshape(-1).astype(np.float32)

        return splinalg.LinearOperator(
            shape=(D * K, Nz * Nx),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.float32,
        )


class MultiAngleOpHFdf:
    """Lazy-builder for per-angle operators, with LRU-style cache.

    Builds each `OpHFdf` instance on first access; reuses thereafter.
    Use `free_cache()` between training stages if memory tight.
    """

    def __init__(
        self,
        grid_z: Tensor,
        grid_x: Tensor,
        steering_angles_rad: list[float],
        **op_kwargs,
    ):
        self.grid_z = grid_z
        self.grid_x = grid_x
        self.angles = list(steering_angles_rad)
        self.op_kwargs = op_kwargs
        self._cache: dict[int, OpHFdf] = {}

    def __len__(self) -> int:
        return len(self.angles)

    def __getitem__(self, idx_angle: int) -> OpHFdf:
        if idx_angle not in self._cache:
            self._cache[idx_angle] = OpHFdf(
                grid_z=self.grid_z,
                grid_x=self.grid_x,
                angle_rad=self.angles[idx_angle],
                **self.op_kwargs,
            )
        return self._cache[idx_angle]

    def free_cache(self) -> None:
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
