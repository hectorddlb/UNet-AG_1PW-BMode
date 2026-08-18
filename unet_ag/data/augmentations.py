"""Data augmentation for sim → exp domain transfer (P10.ii).

Applied at training time on the raw RF + the geometry. Each augmentation has
an ``apply_prob`` and a random magnitude per call.

Supported augmentations:
- ``snr_jitter`` — add AWGN at a randomly chosen SNR ∈ ``snr_db_range`` (dB).
- ``t_start_jitter`` — perturb the Field II time offset by ±``t_start_jitter_ns`` ns.
- ``c_jitter`` — perturb the speed-of-sound by ±``c_jitter_frac`` (relative).
- ``scatterer_density_jitter`` — randomly scale RF amplitude (proxy for
  scatterer density variations) by a multiplicative factor in
  ``scatterer_scale_range``.
- ``pulse_shape_jitter`` (B1-v5) — Gaussian bandpass on RF along temporal axis.
  Center freq jittered ±``pulse_fc_shift_hz`` and -6 dB BW sampled uniformly
  from ``pulse_bw_range_hz``. Closes the spectral gap measured (2026-05-29)
  between Field II training (BW=1.82 MHz) and PICMUS exp (BW=1.04 MHz) —
  exposes the model to the full PICMUS-side BW distribution during train,
  intended to mitigate the OOD resolution penalty seen on PICMUS.

Notes
-----
- Augmentations modify ``rf_1pw`` and ``rf_all`` consistently when both are
  present.
- ``t_start_jitter`` and ``c_jitter`` modify the **geometry** dict; downstream
  ``H_Fdf`` rebuild has to pick up the new values. The trainer handles that.
- ``pulse_shape_jitter`` only mutates the RF, not ``geometry["fc"]`` — the
  label fc stays at the nominal probe center; the filter physically shapes
  the temporal spectrum of the RF samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .rf_loader import RFSample


_BW_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))      # σ = BW / 2.355


@dataclass
class AugConfig:
    apply_prob: float = 0.5
    snr_db_range: tuple[float, float] = (20.0, 40.0)
    snr_apply_prob: float = 0.8
    t_start_jitter_ns: float = 30.0
    t_start_apply_prob: float = 0.5
    c_jitter_frac: float = 0.01            # ±1% speed of sound
    c_apply_prob: float = 0.3
    scatterer_scale_range: tuple[float, float] = (0.8, 1.2)
    scatterer_apply_prob: float = 0.5
    # B1-v5 pulse-shape jitter (Gaussian bandpass on RF along time axis).
    # Range covers measured spectral gap Field II 1.82 MHz → PICMUS exp 1.04 MHz
    # (-6 dB BW). fc shift ±300 kHz covers measured peak range [4.64, 5.04] MHz.
    pulse_bw_range_hz: tuple[float, float] = (1.0e6, 2.5e6)
    pulse_fc_shift_hz: float = 300e3
    pulse_apply_prob: float = 0.0          # off by default; B1-v5 trainer turns it on


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def add_awgn(rf: torch.Tensor, snr_db: float, rng: np.random.Generator) -> torch.Tensor:
    sig_pow = float((rf * rf).mean().item()) + 1e-30
    noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    noise = torch.from_numpy(rng.normal(0.0, np.sqrt(noise_pow), size=rf.shape)).to(rf.dtype)
    return rf + noise


def gaussian_bandpass_rf(
    rf: torch.Tensor, fs: float, fc_hz: float, bw_hz: float
) -> torch.Tensor:
    """Apply a Gaussian-magnitude bandpass along the temporal axis.

    Parameters
    ----------
    rf : 2D ``(D, K)`` or 3D ``(D, K, n_angles)`` real tensor.
        Filtering happens along axis 0 (time).
    fs : float
        Sampling frequency in Hz.
    fc_hz : float
        Passband center frequency in Hz.
    bw_hz : float
        Passband -6 dB full-width in Hz (Gaussian σ = ``bw_hz / 2.355``).
    """
    sigma = bw_hz * _BW_TO_SIGMA
    D = rf.shape[0]
    Y = torch.fft.rfft(rf, dim=0)                                # (D//2+1, ...)
    freqs = torch.fft.rfftfreq(D, d=1.0 / fs).to(rf.device).to(rf.dtype)
    H = torch.exp(-0.5 * ((freqs - fc_hz) / sigma) ** 2)         # (D//2+1,)
    # Broadcast H across remaining dims.
    shape = [H.shape[0]] + [1] * (rf.dim() - 1)
    H = H.view(*shape)
    return torch.fft.irfft(Y * H, n=D, dim=0)


def apply_augmentations(
    sample: RFSample,
    cfg: AugConfig,
    seed: int | None = None,
) -> RFSample:
    """Return a NEW RFSample with augmentations applied (does not mutate input)."""
    rng = _rng(seed)

    rf1 = sample.rf_1pw.clone()
    rf_all = sample.rf_all.clone() if sample.rf_all is not None else None
    geom = dict(sample.geometry)

    if rng.random() < cfg.snr_apply_prob:
        snr = float(rng.uniform(*cfg.snr_db_range))
        rf1 = add_awgn(rf1, snr, rng)
        if rf_all is not None:
            rf_all = add_awgn(rf_all, snr, rng)

    if rng.random() < cfg.scatterer_apply_prob:
        s = float(rng.uniform(*cfg.scatterer_scale_range))
        rf1 = rf1 * s
        if rf_all is not None:
            rf_all = rf_all * s

    if rng.random() < cfg.t_start_apply_prob:
        dt = float(rng.uniform(-cfg.t_start_jitter_ns, cfg.t_start_jitter_ns)) * 1e-9
        geom["t_start"] = float(geom.get("t_start", 0.0)) + dt

    if rng.random() < cfg.c_apply_prob:
        c0 = float(geom["speed_of_sound"])
        df = float(rng.uniform(-cfg.c_jitter_frac, cfg.c_jitter_frac))
        geom["speed_of_sound"] = c0 * (1.0 + df)

    if rng.random() < cfg.pulse_apply_prob:
        fs = float(geom["sampling_freq_hz"])
        fc_nominal = float(geom["fc"])
        fc_shift = float(rng.uniform(-cfg.pulse_fc_shift_hz, cfg.pulse_fc_shift_hz))
        bw = float(rng.uniform(*cfg.pulse_bw_range_hz))
        fc = fc_nominal + fc_shift
        rf1 = gaussian_bandpass_rf(rf1, fs, fc, bw)
        if rf_all is not None:
            rf_all = gaussian_bandpass_rf(rf_all, fs, fc, bw)

    return RFSample(
        rf_1pw=rf1,
        rf_all=rf_all,
        angles_deg=sample.angles_deg,
        geometry=geom,
        phantom_id=sample.phantom_id,
        rf_norm=sample.rf_norm,
    )
