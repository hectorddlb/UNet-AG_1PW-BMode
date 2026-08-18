"""Thin wrapper around ``zhang_dnn.measurement_operator_fdf`` for unet_ag.

Contract (enforced by ``tests/test_h_fdf_wrapper.py``):
- The returned operator has **0 learnable parameters**. ``H_Fdf`` is a fixed
  physical model.
- ``residual_fidelity_loss`` backprops only to ``x_pred``; the operator
  itself never receives gradient. This sidesteps the Phase 4 1.2 OOM (see
  ``memory/feedback_h_in_autograd.md``).

For (γ) propuestas (learnable F-number scalars), rebuild the operator inside
``torch.no_grad()`` every N steps and estimate gradient on the scalars via
finite differences in the outer training loop.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

# Reuse the existing operator implementation (do not duplicate).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from zhang_dnn.measurement_operator_fdf import OpHFdf, MultiAngleOpHFdf  # noqa: E402


DEFAULT_F_NUMBER_ANGLE_DEG = 60.0
DEFAULT_F_NUMBER_UB = 1.5


def _index_t0(geometry: dict) -> int:
    """Sample index at which ``t = 0`` falls, i.e. ``-t_start * fs``.

    ``OpHFdf`` applies ``exp(-i w index_t0 / fs)`` to ``Y`` in the forward and
    its conjugate in the adjoint, exactly the role that ``+index_t0 / f_s``
    plays inside ``schiffner_das.das_pw``. Since the DFT runs over the sample
    index — a time axis of ``t - t_start`` — the extra delay must be
    ``-t_start``, not ``+t_start``.

    The previous ``+t_start * fs`` shifted every Field II reconstruction by
    ``2 * t_start * fs`` samples (144 samples = 5.32 mm at fs = 20.832 MHz).
    Cross-correlating the adjoint against the reference ``das_1pw`` gives
    -5.32 mm with ``+t_start * fs`` and exactly 0.00 mm with this expression.
    PICMUS is unaffected: ``t_start = 0`` there.
    """
    return -int(round(float(geometry.get("t_start", 0.0)) * float(geometry["sampling_freq_hz"])))


def make_op_for_angle(
    angle_rad: float,
    geometry: dict,
    grid_z: Tensor,
    grid_x: Tensor,
    device: str = "cuda",
    nf_chunk: Optional[int] = None,
) -> OpHFdf:
    """Build a single-angle Fd-F operator with Schiffner defaults.

    Notes
    -----
    The operator carries no learnable parameters: F-number, window, and
    normalization are passed by value and stored as plain attributes /
    registered buffers. Verified by ``test_make_op_for_angle_returns_op_with_no_learnable_params``.
    """
    gz = grid_z.to(device) if grid_z.device.type != device.split(":")[0] else grid_z
    gx = grid_x.to(device) if grid_x.device.type != device.split(":")[0] else grid_x
    return OpHFdf(
        grid_z=gz,
        grid_x=gx,
        n_elements=int(geometry["n_elements"]),
        pitch_m=float(geometry["pitch_m"]),
        element_width_m=float(geometry["element_width_m"]),
        sampling_freq_hz=float(geometry["sampling_freq_hz"]),
        speed_of_sound=float(geometry["speed_of_sound"]),
        n_samples=int(geometry["n_samples"]),
        angle_rad=float(angle_rad),
        index_t0=_index_t0(geometry),
        device=device,
        nf_chunk=nf_chunk,
    )


def make_multi_angle_op(
    angles_rad: list[float],
    geometry: dict,
    grid_z: Tensor,
    grid_x: Tensor,
    device: str = "cuda",
    nf_chunk: Optional[int] = None,
) -> MultiAngleOpHFdf:
    """Lazy per-angle operator wrapper.

    Operators are built on first ``op[i]`` access and cached. Call
    ``op.free_cache()`` between phases if memory is tight.
    """
    gz = grid_z.to(device) if grid_z.device.type != device.split(":")[0] else grid_z
    gx = grid_x.to(device) if grid_x.device.type != device.split(":")[0] else grid_x
    return MultiAngleOpHFdf(
        grid_z=gz,
        grid_x=gx,
        steering_angles_rad=list(angles_rad),
        n_elements=int(geometry["n_elements"]),
        pitch_m=float(geometry["pitch_m"]),
        element_width_m=float(geometry["element_width_m"]),
        sampling_freq_hz=float(geometry["sampling_freq_hz"]),
        speed_of_sound=float(geometry["speed_of_sound"]),
        n_samples=int(geometry["n_samples"]),
        index_t0=_index_t0(geometry),
        device=device,
        nf_chunk=nf_chunk,
    )


def hfdf_forward_checkpointed(op: OpHFdf, x: Tensor) -> Tensor:
    """``op.forward(x)`` with per-chunk gradient checkpointing.

    The standard ``OpHFdf.forward`` materialises ``(B=nf_chunk, K, Nz, Nx)``
    real and complex tensors (apod_b, phase_b, kernel_b) inside an inner loop
    over ``Nf // nf_chunk`` chunks. With autograd, every chunk's intermediates
    must be retained until backward — at PICMUS grid 612×388 or 256×128 the
    total exceeds 24 GB and OOMs even with a single 1 PW forward (see runs
    185001 and 185010).

    This function recomputes apod/phase/kernel during backward via
    ``torch.utils.checkpoint.checkpoint``. Memory drops to ~one chunk-worth;
    backward compute roughly doubles.

    Gradient flows to ``x`` only (α contract preserved). The chunks must
    not use in-place assignment to a shared buffer — we accumulate via
    ``torch.cat`` after the loop instead.
    """
    from torch.utils.checkpoint import checkpoint

    if x.shape != (op.Nz, op.Nx):
        raise ValueError(f"x.shape {tuple(x.shape)} != ({op.Nz}, {op.Nx})")
    x_c = x.to(op.dtype) if not x.is_complex() else x

    def _make_chunk(s: int, e: int):
        def chunk_fn(x_in: Tensor) -> Tensor:
            apod_b = op._apod_for_freq_batch(s, e)                       # (B, K, Nz, Nx)
            aw_b = op.aw[s:e]
            phase_b = torch.exp(
                -1j * aw_b[:, None, None, None] * op.tau_tof[None, :, :, :]
            )                                                            # (B, K, Nz, Nx) complex
            kernel_b = apod_b.to(op.dtype) * phase_b
            V_chunk = (kernel_b * x_in[None, None, :, :]).sum(dim=(2, 3))  # (B, K)
            return V_chunk.transpose(0, 1).contiguous()                  # (K, B)
        return chunk_fn

    V_pos_chunks: list[Tensor] = []
    for fi_start in range(0, op.Nf, op.nf_chunk):
        fi_end = min(fi_start + op.nf_chunk, op.Nf)
        fn = _make_chunk(fi_start, fi_end)
        V_slice = checkpoint(fn, x_c, use_reentrant=False)               # (K, B_chunk)
        V_pos_chunks.append(V_slice)

    V_pos = torch.cat(V_pos_chunks, dim=1)                               # (K, Nf)
    V_pos = V_pos * op.t0_phase[None, :]

    V_full = torch.zeros(op.K, op.Ndft, dtype=op.dtype, device=op.device)
    V_full = V_full.index_copy(1, op.idxf, V_pos)
    idx_neg = (op.Ndft - op.idxf) % op.Ndft
    V_full = V_full.index_copy(1, idx_neg, V_pos.conj())

    y_full = torch.fft.ifft(V_full, dim=-1).real                          # (K, Ndft)
    return y_full[:, : op.D].transpose(0, 1).contiguous()                 # (D, K)


def residual_fidelity_loss(
    x_pred: Tensor,
    y_obs: Tensor,
    op_a: OpHFdf,
    reduction: str = "mean",
) -> Tensor:
    """``‖y_obs − H_Fdf · x_pred‖²`` for a single angle.

    The operator is fixed (no learnable params). Gradient flows only to
    ``x_pred``. Output dtype matches the operator's float counterpart.
    """
    if x_pred.dim() != 2:
        raise ValueError(f"x_pred must be (Nz, Nx); got {tuple(x_pred.shape)}")
    if y_obs.dim() != 2:
        raise ValueError(f"y_obs must be (D, K); got {tuple(y_obs.shape)}")

    x_dev = x_pred.to(op_a.device)
    y_dev = y_obs.to(op_a.device).to(op_a.float_dtype)
    y_pred = op_a.forward(x_dev)                                # (D, K) real float
    if y_pred.is_complex():
        y_pred = y_pred.real
    residual = y_dev - y_pred.to(y_dev.dtype)
    sq = residual * residual
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    raise ValueError(f"reduction must be 'mean' or 'sum'; got {reduction!r}")
