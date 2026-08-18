"""Train propuesta B1 — U-Net + Conditional Attention Gate by F-number.

Input pipeline (per phantom):
  1. ``rf_1pw`` (D, K) ← FieldIIMultiAngleLoader, z-score normalised.
  2. ``x0 = OpHFdf.adjoint(rf_1pw)`` (Nz, Nx), computed once per phantom inside
     ``torch.no_grad()`` (no autograd through H — enforces the α contract).
  3. ``f_map = compute_f_number_map_at_fc(grid_z, grid_x, geometry)``,
     (Nz, Nx) float32, broadcast as input channel #2 conditioning.

Target:
  - Schiffner Fd-F 75 PW image read from
    ``results_schiffner_targets_fieldii/<ptype>/<phantom_id>/compounded.npy``.
  - Shapes must match (Nz, Nx). If not, raise — there is no auto-resize.

Loss: MSE + 0.1·L1 (mse_l1_loss).

CLI:
  python -m unet_ag.train.b1_attention_gate \
      --data_root datasets/Field\\ II/field_ii_data_multiangle \
      --target_root results_schiffner_targets_fieldii \
      --output_dir results_v2/b1 \
      --max_phantoms_train 5 --epochs 2 --smoke_steps 6 --device cpu

A larger sbatch template lives in ``slurm/train_b1.sbatch``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch import Tensor

from scipy.signal import hilbert

from ..data.augmentations import AugConfig, apply_augmentations
from ..data.manifest import field_ii_split
from ..data.rf_loader import FieldIIMultiAngleLoader, zscore_rf
from ..losses.supervised import mse_l1_ssim_loss
from ..models.b1_unet_attention_fnum import B1UNetAttentionFNum
from ..physics.f_number_map import compute_f_number_map_band_avg
from ..physics.h_fdf_wrapper import make_op_for_angle
from .base_trainer import parse_common_args, run_training


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


def _iter_split(
    loader: FieldIIMultiAngleLoader,
    indices: list[int],
    target_root: Path,
    grid_z: Tensor,
    grid_x: Tensor,
    device: str,
    aug_cfg: Optional[AugConfig],
    seed_base: int,
) -> Iterator[tuple[dict, Tensor]]:
    Nz, Nx = grid_z.shape[0], grid_x.shape[0]
    for step, idx in enumerate(indices):
        sample = loader[idx]
        if aug_cfg is not None:
            sample = apply_augmentations(sample, aug_cfg, seed=seed_base + step)
            rf, _ = zscore_rf(sample.rf_1pw)
            sample.rf_1pw = rf
        # Decide ptype from phantom_id prefix
        ptype = "points" if sample.phantom_id.startswith("points") else "cysts"
        target_raw = _load_target(target_root, sample.phantom_id, ptype, Nz, Nx)
        if target_raw is None:
            continue                                      # skip phantoms without a Schiffner target yet
        x0_raw = _adjoint_image(sample.rf_1pw, sample.geometry, grid_z, grid_x, device)
        # F1: envelope + log dynamic-range compress to [0, 1]. Both input and
        # target live in the same B-mode space; eliminates the zero attractor.
        x0 = _envelope_log_compress(x0_raw)
        target = _envelope_log_compress(target_raw)
        # F2: band-averaged F(z) so the conditioning map has depth variation
        # (compute_f_number_map_at_fc was constant=1.5 in PICMUS depth range).
        f_map = compute_f_number_map_band_avg(
            grid_z.cpu().numpy(), grid_x.cpu().numpy(), sample.geometry
        )
        f_map_t = torch.from_numpy(f_map).to(device)
        inputs = {
            "x0": x0.to(device).unsqueeze(0).unsqueeze(0),         # (1, 1, Nz, Nx)
            "f_map": f_map_t.unsqueeze(0).unsqueeze(0),            # (1, 1, Nz, Nx)
        }
        yield inputs, target.to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, Nz, Nx)


def _forward_and_loss(model, inputs, target, args):
    pred = model(inputs["x0"], inputs["f_map"])
    loss, br = mse_l1_ssim_loss(
        pred, target,
        mse_weight=args.mse_weight,
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        data_range=1.0,
    )
    return loss, br, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="B1 — UNet + Conditional Attention Gate")
    parser.add_argument("--target_root", type=Path, required=True,
                        help="results_schiffner_targets_fieldii/")
    parser.add_argument("--phantom_type", choices=["points", "cysts", "both"], default="points")
    parser.add_argument("--nz", type=int, default=612)
    parser.add_argument("--nx", type=int, default=388)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--embed_ch", type=int, default=16)
    parser.add_argument("--mse_weight", type=float, default=0.7)
    parser.add_argument("--l1_weight", type=float, default=0.2)
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--no_aug", action="store_true")
    args = parse_common_args(parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loader = FieldIIMultiAngleLoader(args.data_root, phantom_type=args.phantom_type, keep_all_angles=False)
    splits = field_ii_split(args.data_root, phantom_types=[args.phantom_type] if args.phantom_type != "both" else None)
    if args.phantom_type == "both":
        train_files = splits["points"].train + splits["cysts"].train
        val_files = splits["points"].val + splits["cysts"].val
    else:
        train_files = splits[args.phantom_type].train
        val_files = splits[args.phantom_type].val

    name_to_idx = {p.stem: i for i, p in enumerate(loader.file_list)}
    train_indices = [name_to_idx[p.stem] for p in train_files if p.stem in name_to_idx]
    val_indices = [name_to_idx[p.stem] for p in val_files if p.stem in name_to_idx]
    if args.max_phantoms_train:
        train_indices = train_indices[: args.max_phantoms_train]
    if args.max_phantoms_val:
        val_indices = val_indices[: args.max_phantoms_val]

    sample0 = loader[train_indices[0]] if train_indices else loader[0]
    grid_z, grid_x = _build_grid(sample0.geometry, args.nz, args.nx)

    model = B1UNetAttentionFNum(in_ch=1, base_ch=args.base_ch, embed_ch=args.embed_ch).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"B1 params: {n_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    aug_cfg = None if args.no_aug else AugConfig()
    def train_iter():
        return _iter_split(loader, train_indices, args.target_root, grid_z, grid_x, args.device, aug_cfg, seed_base=args.seed * 1000)
    def val_iter():
        return _iter_split(loader, val_indices, args.target_root, grid_z, grid_x, args.device, None, seed_base=0)

    run_training(args, model, optimizer, train_iter, val_iter if val_indices else None, _forward_and_loss)


if __name__ == "__main__":
    main()
