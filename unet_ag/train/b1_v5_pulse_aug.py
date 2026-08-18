"""B1-v5 — B1-v4 arch + L1-anchor + runtime pulse-shape augmentation.

Motivación (2026-05-29). El diagnóstico FT-PICMUS (jobs 187577-582) cerró
null: ningún modo corto post-hoc rescata la OOD penalty sobre PICMUS. La
medición espectral sobre RF (2026-05-29) reveló el gap que el FT no
testeaba: Field II training fc_peak≈4.97 MHz BW(-6 dB)≈1.82 MHz vs PICMUS
exp resolution fc_peak≈5.04 MHz BW≈1.04 MHz. El diagnóstico (B), grid
search 5×5 (fc, BW) sobre el RF PICMUS pre-adjoint, confirmó que el gap
es espectral (algunos pre-filtros mejoran val_total >10% vs identity).

B1-v5 ataca el gap durante TRAIN COMPLETO en lugar de post-hoc: cada
batch del Field II dataloader recibe con prob ``pulse_apply_prob`` un
bandpass gaussiano con BW ∈ ``pulse_bw_range_hz`` y center jittered ±
``pulse_fc_shift_hz``. El modelo aprende invarianza al ancho de banda
del pulso en lugar de memorizar la PSF específica de Field II.

Arch idéntica a B1-v4: ``B1V3Residual`` (residual skip + zero-init
``out_conv`` + ``log_gain`` learnable), mismo target Schiffner_75PW
env+log, misma loss MSE+L1+SSIM+L1-anchor. Único cambio: AugConfig por
default activa el pulse-shape jitter.

Hipótesis a validar:
- (H1) En eval Field II, B1-v5 mantiene la métrica de B1-v4(0.3)
  (no degrada el caso in-distribution; aug regulariza sin matar).
- (H2) En eval PICMUS exp resolution, B1-v5 reduce el penalty
  FWHM_lat (de 4.96 mm vs DAS 2.47 mm) hacia el rango DAS.
- (H3) En eval PICMUS exp contrast, B1-v5 al menos preserva el gCNR
  ganancia de 0.776 ya alcanzado por B1-v4(0.3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..data.augmentations import AugConfig
from ..data.manifest import field_ii_split
from ..data.rf_loader import FieldIIMultiAngleLoader
from ..models.b1_v3_residual import B1V3Residual
from .common import _build_grid
from .b1_v4_anchored import _iter_split, _forward_and_loss
from .base_trainer import parse_common_args, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="B1-v5 — B1-v4 + pulse-shape augmentation")
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--phantom_type", choices=["points", "cysts", "both"], default="points")
    parser.add_argument("--nz", type=int, default=612)
    parser.add_argument("--nx", type=int, default=388)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--embed_ch", type=int, default=16)
    parser.add_argument("--mse_weight", type=float, default=0.7)
    parser.add_argument("--l1_weight", type=float, default=0.2)
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--anchor_weight", type=float, default=0.3,
                        help="B1-v4 winning value (default 0.3).")
    parser.add_argument("--gain_reg_weight", type=float, default=0.0)
    parser.add_argument("--init_gain", type=float, default=1.0)
    parser.add_argument("--freeze_gain", action="store_true")
    parser.add_argument("--no_aug", action="store_true",
                        help="Disable ALL augmentations (incl. pulse jitter). For ablation.")
    parser.add_argument("--target_is_bmode", action="store_true",
                        help="F.3 distillation: targets are already envelope-log-compressed "
                             "B-mode images in [-1, 1] (DRUS teacher output). Skip the trainer's "
                             "hilbert/log-compress and just remap [-1, 1] -> [0, 1].")
    parser.add_argument("--pulse_apply_prob", type=float, default=0.7,
                        help="Per-step probability of applying the bandpass.")
    parser.add_argument("--pulse_bw_min_hz", type=float, default=1.0e6,
                        help="Lower bound on sampled -6 dB BW (Hz). Default covers PICMUS exp 1.04 MHz.")
    parser.add_argument("--pulse_bw_max_hz", type=float, default=2.5e6,
                        help="Upper bound on sampled -6 dB BW (Hz). Default >Field II native 1.82 MHz.")
    parser.add_argument("--pulse_fc_shift_hz", type=float, default=300e3,
                        help="± uniform shift on filter fc, around geometry['fc'].")
    args = parse_common_args(parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loader = FieldIIMultiAngleLoader(args.data_root, phantom_type=args.phantom_type, keep_all_angles=False)
    splits = field_ii_split(args.data_root,
                            phantom_types=[args.phantom_type] if args.phantom_type != "both" else None)
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

    model = B1V3Residual(
        base_ch=args.base_ch, embed_ch=args.embed_ch,
        init_gain=args.init_gain, train_gain=not args.freeze_gain,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_unet = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
    print(f"B1-v5 (B1-v4 + pulse-shape aug, base_ch={args.base_ch}) params: {n_params:,} (unet={n_unet:,})")
    print(f"  init γ : gain={args.init_gain}  freeze={args.freeze_gain}")
    if args.target_is_bmode:
        print(f"  target : B-mode [-1,1]->[0,1] (F.3 distillation, NO re-envelope)")
    print(f"  loss   : mse={args.mse_weight}  l1={args.l1_weight}  ssim={args.ssim_weight}"
          f"  anchor={args.anchor_weight}  gain_reg={args.gain_reg_weight}")
    if args.no_aug:
        print(f"  aug    : DISABLED (--no_aug)")
    else:
        print(f"  aug    : pulse_apply_prob={args.pulse_apply_prob}"
              f"  bw=[{args.pulse_bw_min_hz/1e6:.2f}, {args.pulse_bw_max_hz/1e6:.2f}] MHz"
              f"  fc_shift=±{args.pulse_fc_shift_hz/1e3:.0f} kHz")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.no_aug:
        aug_cfg = None
    else:
        aug_cfg = AugConfig(
            pulse_apply_prob=args.pulse_apply_prob,
            pulse_bw_range_hz=(args.pulse_bw_min_hz, args.pulse_bw_max_hz),
            pulse_fc_shift_hz=args.pulse_fc_shift_hz,
        )

    def train_iter():
        return _iter_split(loader, train_indices, args.target_root, grid_z, grid_x,
                           args.device, aug_cfg, seed_base=args.seed * 1000,
                           target_is_bmode=args.target_is_bmode)

    def val_iter():
        return _iter_split(loader, val_indices, args.target_root, grid_z, grid_x,
                           args.device, None, seed_base=0,
                           target_is_bmode=args.target_is_bmode)

    run_training(args, model, optimizer, train_iter, val_iter if val_indices else None,
                 _forward_and_loss)


if __name__ == "__main__":
    main()
