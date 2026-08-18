"""Evaluate a trained model on PICMUS (sim + exp, resolution + contrast).

Reuses the B1/B1-v3/B1-v4 reconstruction pipeline (``_reconstruct_b1`` from
``evaluate_fieldii``) and writes a CSV with the same column convention so
results line up directly with the Field II eval.

CLI:
  python -m unet_ag.eval.evaluate_picmus \\
      --ckpt        results_v2/b1_v4_a0p3/checkpoints/best.pt \\
      --picmus_root datasets/PICMUS \\
      --mode        both       # sim | exp | both
      --phantom     both       # resolution | contrast | both
      --out         results_v2/b1_v4_a0p3/eval_picmus.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from ..data.rf_loader import PICMUSLoader
from .evaluate_fieldii import (
    _build_grid,
    _envelope_log_compress,
    _load_b1_v3_ckpt,
    _reconstruct_b1,
)
from .masks_picmus import cyst_masks as picmus_cyst_masks
from .masks_picmus import points_wire_targets as picmus_wire_targets
from ..metrics.cnr import compute_cnr
from ..metrics.gcnr import compute_gcnr
from ..metrics.fwhm import aggregate_fwhms_per_wire


def _eval_points(image: np.ndarray, phantom_hdf5: Path,
                 grid_z_m: np.ndarray, grid_x_m: np.ndarray) -> dict:
    wires = picmus_wire_targets(phantom_hdf5)
    dz_mm = float((grid_z_m[1] - grid_z_m[0]) * 1e3)
    dx_mm = float((grid_x_m[1] - grid_x_m[0]) * 1e3)
    res = aggregate_fwhms_per_wire(
        image, wires, dx_mm=dx_mm, dz_mm=dz_mm,
        grid_z_mm=grid_z_m * 1e3, grid_x_mm=grid_x_m * 1e3,
        search_radius_mm=1.5,
    )
    return {
        "fwhm_lat_med_mm": res["lat_median"],
        "fwhm_ax_med_mm":  res["ax_median"],
        "fwhm_n_wires":    int(res["n_total"]),
        "fwhm_n_resolved_lat": int(res["n_resolved_lat"]),
        "fwhm_n_resolved_ax":  int(res["n_resolved_ax"]),
    }


def _eval_cysts(image: np.ndarray, phantom_hdf5: Path,
                grid_z_m: np.ndarray, grid_x_m: np.ndarray) -> dict:
    mi, mo, cysts = picmus_cyst_masks(phantom_hdf5, grid_z_m, grid_x_m)
    if not cysts or mo.sum() == 0:
        return {"cnr_db": float("nan"), "gcnr": float("nan"), "n_cysts": 0}
    cnr = compute_cnr(image,  mi, mo, on_envelope=False)
    gcnr = compute_gcnr(image, mi, mo, on_envelope=False)
    return {"cnr_db": cnr, "gcnr": gcnr, "n_cysts": len(cysts)}


def _eval_image(image: np.ndarray, phantom_hdf5: Path, phantom_type: str,
                grid_z_m: np.ndarray, grid_x_m: np.ndarray) -> dict:
    if phantom_type == "resolution_distortion":
        return _eval_points(image, phantom_hdf5, grid_z_m, grid_x_m)
    return _eval_cysts(image, phantom_hdf5, grid_z_m, grid_x_m)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        type=Path, required=True)
    p.add_argument("--picmus_root", type=Path, required=True)
    p.add_argument("--out",         type=Path, required=True)
    p.add_argument("--mode",    default="both",
                   choices=["sim", "exp", "both"])
    p.add_argument("--phantom", default="both",
                   choices=["resolution_distortion", "contrast_speckle", "both"])
    p.add_argument("--model_type", default="b1_v4",
                   choices=["b1_v3", "b1_v4"],
                   help="Misma arquitectura (B1V3Residual); b1_v4 solo difiere en la perdida.")
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--nz", type=int, default=612)
    p.add_argument("--nx", type=int, default=388)
    p.add_argument("--device", default="cuda")
    p.add_argument("--prefilter_fc_mhz", type=float, default=0.0,
                   help="Gaussian bandpass center freq (MHz) applied to RF before adjoint. 0 disables.")
    p.add_argument("--prefilter_bw_mhz", type=float, default=0.0,
                   help="Gaussian bandpass -6 dB BW (MHz). 0 disables. Best from grid (B): exp/res = (4.60, 0.80).")
    args = p.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"device = {device}")
    print(f"ckpt   = {args.ckpt}")

    model = _load_b1_v3_ckpt(args.ckpt, device, base_ch=args.base_ch)
    with torch.no_grad():
        g = model.gamma if hasattr(model, "gamma") else {}
    if g:
        print(f"  γ (loaded): {dict(g)}")

    modes = ["sim", "exp"] if args.mode == "both" else [args.mode]
    phantoms = (["resolution_distortion", "contrast_speckle"]
                if args.phantom == "both" else [args.phantom])

    rows: list[dict] = []
    for mode in modes:
        for ph in phantoms:
            print(f"\n=== {mode}/{ph} ===")
            try:
                loader = PICMUSLoader(args.picmus_root, mode=mode, phantom=ph,
                                      normalize=True)
            except FileNotFoundError as e:
                print(f"  SKIP — {e}")
                continue
            sample = loader[0]
            grid_z, grid_x = _build_grid(sample.geometry, args.nz, args.nx)
            grid_z_m, grid_x_m = grid_z.numpy(), grid_x.numpy()

            # Optional test-time spectral pre-filter on the RF (no autograd needed).
            # Best filter from grid-search diagnostic (job 192069) for exp/res:
            # (fc=4.60 MHz, BW=0.80 MHz). Applied uniformly to all phantoms when set.
            if args.prefilter_fc_mhz > 0.0 and args.prefilter_bw_mhz > 0.0:
                from ..data.augmentations import gaussian_bandpass_rf
                from ..data.rf_loader import zscore_rf
                fs = float(sample.geometry["sampling_freq_hz"])
                rf_filt = gaussian_bandpass_rf(
                    sample.rf_1pw.detach().cpu().float(),
                    fs, args.prefilter_fc_mhz * 1e6, args.prefilter_bw_mhz * 1e6,
                )
                rf_filt, _ = zscore_rf(rf_filt)
                sample.rf_1pw = rf_filt.to(sample.rf_1pw.device)
                print(f"  prefilter applied: fc={args.prefilter_fc_mhz:.2f} MHz BW={args.prefilter_bw_mhz:.2f} MHz")

            pred_env, das_env = _reconstruct_b1(model, sample, grid_z, grid_x, device)
            print(f"  pred  : min={pred_env.min():.3f}  max={pred_env.max():.3f}  mean={pred_env.mean():.3f}")
            print(f"  das   : min={das_env.min():.3f}  max={das_env.max():.3f}  mean={das_env.mean():.3f}")

            row = {"phantom_id": sample.phantom_id, "mode": mode, "phantom": ph}
            m_pred = _eval_image(pred_env, loader.phantom_path, ph, grid_z_m, grid_x_m)
            m_das  = _eval_image(das_env,  loader.phantom_path, ph, grid_z_m, grid_x_m)
            row.update({f"b1_{k}":      v for k, v in m_pred.items()})
            row.update({f"das_1pw_{k}": v for k, v in m_das.items()})
            rows.append(row)
            print(f"  pred metrics: {m_pred}")
            print(f"  das  metrics: {m_das}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No phantoms evaluated — exiting without writing CSV.")
        return
    all_keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
