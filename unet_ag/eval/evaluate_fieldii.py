"""Field II evaluation — CNR / gCNR / FWHM per phantom for B1 (and friends).

Loads a trained B1 checkpoint, runs inference over the val split, computes:
- For points phantoms: FWHM_lat / FWHM_ax (median + IQR over scatterers).
- For cysts phantoms: CNR (dB) and gCNR (cyst interior vs annular background).

Outputs a CSV plus a console summary. Optionally also evaluates a DAS_1PW
baseline (``H_Fdf.adjoint`` at angle 0°) and the Schiffner_75PW target as
upper-bound reference, computing the same metrics for direct comparison.

CLI:
  python -m unet_ag.eval.evaluate_fieldii \
      --ckpt results_v2/b1/checkpoints/best.pt \
      --data_root "datasets/Field II/field_ii_data_multiangle" \
      --target_root results_schiffner_targets_fieldii \
      --out results_v2/b1/eval_fieldii.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.signal import hilbert

from ..data.manifest import field_ii_split
from ..data.rf_loader import FieldIIMultiAngleLoader
from ..models.b1_v3_residual import B1V3Residual
from ..physics.f_number_map import compute_f_number_map_band_avg
from ..physics.h_fdf_wrapper import make_op_for_angle
from ..metrics.cnr import compute_cnr
from ..metrics.gcnr import compute_gcnr
from ..metrics.fwhm import aggregate_fwhms_per_wire
from .masks_fieldii import cyst_masks, points_wire_targets


DYN_RANGE_DB = 60.0


# ──────────────────────────────────────────────────────────────────
# Preprocessing identical to training (B1 F1)
# ──────────────────────────────────────────────────────────────────


def _envelope_log_compress(img: np.ndarray, dyn_range_db: float = DYN_RANGE_DB) -> np.ndarray:
    env = np.abs(hilbert(img, axis=0))
    # Relative normalisation (see train/common.py): an absolute
    # epsilon swamps Field II native amplitudes (~1e-23) → constant image.
    env_max = float(env.max())
    if env_max <= 0.0:
        return np.zeros(img.shape, dtype=np.float32)
    db = 20.0 * np.log10(env / env_max + 1e-30)
    db = np.clip(db, -dyn_range_db, 0.0)
    return ((db + dyn_range_db) / dyn_range_db).astype(np.float32)


def _build_grid(geometry: dict, nz: int, nx: int):
    c = float(geometry["speed_of_sound"])
    fs = float(geometry["sampling_freq_hz"])
    ns = int(geometry["n_samples"])
    K = int(geometry["n_elements"])
    pitch = float(geometry["pitch_m"])
    z_max = (ns - 1) * c / (2.0 * fs)
    gz = torch.linspace(5e-3, z_max, nz, dtype=torch.float32)
    gx = torch.linspace(-K * pitch / 2.0, +K * pitch / 2.0, nx, dtype=torch.float32)
    return gz, gx


@torch.no_grad()
def _reconstruct_b1(model, sample, grid_z, grid_x, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_envlog, x0_envlog_das_baseline) both (Nz, Nx) float32 in [0,1]."""
    op = make_op_for_angle(0.0, sample.geometry, grid_z, grid_x, device=device, nf_chunk=None)
    y = sample.rf_1pw.to(device).to(op.float_dtype)
    x0 = op.adjoint(y)
    if x0.is_complex():
        x0 = x0.real
    x0_np = x0.float().detach().cpu().numpy()
    x0_env = _envelope_log_compress(x0_np)                                   # DAS_1PW baseline
    x0_t = torch.from_numpy(x0_env).to(device).unsqueeze(0).unsqueeze(0)
    f_map = compute_f_number_map_band_avg(
        grid_z.cpu().numpy(), grid_x.cpu().numpy(), sample.geometry
    )
    f_map_t = torch.from_numpy(f_map).to(device).unsqueeze(0).unsqueeze(0)
    pred = model(x0_t, f_map_t)                                              # (1,1,Nz,Nx)
    return pred[0, 0].cpu().numpy().astype(np.float32), x0_env


# ──────────────────────────────────────────────────────────────────
# Per-phantom metric evaluation
# ──────────────────────────────────────────────────────────────────


def _eval_points(image: np.ndarray, mat_path: Path,
                 grid_z_m: np.ndarray, grid_x_m: np.ndarray) -> dict:
    """FWHM_lat/ax over all wires; image in [0,1] envelope-log."""
    wires = points_wire_targets(mat_path)
    dz_mm = float((grid_z_m[1] - grid_z_m[0]) * 1e3)
    dx_mm = float((grid_x_m[1] - grid_x_m[0]) * 1e3)
    res = aggregate_fwhms_per_wire(
        image, wires, dx_mm=dx_mm, dz_mm=dz_mm,
        grid_z_mm=grid_z_m * 1e3, grid_x_mm=grid_x_m * 1e3,
        search_radius_mm=1.5,
    )
    return {
        "fwhm_lat_med_mm": res["lat_median"],
        "fwhm_ax_med_mm": res["ax_median"],
        "fwhm_n_wires": int(res["n_total"]),
        "fwhm_n_resolved_lat": int(res["n_resolved_lat"]),
        "fwhm_n_resolved_ax": int(res["n_resolved_ax"]),
    }


def _eval_cysts(image: np.ndarray, mat_path: Path,
                grid_z_m: np.ndarray, grid_x_m: np.ndarray) -> dict:
    """CNR + gCNR (cyst interior vs annular tissue ring)."""
    mi, mo, cysts = cyst_masks(mat_path, grid_z_m, grid_x_m)
    if not cysts or mo.sum() == 0:
        return {"cnr_db": float("nan"), "gcnr": float("nan"), "n_cysts": 0}
    cnr = compute_cnr(image, mi, mo, on_envelope=False)
    gcnr = compute_gcnr(image, mi, mo, on_envelope=False)
    return {"cnr_db": cnr, "gcnr": gcnr, "n_cysts": len(cysts)}


def _eval_image(image: np.ndarray, mat_path: Path, ptype: str,
                grid_z_m: np.ndarray, grid_x_m: np.ndarray) -> dict:
    if ptype == "points":
        return _eval_points(image, mat_path, grid_z_m, grid_x_m)
    return _eval_cysts(image, mat_path, grid_z_m, grid_x_m)


# ──────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────


def _load_b1_v3_ckpt(ckpt_path: Path, device: str, base_ch: int = 32) -> B1V3Residual:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model = B1V3Residual(base_ch=base_ch, embed_ch=16).to(device).eval()
    model.load_state_dict(sd, strict=True)
    return model


def _load_target(target_root: Path, phantom_id: str, ptype: str,
                 nz: int, nx: int) -> Optional[np.ndarray]:
    for cand in (target_root / ptype / phantom_id / "compounded.npy",
                 target_root / phantom_id / "compounded.npy"):
        if cand.exists():
            arr = np.load(cand).astype(np.float32)
            if arr.shape != (nz, nx):
                t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                t = torch.nn.functional.interpolate(
                    t, size=(nz, nx), mode="bilinear", align_corners=False
                )[0, 0]
                arr = t.numpy()
            return arr
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--target_root", type=Path, default=None,
                        help="If given, also eval Schiffner_75PW target as upper bound.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model_type", choices=["b1_v3", "b1_v4"], default="b1_v4",
                        help="Misma arquitectura (B1V3Residual): adjunto + mapa de F-number, "
                             "conexion residual y out_conv inicializada a cero. b1_v4 es la "
                             "variante entrenada con anclaje L1 a x_DAS, que es la de la tesis.")
    parser.add_argument("--phantom_type", choices=["points", "cysts", "both"], default="both")
    parser.add_argument("--nz", type=int, default=612)
    parser.add_argument("--nx", type=int, default=388)
    parser.add_argument("--base_ch", type=int, default=32,
                        help="Backbone base channel count — must match training (default 32).")
    parser.add_argument("--max_phantoms", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"device     = {device}")
    print(f"model_type = {args.model_type}")
    print(f"ckpt       = {args.ckpt}")
    print(f"grid       = {args.nz} x {args.nx}")

    loader = FieldIIMultiAngleLoader(args.data_root, phantom_type=args.phantom_type,
                                      keep_all_angles=False)
    # b1_v3 y b1_v4 son la MISMA clase (B1V3Residual); solo difieren en la
    # perdida de entrenamiento, que aqui ya no interviene.
    model = _load_b1_v3_ckpt(args.ckpt, device, base_ch=args.base_ch)
    with torch.no_grad():
        g = model.gamma
    print(f"  residual γ (loaded): gain={g['gain']:.3f}")
    prefix = "b1"
    splits = field_ii_split(
        args.data_root,
        phantom_types=[args.phantom_type] if args.phantom_type != "both" else None,
    )
    if args.phantom_type == "both":
        val_files = splits["points"].val + splits["cysts"].val
    else:
        val_files = splits[args.phantom_type].val

    name_to_idx = {p.stem: i for i, p in enumerate(loader.file_list)}
    val_indices = [name_to_idx[p.stem] for p in val_files if p.stem in name_to_idx]
    if args.max_phantoms:
        val_indices = val_indices[: args.max_phantoms]
    print(f"val phantoms = {len(val_indices)}")

    rows: list[dict] = []
    for j, idx in enumerate(val_indices):
        sample = loader[idx]
        ptype = "points" if sample.phantom_id.startswith("points") else "cysts"
        mat_path = loader.file_list[idx]
        grid_z, grid_x = _build_grid(sample.geometry, args.nz, args.nx)
        grid_z_m, grid_x_m = grid_z.numpy(), grid_x.numpy()

        # entrada: env_log(H^T y) + mapa de F-number
        pred_env, das_env = _reconstruct_b1(model, sample, grid_z, grid_x, device)

        row = {"phantom_id": sample.phantom_id, "ptype": ptype}
        for label, img in [(prefix, pred_env), ("das_1pw", das_env)]:
            stats = _eval_image(img, mat_path, ptype, grid_z_m, grid_x_m)
            for k, v in stats.items():
                row[f"{label}_{k}"] = v

        if args.target_root:
            tgt = _load_target(args.target_root, sample.phantom_id, ptype, args.nz, args.nx)
            if tgt is not None:
                tgt_env = _envelope_log_compress(tgt)
                stats = _eval_image(tgt_env, mat_path, ptype, grid_z_m, grid_x_m)
                for k, v in stats.items():
                    row[f"schiffner75_{k}"] = v

        rows.append(row)
        print(f"  [{j+1}/{len(val_indices)}] {sample.phantom_id} ({ptype})")

    if not rows:
        print("no rows produced — exiting")
        return

    fieldnames = sorted({k for r in rows for k in r.keys()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.out}")

    # Console summary
    def _summarise(key: str, ptype_filter: Optional[str] = None) -> str:
        vals = [r.get(key) for r in rows
                if (ptype_filter is None or r["ptype"] == ptype_filter)
                and r.get(key) is not None and not math.isnan(r[key])]
        if not vals:
            return "n/a"
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f} (n={len(vals)})"

    print("\n=== SUMMARY ===")
    for ptype, metrics in [
        ("points", ["fwhm_lat_med_mm", "fwhm_ax_med_mm",
                    "fwhm_n_resolved_lat", "fwhm_n_resolved_ax", "fwhm_n_wires"]),
        ("cysts",  ["cnr_db", "gcnr"]),
    ]:
        print(f"\n[{ptype}]")
        for m in metrics:
            row_model = _summarise(f"{prefix}_{m}", ptype)
            row_das = _summarise(f"das_1pw_{m}", ptype)
            row_sch = _summarise(f"schiffner75_{m}", ptype)
            print(f"  {m:24s}  {prefix.upper()}: {row_model:30s}  DAS_1PW: {row_das:30s}  Schiffner75: {row_sch}")


if __name__ == "__main__":
    main()
