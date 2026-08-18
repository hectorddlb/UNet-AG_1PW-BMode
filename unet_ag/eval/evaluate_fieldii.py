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
from ..models.a2_unet_fourier import A2UNetFourier
from ..models.b1_unet_attention_fnum import B1UNetAttentionFNum
from ..models.b1_v2_softdas import B1V2SoftDAS
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
    # Relative normalisation (see train/b1_attention_gate.py): an absolute
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


@torch.no_grad()
def _reconstruct_b1_v2(model, sample, grid_z, grid_x, device: str) -> tuple[np.ndarray, np.ndarray]:
    """B1-v2 takes raw RF (after z-score) and the F-number map; soft-DAS is
    internal. Also returns the DAS_1PW baseline via ``H^T y`` for the CSV.
    """
    op = make_op_for_angle(0.0, sample.geometry, grid_z, grid_x, device=device, nf_chunk=None)
    y = sample.rf_1pw.to(device).to(op.float_dtype)
    x0 = op.adjoint(y)
    if x0.is_complex():
        x0 = x0.real
    x0_env = _envelope_log_compress(x0.float().detach().cpu().numpy())

    rf = sample.rf_1pw.to(device).float()
    rf = (rf - rf.mean()) / (rf.std() + 1e-8)
    f_map = compute_f_number_map_band_avg(
        grid_z.cpu().numpy(), grid_x.cpu().numpy(), sample.geometry
    )
    f_map_t = torch.from_numpy(f_map).to(device).unsqueeze(0).unsqueeze(0)
    pred = model(rf, f_map_t)                                                # (1,1,Nz,Nx) ∈ [0,1]
    return pred[0, 0].cpu().numpy().astype(np.float32), x0_env


@torch.no_grad()
def _reconstruct_a2(model, sample, grid_z, grid_x, device: str,
                    rf_n_samples: int = 3500) -> tuple[np.ndarray, np.ndarray]:
    """A2 takes raw RF directly. Also returns the DAS_1PW baseline via H^T y
    so the eval CSV has the same baseline columns as B1's.
    """
    # DAS_1PW baseline (same as B1)
    op = make_op_for_angle(0.0, sample.geometry, grid_z, grid_x, device=device, nf_chunk=None)
    y = sample.rf_1pw.to(device).to(op.float_dtype)
    x0 = op.adjoint(y)
    if x0.is_complex():
        x0 = x0.real
    x0_env = _envelope_log_compress(x0.float().detach().cpu().numpy())

    # A2 input: pad/truncate RF to rf_n_samples, normalize /max(|.|), shape (1,1,D,K)
    rf = sample.rf_1pw.detach().cpu()
    D = rf.shape[0]
    if D > rf_n_samples:
        rf = rf[:rf_n_samples].contiguous()
    elif D < rf_n_samples:
        pad = torch.zeros(rf_n_samples - D, rf.shape[1], dtype=rf.dtype)
        rf = torch.cat([rf, pad], dim=0).contiguous()
    rf_scale = float(rf.abs().max()) + 1e-12
    rf = (rf / rf_scale).to(device).unsqueeze(0).unsqueeze(0)                # (1,1,D,K)
    pred = model(rf)                                                          # (1,1,Nz,Nx) ∈ [0,1]
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


def _load_b1_ckpt(ckpt_path: Path, device: str, base_ch: int = 32) -> B1UNetAttentionFNum:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model = B1UNetAttentionFNum(in_ch=1, base_ch=base_ch, embed_ch=16).to(device).eval()
    model.load_state_dict(sd, strict=True)
    return model


def _load_b1_v2_ckpt(ckpt_path: Path, device: str,
                     sample0, nz: int, nx: int,
                     base_ch: int = 48) -> B1V2SoftDAS:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    grid_z, grid_x = _build_grid(sample0.geometry, nz, nx)
    model = B1V2SoftDAS(
        grid_z.numpy(), grid_x.numpy(), sample0.geometry,
        base_ch=base_ch, embed_ch=16,
    ).to(device).eval()
    model.load_state_dict(sd, strict=True)
    return model


def _load_b1_v3_ckpt(ckpt_path: Path, device: str, base_ch: int = 32) -> B1V3Residual:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model = B1V3Residual(base_ch=base_ch, embed_ch=16).to(device).eval()
    model.load_state_dict(sd, strict=True)
    return model


def _load_a2_ckpt(ckpt_path: Path, device: str, nz: int, nx: int,
                  n_samples: int = 3500, n_elements: int = 128,
                  base_ch: int = 32) -> A2UNetFourier:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model = A2UNetFourier(
        n_samples=n_samples, n_elements=n_elements, nz=nz, nx=nx, base_ch=base_ch,
    ).to(device).eval()
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
    parser.add_argument("--model_type", choices=["b1", "a2", "b1_v2", "b1_v3", "b1_v4"], default="b1",
                        help="b1 = adjoint+f_map; a2 = raw RF (padded); b1_v2 = raw RF + soft-DAS internal; "
                             "b1_v3 = adjoint+f_map with residual skip + zero-init out (identity at ep 0); "
                             "b1_v4 = B1-v3 architecture trained with L1-anchor to x_DAS (same loader as b1_v3).")
    parser.add_argument("--phantom_type", choices=["points", "cysts", "both"], default="both")
    parser.add_argument("--nz", type=int, default=612)
    parser.add_argument("--nx", type=int, default=388)
    parser.add_argument("--rf_n_samples", type=int, default=3500,
                        help="A2 only — must match training value (default 3500).")
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
    if args.model_type == "b1":
        model = _load_b1_ckpt(args.ckpt, device, base_ch=args.base_ch)
        prefix = "b1"
    elif args.model_type == "b1_v2":
        # Build LUT from the first available phantom (matches training-time choice).
        sample0 = loader[0]
        model = _load_b1_v2_ckpt(args.ckpt, device, sample0, nz=args.nz, nx=args.nx,
                                 base_ch=args.base_ch)
        with torch.no_grad():
            g = model.gamma
        print(f"  soft-DAS γ (loaded): F={g['F']:.3f}  edge={g['edge']:.3f}  gain={g['gain']:.3f}")
        prefix = "b1"  # use same CSV column prefix so comparisons line up
    elif args.model_type in ("b1_v3", "b1_v4"):
        # Same model class (B1V3Residual); B1-v4 only differs in training loss.
        model = _load_b1_v3_ckpt(args.ckpt, device, base_ch=args.base_ch)
        with torch.no_grad():
            g = model.gamma
        print(f"  residual γ (loaded): gain={g['gain']:.3f}")
        prefix = "b1"  # CSV columns aligned with B1/B1-v2/B1-v3
    else:
        model = _load_a2_ckpt(args.ckpt, device, nz=args.nz, nx=args.nx,
                              n_samples=args.rf_n_samples, base_ch=args.base_ch)
        prefix = "a2"
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

        if args.model_type in ("b1", "b1_v3", "b1_v4"):
            # B1-v3 / B1-v4 share signatures with B1 baseline: input env_log(H^T y) + f_map.
            pred_env, das_env = _reconstruct_b1(model, sample, grid_z, grid_x, device)
        elif args.model_type == "b1_v2":
            pred_env, das_env = _reconstruct_b1_v2(model, sample, grid_z, grid_x, device)
        else:
            pred_env, das_env = _reconstruct_a2(model, sample, grid_z, grid_x, device,
                                                 rf_n_samples=args.rf_n_samples)

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
