"""Regenerate Schiffner Fd-F Field II targets with the CORRECTED conventions.

Differences vs schiffner_fd-f/scripts/precompute_schiffner_targets_fieldii.py:
  1. das_pw called with tx_sign=+1, tx_reference="origin" — matches the Field II
     simulation (delays = element_x*sin(angle)/c, centred elements, wavefront
     through x=0 at t=0). The old edge/−sin convention misaligned every steered
     angle and produced the lateral streaks seen in the 287 old targets.
  2. index_t0 = −round(t_start*fs) — das_pw semantics are "sample index where
     t=0 falls"; the old script passed +round(t_start*fs) (sign flipped).

With correct compounding, CYSTS targets become usable too (previously garbage),
so the default includes both phantom types. Layout and normalization are
identical to the old script, so the targets are drop-in for the anchor05
trainer (--target_root pointing at the new output_dir).

Recommended SLURM usage (array over ranges, GPU partition), from tesis_unetAG:
  python resultados-finales/scripts/precompute_schiffner_targets_fieldii_fixed.py \
      --data_root "datasets/Field II/field_ii_data_multiangle" \
      --output_dir resultados-finales/targets_schiffner_fixed \
      --range_start 0 --range_end 50 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from schiffner_das.beamforming import das_pw  # noqa: E402
from schiffner_das.f_numbers import GratingAngleLB  # noqa: E402
from schiffner_das.normalizations import NormalizationOn  # noqa: E402
from schiffner_das.windows import Tukey  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir",
                   default="resultados-finales/targets_schiffner_fixed")
    p.add_argument("--phantom_types", nargs="+", default=["points", "cysts"])
    p.add_argument("--nz", type=int, default=612)
    p.add_argument("--nx", type=int, default=388)
    p.add_argument("--z_min", type=float, default=5e-3)
    p.add_argument("--z_max", type=float, default=50e-3)
    p.add_argument("--x_min", type=float, default=-19.2e-3)
    p.add_argument("--x_max", type=float, default=19.2e-3)
    p.add_argument("--pitch", type=float, default=300e-6)
    p.add_argument("--fs", type=float, default=20.832e6)
    p.add_argument("--c", type=float, default=1540.0)
    p.add_argument("--element_width", type=float, default=270e-6)
    p.add_argument("--f_number_grating_deg", type=float, default=60.0)
    p.add_argument("--f_number_lb", type=float, default=1.5)
    p.add_argument("--tukey_roll", type=float, default=0.2)
    p.add_argument("--range_start", type=int, default=0)
    p.add_argument("--range_end", type=int, default=None)
    p.add_argument("--max_angles", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--z_chunk", type=int, default=32)
    p.add_argument("--x_batch", type=int, default=None)
    p.add_argument("--precision", default="float32",
                   choices=["float32", "float64"],
                   help="float32 validado vs float64 (rel<1e-3) — ~muchos x "
                        "más rápido en GPU consumer (FP64 capado)")
    return p.parse_args()


def collect_files(data_root: Path, phantom_types: list[str]):
    pairs = []
    for pt in phantom_types:
        sub = data_root / pt
        if not sub.exists():
            print(f"  WARNING: {sub} not found, skipping")
            continue
        for fp in sorted(sub.glob("*.mat")):
            pairs.append((pt, fp))
    return pairs


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    grid_z = torch.linspace(args.z_min, args.z_max, args.nz, dtype=torch.float64)
    grid_x = torch.linspace(args.x_min, args.x_max, args.nx, dtype=torch.float64)

    f_number = GratingAngleLB(args.f_number_grating_deg, args.f_number_lb)
    window = Tukey(args.tukey_roll)
    normalization = NormalizationOn()

    files = collect_files(data_root, args.phantom_types)
    if not files:
        raise RuntimeError(f"No .mat files under {data_root}")
    range_end = args.range_end if args.range_end is not None else len(files)
    files = files[args.range_start:range_end]
    print(f"Phantoms [{args.range_start}, {range_end}) — {len(files)} files, "
          f"device={device}")

    csv_path = output_dir / f"stats_{args.range_start:04d}_{range_end:04d}.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["phantom_type", "stem", "angle_idx", "angle_deg",
                     "t_start_us", "index_t0", "scale", "img_absmax", "das_s"])

    t0_all = time.perf_counter()
    for pt, fp in files:
        ph_dir = output_dir / pt / fp.stem
        comp_path = ph_dir / "compounded.npy"
        if comp_path.exists():
            print(f"=== {pt}/{fp.stem} — skipped (exists)")
            continue

        m = sio.loadmat(fp)
        rf_all = m["rf_data"]
        angles_deg = np.array(m["angles_deg"]).reshape(-1)
        ts_arr = np.array(m["t_start"]).reshape(-1)
        fs = float(np.array(m["fs"]).squeeze())

        n_angles = rf_all.shape[2]
        if args.max_angles:
            n_angles = min(n_angles, args.max_angles)

        ph_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {pt}/{fp.stem}  (n_angles={n_angles}) ===")
        compounded = torch.zeros(args.nz, args.nx, dtype=torch.cdouble,
                                 device=device)

        for a in range(n_angles):
            angle_rad = math.radians(float(angles_deg[a]))
            t_start = float(ts_arr[a])
            # das_pw semantics: index_t0 = sample index where t=0 falls
            index_t0 = -int(round(t_start * fs))

            rf_np = rf_all[:, :, a].astype(np.float64)
            scale = float(np.max(np.abs(rf_np))) + 1e-30
            rf = torch.from_numpy(rf_np / scale)

            t_d0 = time.perf_counter()
            result = das_pw(
                positions_x=grid_x, positions_z=grid_z, data_RF=rf, f_s=fs,
                steering_angle=angle_rad,
                element_width=args.element_width, element_pitch=args.pitch,
                c_0=args.c, index_t0=index_t0,
                window=window, F_number=f_number, normalization=normalization,
                z_chunk=args.z_chunk, x_batch=args.x_batch, device=device,
                tx_sign=+1.0, tx_reference="origin", precision=args.precision,
            )
            t_d = time.perf_counter() - t_d0
            img = result.image
            compounded = compounded + img.to(compounded.device)
            writer.writerow([pt, fp.stem, a, f"{float(angles_deg[a]):+.4f}",
                             f"{t_start*1e6:+.4f}", index_t0,
                             f"{scale:.6e}",
                             f"{float(img.abs().max()):.6e}", f"{t_d:.2f}"])
            csv_file.flush()

        np.save(comp_path, compounded.real.cpu().numpy().astype(np.float32))
        print(f"  saved {comp_path}")

    csv_file.close()
    print(f"\nDone in {(time.perf_counter()-t0_all)/60:.1f} min. Stats: {csv_path}")


if __name__ == "__main__":
    main()
