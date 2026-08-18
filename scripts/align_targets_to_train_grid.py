"""Re-muestrea los targets Schiffner a la rejilla axial que usa el entrenador.

BUG QUE CORRIGE
---------------
``precompute_schiffner_targets_fieldii*.py`` conforma en una rejilla FIJA
z in [5, 50] mm x 612 filas (dz = 0.0736 mm).  El entrenador y el evaluador,
en cambio, construyen la suya con ``evaluate_fieldii._build_grid``:

    z_max = (n_samples - 1) * c / (2 * fs)      # 54.22 mm con n_samples=1468
    grid_z = linspace(5e-3, z_max, nz)          # dz = 0.0806 mm

Ambas tienen 612 filas, asi que ``_load_target`` NO reinterpola (solo lo hace
si cambia el *shape*) y el target se superpone a la entrada **fila con fila**.
El resultado es un error de escala axial del 9.4 %: un eco a 44 mm cae 45
filas (3.7 mm) mas abajo en el target que en la entrada DAS-1PW, muy por
encima de la FWHM axial (~1.6 mm).  Con targets borrosos el solape parcial lo
disimulaba; con los targets nitidos (convencion TX corregida) target y entrada
dejan de solaparse y la red minimiza la perdida **emborronando**.

QUE HACE
--------
Para cada ``compounded.npy`` (real, tipo RF):
  1. senal analitica por Hilbert;
  2. demodulacion a banda base con la portadora exp(-j 2pi fc 2z/c) -- la
     envolvente compleja esta sobremuestreada ~13x, asi que interpolarla es
     exacto, mientras que interpolar la RF (4.0 muestras/ciclo) no lo seria;
  3. interpolacion lineal a la rejilla del entrenador;
  4. remodulacion y se guarda la parte real (mismo formato que la entrada).
Las filas con z > z_max_target (50 mm) se rellenan con cero: ahi no hay
informacion de target (los dispersores mas profundos estan a ~44 mm).

Uso
---
    python resultados-finales/scripts/align_targets_to_train_grid.py \
        --src resultados-finales/targets_schiffner_fixed \
        --dst resultados-finales/targets_schiffner_gridfix \
        --data_root "datasets/Field II/field_ii_data_multiangle"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import hilbert


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--data_root", type=Path, required=True,
                   help="raiz de los .mat de Field II (para c, fs, fc, n_samples)")
    p.add_argument("--phantom_types", nargs="+", default=["points", "cysts"])
    p.add_argument("--src_z_min", type=float, default=5e-3)
    p.add_argument("--src_z_max", type=float, default=50e-3)
    p.add_argument("--train_z_min", type=float, default=5e-3)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def train_grid_z(geom_mat: dict, nz: int, z_min: float) -> np.ndarray:
    """Replica exactamente evaluate_fieldii._build_grid (eje z)."""
    c = float(np.array(geom_mat["c"]).ravel()[0])
    fs = float(np.array(geom_mat["fs"]).ravel()[0])
    ns = int(np.array(geom_mat["rf_data"]).shape[0])
    z_max = (ns - 1) * c / (2.0 * fs)
    return np.linspace(z_min, z_max, nz, dtype=np.float64)


def resample_axial(rf: np.ndarray, z_src: np.ndarray, z_dst: np.ndarray,
                   fc: float, c: float, taper_m: float = 2e-3) -> np.ndarray:
    """Re-muestrea rf (Nz, Nx) de z_src a z_dst por demodulacion + interp lineal.

    ``taper_m`` aplica un coseno alzado en los ultimos milimetros del tramo
    valido para que el relleno de ceros (z > z_src[-1]) no genere un escalon:
    el Hilbert posterior lo convertiria en ringing. Los dispersores mas
    profundos estan a 44 mm (points) / 45 mm (cysts), asi que un taper de
    48 -> 50 mm no borra senal util.
    """
    analytic = hilbert(rf.astype(np.float64), axis=0)            # (Nz, Nx) complejo
    k = 2.0 * np.pi * fc * 2.0 / c                                # rad/m (ida y vuelta)
    base = analytic * np.exp(-1j * k * z_src[:, None])            # banda base, suave

    inside = (z_dst >= z_src[0]) & (z_dst <= z_src[-1])
    out = np.zeros((z_dst.size, rf.shape[1]), dtype=np.complex128)
    zi = z_dst[inside]
    for ix in range(rf.shape[1]):
        out[inside, ix] = (np.interp(zi, z_src, base[:, ix].real)
                           + 1j * np.interp(zi, z_src, base[:, ix].imag))
    out *= np.exp(+1j * k * z_dst[:, None])

    if taper_m > 0:
        w = np.ones(z_dst.size)
        w[~inside] = 0.0
        edge = z_src[-1]
        ramp = inside & (z_dst > edge - taper_m)
        w[ramp] = 0.5 * (1.0 + np.cos(np.pi * (z_dst[ramp] - (edge - taper_m)) / taper_m))
        out *= w[:, None]
    return out.real.astype(np.float32)


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    n_done = n_skip = 0

    for pt in args.phantom_types:
        src_dir = args.src / pt
        if not src_dir.exists():
            print(f"  AVISO: {src_dir} no existe, se omite")
            continue
        for ph_dir in sorted(d for d in src_dir.iterdir() if d.is_dir()):
            src_npy = ph_dir / "compounded.npy"
            if not src_npy.exists():
                continue
            dst_npy = args.dst / pt / ph_dir.name / "compounded.npy"
            if dst_npy.exists() and not args.overwrite:
                n_skip += 1
                continue

            mat_path = args.data_root / pt / f"{ph_dir.name}.mat"
            if not mat_path.exists():
                print(f"  AVISO: falta {mat_path}, se omite {ph_dir.name}")
                continue
            m = sio.loadmat(mat_path, variable_names=["c", "fs", "fc", "rf_data"])
            c = float(np.array(m["c"]).ravel()[0])
            fc = float(np.array(m["fc"]).ravel()[0])

            rf = np.load(src_npy)
            nz, _ = rf.shape
            z_src = np.linspace(args.src_z_min, args.src_z_max, nz, dtype=np.float64)
            z_dst = train_grid_z(m, nz, args.train_z_min)

            out = resample_axial(rf, z_src, z_dst, fc, c)
            dst_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst_npy, out)
            n_done += 1
            if n_done % 25 == 0:
                print(f"  {pt}: {n_done} escritos ({time.perf_counter()-t0:.0f} s)")

    print(f"\nListo: {n_done} re-muestreados, {n_skip} ya existian "
          f"({(time.perf_counter()-t0)/60:.1f} min) -> {args.dst}")


if __name__ == "__main__":
    main()
