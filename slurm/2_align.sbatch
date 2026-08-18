#!/bin/bash
#SBATCH --job-name=ag_align
#SBATCH --partition=C1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/align_%j.out
#SBATCH --error=slurm_logs/align_%j.err

# Paso 2 — re-muestrear los targets a la rejilla axial del entrenador.
#
# NO ES OPCIONAL. El generador de targets conforma en z ∈ [5, 50] mm con 612
# filas; el entrenador construye la suya como z ∈ [5, (ns-1)·c/(2·fs)] =
# [5, 54.22] mm, tambien con 612 filas. Mismo shape, distinta escala: el
# cargador de targets no reinterpola (solo lo hace si cambia el shape) y el
# target queda desplazado un 9.4 % en profundidad. Un eco a 44 mm cae 3.7 mm
# fuera de sitio, muy por encima de la FWHM axial (~1.6 mm), y la red aprende
# a emborronar para minimizar la perdida.
#
# Es CPU puro (Hilbert + interpolacion), asi que no compite por la GPU.
#
#   sbatch slurm/2_align.sbatch

set -euo pipefail
mkdir -p slurm_logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-unet_ag}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DATA_ROOT="${DATA_ROOT:?exporta DATA_ROOT=<ruta a field_ii_data_multiangle>}"
SRC="${SRC:-targets_schiffner_raw}"
DST="${DST:-targets_schiffner_aligned}"

python -u scripts/align_targets_to_train_grid.py \
    --src "$SRC" --dst "$DST" --data_root "$DATA_ROOT"

echo "  points: $(ls "$DST/points" 2>/dev/null | wc -l)/300"
echo "  cysts : $(ls "$DST/cysts"  2>/dev/null | wc -l)/300"
echo "=== Done — $(date) ==="
