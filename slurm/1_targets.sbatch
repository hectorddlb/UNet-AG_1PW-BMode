#!/bin/bash
#SBATCH --job-name=ag_targets
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/targets_%A_%a.out
#SBATCH --error=slurm_logs/targets_%A_%a.err
#SBATCH --array=0-5

# Paso 1 — targets Schiffner Fd-F 75 PW sobre los fantomas de Field II.
#
# Es el supervisor del entrenamiento: la compresion coherente de 75 ondas
# planas con apodizacion de F-number dependiente de frecuencia. Se precomputa
# una vez porque no cambia entre corridas.
#
# El array reparte los 300 fantomas de cada tipo en 6 tramos de 50. Cada tramo
# tarda ~2 h en una TITAN RTX; los quistes (50.000 dispersores) dominan.
#
#   sbatch slurm/1_targets.sbatch
#
# Sin SLURM, el equivalente secuencial es:
#   python scripts/precompute_schiffner_targets.py \
#       --data_root "<...>/field_ii_data_multiangle" \
#       --output_dir targets_schiffner_raw --device cuda

set -euo pipefail
mkdir -p slurm_logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-unet_ag}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DATA_ROOT="${DATA_ROOT:?exporta DATA_ROOT=<ruta a field_ii_data_multiangle>}"
OUT="${OUT:-targets_schiffner_raw}"

START=$(( SLURM_ARRAY_TASK_ID * 50 ))
END=$(( START + 50 ))

echo "=== targets ${SLURM_ARRAY_TASK_ID}: fantomas [${START}, ${END}) ==="
python -u scripts/precompute_schiffner_targets.py \
    --data_root   "$DATA_ROOT" \
    --output_dir  "$OUT" \
    --range_start "$START" --range_end "$END" \
    --device cuda
echo "=== Done — $(date) ==="
