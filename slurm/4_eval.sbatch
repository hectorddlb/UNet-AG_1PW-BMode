#!/bin/bash
#SBATCH --job-name=ag_eval
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/eval_%j.out
#SBATCH --error=slurm_logs/eval_%j.err

# Paso 4 — validacion.
#
#   Field II (dentro de dominio) : FWHM lateral/axial en los fantomas de wires,
#                                  CNR y gCNR en los de quistes, sobre el split
#                                  de test. Compara contra DAS 1 PW y
#                                  Schiffner Fd-F 75 PW en el mismo CSV.
#   PICMUS  (fuera de dominio)   : los 4 fantomas del benchmark
#                                  (sim/exp x resolution_distortion/contrast_speckle).
#
# El modelo no ve un solo dato de PICMUS durante el entrenamiento: es una
# transferencia limpia de simulacion a adquisicion real.
#
#   sbatch --export=ALL,RUN=results/unet_ag_beta05 slurm/4_eval.sbatch

set -euo pipefail
mkdir -p slurm_logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-unet_ag}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DATA_ROOT="${DATA_ROOT:?exporta DATA_ROOT=<ruta a field_ii_data_multiangle>}"
PICMUS_ROOT="${PICMUS_ROOT:?exporta PICMUS_ROOT=<ruta a la raiz de PICMUS>}"
TARGET_ROOT="${TARGET_ROOT:-targets_schiffner_aligned}"
RUN="${RUN:-results/unet_ag_beta05}"
CKPT="${CKPT:-$RUN/checkpoints/best.pt}"

echo "=== eval ${SLURM_JOB_ID:-local} — $(date) ==="
echo "  ckpt: $CKPT"

python -u -m unet_ag.eval.evaluate_fieldii \
    --ckpt        "$CKPT" \
    --data_root   "$DATA_ROOT" \
    --target_root "$TARGET_ROOT" \
    --out         "$RUN/eval_fieldii.csv" \
    --model_type  b1_v4 --base_ch 32 --phantom_type both \
    --nz 612 --nx 388 --device cuda

python -u -m unet_ag.eval.evaluate_picmus \
    --ckpt        "$CKPT" \
    --picmus_root "$PICMUS_ROOT" \
    --out         "$RUN/eval_picmus.csv" \
    --mode both --phantom both --model_type b1_v4 --base_ch 32 \
    --nz 612 --nx 388 \
    --prefilter_fc_mhz 0.0 --prefilter_bw_mhz 0.0 --device cuda

echo "=== Done — $(date) ==="
