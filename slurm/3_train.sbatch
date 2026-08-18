#!/bin/bash
#SBATCH --job-name=ag_train
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --output=slurm_logs/train_%j.out
#SBATCH --error=slurm_logs/train_%j.err

# Paso 3 — entrenar UNet-AG.
#
# Configuracion de la tesis (la que produjo la fila UNet-AG de la tabla):
#   phantom_type = points, anchor_weight = 0.5, 20 epocas, seed 42,
#   pulse_apply_prob = 0.7, rejilla 612x388, base_ch 32.
# ~9 h en una TITAN RTX.
#
#   sbatch slurm/3_train.sbatch
#   sbatch --export=ALL,SEED=1,OUTPUT_DIR=results/seed1 slurm/3_train.sbatch
#
# Smoke de 10 min para comprobar la instalacion antes de quemar 9 h:
#   sbatch --export=ALL,EPOCHS=2,MAX_PHANTOMS_TRAIN=4,MAX_PHANTOMS_VAL=2,SMOKE_STEPS=4 \
#          slurm/3_train.sbatch

set -euo pipefail
mkdir -p slurm_logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-unet_ag}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DATA_ROOT="${DATA_ROOT:?exporta DATA_ROOT=<ruta a field_ii_data_multiangle>}"
TARGET_ROOT="${TARGET_ROOT:-targets_schiffner_aligned}"
OUTPUT_DIR="${OUTPUT_DIR:-results/unet_ag_beta05}"
PHANTOM_TYPE="${PHANTOM_TYPE:-points}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.5}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-42}"
PULSE_APPLY_PROB="${PULSE_APPLY_PROB:-0.7}"

EXTRA=()
[ -n "${MAX_PHANTOMS_TRAIN:-}" ] && EXTRA+=(--max_phantoms_train "$MAX_PHANTOMS_TRAIN")
[ -n "${MAX_PHANTOMS_VAL:-}"   ] && EXTRA+=(--max_phantoms_val   "$MAX_PHANTOMS_VAL")
[ -n "${SMOKE_STEPS:-}"        ] && EXTRA+=(--smoke_steps        "$SMOKE_STEPS")
[ "${RESUME:-0}" = "1" ]         && EXTRA+=(--resume)

echo "=== train ${SLURM_JOB_ID:-local} on $(hostname) — $(date) ==="
echo "  data    : $DATA_ROOT"
echo "  targets : $TARGET_ROOT"
echo "  salida  : $OUTPUT_DIR"
echo "  beta    : $ANCHOR_WEIGHT   phantoms: $PHANTOM_TYPE   seed: $SEED"

python -u -m unet_ag.train.b1_v5_pulse_aug \
    --data_root        "$DATA_ROOT" \
    --target_root      "$TARGET_ROOT" \
    --output_dir       "$OUTPUT_DIR" \
    --phantom_type     "$PHANTOM_TYPE" \
    --anchor_weight    "$ANCHOR_WEIGHT" \
    --epochs           "$EPOCHS" \
    --seed             "$SEED" \
    --pulse_apply_prob "$PULSE_APPLY_PROB" \
    --nz 612 --nx 388 --base_ch 32 --embed_ch 16 \
    --mse_weight 0.7 --l1_weight 0.2 --ssim_weight 0.1 \
    --lr 1e-3 --weight_decay 1e-4 \
    --device cuda \
    "${EXTRA[@]}"

echo "=== Done — $(date) ==="
