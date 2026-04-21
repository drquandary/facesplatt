#!/usr/bin/env bash
#SBATCH --job-name=facelift_cfad
#SBATCH --partition=b200-mig45
#SBATCH --qos=mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/facelift_%j.out
#SBATCH --error=logs/facelift_%j.err

# facesplatt / FaceLift — process all cfad/*.png → outputs/<image>/gaussians.ply
set -euo pipefail

FP_ROOT="${BETTY_ROOT:-/vast/home/j/jvadala/facesplatt}"
ENV_PATH="${FP_ROOT}/envs/facelift"
FACELIFT_REPO="${FP_ROOT}/FaceLift"

# Override INPUT_DIR via --export to sbatch for smoke-test runs.
INPUT_DIR="${INPUT_DIR:-${FP_ROOT}/cfad}"
OUTPUT_DIR="${OUTPUT_DIR:-${FP_ROOT}/outputs}"
HF_CACHE="${FP_ROOT}/hf_cache"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
echo "=== GPUs: ${CUDA_VISIBLE_DEVICES:-none} ==="
mkdir -p logs "${OUTPUT_DIR}" "${HF_CACHE}" "${HF_CACHE}/u2net"

if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/Z98-lmod.sh ] && source /etc/profile.d/Z98-lmod.sh
fi

# --- Modules ---
module purge
module load anaconda3/2023.09-0
module load cuda/12.8.1
module load gcc/13.3.0
source activate "${ENV_PATH}"

# --- Cache paths → project storage ---
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export U2NET_HOME="${HF_CACHE}/u2net"   # rembg background removal weights
export TOKENIZERS_PARALLELISM=false

# --- Run inference ---
cd "${FACELIFT_REPO}"
python inference.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --auto_crop \
  --seed 4 \
  --guidance_scale_2D 3.0 \
  --step_2D 50

echo "=== inference done at $(date) ==="

# --- Quick summary ---
n_ply=$(find "${OUTPUT_DIR}" -name 'gaussians.ply' | wc -l)
echo "=== produced ${n_ply} .ply files ==="
