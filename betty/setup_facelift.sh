#!/usr/bin/env bash
# Runs on Betty LOGIN NODE (no GPU needed — just pip + clone).
# Creates a project-local conda env with FaceLift's dependencies and clones the repo.
# Safe to re-run; skips steps that already completed.
set -euo pipefail

FP_ROOT="${BETTY_ROOT:-/vast/home/j/jvadala/facesplatt}"
ENV_PATH="${FP_ROOT}/envs/facelift"
PROJECT_ROOT="${FP_ROOT}"
FACELIFT_REPO="${FP_ROOT}/FaceLift"

echo "▶ Betty FaceLift setup"
echo "  project:   ${PROJECT_ROOT}"
echo "  env:       ${ENV_PATH}"
echo "  FaceLift:  ${FACELIFT_REPO}"

mkdir -p "${PROJECT_ROOT}/envs" "${FP_ROOT}" "${FP_ROOT}/outputs" "${FP_ROOT}/hf_cache"

# --- Modules (needed for diff-gaussian-rasterization's CUDA compile) ---
module purge
module load anaconda3/2023.09-0
module load cuda/12.8.1
module load gcc/13.3.0
echo "✓ modules loaded: anaconda3, cuda/12.8.1, gcc/13.3.0"

# --- Clone FaceLift ---
if [ ! -d "${FACELIFT_REPO}/.git" ]; then
  git clone https://github.com/weijielyu/FaceLift.git "${FACELIFT_REPO}"
  echo "✓ cloned FaceLift"
else
  echo "✓ FaceLift already cloned"
fi

# --- Conda env ---
if [ ! -d "${ENV_PATH}" ]; then
  echo "▶ creating conda env (python 3.11)..."
  conda create -p "${ENV_PATH}" python=3.11 -y
fi
source activate "${ENV_PATH}"
echo "✓ env active: ${CONDA_PREFIX}"

# --- Python deps (mirrors FaceLift setup_env.sh but omits sudo + video) ---
python -m pip install --upgrade pip

# PyTorch w/ CUDA 12.4 wheels — ABI-compatible with Betty's cuda/12.8.1
pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu124

# Core ML
pip install \
  transformers==4.44.2 \
  diffusers[torch]==0.30.3 \
  huggingface-hub==0.35.3 \
  xformers==0.0.27.post2 \
  accelerate==0.33.0

# Vision
pip install Pillow==10.4.0 opencv-python==4.10.0.84 \
  scikit-image==0.21.0 lpips==0.1.4
pip install facenet-pytorch --no-deps
pip install rembg

# Numeric
pip install numpy==1.26.4 matplotlib==3.7.5 scikit-learn==1.3.2 \
  einops==0.8.0 jaxtyping==0.2.19 pytorch-msssim==1.0.0

# Utilities
pip install easydict==1.13 pyyaml==6.0.2 \
  termcolor==2.4.0 plyfile==1.0.3 tqdm rich

# The CUDA-compiled rasterizer — requires cuda/gcc modules to be loaded (done above)
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization

echo "✓ python packages installed"

# --- Quick sanity check ---
python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda built={torch.version.cuda}  cuda available={torch.cuda.is_available()}")
import diffusers, transformers, xformers
print(f"diffusers {diffusers.__version__}  transformers {transformers.__version__}  xformers {xformers.__version__}")
try:
    import diff_gaussian_rasterization
    print("diff_gaussian_rasterization: OK")
except Exception as e:
    print("diff_gaussian_rasterization: FAIL —", e)
PY

echo ""
echo "✓ setup complete. Next step:"
echo "    cd ${FP_ROOT}/betty && sbatch run_facelift.sbatch"
