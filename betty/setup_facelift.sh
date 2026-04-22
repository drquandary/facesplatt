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

# Lmod (`module` command) is defined by /etc/profile.d/Z98-lmod.sh on Betty — only
# sourced for login shells. When we run this script via `ssh 'bash ...'`, bash is
# non-login, so we must source it ourselves.
if ! command -v module >/dev/null 2>&1; then
  # Lmod init script references FPATH — fine normally, but our `set -u` flags it.
  set +u
  if [ -f /etc/profile.d/Z98-lmod.sh ]; then
    source /etc/profile.d/Z98-lmod.sh
  elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
  fi
  set -u
fi

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
pip install rembg onnxruntime
pip install videoio==0.3.0 ffmpeg-python==0.2.0

# Numeric
pip install numpy==1.26.4 matplotlib==3.7.5 scikit-learn==1.3.2 \
  einops==0.8.0 jaxtyping==0.2.19 pytorch-msssim==1.0.0

# Utilities
pip install easydict==1.13 pyyaml==6.0.2 \
  termcolor==2.4.0 plyfile==1.0.3 tqdm rich

# The CUDA-compiled rasterizer.
# R14: --no-build-isolation because setup.py imports torch at top; isolated build
#      venvs have no torch → ModuleNotFoundError.
# R15: TORCH_CUDA_ARCH_LIST required because login nodes have no GPU, so torch
#      can't auto-detect arch → IndexError in _get_cuda_arch_flags. Value covers
#      A100/H100/L40 natively; B200 (Blackwell sm_100) falls through to PTX JIT.
# R16: The 2023-era headers don't `#include <cstdint>`. CUDA 12.4 tolerated it;
#      CUDA 12.8's stricter STL headers reject `std::uintptr_t` / `uint32_t`.
#      Clone locally, inject the include, install from path.
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0+PTX"
pip install --no-build-isolation ninja

DGR_DIR="${FP_ROOT}/dgr"
if [ ! -d "${DGR_DIR}" ]; then
  git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization "${DGR_DIR}"
fi
cd "${DGR_DIR}"
for f in cuda_rasterizer/*.h cuda_rasterizer/*.cu rasterize_points.h rasterize_points.cu; do
  [ -f "$f" ] || continue
  if ! grep -q "#include <cstdint>" "$f"; then
    if grep -q "#pragma once" "$f"; then
      sed -i '/#pragma once/a #include <cstdint>' "$f"
    else
      sed -i '1i #include <cstdint>' "$f"
    fi
  fi
done
pip install --no-build-isolation --no-cache-dir .
cd - >/dev/null

echo "✓ python packages installed"

# R18: Betty's conda-forge ffmpeg has no libx264; videoio.videosave() fails
# with ValueError. FaceLift's imageseq2video is only for the demo turntable
# .mp4 — we only care about gaussians.ply. Patch inference.py to skip both
# render_turntable() and imageseq2video() entirely.
export FACELIFT_REPO
python3 "$(dirname "${BASH_SOURCE[0]}")/fix_skip_turntable.py" 2>/dev/null || true

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
