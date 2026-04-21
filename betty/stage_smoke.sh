#!/usr/bin/env bash
# Stage a small subset of cfad/ for a smoke test run.
# Usage: bash stage_smoke.sh [N]    (default 5 faces, frontal-pose only)
set -euo pipefail

FP_ROOT="${BETTY_ROOT:-/vast/home/j/jvadala/facesplatt}"
N="${1:-5}"
SMOKE_DIR="${FP_ROOT}/cfad_smoke"

rm -rf "${SMOKE_DIR}"
mkdir -p "${SMOKE_DIR}"

# Prefer frontal shots (pose code 111) for predictable landmark/depth behavior.
cd "${FP_ROOT}/cfad"
ls *-111-*.png 2>/dev/null | head -"${N}" | while read -r f; do
  cp "$f" "${SMOKE_DIR}/"
done

echo "✓ staged $(ls "${SMOKE_DIR}"/*.png | wc -l) faces in ${SMOKE_DIR}"
ls -la "${SMOKE_DIR}/"
