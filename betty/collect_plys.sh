#!/usr/bin/env bash
# Flatten outputs/<image>/gaussians.ply → splats/<image>.ply for clean rsync download.
set -euo pipefail

FP_ROOT="${BETTY_ROOT:-/vast/home/j/jvadala/facesplatt}"
OUT="${FP_ROOT}/outputs"
DEST="${FP_ROOT}/splats"

mkdir -p "${DEST}"
n=0
for d in "${OUT}"/*/; do
  name=$(basename "${d%/}")
  src="${d}gaussians.ply"
  if [ -f "${src}" ]; then
    cp -f "${src}" "${DEST}/${name}.ply"
    n=$((n+1))
  fi
done

# Also copy the input / multiview / output preview per face (tiny PNGs, useful for the gallery HUD)
PREVIEW="${FP_ROOT}/previews"
mkdir -p "${PREVIEW}"
for d in "${OUT}"/*/; do
  name=$(basename "${d%/}")
  [ -f "${d}output.png" ] && cp -f "${d}output.png" "${PREVIEW}/${name}.png"
done

echo "✓ collected ${n} .ply files into ${DEST}"
du -sh "${DEST}" "${PREVIEW}"
