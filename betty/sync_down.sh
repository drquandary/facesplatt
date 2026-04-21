#!/usr/bin/env bash
# Pull splats/ + previews/ from Betty → local.
# Run from facesplatt repo root.
set -euo pipefail

BETTY_USER="${BETTY_USER:-jvadala}"
BETTY_HOST="${BETTY_HOST:-login.betty.parcc.upenn.edu}"
REMOTE_ROOT="${BETTY_ROOT:-/vast/home/j/jvadala/facesplatt}"

mkdir -p splats previews

rsync -avz --progress \
  "${BETTY_USER}@${BETTY_HOST}:${REMOTE_ROOT}/splats/" splats/

rsync -avz --progress \
  "${BETTY_USER}@${BETTY_HOST}:${REMOTE_ROOT}/previews/" previews/

echo "✓ Downloaded: $(ls splats/*.ply 2>/dev/null | wc -l | tr -d ' ') .ply files, $(du -sh splats/ | cut -f1) total"
