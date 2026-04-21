#!/usr/bin/env bash
# Push cfad/ + betty/ from local Mac → Betty project storage.
# Run from the facesplatt repo root.
set -euo pipefail

BETTY_USER="${BETTY_USER:-jvadala}"
BETTY_PROJECT="${BETTY_PROJECT:-jvadala-facesplatt}"
BETTY_HOST="${BETTY_HOST:-login.betty.parcc.upenn.edu}"
REMOTE_ROOT="/vast/projects/${BETTY_PROJECT}/facesplatt"

echo "▶ rsync local → Betty:${REMOTE_ROOT}"

# Create the remote dir in case it's first run.
ssh "${BETTY_USER}@${BETTY_HOST}" "mkdir -p ${REMOTE_ROOT}/cfad ${REMOTE_ROOT}/betty"

rsync -avz --progress \
  --exclude='.DS_Store' \
  cfad/ "${BETTY_USER}@${BETTY_HOST}:${REMOTE_ROOT}/cfad/"

rsync -avz --progress \
  --exclude='.DS_Store' \
  betty/ "${BETTY_USER}@${BETTY_HOST}:${REMOTE_ROOT}/betty/"

echo "✓ Uploaded $(ls cfad/*.png | wc -l | tr -d ' ') face images and the job scripts."
echo "  Next: ssh ${BETTY_USER}@${BETTY_HOST} and run betty/setup_facelift.sh"
