#!/usr/bin/env bash
# Incremental pull of .ply files from Betty → local, deleting remote copies
# after safe receipt. Lets a full 120-face FaceLift run fit in a tight home quota.
#
# Usage:
#   bash betty/incremental_pull.sh                # polls every 60s forever
#   INTERVAL=30 bash betty/incremental_pull.sh    # tighter poll
#   LOCAL=splats bash betty/incremental_pull.sh   # custom local dir
set -u

BETTY_HOST="${BETTY_HOST:-login.betty.parcc.upenn.edu}"
REMOTE_ROOT="${BETTY_ROOT:-/vast/home/j/jvadala/facesplatt}"
REMOTE_OUTPUTS="${REMOTE_OUTPUTS:-${REMOTE_ROOT}/outputs}"
LOCAL="${LOCAL:-splats}"
INTERVAL="${INTERVAL:-60}"

mkdir -p "${LOCAL}"

echo "▶ incremental pull"
echo "  remote: ${BETTY_HOST}:${REMOTE_OUTPUTS}"
echo "  local:  ${LOCAL}"
echo "  poll:   every ${INTERVAL}s (Ctrl-C to stop)"

while :; do
  # List subdirs that have a completed gaussians.ply. We use -size +1M to ensure
  # the file isn't mid-write (PLYs end up >>1MB).
  faces=$(ssh -o BatchMode=yes "${BETTY_HOST}" \
    "find ${REMOTE_OUTPUTS} -maxdepth 2 -name gaussians.ply -size +1M 2>/dev/null | xargs -r -n1 dirname | xargs -r -n1 basename" \
    2>/dev/null | sort -u)

  if [ -z "${faces}" ]; then
    ts=$(date +%H:%M:%S)
    echo "  [${ts}] no new complete .ply yet"
  else
    for face in ${faces}; do
      dest="${LOCAL}/${face}.ply"
      if [ -f "${dest}" ]; then
        continue  # already pulled this one
      fi
      echo "  ↓ pulling ${face}.ply"
      if rsync -az --partial \
          "${BETTY_HOST}:${REMOTE_OUTPUTS}/${face}/gaussians.ply" \
          "${dest}.tmp" 2>/dev/null; then
        mv "${dest}.tmp" "${dest}"
        size=$(du -h "${dest}" | cut -f1)
        echo "    ✓ ${size} landed locally"
        # Safe to remove remote dir (input.png, multiview.png, output.png, turntable.mp4, ply)
        ssh -o BatchMode=yes "${BETTY_HOST}" "rm -rf ${REMOTE_OUTPUTS}/${face}" 2>/dev/null \
          && echo "    ✓ remote cleaned" \
          || echo "    ! remote cleanup failed (not fatal)"
      else
        echo "    ✗ rsync failed; will retry next tick"
        rm -f "${dest}.tmp"
      fi
    done
  fi

  # Summary
  have=$(ls "${LOCAL}"/*.ply 2>/dev/null | wc -l | tr -d ' ')
  totalsize=$(du -sh "${LOCAL}" 2>/dev/null | cut -f1)
  echo "  so far: ${have} .ply files, ${totalsize} local"

  sleep "${INTERVAL}"
done
