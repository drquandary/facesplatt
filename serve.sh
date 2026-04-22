#!/usr/bin/env bash
# Serve facesplatt locally + optional ngrok tunnel for remote sharing.
#
# Usage:
#   ./serve.sh              # local only at http://localhost:8000/
#   ./serve.sh --share      # also start ngrok and print the public URL
#
# Requirements for --share: `brew install ngrok` and `ngrok config add-authtoken <token>`
set -e
cd "$(dirname "$0")"

PORT="${PORT:-8000}"

# Kill any prior server on the same port (otherwise address-in-use).
lsof -ti:"${PORT}" | xargs -r kill 2>/dev/null || true

echo "▶ static server on http://localhost:${PORT}/"
python3 -m http.server "${PORT}" > /tmp/facesplatt-server.log 2>&1 &
SERVER_PID=$!
sleep 1

if [ "${1:-}" = "--share" ]; then
  if ! command -v ngrok >/dev/null 2>&1; then
    echo "!! ngrok not installed. Install with: brew install ngrok"
    echo "   Then: ngrok config add-authtoken <your-token> (from https://dashboard.ngrok.com)"
    exit 1
  fi
  echo "▶ ngrok tunnel starting…"
  # Run ngrok in foreground; its web UI shows the public URL at http://127.0.0.1:4040
  ngrok http "${PORT}"
  # After ngrok exits, clean up the server too.
  kill "${SERVER_PID}" 2>/dev/null || true
else
  echo "  open http://localhost:${PORT}/"
  echo "  (or: ./serve.sh --share  to get a public ngrok URL)"
  echo ""
  echo "Press Ctrl-C to stop."
  wait "${SERVER_PID}"
fi
