#!/usr/bin/env bash
# Webcam APIs require a secure context — localhost counts.
# Usage: ./serve.sh  then open http://localhost:8000/
set -e
cd "$(dirname "$0")"
exec python3 -m http.server 8000
