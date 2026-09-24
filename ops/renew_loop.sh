#!/bin/bash
# Keep every Colab CLI session's runtime-proxy token fresh (workaround for CLI issue #106).
PY=~/.local/share/uv/tools/google-colab-cli/bin/python
cd "$(dirname "$0")"
while true; do
  $PY renew_tokens.py 2>&1 | tail -n 20
  sleep 300
done
