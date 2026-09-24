#!/bin/bash
# Kernel heartbeat: run a trivial cell on every one of my sessions every 4 minutes.
# Observation (24 Sep): runtimes that only received the CLI's tunnel keep-alive were reclaimed ~15-25 min after
# the last real kernel execution, while the one session with a job executing code every 10 min survived for hours.
C=~/.local/bin/colab
PY=~/.local/share/uv/tools/google-colab-cli/bin/python
while true; do
  for s in $($PY -c 'from colab_cli.common import state
print(" ".join(n for n in state.store.list() if not n.startswith("power-train")))' 2>/dev/null); do
    echo "print(1)" | timeout 90 $C exec -s $s >/dev/null 2>&1 && r=ok || r=FAIL
    echo "$(date -u +%T) heartbeat $s $r"
  done
  sleep 240
done
