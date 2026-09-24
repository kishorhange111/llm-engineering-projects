# ops — running jobs on Google Colab GPUs from a terminal

| file | what it does |
|---|---|
| `renew_tokens.py` | **Workaround for [google-colab-cli#106](https://github.com/googlecolab/google-colab-cli/issues/106).** The runtime-proxy token stored by `colab new` expires after 3600 s and is never renewed; after ~1 h every `exec`/`download` gets 404, the CLI deletes the session and kills its keep-alive, and Colab reclaims the VM mid-training. This script fetches fresh tokens from the assignments API and writes them into the CLI's session state. |
| `renew_loop.sh` | runs `renew_tokens.py` every 5 minutes |
| `queue.sh` / `queue_project.py` | queue a project on a runtime: wait for the running job's PID to exit, smoke-test the next project (`QUICK=1`), then run it in full |
| `heartbeat.sh` | **Kernel heartbeat.** Runtimes that only received the CLI's tunnel keep-alive were reclaimed ~15–25 min after the last real kernel execution (6 mass losses in one day), while a runtime with a job executing code every 10 min survived for hours. Running a trivial cell on every session every 4 min stopped the losses completely. |
| `watchdog.sh` | every 3 min: flags `LOST` sessions (server assignment gone) and, the moment a project's result file appears, **downloads its whole result folder** — results survive even if the VM is reclaimed seconds later |
| `new_session.sh` | allocate a runtime, optionally rejecting a region and retrying |
| `chain_any.py` | generic launcher: unpack code, install deps, run a queue of projects in the background with `;`-separated steps so one failure never blocks the rest |

```bash
nohup bash ops/renew_loop.sh > renew.log 2>&1 &
bash ops/queue.sh <session> "<running job cmdline>" 03_vlm_chartqa_lora/train.py vlm "qwen-vl-utils"
```

Why wait on a PID rather than `pgrep -f <pattern>`? A waiter launched through `bash -c '... pgrep -f pattern ...'` contains the pattern in its own command line and would wait on itself forever.
