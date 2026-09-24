# ops — running jobs on Google Colab GPUs from a terminal

| file | what it does |
|---|---|
| `renew_tokens.py` | **Workaround for [google-colab-cli#106](https://github.com/googlecolab/google-colab-cli/issues/106).** The runtime-proxy token stored by `colab new` expires after 3600 s and is never renewed; after ~1 h every `exec`/`download` gets 404, the CLI deletes the session and kills its keep-alive, and Colab reclaims the VM mid-training. This script fetches fresh tokens from the assignments API and writes them into the CLI's session state. |
| `renew_loop.sh` | runs `renew_tokens.py` every 5 minutes |
| `queue.sh` / `queue_project.py` | queue a project on a runtime: wait for the running job's PID to exit, smoke-test the next project (`QUICK=1`), then run it in full |

```bash
nohup bash ops/renew_loop.sh > renew.log 2>&1 &
bash ops/queue.sh <session> "<running job cmdline>" 03_vlm_chartqa_lora/train.py vlm "qwen-vl-utils"
```

Why wait on a PID rather than `pgrep -f <pattern>`? A waiter launched through `bash -c '... pgrep -f pattern ...'` contains the pattern in its own command line and would wait on itself forever.
