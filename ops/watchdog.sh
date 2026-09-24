#!/bin/bash
# Watchdog for all training sessions. Every 3 minutes:
#   * LOST <session>      - a session of mine is no longer assigned on the Colab server
#   * FINISHED <project>  - a result file appeared on a VM (checked via the contents API, no kernel needed)
#   * ALIVE ...           - heartbeat line
# Usage: bash ops/watchdog.sh >> ops/monitor.log 2>&1
C=~/.local/bin/colab
PY=~/.local/share/uv/tools/google-colab-cli/bin/python
cd "$(dirname "$0")"
# session : remote result file : project label
WATCH="
seq:/content/genai/01_rag_retriever_finetuning/results/metrics.json:rag_retriever
seq:/content/grpo_gsm8k/metrics.json:grpo_reasoning
p5:/content/function_calling/metrics.json:function_calling
seq:/content/whisper_mr/metrics.json:whisper_marathi
seq:/content/dlp/results/02_shiftvit_cifar10/metrics.json:02_shiftvit
small:/content/dlp/results/09_gpt2_lora_finetuning/metrics.json:09_gpt2_lora
t1:/content/dlp/results/11_neural_style_transfer/metrics.json:11_style_transfer
small:/content/dlp/results/12_adain_style_transfer/metrics.json:12_adain
t1:/content/dlp/results/13_music_transformer_midi/metrics.json:13_music_transformer
llm2:/content/out/summary.json:llm_sft
llm2:/content/dpo/dpo_summary.json:llm_dpo
llm2:/content/eval/summary.json:llm_eval
p5:/content/guardrail/metrics.json:guardrail
seq:/content/sql_distill/metrics.json:sql_distillation
seq:/content/serving/metrics.json:serving_benchmark
"
declare -A done_
while true; do
  live=$($PY -c 'from colab_cli.common import state
print(" ".join(a.endpoint for a in state.client.list_assignments()))' 2>/dev/null)
  if [ -n "$live" ]; then
    mine=$($PY -c 'from colab_cli.common import state
print(" ".join(f"{n}={s.endpoint}" for n, s in state.store.list().items()))' 2>/dev/null)
    for pair in $mine; do
      n=${pair%%=*}; e=${pair#*=}
      [[ " $live " == *" $e "* ]] || echo "$(date -u +%T) LOST $n ($e)"
    done
  else
    echo "$(date -u +%T) WARN cannot reach Colab API (network?)"
  fi
  for line in $WATCH; do
    s=${line%%:*}; rest=${line#*:}; f=${rest%:*}; label=${rest##*:}
    [ -n "${done_[$label]}" ] && continue
    if timeout 60 $C ls -s $s "$(dirname $f)" 2>/dev/null | grep -qx "$(basename $f)"; then
      # Download every file in the result folder immediately - VMs can disappear at any moment.
      dir=$(dirname $f); dest=/mnt/e/Resumes/staging/$label; mkdir -p $dest; ok=1
      for x in $(timeout 60 $C ls -s $s "$dir" 2>/dev/null | grep -v '/$'); do
        timeout 300 $C download -s $s "$dir/$x" "$dest/$x" >/dev/null 2>&1 || ok=0
      done
      if [ $ok = 1 ]; then
        done_[$label]=1
        echo "$(date -u +%T) FINISHED $label on $s -> saved to E:\\Resumes\\staging\\$label"
      else
        echo "$(date -u +%T) FINISHED $label on $s but download incomplete; will retry"
      fi
    fi
  done
  echo "$(date -u +%T) ALIVE finished=[${!done_[*]}]"
  sleep 180
done
