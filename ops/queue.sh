# Usage: bash ops/queue.sh <session> <wait_for> <script> <name> "<pip packages>"
set -e
C=~/.local/bin/colab
sed -e "s#__WAIT_FOR__#$2#" -e "s#__SCRIPT__#$3#" -e "s#__NAME__#$4#" -e "s#__PIP__#$5#" ops/queue_project.py > /tmp/q_$1_$4.py
tar -czf /tmp/genai.tgz 01_rag_retriever_finetuning 02_qlora_text_to_sql 03_vlm_chartqa_lora 04_whisper_marathi_asr
$C upload -s $1 /tmp/genai.tgz /content/genai.tgz
timeout 200 $C exec -s $1 -f /tmp/q_$1_$4.py
