# (runs on a VM) Generic launcher: unpack repos, install deps, run STEPS one after another in the background.
# __NAME__ and __STEPS__ (steps separated by '|||') are substituted before upload.
import subprocess

NAME = "__NAME__"
STEPS = "__STEPS__".split("|||")
print(subprocess.run(
    "mkdir -p /content/dlp /content/genai; [ -f /content/dlp.tgz ] && tar -xzf /content/dlp.tgz -C /content/dlp; "
    "[ -f /content/genai.tgz ] && tar -xzf /content/genai.tgz -C /content/genai; "
    "pip install -q trl peft jiwer rdkit pretty_midi gdown nltk onnx onnxruntime 'protobuf==6.31.1' 2>&1 | grep -vi grpcio | tail -1; "
    "pip uninstall -y -q torchao 2>&1 | tail -1; "
    "python -c 'import torch, trl, peft; print(\"imports ok\", torch.cuda.get_device_name(0))'",
    shell=True, capture_output=True, text=True).stdout[-500:])
open(f"/content/{NAME}_chain.sh", "w").write("\n".join(STEPS) + "\n")
subprocess.Popen(f"nohup bash /content/{NAME}_chain.sh > /content/{NAME}_chain.out 2>&1 &", shell=True)
print(f"{NAME} chain launched: {len(STEPS)} steps")
