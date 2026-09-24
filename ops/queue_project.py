# (runs on a VM) Queue a project: wait for the currently running job (found by WAIT_FOR in its command line)
# to exit, then smoke-test the new project and, if that passes, run it in full.
# Edit the three constants before `colab exec -f`.
import subprocess

WAIT_FOR = "__WAIT_FOR__"          # substring of the running python job's command line
SCRIPT = "__SCRIPT__"              # path relative to /content/genai
NAME = "__NAME__"                  # log name
PIP = "__PIP__"

print(subprocess.run(f"pip install -q {PIP} 2>&1 | tail -1; mkdir -p /content/genai && "
                     "tar -xzf /content/genai.tgz -C /content/genai --keep-newer-files 2>/dev/null; ls /content/genai",
                     shell=True, capture_output=True, text=True).stdout)
rows = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout.splitlines()[1:]
pids = [r.split()[0] for r in rows if WAIT_FOR in r and "/bin/sh" not in r and "grep" not in r]
wait = f"while kill -0 {pids[0]} 2>/dev/null; do sleep 30; done; " if pids else ""
subprocess.Popen(f"cd /content/genai && nohup bash -c '{wait}"
                 f"QUICK=1 OUT=/content/{NAME}_quick python {SCRIPT} > {NAME}_quick.log 2>&1 && "
                 f"python {SCRIPT} > {NAME}.log 2>&1' > /dev/null 2>&1 &", shell=True)
print(f"queued {NAME}" + (f" after pid {pids[0]}" if pids else " (starting now)"))
