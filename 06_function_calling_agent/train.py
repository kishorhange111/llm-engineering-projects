"""
Fine-tuning a small LLM for function calling (the core skill of AI agents)
==========================================================================

TASK: given a set of available tools (JSON schemas) and a user request,
decide whether a tool is needed and, if so, emit the exact call:
    <functioncall>{"name": "get_exchange_rate", "arguments": {"base": "USD", "target": "INR"}}</functioncall>
or answer in plain text when no tool applies. This is what agent frameworks
(LangChain / LangGraph / CrewAI tools, OpenAI function calling, MCP tool use)
rely on - a wrong tool name or a malformed argument breaks the whole agent.

MODEL: Qwen2.5-1.5B-Instruct - small enough to run cheaply and locally,
which is exactly where fine-tuning pays off for agents.
DATA: glaive-function-calling-v2 (~113k conversations). We take each
conversation's first user turn and the assistant's first response (a function
call, or a plain answer when no function fits).
METHOD: LoRA (r=16) on all attention + MLP projections, bf16, loss on the
assistant response only.

METRICS on held-out conversations (same prompt before/after):
    call decision accuracy  - called a tool exactly when it should (and not otherwise)
    tool name accuracy      - right function, among cases that need one
    full-call exact match   - right function AND arguments equal as JSON
    valid JSON rate         - parsable call when a call was attempted
"""
import json
import os
import re
import time

import torch

QUICK = os.environ.get("QUICK") == "1"
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
OUT = os.environ.get("OUT", "/content/function_calling")
N_TRAIN = 300 if QUICK else int(os.environ.get("N_TRAIN", "15000"))
N_TEST = 60 if QUICK else int(os.environ.get("N_TEST", "600"))
MAX_LEN = 1024
TIME_BUDGET_HOURS = float(os.environ.get("TIME_BUDGET_HOURS", "0.8"))
INSTRUCTION = ("If one of the functions is needed, reply ONLY with <functioncall>{\"name\": ..., \"arguments\": {...}}"
               "</functioncall>. Otherwise answer the user directly.")
CALL_RE = re.compile(r"<functioncall>\s*(\{.*\})\s*(?:</functioncall>|$)", re.S)


# ---------------------------------------------------------------------------
# Data: parse glaive's raw text format into (system, user, target)
# ---------------------------------------------------------------------------
def parse(ex):
    system = ex["system"].replace("SYSTEM:", "").strip()
    chat = ex["chat"]
    m = re.search(r"USER:(.*?)ASSISTANT:(.*?)(?:<\|endoftext\|>|FUNCTION RESPONSE:|USER:|$)", chat, re.S)
    if not m:
        return {"ok": False, "system": "", "user": "", "target": "", "call": ""}
    user, reply = m.group(1).strip(), m.group(2).strip()
    call = ""
    if "<functioncall>" in reply:
        fm = re.search(r'"name":\s*"([^"]+)".*?"arguments":\s*\'(.*)\'\s*\}', reply, re.S)
        try:
            call = json.dumps({"name": fm.group(1), "arguments": json.loads(fm.group(2))}, ensure_ascii=False)
        except Exception:
            return {"ok": False, "system": "", "user": "", "target": "", "call": ""}
        target = f"<functioncall>{call}</functioncall>"
    else:
        target = reply
    return {"ok": bool(user and target), "system": system, "user": user, "target": target, "call": call}


def messages(ex, with_answer):
    msgs = [{"role": "system", "content": f"{ex['system']}\n\n{INSTRUCTION}"}, {"role": "user", "content": ex["user"]}]
    if with_answer:
        msgs.append({"role": "assistant", "content": ex["target"]})
    return msgs


def parse_call(text):
    m = CALL_RE.search(text)
    if not m:
        return None, False
    try:
        c = json.loads(m.group(1))
        return c, isinstance(c, dict) and "name" in c
    except Exception:
        return None, False


@torch.no_grad()
def evaluate(model, tok, test, label):
    model.eval()
    tok.padding_side = "left"
    n_call = decision = name_ok = exact = attempted = valid = 0
    samples, t = [], time.time()
    for i in range(0, len(test), 16):
        batch = test.select(range(i, min(i + 16, len(test))))
        texts = [tok.apply_chat_template(messages(ex, False), add_generation_prompt=True, tokenize=False) for ex in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
        for ex, g in zip(batch, out[:, enc["input_ids"].shape[1]:]):
            text = tok.decode(g, skip_special_tokens=True)
            pred, ok_json = parse_call(text)
            should = bool(ex["call"])
            called = "<functioncall>" in text
            decision += called == should
            if called:
                attempted += 1
                valid += ok_json
            if should:
                n_call += 1
                gold = json.loads(ex["call"])
                if ok_json and pred.get("name") == gold["name"]:
                    name_ok += 1
                    exact += pred.get("arguments") == gold["arguments"]
            if len(samples) < 10:
                samples.append({"user": ex["user"][:300], "gold": ex["target"][:300], "pred": text[:300]})
    res = {"call_decision_accuracy": decision / len(test), "tool_name_accuracy": name_ok / max(n_call, 1),
           "full_call_exact_match": exact / max(n_call, 1), "valid_json_rate": valid / max(attempted, 1),
           "test_examples": len(test), "examples_needing_a_call": n_call, "eval_minutes": (time.time() - t) / 60}
    print(label, res, flush=True)
    return res, samples


def main():
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    os.makedirs(OUT, exist_ok=True)
    raw = load_dataset("glaiveai/glaive-function-calling-v2", split="train").shuffle(seed=42)
    raw = raw.select(range(min(len(raw), N_TRAIN + 3 * N_TEST + 2000)))
    data = raw.map(parse, remove_columns=raw.column_names).filter(lambda e: e["ok"])
    test = data.select(range(N_TEST))
    train = data.select(range(N_TEST, min(len(data), N_TEST + N_TRAIN)))
    print(f"train {len(train)}, test {len(test)} ({sum(bool(c) for c in test['call'])} need a call)")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    base_res, base_samples = evaluate(model, tok, test, "zero-shot")

    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()

    def encode(ex):
        prompt = tok.apply_chat_template(messages(ex, False), add_generation_prompt=True, tokenize=False)
        full = tok.apply_chat_template(messages(ex, True), tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"][:MAX_LEN]
        return {"input_ids": f_ids, "labels": ([-100] * len(p_ids) + f_ids[len(p_ids):])[:len(f_ids)]}

    train_tok = train.map(encode, remove_columns=train.column_names).filter(lambda e: any(l != -100 for l in e["labels"]))

    def collate(b):
        w = max(len(x["input_ids"]) for x in b)
        pad = lambda s, v: s + [v] * (w - len(s))
        return {"input_ids": torch.tensor([pad(x["input_ids"], tok.pad_token_id) for x in b]),
                "labels": torch.tensor([pad(x["labels"], -100) for x in b]),
                "attention_mask": torch.tensor([pad([1] * len(x["input_ids"]), 0) for x in b])}

    class TimeBudget(TrainerCallback):
        def __init__(self):
            self.deadline = time.time() + TIME_BUDGET_HOURS * 3600

        def on_step_end(self, args, state, control, **k):
            if time.time() > self.deadline:
                control.should_training_stop = True
            return control

    tok.padding_side = "right"
    # batch 4 x accumulation 4 = effective batch 16; a batch of 16 x 1024 tokens x 151k-token vocabulary
    # produced a 9 GB logits tensor and ran out of memory on a 24 GB L4.
    args = TrainingArguments(output_dir=OUT, num_train_epochs=1, per_device_train_batch_size=4,
                             gradient_accumulation_steps=4, learning_rate=2e-4, lr_scheduler_type="cosine",
                             warmup_steps=0.05, bf16=True, gradient_checkpointing=True, logging_steps=20,
                             save_strategy="no", remove_unused_columns=False, report_to="none", seed=42)
    trainer = Trainer(model=model, args=args, train_dataset=train_tok, data_collator=collate, callbacks=[TimeBudget()])
    t = time.time()
    result = trainer.train()
    train_min = (time.time() - t) / 60

    tuned_res, tuned_samples = evaluate(model, tok, test, "LoRA fine-tuned")
    summary = {"model": MODEL, "train_examples": len(train_tok), "trainable_params": trainable, "total_params": total,
               "train_minutes": train_min, "final_train_loss": result.training_loss,
               "gpu": torch.cuda.get_device_name(0), "zero_shot": base_res, "fine_tuned": tuned_res}
    json.dump(summary, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    json.dump({"zero_shot": base_samples, "fine_tuned": tuned_samples}, open(os.path.join(OUT, "samples.json"), "w"),
              indent=2, ensure_ascii=False)
    json.dump(trainer.state.log_history, open(os.path.join(OUT, "log_history.json"), "w"))
    print(json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
