"""
QLoRA fine-tuning of a 7B LLM for text-to-SQL, evaluated by *executing* the SQL
===============================================================================

TASK: "Which customers from Germany spent more than 500 last year?"  +  table schema
      ->  SELECT ... FROM ... WHERE ...
Natural-language access to databases is one of the most requested enterprise
LLM features (analytics copilots, "chat with your data", MCP database tools).

QLoRA (Dettmers et al., 2023) - fine-tune a 7-billion-parameter model on one GPU:
    * The base model is loaded in 4-bit NF4 ("NormalFloat4", an information-
      theoretically optimal 4-bit type for normally distributed weights) with
      double quantization -> 7B weights take ~5 GB instead of ~15 GB (bf16).
    * The 4-bit weights are FROZEN. Small LoRA adapters (rank 16) are added to
      every linear layer of every Transformer block and trained in bf16.
    * Only ~0.5% of parameters receive gradients/optimizer state.
    * Paged optimizers + gradient checkpointing keep memory spikes in check.

EVALUATION - execution accuracy, not string matching:
    Every test example ships with CREATE TABLE + INSERT statements. We build the
    database in SQLite, run the gold query and the model's query, and compare
    result sets. Two different-looking queries that return the same rows are
    both correct; a query that looks right but returns the wrong rows is wrong.
    The same metric is used by the Spider and BIRD text-to-SQL benchmarks.
    Compared: the base instruct model (zero-shot, same prompt) vs. QLoRA-tuned.

Data: gretelai/synthetic_text_to_sql - 105k examples across 100 domains.
"""
import json
import os
import re
import sqlite3
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainerCallback, TrainingArguments)

QUICK = os.environ.get("QUICK") == "1"
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
OUT = os.environ.get("OUT", "/content/qlora_sql")
N_TRAIN = 300 if QUICK else int(os.environ.get("N_TRAIN", "12000"))
N_TEST = 40 if QUICK else int(os.environ.get("N_TEST", "500"))
MAX_LEN = 768
TIME_BUDGET_HOURS = float(os.environ.get("TIME_BUDGET_HOURS", "1.2"))
BF16 = torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if BF16 else torch.float16
SYSTEM = "You are an expert SQL assistant. Given a database schema and a question, reply with one SQLite query only."


# ---------------------------------------------------------------------------
# Data + prompt
# ---------------------------------------------------------------------------
def schema_only(context):
    """Show the model the CREATE TABLE statements (the schema), not the data rows."""
    return "\n".join(s.strip() + ";" for s in context.split(";") if s.strip().upper().startswith("CREATE"))


def messages(ex, with_answer):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Schema:\n{schema_only(ex['sql_context'])}\n\nQuestion: {ex['sql_prompt']}"}]
    if with_answer:
        msgs.append({"role": "assistant", "content": ex["sql"]})
    return msgs


def run_sql(context, query):
    """Build an in-memory SQLite DB from the example's context, run the query, return sorted rows (or None on error)."""
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(context)
        rows = con.execute(query).fetchall()
        return sorted(map(repr, rows))   # order-insensitive comparison, type-stable
    except Exception:
        return None
    finally:
        con.close()


def executable(ex):
    """Keep test examples whose gold SQL runs in SQLite and returns something (so the test is meaningful)."""
    rows = run_sql(ex["sql_context"], ex["sql"])
    return bool(rows)


def clean_sql(text):
    text = re.sub(r"^```(?:sql)?|```$", "", text.strip(), flags=re.I | re.M).strip()
    return text.split(";")[0].strip() + ";"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, tok, test, label):
    model.eval()
    tok.padding_side = "left"             # decoder-only models must be left-padded for batched generation
    correct = valid = exact = 0
    samples = []
    start = time.time()
    for i in range(0, len(test), 16):
        batch = test.select(range(i, min(i + 16, len(test))))
        prompts = [tok.apply_chat_template(messages(ex, False), add_generation_prompt=True, tokenize=False) for ex in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
        for ex, g in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
            pred = clean_sql(tok.decode(g, skip_special_tokens=True))
            got, gold = run_sql(ex["sql_context"], pred), run_sql(ex["sql_context"], ex["sql"])
            valid += got is not None
            correct += got is not None and got == gold
            exact += " ".join(pred.lower().split()).rstrip(";") == " ".join(ex["sql"].lower().split()).rstrip(";")
            if len(samples) < 12:
                samples.append({"question": ex["sql_prompt"], "gold": ex["sql"], "pred": pred, "correct": got == gold})
    n = len(test)
    res = {"execution_accuracy": correct / n, "valid_sql_rate": valid / n, "exact_match": exact / n,
           "eval_minutes": (time.time() - start) / 60}
    print(label, res, flush=True)
    return res, samples


class TimeBudget(TrainerCallback):
    def __init__(self, hours):
        self.deadline = time.time() + hours * 3600

    def on_step_end(self, args, state, control, **kw):
        if time.time() > self.deadline:
            print(f"[time budget] stopping at step {state.global_step}")
            control.should_training_stop = True
        return control


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = load_dataset("gretelai/synthetic_text_to_sql")
    train = raw["train"].shuffle(seed=42).select(range(N_TRAIN))
    test = raw["test"].shuffle(seed=42).select(range(N_TEST * 3)).filter(executable)
    test = test.select(range(min(N_TEST, len(test))))
    print(f"train {len(train)}, executable test {len(test)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 4-bit NF4 quantization with double quantization; matmuls computed in bf16 (fp16 on older GPUs).
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=DTYPE)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, dtype=DTYPE, device_map={"": 0})
    print(f"GPU memory after loading 4-bit model: {torch.cuda.memory_allocated() / 2**30:.1f} GB")

    # ---- Baseline: zero-shot base model ----
    base_res, base_samples = evaluate(model, tok, test, "base (zero-shot)")

    # ---- QLoRA ----
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    tok.padding_side = "right"

    def encode(ex):
        """Loss only on the SQL answer: prompt tokens get label -100."""
        prompt = tok.apply_chat_template(messages(ex, False), add_generation_prompt=True, tokenize=False)
        full = tok.apply_chat_template(messages(ex, True), tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"][:MAX_LEN]
        return {"input_ids": f_ids, "labels": [-100] * len(p_ids) + f_ids[len(p_ids):]}

    train_tok = train.map(encode, remove_columns=train.column_names)

    def collate(batch):
        w = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (w - len(seq))
        return {"input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
                "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
                "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch])}

    args = TrainingArguments(
        output_dir=OUT, num_train_epochs=1, per_device_train_batch_size=8, gradient_accumulation_steps=2,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=0.05, max_grad_norm=0.3,
        optim="paged_adamw_8bit", bf16=BF16, fp16=not BF16, logging_steps=10, save_strategy="no",
        remove_unused_columns=False, report_to="none", seed=42)
    trainer = Trainer(model=model, args=args, train_dataset=train_tok, data_collator=collate,
                      callbacks=[TimeBudget(TIME_BUDGET_HOURS)])
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    result = trainer.train()
    train_min = (time.time() - t) / 60
    peak_gb = torch.cuda.max_memory_allocated() / 2**30

    model.config.use_cache = True
    tuned_res, tuned_samples = evaluate(model, tok, test, "QLoRA fine-tuned")
    model.save_pretrained(os.path.join(OUT, "adapter"))   # only the LoRA weights (~150 MB), not the 7B base

    summary = {"model": MODEL, "train_examples": len(train), "test_examples": len(test),
               "trainable_params": trainable, "total_params": total, "trainable_percent": 100 * trainable / total,
               "train_minutes": train_min, "peak_train_gpu_memory_gb": peak_gb,
               "final_train_loss": result.training_loss, "gpu": torch.cuda.get_device_name(0),
               "base": base_res, "qlora": tuned_res}
    json.dump(summary, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    json.dump({"base": base_samples, "qlora": tuned_samples}, open(os.path.join(OUT, "samples.json"), "w"), indent=2)
    json.dump(trainer.state.log_history, open(os.path.join(OUT, "log_history.json"), "w"))
    print(json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
