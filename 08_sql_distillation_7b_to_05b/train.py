"""
Distilling a 7B LLM's text-to-SQL skill into a 0.5B model (sequence-level knowledge distillation)
==============================================================================================

QUESTION: Can a model 15x smaller learn a task from a big model's OUTPUTS alone -
no human labels - and how close does it get? This is how many production
teams cut LLM serving cost: use a large model as a teacher once, deploy a
small student forever.

SETUP (same data and execution-based metric as project 02):
    teacher : Qwen2.5-7B-Instruct, zero-shot, bf16 - writes SQL for 12k training questions
    student : Qwen2.5-0.5B-Instruct, full fine-tuning
    arms    : (a) student zero-shot
              (b) student trained on the TEACHER's SQL        <- distillation, no human labels
              (c) student trained on the teacher's SQL, keeping only queries that EXECUTE
                  without error (cheap automatic filtering - still no gold labels)
              (d) student trained on the human GOLD SQL        <- upper bound
METRICS: execution accuracy on 500 held-out questions (SQL is run in SQLite and
the result sets compared), plus generation throughput of teacher vs student.

Sequence-level KD (Kim & Rush, 2016): the student imitates the teacher's final
outputs rather than its per-token probabilities - which also works with a
teacher you can only call through an API.
"""
import json
import os
import re
import sqlite3
import time

import torch

QUICK = os.environ.get("QUICK") == "1"
TEACHER = "Qwen/Qwen2.5-7B-Instruct"
STUDENT = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = os.environ.get("OUT", "/content/sql_distill")
N_TRAIN = 300 if QUICK else int(os.environ.get("N_TRAIN", "6000"))   # teacher labelling is the expensive step
N_TEST = 40 if QUICK else 500
SYSTEM = "You are an expert SQL assistant. Given a database schema and a question, reply with one SQLite query only."


def schema_only(ctx):
    return "\n".join(s.strip() + ";" for s in ctx.split(";") if s.strip().upper().startswith("CREATE"))


def messages(q, ctx, answer=None):
    m = [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"Schema:\n{schema_only(ctx)}\n\nQuestion: {q}"}]
    if answer is not None:
        m.append({"role": "assistant", "content": answer})
    return m


def run_sql(ctx, query):
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(ctx)
        return sorted(map(repr, con.execute(query).fetchall()))
    except Exception:
        return None
    finally:
        con.close()


def clean_sql(text):
    text = re.sub(r"^```(?:sql)?|```$", "", text.strip(), flags=re.I | re.M).strip()
    return text.split(";")[0].strip() + ";"


@torch.no_grad()
def generate(model, tok, rows, bs=48, max_new=200):
    tok.padding_side = "left"
    outs, t, n_tok = [], time.time(), 0
    for i in range(0, len(rows), bs):
        b = rows[i:i + bs]
        texts = [tok.apply_chat_template(messages(q, c), add_generation_prompt=True, tokenize=False) for q, c in b]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        new = g[:, enc["input_ids"].shape[1]:]
        n_tok += int((new != tok.pad_token_id).sum())
        outs += [clean_sql(tok.decode(x, skip_special_tokens=True)) for x in new]
    secs = time.time() - t
    return outs, {"seconds": secs, "tokens_per_second": n_tok / secs}


def exec_accuracy(preds, rows):
    return sum(run_sql(c, p) is not None and run_sql(c, p) == run_sql(c, gold) for p, (q, c, gold) in zip(preds, rows)) / len(rows)


def finetune(name, pairs):
    """Full fine-tuning of the 0.5B student on (question, context, sql) triples; loss on the SQL only."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(STUDENT)
    model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16).to("cuda")

    def enc(ex):
        p = tok.apply_chat_template(messages(ex["q"], ex["c"]), add_generation_prompt=True, tokenize=False)
        f = tok.apply_chat_template(messages(ex["q"], ex["c"], ex["a"]), tokenize=False)
        pi, fi = tok(p, add_special_tokens=False)["input_ids"], tok(f, add_special_tokens=False)["input_ids"][:768]
        return {"input_ids": fi, "labels": ([-100] * len(pi) + fi[len(pi):])[:len(fi)]}

    ds = Dataset.from_dict({"q": [x[0] for x in pairs], "c": [x[1] for x in pairs], "a": [x[2] for x in pairs]})
    ds = ds.map(enc, remove_columns=["q", "c", "a"])

    def collate(b):
        w = max(len(x["input_ids"]) for x in b)
        pad = lambda s, v: s + [v] * (w - len(s))
        return {"input_ids": torch.tensor([pad(x["input_ids"], tok.pad_token_id) for x in b]),
                "labels": torch.tensor([pad(x["labels"], -100) for x in b]),
                "attention_mask": torch.tensor([pad([1] * len(x["input_ids"]), 0) for x in b])}

    tok.padding_side = "right"
    args = TrainingArguments(output_dir=f"{OUT}/{name}", num_train_epochs=1, per_device_train_batch_size=16,
                             learning_rate=2e-5, lr_scheduler_type="cosine", warmup_steps=0.05, bf16=True,
                             logging_steps=50, save_strategy="no", remove_unused_columns=False, report_to="none", seed=42)
    t = time.time()
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collate).train()
    return model, tok, (time.time() - t) / 60


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(OUT, exist_ok=True)
    raw = load_dataset("gretelai/synthetic_text_to_sql")
    train = raw["train"].shuffle(seed=42).select(range(N_TRAIN))
    test = raw["test"].shuffle(seed=42).select(range(N_TEST * 3))
    test_rows = [(e["sql_prompt"], e["sql_context"], e["sql"]) for e in test if run_sql(e["sql_context"], e["sql"])][:N_TEST]
    train_rows = [(e["sql_prompt"], e["sql_context"], e["sql"]) for e in train]
    res = {}

    # ---- teacher labels the training set (and is evaluated on the test set) ----
    ttok = AutoTokenizer.from_pretrained(TEACHER)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16).to("cuda")
    t_test, t_speed = generate(teacher, ttok, [(q, c) for q, c, _ in test_rows])
    res["teacher_7b_zero_shot"] = {"execution_accuracy": exec_accuracy(t_test, test_rows), **t_speed}
    t = time.time()
    t_train, _ = generate(teacher, ttok, [(q, c) for q, c, _ in train_rows], bs=64)
    label_min = (time.time() - t) / 60
    del teacher
    torch.cuda.empty_cache()
    executes = [run_sql(c, a) is not None for (q, c, _), a in zip(train_rows, t_train)]
    print(f"teacher labelled {len(t_train)} questions in {label_min:.1f} min; {sum(executes)} of them execute")

    # ---- student arms ----
    stok = AutoTokenizer.from_pretrained(STUDENT)
    student = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16).to("cuda")
    s_test, s_speed = generate(student, stok, [(q, c) for q, c, _ in test_rows])
    res["student_0.5b_zero_shot"] = {"execution_accuracy": exec_accuracy(s_test, test_rows), **s_speed}
    del student

    arms = {
        "student_distilled_teacher_sql": [(q, c, a) for (q, c, _), a in zip(train_rows, t_train)],
        "student_distilled_executable_only": [(q, c, a) for (q, c, _), a, ok in zip(train_rows, t_train, executes) if ok],
        "student_trained_on_gold_sql": train_rows,
    }
    for name, pairs in arms.items():
        model, tok, minutes = finetune(name, pairs)
        preds, speed = generate(model, tok, [(q, c) for q, c, _ in test_rows])
        res[name] = {"execution_accuracy": exec_accuracy(preds, test_rows), "train_examples": len(pairs),
                     "train_minutes": minutes, **speed}
        print(name, res[name], flush=True)
        del model
        torch.cuda.empty_cache()

    summary = {"teacher": TEACHER, "student": STUDENT, "test_questions": len(test_rows),
               "teacher_labelling_minutes": label_min, "teacher_labels_executable": sum(executes) / len(executes),
               "gpu": torch.cuda.get_device_name(0), "results": res}
    json.dump(summary, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
