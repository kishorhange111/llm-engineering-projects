"""
Fine-tuning a vision-language model (Qwen2-VL-2B) with LoRA for chart question answering
========================================================================================

TASK: look at a chart image and answer a question about it
      ("What was the revenue of company B in 2019?" -> "23.4").
Chart and document understanding is one of the most common enterprise uses
of multimodal LLMs (reports, dashboards, invoices, slides).

MODEL: Qwen2-VL-2B-Instruct
    * A Vision Transformer encodes the image at (close to) native resolution
      into visual tokens (2D rotary position embeddings, dynamic number of
      tokens per image - here capped at ~512).
    * The visual tokens are inserted into the prompt of a 1.5B language model,
      which answers in text. One Transformer reads image and question together.

FINE-TUNING: LoRA (rank 16) on the language model's attention and MLP
projections; the vision encoder stays frozen. bf16 on one A100. The loss is
computed only on the answer tokens.

METRIC: "relaxed accuracy", the official ChartQA metric - a numeric answer is
correct within 5% of the gold value, text answers must match exactly
(case-insensitive). Evaluated on held-out ChartQA *test* questions, before and
after fine-tuning, with the same prompt.
"""
import json
import os
import time

import torch

QUICK = os.environ.get("QUICK") == "1"
MODEL = os.environ.get("MODEL", "Qwen/Qwen2-VL-2B-Instruct")
OUT = os.environ.get("OUT", "/content/vlm_chartqa")
N_TRAIN = 200 if QUICK else int(os.environ.get("N_TRAIN", "8000"))
N_TEST = 40 if QUICK else int(os.environ.get("N_TEST", "600"))
MAX_PIXELS = 512 * 28 * 28          # <= ~512 visual tokens per chart
TIME_BUDGET_HOURS = float(os.environ.get("TIME_BUDGET_HOURS", "1.0"))
INSTRUCTION = "Answer the question with a single word or number."


def messages(ex, with_answer):
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": f"{ex['query']}\n{INSTRUCTION}"}]}]
    if with_answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": str(ex["label"][0])}]})
    return msgs


def relaxed_match(pred, gold, tol=0.05):
    """ChartQA relaxed accuracy: numbers within 5 %, otherwise case-insensitive exact match."""
    clean = lambda s: str(s).strip().strip(".").replace(",", "").replace("%", "").replace("$", "").lower()
    p, g = clean(pred), clean(gold)
    try:
        pf, gf = float(p), float(g)
        return abs(pf - gf) <= tol * abs(gf) if gf != 0 else pf == gf
    except ValueError:
        return p == g


@torch.no_grad()
def evaluate(model, processor, test, label):
    model.eval()
    processor.tokenizer.padding_side = "left"
    correct, samples, t = 0, [], time.time()
    for i in range(0, len(test), 16):
        batch = test.select(range(i, min(i + 16, len(test))))
        texts = [processor.apply_chat_template(messages(ex, False), add_generation_prompt=True, tokenize=False)
                 for ex in batch]
        enc = processor(text=texts, images=[[ex["image"]] for ex in batch], padding=True,
                        return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=24, do_sample=False)
        preds = processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for ex, p in zip(batch, preds):
            ok = relaxed_match(p, ex["label"][0])
            correct += ok
            if len(samples) < 12:
                samples.append({"question": ex["query"], "gold": ex["label"][0], "pred": p.strip(), "correct": bool(ok)})
    res = {"relaxed_accuracy": correct / len(test), "eval_minutes": (time.time() - t) / 60}
    print(label, res, flush=True)
    return res, samples


def main():
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration, Trainer, TrainerCallback,
                              TrainingArguments)

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(42)
    ds = load_dataset("HuggingFaceM4/ChartQA")
    train = ds["train"].shuffle(seed=42).select(range(N_TRAIN))
    test = ds["test"].shuffle(seed=42).select(range(N_TEST))
    print(f"train {len(train)}, test {len(test)}")

    processor = AutoProcessor.from_pretrained(MODEL, min_pixels=128 * 28 * 28, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa").to("cuda")

    base_res, base_samples = evaluate(model, processor, test, "zero-shot")

    # LoRA on the language model only: these module names exist in the LLM, not in the vision tower.
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    tok = processor.tokenizer
    assistant_start = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]

    def collate(examples):
        tok.padding_side = "right"
        texts = [processor.apply_chat_template(messages(ex, True), tokenize=False) for ex in examples]
        batch = processor(text=texts, images=[[ex["image"]] for ex in examples], padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        n = len(assistant_start)
        for row, ids in enumerate(batch["input_ids"].tolist()):
            # Mask everything up to and including the last "<|im_start|>assistant\n": loss only on the answer.
            starts = [j for j in range(len(ids) - n + 1) if ids[j:j + n] == assistant_start]
            labels[row, :starts[-1] + n] = -100
        batch["labels"] = labels
        return batch

    class TimeBudget(TrainerCallback):
        def __init__(self):
            self.deadline = time.time() + TIME_BUDGET_HOURS * 3600

        def on_step_end(self, args, state, control, **kw):
            if time.time() > self.deadline:
                control.should_training_stop = True
            return control

    args = TrainingArguments(
        output_dir=OUT, num_train_epochs=1, per_device_train_batch_size=8, gradient_accumulation_steps=2,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=0.05, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, save_strategy="no", remove_unused_columns=False, dataloader_num_workers=2,
        report_to="none", seed=42)
    trainer = Trainer(model=model, args=args, train_dataset=train, data_collator=collate, callbacks=[TimeBudget()])
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    result = trainer.train()
    train_min = (time.time() - t) / 60

    tuned_res, tuned_samples = evaluate(model, processor, test, "LoRA fine-tuned")
    model.save_pretrained(os.path.join(OUT, "adapter"))
    summary = {"model": MODEL, "train_examples": len(train), "test_examples": len(test),
               "trainable_params": trainable, "total_params": total, "trainable_percent": 100 * trainable / total,
               "train_minutes": train_min, "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
               "final_train_loss": result.training_loss, "gpu": torch.cuda.get_device_name(0),
               "zero_shot": base_res, "fine_tuned": tuned_res}
    json.dump(summary, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    json.dump({"zero_shot": base_samples, "fine_tuned": tuned_samples}, open(os.path.join(OUT, "samples.json"), "w"),
              indent=2)
    json.dump(trainer.state.log_history, open(os.path.join(OUT, "log_history.json"), "w"))
    print(json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
