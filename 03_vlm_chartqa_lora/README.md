# 03 · LoRA Fine-tuning a Vision-Language Model for Chart Question Answering (Qwen2-VL-2B, ChartQA)

Look at a chart, answer a question about it — the core of multimodal copilots over reports, dashboards and slides.

## Approach
| | |
|---|---|
| Model | **Qwen2-VL-2B-Instruct** (2.2 B params): Vision Transformer at near-native resolution → visual tokens inserted into the prompt of a 1.5 B language model |
| Resolution | 128–512 × 28² pixels per chart → at most ~512 visual tokens |
| Method | **LoRA** (r = 16, α = 32) on all attention + MLP projections of the language model; vision encoder frozen |
| Trainable | **18.5 M parameters = 0.83 %** |
| Data | ChartQA — 8,000 training questions; **600 held-out test questions** |
| Prompt | image + question + "Answer the question with a single word or number." |
| Training | 1 epoch, LR 1e-4 cosine, bf16, gradient checkpointing, loss on answer tokens only |
| Hardware | 1 × A100 40 GB · **24.7 min** · peak 14.2 GB |

**Metric — relaxed accuracy** (the official ChartQA metric): numeric answers count as correct within 5 % of the gold value; text answers must match exactly (case-insensitive).

## Results (600 test questions, identical prompt)
| Model | Relaxed accuracy |
|---|---|
| Qwen2-VL-2B-Instruct, zero-shot | 76.2 % |
| + LoRA fine-tuning on 8k ChartQA questions | **77.5 %** |

**Honest reading:** +1.3 points, but with 600 test questions the 95 % confidence interval is about ±3.4 points (±1.7 standard error), so the gain is **not statistically significant**. The zero-shot model is already strong because Qwen2-VL's pre-training included chart and document data. The interesting results are therefore *what* fails:

### Where it still fails — real test examples
| Question | Gold | Zero-shot | Fine-tuned |
|---|---|---|---|
| Which party is least likely to approve NSA surveillance? | Independent | Independent ✓ | Independent ✓ |
| Pinterest revenue in Q4 2020? | 706 | 706 ✓ | 706 ✓ |
| International mDAU users, last quarter of 2021? | 162 | 159 ✓ (within 5 %) | 158.5 ✓ |
| Difference between the highest and lowest unemployment rate? | 10.53 | 7.18 ✗ | 7.59 ✗ |
| Difference in value between Papua New Guinea and Luxembourg? | 0.08 | 0.04 ✗ | 0.04 ✗ |
| Which province had the highest relative incidence of coronavirus? | Bolzano | Lombardy ✗ | Lombardy ✗ |

Reading a single value off a chart is essentially solved at this size; **multi-step arithmetic over several chart values** is where both models fail — and a 2B model answering in one token has no room to "work it out". The last example shows a plausible *world-knowledge* answer overriding what the chart shows.

More in [`results/samples.json`](results/samples.json); metrics in [`results/metrics.json`](results/metrics.json).

## Discussion points
- **Baseline first:** without the zero-shot measurement, "77.5 % after fine-tuning" would look like a big win. Measuring first shows the adaptation gain is marginal — the decision would be *not* to ship a fine-tune here without a better approach.
- **What would move the number:** let the model reason before answering (chain-of-thought / program-of-thought: extract the values, then compute in Python), train on rationales rather than bare answers, higher image resolution for dense charts, or a larger VLM (7B) — and evaluate human-written vs. machine-generated questions separately (the human split is much harder).
- **Why LoRA on the language model only:** the vision encoder already produces good chart features; adapting the LLM changes how it *uses* them, at 0.83 % of the parameters and 14 GB of memory.
- **Statistical care:** with n = 600 and p ≈ 0.77, SE = √(p(1−p)/n) ≈ 1.7 pts. Small deltas need bigger test sets or paired significance tests (McNemar).

## Run
```bash
pip install torch transformers datasets peft qwen-vl-utils
python 03_vlm_chartqa_lora/train.py        # ~30 min incl. evaluation on an A100
QUICK=1 python 03_vlm_chartqa_lora/train.py
```
