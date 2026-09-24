# 08 · Distilling a 7B LLM's Text-to-SQL Skill into a 0.5B Model — Without Human Labels

Production teams cut LLM cost by using a large model **once** as a teacher and deploying a small student. How close does a 15× smaller model get when it learns only from the teacher's outputs — and can we clean those outputs automatically?

## Setup (same data and execution-based metric as [project 02](../../fine-tuning/qlora-text-to-sql))
| | |
|---|---|
| Teacher | **Qwen2.5-7B-Instruct**, zero-shot — writes SQL for 6,000 training questions (11 min on an A100) |
| Student | **Qwen2.5-0.5B-Instruct**, full fine-tuning, 1 epoch |
| Arms | **(a)** student zero-shot · **(b)** trained on **all** teacher SQL · **(c)** trained only on teacher SQL that **executes without error** (automatic filter, still no gold labels) · **(d)** trained on human **gold** SQL (upper bound) |
| Metric | **execution accuracy** on 500 held-out questions — SQL is run in SQLite and result sets compared |

Sequence-level knowledge distillation (Kim & Rush, 2016): the student imitates the teacher's final outputs — which also works with a teacher you can only call through an API.

## Results (500 held-out questions)
| Model | Training labels | Execution accuracy |
|---|---|---|
| Teacher 7B, zero-shot | — | 59.4 % |
| Student 0.5B, zero-shot | — | 43.4 % |
| Student 0.5B | all teacher SQL (6,000) | 50.4 % |
| **Student 0.5B** | **teacher SQL that executes (5,189)** | **52.2 %** |
| Student 0.5B | human gold SQL (6,000) | 52.4 % |

- **Distillation closes 55 % of the student→teacher gap** (43.4 → 52.2 vs. 59.4) with **zero human labels**.
- **A one-line quality filter matters:** dropping the 13.5 % of teacher queries that fail to execute lifts the student from 50.4 → 52.2 % — **statistically indistinguishable from training on human gold labels (52.4 %)**.
- The 0.5B student needs ~1 GB of weights vs. ~15 GB for the teacher — it fits on a CPU or a small GPU.

## Discussion points
- **Why filtered teacher data ≈ gold data:** the student is capacity-limited, not label-limited; teacher outputs that run are "good enough" labels, and the teacher's style is consistent (easier to imitate than heterogeneous human SQL).
- **Beyond this:** execution-*correctness* filtering (compare against a reference answer or a self-consistency vote), sampling several teacher outputs per question, and token-level KD with teacher logits (needs white-box access).
- **Throughput note:** with plain Hugging Face `generate()` the student generated only ~1.3× faster than the teacher (448 vs. 353 tok/s) — small models are bottlenecked by Python/launch overhead, not FLOPs. A serving engine with continuous batching (e.g. vLLM) is needed to turn the 15× smaller model into a proportional throughput and cost win.

## Run
```bash
pip install torch transformers datasets
python fine-tuning/llm-distillation-text-to-sql/train.py     # ~45 min on an A100
```
