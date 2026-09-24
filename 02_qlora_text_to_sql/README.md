# 02 · QLoRA Fine-tuning of a 7B LLM for Text-to-SQL — Scored by Executing the SQL

Turn a natural-language question plus a table schema into a SQLite query — the core of "chat with your database" copilots and LLM database tools (e.g. MCP servers over enterprise data).

## Approach
| | |
|---|---|
| Base model | **Qwen2.5-7B-Instruct** (7.6 B parameters) |
| Method | **QLoRA** — base weights frozen in 4-bit **NF4** with double quantization; LoRA adapters (r = 16, α = 32) on all attention + MLP projections of every layer |
| Trainable | **40.4 M parameters = 0.53 %** of the model |
| Data | `gretelai/synthetic_text_to_sql` — 12,000 training examples across 100 domains |
| Prompt | system instruction + `CREATE TABLE` schema + question → one SQLite query |
| Training | 1 epoch, LR 2e-4 cosine, paged 8-bit AdamW, bf16 compute, gradient checkpointing, loss on SQL tokens only |
| Hardware | 1 × NVIDIA A100 40 GB · **17 min** · peak **14.7 GB** GPU memory (4-bit model alone: 5.2 GB) |

## Evaluation: execution accuracy
Each test example ships with `CREATE TABLE` + `INSERT` statements. The database is built in SQLite, **both the gold query and the model's query are executed, and their result sets are compared** — the same principle as the Spider / BIRD benchmarks. Two different-looking queries returning the same rows are both correct; a plausible-looking query returning wrong rows is wrong.
Test set: 500 held-out examples whose gold SQL runs in SQLite and returns rows.

## Results (500 test questions, same prompt for both)
| Model | Execution accuracy | Exact-match SQL | Valid SQL |
|---|---|---|---|
| Qwen2.5-7B-Instruct, zero-shot | 60.0 % | 26.0 % | 94.2 % |
| **+ QLoRA (17 min on one GPU)** | **66.2 %** | **40.0 %** | 94.0 % |

**+6.2 points execution accuracy** (a 15.5 % relative reduction in wrong answers) from 17 minutes of single-GPU training.

### What changed — real test examples
| Question | Zero-shot 7B | QLoRA | |
|---|---|---|---|
| total quantity of size 8 and 9 women's shoes sold in the UK | `... product = 'women's shoes' AND country = 'United Kingdom'` ✗ (guessed values, broken quote) | `SELECT SUM(quantity) FROM sales_2 WHERE size IN (8, 9) AND country = 'UK'` ✓ | learned data conventions |
| unique donors per cause area | 3-way join with a sub-query ✗ | `SELECT d.cause_area, COUNT(DISTINCT d.id) ... GROUP BY d.cause_area` ✓ | simpler, correct |
| percentage of mobile vs broadband subscribers per region | window-function formula ✗ | join on the wrong key ✗ | multi-join aggregation is still hard |

More in [`results/samples.json`](results/samples.json); raw metrics in [`results/metrics.json`](results/metrics.json).

## Discussion points
- **Why QLoRA?** Full fine-tuning a 7B model needs well over 100 GB of GPU memory (fp32 master weights + gradients + two Adam states ≈ 16 bytes/parameter, before activations); QLoRA did it in 14.7 GB. NF4 is information-theoretically optimal for normally distributed weights; double quantization also compresses the quantization constants.
- **Why execution accuracy?** Exact string match punishes harmless differences (aliases, column order) — here exact match understates quality (40 %) versus execution (66 %).
- **Honest caveat:** part of the gain is the model learning this dataset's *conventions* (value spellings like `'UK'`, terse `SELECT` style). On a real enterprise database the same effect is desirable — but it also means evaluation must use held-out data from the target schema.
- **Remaining failures:** multi-table joins with nested aggregation, and questions whose filter values are not visible in the schema. Next steps: include sample rows / column value hints in the prompt (schema linking), self-consistency (sample several queries, pick the majority result), execution-guided retry on SQL errors, and evaluating on Spider/BIRD.

## Run
```bash
pip install torch transformers datasets peft bitsandbytes accelerate
python 02_qlora_text_to_sql/train.py                                   # 7B, ~25 min incl. evaluation on an A100
MODEL=Qwen/Qwen2.5-3B-Instruct python 02_qlora_text_to_sql/train.py    # smaller GPUs
QUICK=1 python 02_qlora_text_to_sql/train.py                           # smoke test
```
