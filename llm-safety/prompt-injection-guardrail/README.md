# Prompt-Injection & Jailbreak Guardrail Classifier — and Why In-Distribution Accuracy Lies

Prompt injection ("ignore all previous instructions and…", role-play jailbreaks, hidden instructions in retrieved documents) is **#1 in the OWASP Top 10 for LLM applications**. A common defence is a small, fast classifier that screens every user input and every retrieved chunk *before* it reaches the LLM. This project builds one — and evaluates it the way an attacker would: on attacks it has never seen.

## Setup
| | |
|---|---|
| Training data | `xTRam1/safe-guard-prompt-injection` — 8,236 prompts (benign vs. injection) |
| Test sets | **in-distribution** test split (2,060) · **out-of-distribution**: `deepset/prompt-injections` (662) and `jackhhao/jailbreak-classification` (1,306) — different authors, attack styles and benign prompts |
| Models | **1.** TF-IDF + logistic regression (baseline) · **2.** **ModernBERT-base** fine-tuned (149 M, 3 epochs, fp32, 6.9 min on an L4) · **3.** `protectai/deberta-v3-base-prompt-injection-v2`, an open-source guardrail, zero-shot |
| Metrics | precision, recall, F1, ROC-AUC, **false-positive rate on benign prompts**, GPU latency (batch 1) |

## Results
| Model | In-distribution F1 | deepset (OOD) recall / F1 | jailbreak set (OOD) F1 / **false positives** |
|---|---|---|---|
| TF-IDF + LR | 0.986 | 8.4 % / 0.15 | 0.955 / 1.9 % |
| **ModernBERT fine-tuned (ours)** | **0.997** | 13.7 % / 0.24 | 0.906 / **20.5 %** |
| ProtectAI guardrail (zero-shot) | 0.909 | **41.4 % / 0.58** | 0.892 / **0.6 %** |

Latency (batch 1, L4 GPU): ModernBERT **20.5 ms**, ProtectAI DeBERTa-v3 26.9 ms.

## What the numbers say
1. **Near-perfect in-distribution, weak out-of-distribution.** Our classifier scores F1 0.997 on its own test split — and catches only 14 % of the deepset injections while flagging **1 in 5 harmless prompts** in the jailbreak set. It learned the training set's *style*, not the concept of an attack.
2. **Data diversity beats model choice.** The ProtectAI model, trained on a broad mix of sources, has the lowest in-distribution score but by far the best generalisation (3× our recall on deepset, 30× fewer false positives on the jailbreak set).
3. **A TF-IDF baseline looks excellent in-distribution (0.986)** — a warning sign that the test split shares surface patterns with training.

**Takeaway:** for security classifiers, a held-out split from the same dataset is not an evaluation. Test against attacks from *other* sources, track the false-positive rate on real benign traffic, and treat a guardrail as one layer of defence (plus least-privilege tools, output filtering, human confirmation for risky actions) — never the only one.

## Next steps
- Train on a union of diverse sources (+ synthetic paraphrases of attacks), evaluate leave-one-source-out.
- Calibrate the threshold per deployment for a target false-positive rate.
- Add indirect-injection examples (instructions hidden in web pages / RAG documents).

## Engineering notes
DeBERTa-v3 was the first choice: trained in bf16 it collapsed to predicting one class (AUC 0.50), and in fp32 under transformers v5 its gradients became NaN after ~100 steps (loss stuck at 0). Diagnosing it from the training logs rather than publishing the numbers led to switching to ModernBERT (Answer.AI, 2024), which trained cleanly.

## Run
```bash
pip install torch transformers datasets scikit-learn
python llm-safety/prompt-injection-guardrail/train.py      # ~15 min on an L4
```
