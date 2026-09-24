# 05 · Reinforcement Learning for Reasoning: GRPO on GSM8K (DeepSeek-R1 style)

Can we improve an LLM's math reasoning **without showing it any worked solutions** — only rewarding correct final answers? This is *reinforcement learning with verifiable rewards* (RLVR), the recipe behind DeepSeek-R1-style reasoning models, run at single-GPU scale.

## GRPO in one paragraph
For every question the model samples a **group of 8 answers**. Each gets a rule-based reward: **+1.0** if the final answer is correct, **+0.5** if it follows `<think>…</think><answer>…</answer>`. The advantage of an answer is its reward minus the group mean, divided by the group std — answers better than their siblings are reinforced. Unlike PPO there is **no value/critic network** (the group is the baseline), and a KL penalty (β = 0.04) keeps the policy near the starting model.

## Setup
| | |
|---|---|
| Model | Qwen2.5-1.5B-Instruct |
| Data | GSM8K grade-school math; RL on training questions, evaluation on **500 held-out test questions** |
| Training | Hugging Face TRL `GRPOTrainer`, 307 steps × 4 questions × 8 samples (1,228 questions, ~9,800 sampled solutions), LR 2e-6, bf16, **72 min on one A100** |
| Evaluation | greedy decoding, same prompt before/after: answer accuracy + format compliance |

## Results (500 held-out questions)
| Metric | Before RL | After GRPO |
|---|---|---|
| **Format compliance** (`<think>` + `<answer>` structure) | 55.0 % | **97.6 %** |
| **Answer accuracy** | 67.6 % | 67.6 % |

**Honest result:** in 307 steps GRPO fully learned the *format* reward (55 → 98 %) but did **not** move accuracy. This is the typical first phase of RLVR: the cheap, dense signal (format) is learned first; correctness gains need many more steps, larger groups or harder questions. During training the correctness reward hovered around 0.69–0.78 per group, i.e. the model rarely produced *new* correct answers it could not already produce.

## Discussion points
- **Why accuracy didn't move:** a 1.5B instruct model already solves ~68 % of GSM8K; with 8 samples per question most groups are all-correct or all-wrong (zero variance → zero advantage → no gradient; TRL reported 10–30 % of groups with zero reward std). Signal only comes from questions the model *sometimes* gets right.
- **What would help:** more steps (DeepSeek-R1 used thousands), larger groups (16–64), curriculum toward questions with mixed success, a smaller format reward (it can dominate), and vLLM-accelerated sampling (HF `generate()` sampling dominated the ~12–15 s step time here).
- **What it demonstrates:** the full RLVR pipeline — verifiable reward functions, group-relative advantages, KL regularisation, and evaluation that separates *format* from *correctness* so a reward-hacking-looking improvement is not mistaken for better reasoning.

## Run
```bash
pip install torch transformers datasets trl
python 05_grpo_reasoning_gsm8k/train.py      # ~80 min on an A100 (time-budgeted)
```
