# 06 · Fine-tuning a Small LLM for Function Calling (the Core Skill of AI Agents)

Agent frameworks (LangChain / LangGraph / CrewAI tools, OpenAI-style function calling, MCP tool use) depend on one capability: given tool schemas and a user request, **decide whether a tool is needed and emit an exactly-correct call**. A wrong tool name or a malformed argument breaks the agent.

## Approach
| | |
|---|---|
| Model | **Qwen2.5-1.5B-Instruct** — small enough to run cheaply/locally, which is where fine-tuning pays off |
| Data | glaive-function-calling-v2 → 14,989 training conversations (first user turn + first assistant action) · **600 held-out** test cases, 192 of which require a tool call |
| Target format | `<functioncall>{"name": …, "arguments": {…}}</functioncall>` — or a plain-text answer when no tool fits |
| Method | LoRA r = 16 on all attention + MLP projections (18.5 M trainable params), bf16, loss on the assistant response only, batch 4 × grad-accum 4 |
| Hardware | 1 × NVIDIA L4 · 48 min |

## Results (600 held-out cases, same prompt before/after)
| Metric | Zero-shot | **LoRA fine-tuned** |
|---|---|---|
| Call-decision accuracy (call a tool exactly when needed) | 71.8 % | **96.0 %** |
| Tool-name accuracy (cases needing a call) | 16.7 % | **100 %** |
| **Full-call exact match** (right tool **and** arguments, compared as JSON) | 16.7 % | **94.8 %** |
| Valid JSON when a call is attempted | 62.3 % | **100 %** |

**From "unusable as an agent" to 95 % exactly-correct tool calls in 48 minutes on one GPU.**

## Discussion points
- **Why the zero-shot model fails:** it knows *what* to do but not the required protocol — it wraps calls in prose, invents keys, or emits invalid JSON. Fine-tuning mostly teaches the **contract** (format, when to abstain), which is exactly what agents need to be reliable.
- **Abstention matters:** 96 % call-decision accuracy includes correctly *not* calling a tool for chit-chat — false tool calls are as harmful as missed ones.
- **Caveat:** test and training data come from the same generator (glaive). Real-world robustness needs out-of-distribution evaluation, e.g. the Berkeley Function-Calling Leaderboard, multi-turn/parallel calls and tool errors.
- **Production pattern:** small fine-tuned model for routing / tool calls + constrained decoding (JSON schema grammar) for guaranteed validity + a larger model only for open-ended answers.

## Run
```bash
pip install torch transformers datasets peft
python agents/function-calling-llm/train.py        # ~65 min on an L4 incl. evaluations
```
