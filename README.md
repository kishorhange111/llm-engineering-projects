# Generative AI Engineering

Hands-on projects covering the core skills of building production LLM systems — **fine-tuning (LoRA / QLoRA / distillation), reinforcement learning, RAG, agents, multimodal models and LLM safety**. Every project is trained on cloud GPUs and evaluated with a task-appropriate metric against a measured baseline, with failures analysed rather than hidden.

## Fine-tuning
| Project | Techniques | Result |
|---|---|---|
| [QLoRA text-to-SQL (7B)](fine-tuning/qlora-text-to-sql) | Qwen2.5-7B, 4-bit NF4 QLoRA, SQL **executed** in SQLite for scoring | execution accuracy **60.0 % → 66.2 %** · 17 min on one GPU · 0.53 % of params trained |
| [Distilling 7B text-to-SQL into 0.5B](fine-tuning/llm-distillation-text-to-sql) | sequence-level knowledge distillation, execution-filtered teacher labels | student **43.4 % → 52.2 %** with **no human labels** — on par with gold-label training (52.4 %) |

## Reinforcement learning
| Project | Techniques | Result |
|---|---|---|
| [GRPO for math reasoning](reinforcement-learning/grpo-math-reasoning) | DeepSeek-R1-style RL with verifiable rewards, TRL `GRPOTrainer`, GSM8K | format compliance **55 % → 98 %**; accuracy unchanged (67.6 %) — analysed |

## Agents
| Project | Techniques | Result |
|---|---|---|
| [Function-calling LLM](agents/function-calling-llm) | Qwen2.5-1.5B + LoRA, tool-call exact match, JSON validity, abstention | exact tool calls **16.7 % → 94.8 %**, valid JSON **100 %** |

## Retrieval-augmented generation
| Project | Techniques | Result |
|---|---|---|
| [RAG retrieval, measured](rag/retrieval-evaluation-and-finetuning) | BM25 vs. bge embeddings + FAISS vs. contrastive fine-tuning vs. cross-encoder reranking; nDCG / Recall / MRR | dense **0.403 vs 0.217** nDCG@10 (BM25); reranker *hurt* — measured, not assumed |

## Multimodal
| Project | Techniques | Result |
|---|---|---|
| [Vision-language model for chart QA](multimodal/vision-language-chart-qa) | Qwen2-VL-2B + LoRA, ChartQA relaxed accuracy, error analysis | 76.2 % → 77.5 % (within noise — strong zero-shot baseline) |
| [Whisper for Marathi speech recognition](multimodal/whisper-marathi-speech-recognition) | Whisper-small fine-tuning on 11.9 h of FLEURS Marathi | **CER 53.6 % → 14.7 %**, WER 121 % → 45 % |

## LLM safety
| Project | Techniques | Result |
|---|---|---|
| [Prompt-injection guardrail](llm-safety/prompt-injection-guardrail) | ModernBERT classifier vs. TF-IDF vs. open-source guardrail, **out-of-distribution** evaluation | F1 0.997 in-distribution but 14 % recall on unseen attacks — why guardrails need OOD testing |

## Principles
- **Baseline first:** every fine-tuned model is compared with the same model zero-shot (or a classical method) on the same held-out data and prompt.
- **Metrics that match the task:** SQL is executed and result sets compared; retrieval uses nDCG/Recall/MRR; charts use ChartQA relaxed accuracy; speech uses WER/CER; security uses out-of-distribution recall and false-positive rate.
- **Show failures:** each README lists what still goes wrong and what to try next.
- **Cost-aware:** hard time budgets per run; LoRA/QLoRA so a single GPU is enough.

## Layout
Each project folder contains a self-contained, heavily commented `train.py`, a `README.md` (approach, results, discussion) and `results/` with the raw metrics and samples. Trained on Google Colab GPUs (NVIDIA A100 / L4) with PyTorch and Hugging Face `transformers`, `peft`, `trl`, `datasets`.

Related: [**mini-chatgpt**](https://github.com/kishorhange111/mini-chatgpt) — base LLM → SFT → DPO → benchmarks · [**deep-learning-portfolio**](https://github.com/kishorhange111/deep-learning-portfolio) — computer vision, NLP and generative models.
