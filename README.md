# LLM Engineering Projects

Hands-on projects on the core skills of building production LLM systems: **parameter-efficient fine-tuning, retrieval for RAG, multimodal models and speech** — each trained on cloud GPUs and evaluated with task-appropriate, measured metrics against a baseline.

| # | Project | Techniques | Result |
|---|---|---|---|
| 01 | [RAG retrieval, measured](01_rag_retriever_finetuning) | BM25, bge embeddings + FAISS, contrastive fine-tuning, cross-encoder reranking, nDCG/Recall/MRR | dense **0.403 nDCG@10 vs 0.217 BM25**; fine-tuning +2.4 pts Recall@100; reranker *hurt* — measured, not assumed |
| 02 | [QLoRA text-to-SQL](02_qlora_text_to_sql) | Qwen2.5-7B, 4-bit NF4 QLoRA, execution-based evaluation | execution accuracy **60.0 % → 66.2 %** in 17 min on one GPU, 0.53 % of parameters trained |
| 03 | [Vision-language model for chart QA](03_vlm_chartqa_lora) | Qwen2-VL-2B, LoRA, ChartQA relaxed accuracy, error analysis | **76.2 % → 77.5 %** (within noise — strong zero-shot baseline; failures analysed) |

*In progress (results will be added as runs finish): Whisper fine-tuning for Marathi speech recognition (FLEURS) · GRPO reinforcement learning for math reasoning (GSM8K) · function-calling agent fine-tuning.*

## Principles
- **Baseline first:** every fine-tuned model is compared with the same model zero-shot (or a classical method) on the same held-out data and prompt.
- **Metrics that match the task:** SQL is *executed* and result sets compared; retrieval uses nDCG/Recall/MRR; charts use ChartQA relaxed accuracy; speech uses WER/CER.
- **Show failures:** each README lists what still goes wrong and what to try next.
- **Cost-aware:** hard time budgets per run; LoRA/QLoRA so a single GPU is enough.

## Infrastructure
Training ran on Google Colab A100/L4/T4 runtimes driven from a terminal with the [Colab CLI](https://github.com/googlecolab/google-colab-cli). [`ops/`](ops) contains the job-queue helpers and a workaround for a CLI bug that silently lost runtimes after one hour ([issue #106](https://github.com/googlecolab/google-colab-cli/issues/106)): the runtime-proxy token is renewed from the assignments API every 5 minutes.

Related: [deep-learning-projects](https://github.com/kishorhange111/deep-learning-projects) — computer vision and NLP projects (U-Net, image captioning, Siamese networks, BERT).
