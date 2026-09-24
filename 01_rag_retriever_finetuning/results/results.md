FiQA-2018 test set: 648 questions over 57,638 passages

| retriever | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.217 | 0.278 | 0.474 | 0.270 |
| bge-small (off-the-shelf) | 0.403 | 0.464 | 0.696 | 0.488 |
| bge-small fine-tuned on FiQA | 0.393 | 0.456 | 0.720 | 0.472 |
| fine-tuned + bge-reranker (top-50) | 0.355 | 0.436 | 0.720 | 0.425 |
