"""
RAG retrieval, measured: BM25 vs. dense embeddings vs. fine-tuned embeddings vs. + cross-encoder reranking
=========================================================================================================

WHY: In Retrieval-Augmented Generation the LLM can only be as good as the
passages it is given. Most RAG failures are *retrieval* failures, so the
retriever deserves its own evaluation - before any LLM is involved.

TASK: financial question answering (BEIR / FiQA-2018). For each question,
find the relevant answers among 57,638 forum posts. Ground-truth relevance
labels ("qrels") make retrieval quality measurable.

PIPELINE STAGES COMPARED (same 648 test questions for every stage):
    1. BM25            - classic keyword search (what Elasticsearch does by default).
    2. Dense retrieval - BAAI/bge-small-en-v1.5 (33M params) embeds questions and
                         passages into 384-d vectors; nearest neighbours by cosine
                         similarity in a FAISS index. Matches meaning, not just words.
    3. Fine-tuned      - the same embedding model adapted to the finance domain on the
                         FiQA *training* questions with Multiple-Negatives-Ranking loss:
                         for a batch of (question, answer) pairs, every other answer in
                         the batch is a negative -> a contrastive, InfoNCE-style objective.
    4. + Reranker      - a cross-encoder (BAAI/bge-reranker-base) re-scores the top-50
                         candidates by reading question and passage *together*. Too slow
                         for 57k passages, very accurate for 50: the standard
                         "retrieve fast, then rerank" production design.

METRICS (standard IR metrics, computed from scratch below):
    nDCG@10   - ranking quality of the top 10, rewards relevant hits near the top
    Recall@10 / Recall@100 - share of relevant passages found in the top k
                  (Recall@k bounds what the LLM can ever see in a RAG prompt)
    MRR@10    - how high the FIRST relevant passage appears
"""
import json
import os
import time

import numpy as np

QUICK = os.environ.get("QUICK") == "1"
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
EMBEDDER = "BAAI/bge-small-en-v1.5"
RERANKER = "BAAI/bge-reranker-base"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # bge's query instruction
EPOCHS = 1 if QUICK else 3
BATCH = 64
RERANK_TOP = 50


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_fiqa():
    from datasets import load_dataset

    corpus = load_dataset("mteb/fiqa", "corpus", split="corpus")
    queries = load_dataset("mteb/fiqa", "queries", split="queries")
    qrels = load_dataset("mteb/fiqa", "default")
    doc_ids = [str(i) for i in corpus["_id"]]
    docs = [(t + " " + x).strip() if t else x for t, x in zip(corpus["title"], corpus["text"])]
    qtext = {str(i): t for i, t in zip(queries["_id"], queries["text"])}

    def rel(split):
        r = {}
        for q, d, s in zip(qrels[split]["query-id"], qrels[split]["corpus-id"], qrels[split]["score"]):
            if s > 0:
                r.setdefault(str(q), set()).add(str(d))
        return r

    return doc_ids, docs, qtext, rel("train"), rel("test")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(ranked, relevant):
    """ranked: {qid: [doc ids, best first]}, relevant: {qid: set(doc ids)}."""
    ndcg, r10, r100, mrr = [], [], [], []
    for q, rel in relevant.items():
        docs = ranked[q]
        gains = [1.0 if d in rel else 0.0 for d in docs[:10]]
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), 10)))
        ndcg.append(dcg / idcg)
        r10.append(len(rel & set(docs[:10])) / len(rel))
        r100.append(len(rel & set(docs[:100])) / len(rel))
        first = next((i for i, d in enumerate(docs[:10]) if d in rel), None)
        mrr.append(0.0 if first is None else 1 / (first + 1))
    return {"nDCG@10": float(np.mean(ndcg)), "Recall@10": float(np.mean(r10)),
            "Recall@100": float(np.mean(r100)), "MRR@10": float(np.mean(mrr))}


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------
def bm25_search(docs, doc_ids, qids, qtext, k=100):
    import re

    from rank_bm25 import BM25Okapi

    tokenize = lambda s: re.findall(r"\w+", s.lower())
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    out = {}
    for q in qids:
        scores = bm25.get_scores(tokenize(qtext[q]))
        top = np.argpartition(-scores, k)[:k]
        out[q] = [doc_ids[i] for i in top[np.argsort(-scores[top])]]
    return out


def dense_search(model, docs, doc_ids, qids, qtext, k=100):
    """Embed every passage once, build a FAISS inner-product index, search all test questions."""
    import faiss

    t = time.time()
    doc_emb = model.encode(docs, batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatIP(doc_emb.shape[1])   # exact search; normalized vectors -> inner product = cosine
    index.add(doc_emb)
    index_s = time.time() - t
    q_emb = model.encode([QUERY_PREFIX + qtext[q] for q in qids], batch_size=256, normalize_embeddings=True,
                         show_progress_bar=False).astype("float32")
    t = time.time()
    _, idx = index.search(q_emb, k)
    search_ms = 1000 * (time.time() - t) / len(qids)
    return {q: [doc_ids[i] for i in row] for q, row in zip(qids, idx)}, index_s, search_ms


def rerank(ranked, docs_by_id, qtext, top=RERANK_TOP):
    """Cross-encoder: score (question, passage) pairs jointly, re-order the top candidates."""
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(RERANKER, max_length=512)
    out, t = {}, time.time()
    for q, docs in ranked.items():
        cand = docs[:top]
        scores = ce.predict([(qtext[q], docs_by_id[d]) for d in cand], batch_size=64, show_progress_bar=False)
        out[q] = [cand[i] for i in np.argsort(-scores)] + docs[top:]
    return out, 1000 * (time.time() - t) / len(ranked)


def finetune(train_rel, qtext, docs_by_id):
    """Domain-adapt the embedding model with in-batch negatives (MultipleNegativesRankingLoss)."""
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    pairs = [(QUERY_PREFIX + qtext[q], docs_by_id[d]) for q, ds in train_rel.items() for d in ds if d in docs_by_id]
    if QUICK:
        pairs = pairs[:2000]
    print(f"training pairs: {len(pairs)}")
    ds = Dataset.from_dict({"anchor": [p[0] for p in pairs], "positive": [p[1] for p in pairs]})
    model = SentenceTransformer(EMBEDDER)
    model.max_seq_length = 256
    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.expanduser("~/checkpoints/bge_fiqa"), num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, learning_rate=2e-5, warmup_steps=0.1,
        fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,   # no duplicate passages -> no false in-batch negatives
        logging_steps=50, save_strategy="no", report_to="none", seed=42)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                                         loss=losses.MultipleNegativesRankingLoss(model))
    t = time.time()
    trainer.train()
    return model, (time.time() - t) / 60, len(pairs)


def main():
    from sentence_transformers import SentenceTransformer

    os.makedirs(OUT, exist_ok=True)
    doc_ids, docs, qtext, train_rel, test_rel = load_fiqa()
    if QUICK:
        test_rel = dict(list(test_rel.items())[:50])
        keep = {d for ds in test_rel.values() for d in ds} | set(doc_ids[:5000])
        sel = [i for i, d in enumerate(doc_ids) if d in keep]
        doc_ids, docs = [doc_ids[i] for i in sel], [docs[i] for i in sel]
    docs_by_id = dict(zip(doc_ids, docs))
    qids = list(test_rel)
    print(f"corpus {len(docs)} passages, {len(qids)} test questions, {len(train_rel)} train questions")

    results, notes = {}, {}
    results["BM25"] = evaluate(bm25_search(docs, doc_ids, qids, qtext), test_rel)
    print("BM25", results["BM25"])

    base = SentenceTransformer(EMBEDDER)
    ranked, index_s, ms = dense_search(base, docs, doc_ids, qids, qtext)
    results["bge-small (off-the-shelf)"] = evaluate(ranked, test_rel)
    notes["index_build_seconds"] = index_s
    print("dense", results["bge-small (off-the-shelf)"])

    tuned, train_min, n_pairs = finetune(train_rel, qtext, docs_by_id)
    ranked, _, ms = dense_search(tuned, docs, doc_ids, qids, qtext)
    results["bge-small fine-tuned on FiQA"] = evaluate(ranked, test_rel)
    notes.update(finetune_minutes=train_min, finetune_pairs=n_pairs, faiss_search_ms_per_query=ms)
    print("fine-tuned", results["bge-small fine-tuned on FiQA"])

    reranked, rerank_ms = rerank(ranked, docs_by_id, qtext)
    results["fine-tuned + bge-reranker (top-50)"] = evaluate(reranked, test_rel)
    notes["rerank_ms_per_query"] = rerank_ms
    print("reranked", results["fine-tuned + bge-reranker (top-50)"])

    json.dump({"results": results, "notes": notes}, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    with open(os.path.join(OUT, "results.md"), "w") as f:
        f.write(f"FiQA-2018 test set: {len(qids)} questions over {len(docs):,} passages\n\n")
        f.write("| retriever | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |\n|---|---|---|---|---|\n")
        for name, m in results.items():
            f.write(f"| {name} | {m['nDCG@10']:.3f} | {m['Recall@10']:.3f} | {m['Recall@100']:.3f} | {m['MRR@10']:.3f} |\n")
    plot(results)
    print(json.dumps(notes, indent=2))
    print("DONE")


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (metric, color) in enumerate([("nDCG@10", "#2a7ab9"), ("Recall@10", "#7fb3d5"), ("Recall@100", "#bbb")]):
        ax.bar(x + (i - 1) * 0.27, [results[n][metric] for n in names], 0.27, label=metric, color=color)
    ax.set_xticks(x, [n.replace(" (", "\n(").replace(" +", "\n+").replace(" fine", "\nfine") for n in names], fontsize=8)
    ax.legend()
    ax.set_title("FiQA-2018 retrieval quality by pipeline stage")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "retrieval_comparison.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
