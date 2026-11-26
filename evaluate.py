import time
import numpy as np
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from dataset_builder import load_scifact_splits
from train import encode_texts


def calc_metrics(retrieved_ids, relevant_ids, k=10):
    top_k = retrieved_ids[:k]
    rel_set = set(relevant_ids)

    hits = len(set(top_k) & rel_set)
    recall = hits / len(rel_set) if rel_set else 0.0

    rr = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in rel_set:
            rr = 1.0 / rank
            break

    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in rel_set:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return recall, rr, ndcg


def batch_encode(texts, tokenizer, model, device, batch_size=64, max_length=256, desc="Encoding"):
    all_embeddings = []
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            emb = encode_texts(model, encoded)
            all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings).astype("float32")


def evaluate_model(model_path, corpus, test_queries, test_qrels, k=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)

    doc_ids = list(corpus.keys())
    doc_texts = [corpus[d] for d in doc_ids]

    doc_embeddings = batch_encode(
        doc_texts, tokenizer, model, device, desc=f"Corpus ({model_path})"
    )

    _, dim = doc_embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)

    q_ids = list(test_queries.keys())
    q_texts = [
        f"Represent this sentence for searching relevant passages: {test_queries[q]}"
        for q in q_ids
    ]

    t_start = time.time()
    q_embeddings = batch_encode(
        q_texts, tokenizer, model, device, desc=f"Queries ({model_path})"
    )
    _, indices = index.search(q_embeddings, k)
    latency_ms = ((time.time() - t_start) / len(q_ids)) * 1000

    recalls, mrrs, ndcgs = [], [], []
    for i, q_id in enumerate(q_ids):
        retrieved_ids = [doc_ids[idx] for idx in indices[i]]
        r, rr, n = calc_metrics(retrieved_ids, test_qrels[q_id], k=k)
        recalls.append(r)
        mrrs.append(rr)
        ndcgs.append(n)

    return {
        "recall": float(np.mean(recalls)),
        "mrr": float(np.mean(mrrs)),
        "ndcg": float(np.mean(ndcgs)),
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    corpus, _, _, test_queries, test_qrels = load_scifact_splits()

    base_model = "BAAI/bge-small-en-v1.5"
    tuned_model = "output/best-bge-scifact"

    base_res = evaluate_model(base_model, corpus, test_queries, test_qrels, k=10)

    tuned_res = evaluate_model(tuned_model, corpus, test_queries, test_qrels, k=10)

    print("Zero-Shot vs. Fine-Tuned")
    header = f"{'Metric':<15} | {'Zero-Shot BGE':<15} | {'Fine-Tuned BGE':<15} | {'Delta':<15}"
    print(header)

    for key, label in [("recall", "Recall@10"), ("mrr", "MRR@10"), ("ndcg", "NDCG@10")]:
        b_val = base_res[key]
        t_val = tuned_res[key]
        diff = t_val - b_val
        sign = "+" if diff >= 0 else ""
        print(f"{label:<15} | {b_val:<15.4f} | {t_val:<15.4f} | {sign}{diff:.4f}")

    print(
        f"{'Latency (ms/q)':<15} | {base_res['latency_ms']:<15.2f} | "
        f"{tuned_res['latency_ms']:<15.2f} | {tuned_res['latency_ms'] - base_res['latency_ms']:+.2f} ms"
    )
