from datasets import load_dataset
from collections import defaultdict
import os
import numpy as np
import faiss
from datasets import Dataset
from sentence_transformers import SentenceTransformer

def load_scifact_splits():
    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")
    train_qrels_ds = load_dataset("BeIR/scifact-qrels", split="train")
    test_qrels_ds = load_dataset("BeIR/scifact-qrels", split="test")
    corpus = {}
    queries = {}
    train_qrels = defaultdict(list)
    test_qrels = defaultdict(list)
    for row in corpus_ds:
        corpus[str(row["_id"])] = f"{row['title']}. {row['text']}".strip()
    for row in queries_ds:
        queries[str(row["_id"])] = row["text"]
    for row in train_qrels_ds:
        if row["score"] > 0:
            q_id = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            train_qrels[q_id].append(doc_id)
    for row in test_qrels_ds:
        if row["score"] > 0:
            q_id = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            test_qrels[q_id].append(doc_id)
    train_queries = {q_id: queries[q_id] for q_id in train_qrels if q_id in queries}
    test_queries = {q_id: queries[q_id] for q_id in test_qrels if q_id in queries}

    assert len(set(train_queries.keys()) & set(test_queries.keys())) == 0, "Data leakage!"   

    return corpus, train_queries, dict(train_qrels), test_queries, dict(test_qrels)

def mine_hard_negatives(corpus, train_queries, train_qrels, model_name="BAAI/bge-small-en-v1.5", top_k=15):
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[doc_id] for doc_id in doc_ids]

    model = SentenceTransformer(model_name)

    if os.path.exists("doc_embeddings.npy"):
        d_embeddings = np.load("doc_embeddings.npy")
    else:
        d_embeddings = model.encode(
            doc_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        d_embeddings = np.array(d_embeddings, dtype="float32")
        np.save("doc_embeddings.npy", d_embeddings)
    size = d_embeddings.shape[1]
    index = faiss.IndexFlatIP(size)
    index.add(d_embeddings)

    q_ids = list(train_queries.keys())
    q_texts = [f"Represent this sentence for searching relevant passages: {train_queries[q]}" for q in q_ids]
    q_embeddings = model.encode(
        q_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   
    )
    q_embeddings = np.array(q_embeddings, dtype="float32")
    scores, indices = index.search(q_embeddings, top_k)
    triplets = []   
    for i, q_id in enumerate(q_ids):
        pos_ids = set(train_qrels[q_id])

        for idx in indices[i]:
            cand_id = doc_ids[idx]
            if cand_id not in pos_ids:
                hard_neg_id = cand_id
                break
        for pos_id in pos_ids:
            triplets.append({"anchor": q_texts[i], "positive": corpus[pos_id], "negative": corpus[hard_neg_id]})

    return Dataset.from_list(triplets)
    
if __name__ == "__main__":
    corpus, train_queries, train_qrels, test_queries, test_qrels = load_scifact_splits()
    train_dataset = mine_hard_negatives(corpus, train_queries, train_qrels)

    first_sample = next(iter(train_dataset))
    print(f"Total training triplets: {len(train_dataset)}")
    #print("Sample anchor:  ", first_sample["anchor"])
    #print("Sample positive:", first_sample["positive"][:180], "...")
    #print("Sample negative:", first_sample["negative"][:180], "...")
    #print(f"Total training triplets: {len(train_dataset)}")
