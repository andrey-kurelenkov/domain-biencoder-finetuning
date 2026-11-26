# Domain Adaptation of Bi-Encoder on SciFact (Pure PyTorch)

Fine-tuning pipeline for adapting a lightweight dense retriever (`BAAI/bge-small-en-v1.5`, 33M parameters) to the biomedical **BEIR SciFact** domain using **FAISS Hard Negative Mining** and a custom **PyTorch** contrastive training loop (`MultipleNegativesRankingLoss`).

## Motivation

Building upon our previous retrieval benchmark ([`hybrid-search-reranker`](https://github.com/andrey-kurelenkov/hybrid-search-reranker)), adding a Cross-Encoder improved `Recall@10` to `0.8414` but suffered from domain shift and high latency (~2,611 ms/query). This project tests domain adaptation of a fast Bi-Encoder on SciFact triplets to outperform both zero-shot retrieval and heavy cross-encoders while keeping latency under 20 ms.

## Architecture & Training Pipeline

```text
SciFact Train Split (809 queries, 5,183 abstracts) ──► FAISS Hard Negative Mining ──► Training Triplets ──► Pure PyTorch Training Loop (3 Epochs, AdamW, MNRL) ──► Evaluation on SciFact Test Split (300 queries)
```

## Training Dynamics & Benchmark Results

* **Training (3 Epochs, T4 GPU):** Train loss drops from `1.0086` (Epoch 1) to `0.4799` (Epoch 3), with validation loss improving from `0.8527` to `0.6834`.
* **SciFact Test Set Performance:** Fine-tuning improves `Recall@10` from `0.8162` to `0.8877` (+7.14%), `MRR@10` from `0.6745` to `0.7471` (+7.27%), and `NDCG@10` from `0.7034` to `0.7803` (+7.69%), while maintaining an ultra-fast query latency of **18.51 ms**.

## Key Engineering Takeaways

* **Superior Domain Adaptation:** Surpasses hybrid RRF and cross-encoder setups while operating ~140x faster.
* **Hard Negatives:** Forced the model to capture biomedical semantics rather than surface-level keyword overlap.
* **Zero Train-Serve Skew:** Unified `CLS-pooling` and L2 normalization across training and evaluation scripts.

## Repository Structure & Usage

```text
├── dataset_builder.py   # Strict train/test split & FAISS hard negative mining
├── train.py             # Custom PyTorch Dataset & contrastive training loop
├── evaluate.py          # Held-out test set evaluation & latency benchmarks
└── requirements.txt     # Pinned project dependencies
```

```bash
git clone https://github.com/andrey-kurelenkov/domain-biencoder-finetuning
cd domain-biencoder-finetuning
pip install -r requirements.txt
python dataset_builder.py
python train.py
python evaluate.py
```
