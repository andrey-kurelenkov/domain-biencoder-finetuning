import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from datasets import load_from_disk
from tqdm import tqdm

class TripletDataset(Dataset):
    def __init__ (self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, tokenizer, max_length=256):
    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives = [item["negative"] for item in batch]

    anchor_enc = tokenizer(
        anchors, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    pos_enc = tokenizer(
        positives, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    neg_enc = tokenizer(
        negatives, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return anchor_enc, pos_enc, neg_enc

def encode_texts(model, encoded_inputs):
    outputs = model(**encoded_inputs)
    cls_emb = outputs.last_hidden_state[:, 0]
    return F.normalize(cls_emb, p=2, dim=1)

def multiple_negatives_ranking_loss(anchor_emb, pos_emb, neg_emb, scale=20.0):
    candidates_emb = torch.cat([pos_emb,neg_emb], dim=0)
    scores = torch.matmul(anchor_emb,candidates_emb.T) * scale
    labels = torch.arange(anchor_emb.size(0), device=anchor_emb.device)

    return F.cross_entropy(scores, labels)

from dataset_builder import load_scifact_splits, mine_hard_negatives


def train_biencoder(
    model_name="BAAI/bge-small-en-v1.5",
    save_dir="output/best-bge-scifact",
    epochs=3,
    batch_size=16,
    lr=2e-5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    if os.path.exists("scifact_train_triplets"):
        raw_ds = load_from_disk("scifact_train_triplets")
    else:
        corpus, train_queries, train_qrels, _, _ = load_scifact_splits()
        raw_ds = mine_hard_negatives(corpus, train_queries, train_qrels, model_name=model_name)
        raw_ds.save_to_disk("scifact_train_triplets")

    splits = raw_ds.train_test_split(test_size=0.1, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    train_loader = DataLoader(
        TripletDataset(splits["train"]),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    val_loader = DataLoader(
        TripletDataset(splits["test"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for anchor_enc, pos_enc, neg_enc in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            anchor_enc = {k: v.to(device) for k, v in anchor_enc.items()}
            pos_enc = {k: v.to(device) for k, v in pos_enc.items()}
            neg_enc = {k: v.to(device) for k, v in neg_enc.items()}

            optimizer.zero_grad()

            anchor_emb = encode_texts(model, anchor_enc)
            pos_emb = encode_texts(model, pos_enc)
            neg_emb = encode_texts(model, neg_enc)

            loss = multiple_negatives_ranking_loss(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for anchor_enc, pos_enc, neg_enc in val_loader:
                anchor_enc = {k: v.to(device) for k, v in anchor_enc.items()}
                pos_enc = {k: v.to(device) for k, v in pos_enc.items()}
                neg_enc = {k: v.to(device) for k, v in neg_enc.items()}

                anchor_emb = encode_texts(model, anchor_enc)
                pos_emb = encode_texts(model, pos_enc)
                neg_emb = encode_texts(model, neg_enc)

                val_loss = multiple_negatives_ranking_loss(anchor_emb, pos_emb, neg_emb)
                val_losses.append(val_loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"New best model {save_dir} (Val Loss: {best_val_loss:.4f})")


if __name__ == "__main__":
    train_biencoder()


