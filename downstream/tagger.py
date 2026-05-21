"""
downstream/tagger.py
====================
Bilingual biLSTM POS tagger for zero-shot cross-lingual transfer.

Architecture
------------
- Character embedding layer (CNN over characters → character-level token repr)
- Word embedding layer (randomly initialised, trained from scratch)
- Bidirectional LSTM
- Linear output → UPOS tag distribution

This mirrors the architecture used by Lin et al. (2019) and Rice et al. (2025)
for source-language selection studies, making our results directly comparable.

No pretrained embeddings are used.  This is deliberate: the absence of
multilingual pretraining means transfer performance is purely a function of
cross-lingual structural overlap, making the signal from typological embeddings
maximally visible.

Usage
-----
    from downstream.tagger import BiLSTMTagger, train_tagger, evaluate_tagger
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

PAD_TOK = "<PAD>"
UNK_TOK = "<UNK>"
PAD_CHAR = "\x00"
UNK_CHAR = "\x01"


class Vocab:
    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.id2token: List[str] = []
        self._frozen = False

    def add(self, token: str) -> int:
        if token not in self.token2id:
            if self._frozen:
                return self.token2id.get(UNK_TOK, 1)
            self.token2id[token] = len(self.id2token)
            self.id2token.append(token)
        return self.token2id[token]

    def freeze(self):
        self._frozen = True

    def __len__(self):
        return len(self.id2token)

    def encode(self, token: str) -> int:
        return self.token2id.get(token, self.token2id.get(UNK_TOK, 1))


def build_vocabs(
    word_seqs: List[List[str]],
    tag_seqs: List[List[str]],
    min_word_freq: int = 2,
) -> Tuple[Vocab, Vocab, Vocab]:
    """
    Build word, character, and tag vocabularies from training data.
    """
    word_counts = Counter(w for sent in word_seqs for w in sent)
    char_set = set(c for sent in word_seqs for w in sent for c in w)

    word_vocab = Vocab()
    word_vocab.add(PAD_TOK)
    word_vocab.add(UNK_TOK)
    for w, cnt in word_counts.most_common():
        if cnt >= min_word_freq:
            word_vocab.add(w)
    word_vocab.freeze()

    char_vocab = Vocab()
    char_vocab.add(PAD_CHAR)
    char_vocab.add(UNK_CHAR)
    for c in sorted(char_set):
        char_vocab.add(c)
    char_vocab.freeze()

    tag_vocab = Vocab()
    tag_vocab.add(PAD_TOK)
    for t in sorted(set(t for sent in tag_seqs for t in sent)):
        tag_vocab.add(t)
    tag_vocab.freeze()

    return word_vocab, char_vocab, tag_vocab


# ---------------------------------------------------------------------------
# PyTorch model  (imported lazily so the module loads without torch)
# ---------------------------------------------------------------------------

def _build_model(
    word_vocab_size: int,
    char_vocab_size: int,
    tag_vocab_size: int,
    word_dim: int = 100,
    char_dim: int = 30,
    char_cnn_filters: int = 50,
    char_cnn_kernel: int = 3,
    lstm_hidden: int = 200,
    lstm_layers: int = 2,
    dropout: float = 0.33,
):
    """
    Build a biLSTM POS tagger using PyTorch.

    Architecture following Plank et al. (2016) / Straka & Straková (2017)
    and the setup used in Lin et al. (2019) for cross-lingual transfer.
    """
    import torch
    import torch.nn as nn

    class CharCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
            self.conv = nn.Conv1d(char_dim, char_cnn_filters,
                                  kernel_size=char_cnn_kernel, padding=1)
            self.drop = nn.Dropout(dropout)

        def forward(self, char_ids):
            # char_ids: (batch, seq_len, max_word_len)
            B, S, W = char_ids.shape
            x = self.embed(char_ids.view(B * S, W))    # (B*S, W, char_dim)
            x = x.transpose(1, 2)                       # (B*S, char_dim, W)
            x = torch.relu(self.conv(x))                # (B*S, filters, W)
            x = x.max(dim=2).values                     # (B*S, filters)
            x = self.drop(x)
            return x.view(B, S, char_cnn_filters)

    class BiLSTMTaggerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embed = nn.Embedding(word_vocab_size, word_dim, padding_idx=0)
            self.char_cnn = CharCNN()
            self.drop = nn.Dropout(dropout)
            input_dim = word_dim + char_cnn_filters
            self.lstm = nn.LSTM(
                input_dim, lstm_hidden // 2,
                num_layers=lstm_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            self.out = nn.Linear(lstm_hidden, tag_vocab_size)

        def forward(self, word_ids, char_ids, lengths):
            # word_ids: (B, S),  char_ids: (B, S, W),  lengths: (B,)
            w = self.drop(self.word_embed(word_ids))    # (B, S, word_dim)
            c = self.char_cnn(char_ids)                 # (B, S, filters)
            x = torch.cat([w, c], dim=-1)               # (B, S, input_dim)
            x = self.drop(x)

            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            logits = self.out(self.drop(out))           # (B, S, n_tags)
            return logits

    return BiLSTMTaggerModel()


# ---------------------------------------------------------------------------
# Data encoding utilities
# ---------------------------------------------------------------------------

def encode_sentence(
    words: List[str],
    tags: List[str],
    word_vocab: Vocab,
    char_vocab: Vocab,
    tag_vocab: Vocab,
    max_word_len: int = 30,
) -> Tuple[List[int], List[List[int]], List[int]]:
    """
    Encode a single sentence into (word_ids, char_ids, tag_ids).
    """
    word_ids = [word_vocab.encode(w) for w in words]
    char_ids = [
        [char_vocab.encode(c) for c in w[:max_word_len]] +
        [0] * max(0, max_word_len - len(w))
        for w in words
    ]
    tag_ids = [tag_vocab.encode(t) for t in tags]
    return word_ids, char_ids, tag_ids


def collate_batch(
    batch: List[Tuple],
    device="cpu",
):
    """
    Collate a list of (word_ids, char_ids, tag_ids) into padded tensors.
    """
    import torch

    lengths = [len(w) for w, c, t in batch]
    max_len = max(lengths)
    max_wlen = max(len(c[0]) for _, c, _ in batch)

    word_ids_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    char_ids_pad = torch.zeros(len(batch), max_len, max_wlen, dtype=torch.long)
    tag_ids_pad  = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (w, c, t) in enumerate(batch):
        L = len(w)
        word_ids_pad[i, :L] = torch.tensor(w, dtype=torch.long)
        for j, cj in enumerate(c):
            cj = cj[:max_wlen]
            char_ids_pad[i, j, :len(cj)] = torch.tensor(cj, dtype=torch.long)
        tag_ids_pad[i, :L] = torch.tensor(t, dtype=torch.long)

    return (word_ids_pad.to(device),
            char_ids_pad.to(device),
            tag_ids_pad.to(device),
            torch.tensor(lengths, dtype=torch.long).to(device))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tagger(
    word_seqs_train: List[List[str]],
    tag_seqs_train: List[List[str]],
    word_vocab: Optional[Vocab] = None,
    char_vocab: Optional[Vocab] = None,
    tag_vocab: Optional[Vocab] = None,
    n_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 5,
    word_seqs_dev: Optional[List[List[str]]] = None,
    tag_seqs_dev: Optional[List[List[str]]] = None,
    verbose: bool = True,
) -> Tuple[object, Vocab, Vocab, Vocab]:
    """
    Train a biLSTM POS tagger from scratch.

    Parameters
    ----------
    word_seqs_train, tag_seqs_train  : training data
    word_vocab, char_vocab, tag_vocab : pass pre-built vocabs for cross-lingual
                                        eval (so tag set is fixed)
    patience    : early stopping patience on dev accuracy
    word_seqs_dev, tag_seqs_dev : optional dev data for early stopping

    Returns
    -------
    model, word_vocab, char_vocab, tag_vocab
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    if word_vocab is None or char_vocab is None or tag_vocab is None:
        word_vocab, char_vocab, tag_vocab = build_vocabs(
            word_seqs_train, tag_seqs_train)

    # Encode training data
    train_encoded = [
        encode_sentence(w, t, word_vocab, char_vocab, tag_vocab)
        for w, t in zip(word_seqs_train, tag_seqs_train)
    ]

    dev_encoded = None
    if word_seqs_dev and tag_seqs_dev:
        dev_encoded = [
            encode_sentence(w, t, word_vocab, char_vocab, tag_vocab)
            for w, t in zip(word_seqs_dev, tag_seqs_dev)
        ]

    model = _build_model(
        word_vocab_size=len(word_vocab),
        char_vocab_size=len(char_vocab),
        tag_vocab_size=len(tag_vocab),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_dev_acc = -1.0
    best_state = None
    patience_count = 0

    for epoch in range(n_epochs):
        model.train()
        indices = np.random.permutation(len(train_encoded))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch = [train_encoded[i] for i in batch_idx]
            word_ids, char_ids, tag_ids, lengths = collate_batch(batch, device)

            logits = model(word_ids, char_ids, lengths)
            B, S, T = logits.shape
            loss = criterion(logits.view(B * S, T), tag_ids.view(B * S))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Dev evaluation
        if dev_encoded:
            dev_acc = _eval_accuracy(model, dev_encoded, batch_size, device)
            scheduler.step(1.0 - dev_acc)
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            if verbose:
                print(f"    Epoch {epoch+1:3d}  loss={avg_loss:.4f}  "
                      f"dev_acc={dev_acc:.4f}  patience={patience_count}/{patience}")
            if patience_count >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break
        else:
            if verbose and (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1:3d}  loss={avg_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, word_vocab, char_vocab, tag_vocab


def _eval_accuracy(model, encoded_data, batch_size=32, device="cpu") -> float:
    import torch
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for start in range(0, len(encoded_data), batch_size):
            batch = encoded_data[start:start + batch_size]
            word_ids, char_ids, tag_ids, lengths = collate_batch(batch, device)
            logits = model(word_ids, char_ids, lengths)
            preds = logits.argmax(-1)
            mask = tag_ids != -100
            correct += (preds[mask] == tag_ids[mask]).sum().item()
            total += mask.sum().item()
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_tagger(
    model,
    word_seqs_test: List[List[str]],
    tag_seqs_test: List[List[str]],
    word_vocab: Vocab,
    char_vocab: Vocab,
    tag_vocab: Vocab,
    device: str = "cpu",
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Evaluate a trained tagger on test data (possibly from a different language).

    Returns
    -------
    metrics : {"token_accuracy": float, "sentence_accuracy": float}
    """
    import torch

    encoded = [
        encode_sentence(w, t, word_vocab, char_vocab, tag_vocab)
        for w, t in zip(word_seqs_test, tag_seqs_test)
    ]

    model.eval()
    tok_correct = tok_total = 0
    sent_correct = sent_total = 0

    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start:start + batch_size]
            word_ids, char_ids, tag_ids, lengths = collate_batch(batch, device)
            logits = model(word_ids, char_ids, lengths)
            preds = logits.argmax(-1)
            mask = tag_ids != -100

            tok_correct += (preds[mask] == tag_ids[mask]).sum().item()
            tok_total   += mask.sum().item()

            for i in range(len(batch)):
                L = int(lengths[i])
                sent_pred = preds[i, :L]
                sent_true = tag_ids[i, :L]
                m = sent_true != -100
                if m.all():
                    sent_correct += (sent_pred == sent_true).all().item()
                sent_total += 1

    return {
        "token_accuracy": tok_correct / max(tok_total, 1),
        "sentence_accuracy": sent_correct / max(sent_total, 1),
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_tagger(model, word_vocab, char_vocab, tag_vocab, out_dir: str):
    import torch
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    for name, vocab in [("word_vocab", word_vocab),
                         ("char_vocab", char_vocab),
                         ("tag_vocab",  tag_vocab)]:
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump({"token2id": vocab.token2id,
                       "id2token": vocab.id2token}, f)


def load_tagger(out_dir: str, device: str = "cpu"):
    import torch

    vocabs = []
    for name in ["word_vocab", "char_vocab", "tag_vocab"]:
        with open(os.path.join(out_dir, f"{name}.json")) as f:
            d = json.load(f)
        v = Vocab()
        v.token2id = d["token2id"]
        v.id2token = d["id2token"]
        v.freeze()
        vocabs.append(v)
    word_vocab, char_vocab, tag_vocab = vocabs

    model = _build_model(
        word_vocab_size=len(word_vocab),
        char_vocab_size=len(char_vocab),
        tag_vocab_size=len(tag_vocab),
    )
    state = torch.load(os.path.join(out_dir, "model.pt"),
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    return model, word_vocab, char_vocab, tag_vocab
