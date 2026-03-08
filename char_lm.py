"""
Character-level LSTM Language Model with Language Embeddings
=============================================================
Implements the architecture from Östling & Tiedemann (2017),
as used in the semi-supervised extension of Bjerva et al. (2019).

Architecture:
    - 2-layer stacked character LSTM (1024-dim hidden states)
    - 128-dim character embeddings
    - 64-dim language embeddings, concatenated to char embeddings
      at every time step
    - Softmax output layer over the character vocabulary
    - Trained with Adam, early stopping

After training, the 64-dim language embeddings are extracted and
used as the pre-trained representations λ̃^ℓ in the semi-supervised
typological matrix factorisation model.
"""

import os
import json
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# 1. Dataset
# ============================================================================

class MultilingualCharDataset(Dataset):
    """
    Produces (lang_id, char_sequence) training samples by drawing
    fixed-length character windows from each language's text.

    Parameters
    ----------
    texts : dict mapping lang_id (str) → full text (str)
    char2idx : dict mapping character → index
    lang2idx : dict mapping lang_id → index
    seq_len : int – length of each character sequence
    samples_per_lang : int – how many windows to draw per language
    """

    def __init__(self, texts: Dict[str, str], char2idx: dict,
                 lang2idx: dict, seq_len: int = 200,
                 samples_per_lang: int = 1000):
        self.samples = []
        self.char2idx = char2idx
        self.unk_idx = char2idx.get("<UNK>", 0)

        for lang_id, text in texts.items():
            if lang_id not in lang2idx:
                continue
            lidx = lang2idx[lang_id]
            chars = [char2idx.get(c, self.unk_idx) for c in text]
            if len(chars) < seq_len + 1:
                continue

            # Draw random windows
            rng = np.random.default_rng(hash(lang_id) % 2**31)
            max_start = len(chars) - seq_len - 1
            starts = rng.integers(0, max_start, size=samples_per_lang)
            for s in starts:
                inp = chars[s : s + seq_len]
                tgt = chars[s + 1 : s + seq_len + 1]
                self.samples.append((lidx, inp, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lidx, inp, tgt = self.samples[idx]
        return (torch.tensor(lidx, dtype=torch.long),
                torch.tensor(inp, dtype=torch.long),
                torch.tensor(tgt, dtype=torch.long))


def build_vocab(texts: Dict[str, str],
                min_freq: int = 5,
                max_vocab: int = 500) -> dict:
    """Build character vocabulary from all texts."""
    counter = Counter()
    for text in texts.values():
        counter.update(text[:500_000])  # sample first 500k chars

    # Keep most common characters
    most_common = counter.most_common(max_vocab)
    char2idx = {"<PAD>": 0, "<UNK>": 1}
    for ch, freq in most_common:
        if freq >= min_freq:
            char2idx[ch] = len(char2idx)
    print(f"Character vocabulary size: {len(char2idx)}")
    return char2idx


# ============================================================================
# 2. Model
# ============================================================================

class CharLSTM_LM(nn.Module):
    """
    Character-level LSTM language model with language embeddings.

    At each time step, the input is:
        [char_embedding ; lang_embedding]
    which is fed into a 2-layer LSTM. The output is projected to
    a softmax over the character vocabulary.

    Parameters
    ----------
    n_chars : int – character vocabulary size
    n_langs : int – number of languages
    char_emb_dim : int – character embedding dimension (128)
    lang_emb_dim : int – language embedding dimension (64)
    hidden_dim : int – LSTM hidden dimension (1024)
    n_layers : int – number of LSTM layers (2)
    dropout : float
    """

    def __init__(self, n_chars: int, n_langs: int,
                 char_emb_dim: int = 128,
                 lang_emb_dim: int = 64,
                 hidden_dim: int = 1024,
                 n_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.char_emb = nn.Embedding(n_chars, char_emb_dim, padding_idx=0)
        self.lang_emb = nn.Embedding(n_langs, lang_emb_dim)

        self.lstm = nn.LSTM(
            input_size=char_emb_dim + lang_emb_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, n_chars)
        self.dropout = nn.Dropout(dropout)

    def forward(self, lang_idx, char_seq):
        """
        Parameters
        ----------
        lang_idx : (B,) – language index per sample
        char_seq : (B, T) – character indices

        Returns
        -------
        logits : (B, T, n_chars) – per-character log-probabilities
        """
        B, T = char_seq.shape

        char_e = self.char_emb(char_seq)            # (B, T, char_emb_dim)
        lang_e = self.lang_emb(lang_idx)             # (B, lang_emb_dim)
        lang_e = lang_e.unsqueeze(1).expand(-1, T, -1)  # (B, T, lang_emb_dim)

        x = torch.cat([char_e, lang_e], dim=-1)      # (B, T, char+lang)
        x = self.dropout(x)

        lstm_out, _ = self.lstm(x)                    # (B, T, hidden_dim)
        logits = self.output_proj(self.dropout(lstm_out))  # (B, T, n_chars)
        return logits

    def get_lang_embeddings(self) -> np.ndarray:
        """Extract language embeddings as a numpy array."""
        return self.lang_emb.weight.detach().cpu().numpy()


# ============================================================================
# 3. Training
# ============================================================================

def train_charlm(
    texts: Dict[str, str],
    output_dir: str = "charlm_output",
    char_emb_dim: int = 128,
    lang_emb_dim: int = 64,
    hidden_dim: int = 1024,
    n_layers: int = 2,
    seq_len: int = 200,
    samples_per_lang: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 5,
    device: str = "cpu",
) -> np.ndarray:
    """
    Train the character-level LM and return language embeddings.

    Parameters
    ----------
    texts : dict mapping lang_id → text string
    output_dir : str – where to save model checkpoint and embeddings
    (other params match the paper's hyperparameters)

    Returns
    -------
    lang_embeddings : np.ndarray of shape (n_langs, lang_emb_dim)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build vocabularies
    char2idx = build_vocab(texts)
    lang_ids = sorted(texts.keys())
    lang2idx = {lid: i for i, lid in enumerate(lang_ids)}

    n_chars = len(char2idx)
    n_langs = len(lang2idx)

    # Save mappings
    with open(os.path.join(output_dir, "char2idx.json"), "w") as f:
        json.dump(char2idx, f)
    with open(os.path.join(output_dir, "lang2idx.json"), "w") as f:
        json.dump(lang2idx, f)

    # Create dataset (80/20 split for early stopping)
    full_ds = MultilingualCharDataset(
        texts, char2idx, lang2idx, seq_len, samples_per_lang)

    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build model
    model = CharLSTM_LM(n_chars, n_langs, char_emb_dim, lang_emb_dim,
                         hidden_dim, n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"\nTraining char-LM: {n_langs} languages, "
          f"{n_chars} chars, {n_total} samples")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches = 0
        for lang_idx, inp, tgt in train_loader:
            lang_idx = lang_idx.to(device)
            inp = inp.to(device)
            tgt = tgt.to(device)

            logits = model(lang_idx, inp)
            loss = criterion(logits.reshape(-1, n_chars), tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train = train_loss / n_batches

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for lang_idx, inp, tgt in val_loader:
                lang_idx = lang_idx.to(device)
                inp = inp.to(device)
                tgt = tgt.to(device)
                logits = model(lang_idx, inp)
                loss = criterion(logits.reshape(-1, n_chars), tgt.reshape(-1))
                val_loss += loss.item()
                n_val_batches += 1

        avg_val = val_loss / max(n_val_batches, 1)
        print(f"  Epoch {epoch+1}/{max_epochs}  "
              f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "best_model.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model and extract embeddings
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "best_model.pt"),
                    map_location=device))
    lang_embeddings = model.get_lang_embeddings()

    # Save embeddings
    emb_path = os.path.join(output_dir, "lang_embeddings.npy")
    np.save(emb_path, lang_embeddings)

    idx2lang_path = os.path.join(output_dir, "idx2lang.json")
    with open(idx2lang_path, "w") as f:
        json.dump({str(i): lid for lid, i in lang2idx.items()}, f)

    print(f"\nSaved {lang_embeddings.shape} language embeddings to {emb_path}")
    return lang_embeddings


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train char-level LM on ParaBible data")
    parser.add_argument("--parabible_dir", type=str, required=True,
                        help="Directory with Bible text files")
    parser.add_argument("--output_dir", type=str, default="charlm_output")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--lang_emb_dim", type=int, default=64)
    parser.add_argument("--char_emb_dim", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--samples_per_lang", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Import loading function
    from data_preparation import load_parabible_texts
    texts = load_parabible_texts(args.parabible_dir)

    train_charlm(
        texts,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        lang_emb_dim=args.lang_emb_dim,
        char_emb_dim=args.char_emb_dim,
        seq_len=args.seq_len,
        samples_per_lang=args.samples_per_lang,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
    )
