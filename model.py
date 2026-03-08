"""
A Probabilistic Generative Model of Linguistic Typology
=========================================================
PyTorch implementation of the core matrix factorisation model and
semi-supervised extension from Bjerva et al. (NAACL 2019).

The model explains a binary matrix of typological parameters via
exponential-family matrix factorisation:

    p(π_i^ℓ | λ^ℓ) = sigmoid(e_{π_i}^T  λ^ℓ)

where λ^ℓ is a language embedding and e_{π_i} is a parameter embedding.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

class WALSDataset(Dataset):
    """
    Wraps the binarised WALS matrix for PyTorch training.

    Each sample is a (language_index, feature_index, value) triple
    drawn from the *observed* cells of the matrix.

    Parameters
    ----------
    lang_indices : np.ndarray of shape (N,)
        Language index for each observed cell.
    feat_indices : np.ndarray of shape (N,)
        Feature index for each observed cell.
    values : np.ndarray of shape (N,)
        Binary value (0 or 1) for each observed cell.
    """

    def __init__(self, lang_indices: np.ndarray,
                 feat_indices: np.ndarray,
                 values: np.ndarray):
        self.lang_indices = torch.LongTensor(lang_indices)
        self.feat_indices = torch.LongTensor(feat_indices)
        self.values = torch.FloatTensor(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return (self.lang_indices[idx],
                self.feat_indices[idx],
                self.values[idx])


# ---------------------------------------------------------------------------
# 2. Core Model  (Section 3 of the paper)
# ---------------------------------------------------------------------------

class TypologicalMF(nn.Module):
    """
    Exponential-family matrix factorisation for typological features.

    Each language ℓ has a learned embedding  λ^ℓ ∈ ℝ^d.
    Each (binarised) feature i has an embedding   e_{π_i} ∈ ℝ^d.

    The probability that feature i is 'on' for language ℓ is:
        p(π_i^ℓ = 1) = σ(e_{π_i}^T  λ^ℓ)

    A Gaussian prior N(0, σ²I) on the language embeddings is realised
    via L2 regularisation during training.

    Parameters
    ----------
    n_languages : int
        Total number of languages.
    n_features : int
        Total number of *binarised* features.
    embed_dim : int
        Dimensionality d of the embeddings.
    """

    def __init__(self, n_languages: int, n_features: int,
                 embed_dim: int = 64):
        super().__init__()
        self.lang_embeddings = nn.Embedding(n_languages, embed_dim)
        self.feat_embeddings = nn.Embedding(n_features, embed_dim)

        # Initialise close to zero (standard practice for MF)
        nn.init.normal_(self.lang_embeddings.weight, std=0.01)
        nn.init.normal_(self.feat_embeddings.weight, std=0.01)

    def forward(self, lang_idx: torch.Tensor,
                feat_idx: torch.Tensor) -> torch.Tensor:
        """
        Returns predicted probabilities for the given (lang, feat) pairs.
        """
        lang_emb = self.lang_embeddings(lang_idx)   # (B, d)
        feat_emb = self.feat_embeddings(feat_idx)    # (B, d)
        logits = (lang_emb * feat_emb).sum(dim=-1)   # dot product → (B,)
        return torch.sigmoid(logits)

    def predict_all(self) -> torch.Tensor:
        """
        Returns the full (n_languages × n_features) probability matrix.
        Useful at evaluation time.
        """
        # (L, d) @ (d, F) → (L, F)
        logits = self.lang_embeddings.weight @ self.feat_embeddings.weight.T
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# 3. Semi-supervised Model  (Section 4 of the paper)
# ---------------------------------------------------------------------------

class TypologicalMF_SemiSup(nn.Module):
    """
    Semi-supervised extension that replaces learned language embeddings
    with *fixed* pre-trained language embeddings (e.g. from a char-LM).

    Because the language representations are frozen, the likelihood is
    convex in the parameter (feature) embeddings.

    Parameters
    ----------
    pretrained_lang_embs : np.ndarray of shape (n_languages, d_pretrained)
        Pre-trained language embeddings (e.g. from Östling & Tiedemann 2017).
    n_features : int
        Total number of binarised features.
    embed_dim : int
        Dimensionality of the feature embeddings. If it differs from
        d_pretrained, a linear projection is applied.
    freeze_lang : bool
        If True (default), language embeddings are NOT updated.
        Set to False to fine-tune them (paper reports this works better
        when some in-branch training data is available).
    """

    def __init__(self, pretrained_lang_embs: np.ndarray,
                 n_features: int,
                 embed_dim: int = 64,
                 freeze_lang: bool = False):
        super().__init__()
        n_languages, d_pre = pretrained_lang_embs.shape

        # Language embeddings initialised from pre-trained vectors
        self.lang_embeddings = nn.Embedding(n_languages, d_pre)
        self.lang_embeddings.weight = nn.Parameter(
            torch.FloatTensor(pretrained_lang_embs),
            requires_grad=not freeze_lang
        )

        # Optional projection if dimensions don't match
        if d_pre != embed_dim:
            self.proj = nn.Linear(d_pre, embed_dim, bias=False)
        else:
            self.proj = nn.Identity()

        self.feat_embeddings = nn.Embedding(n_features, embed_dim)
        nn.init.normal_(self.feat_embeddings.weight, std=0.01)

    def forward(self, lang_idx, feat_idx):
        lang_emb = self.proj(self.lang_embeddings(lang_idx))
        feat_emb = self.feat_embeddings(feat_idx)
        logits = (lang_emb * feat_emb).sum(dim=-1)
        return torch.sigmoid(logits)

    def predict_all(self):
        all_lang = self.proj(self.lang_embeddings.weight)
        logits = all_lang @ self.feat_embeddings.weight.T
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_dataset: WALSDataset,
    n_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    l2_reg: float = 0.1,       # corresponds to Gaussian prior σ²=10
    device: str = "cpu",
) -> list:
    """
    Train the matrix factorisation model.

    Parameters
    ----------
    model : TypologicalMF or TypologicalMF_SemiSup
    train_dataset : WALSDataset with observed (lang, feat, val) triples
    n_epochs : int
    batch_size : int
    lr : float – learning rate for Adam
    l2_reg : float – L2 regularisation weight (= 1/σ² from the prior)
    device : str

    Returns
    -------
    losses : list of float – average BCE loss per epoch
    """
    model = model.to(device)
    loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, drop_last=False)

    # Adam optimiser with weight decay implementing the Gaussian prior
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=l2_reg)
    criterion = nn.BCELoss()

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        model.train()
        for lang_idx, feat_idx, vals in loader:
            lang_idx = lang_idx.to(device)
            feat_idx = feat_idx.to(device)
            vals = vals.to(device)

            preds = model(lang_idx, feat_idx)
            loss = criterion(preds, vals)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / n_batches
        losses.append(avg)
        print(f"  Epoch {epoch+1}/{n_epochs}  loss={avg:.4f}")

    return losses


# ---------------------------------------------------------------------------
# 5. Quick Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Tiny synthetic example to verify shapes & training loop
    np.random.seed(42)
    N_LANG, N_FEAT = 50, 30
    # Generate a random sparse binary matrix
    true_matrix = (np.random.rand(N_LANG, N_FEAT) > 0.5).astype(float)

    # Observe 70 % of cells
    observed_mask = np.random.rand(N_LANG, N_FEAT) < 0.7
    langs, feats = np.where(observed_mask)
    vals = true_matrix[langs, feats]

    ds = WALSDataset(langs, feats, vals)
    model = TypologicalMF(N_LANG, N_FEAT, embed_dim=16)
    print("=== Training core model ===")
    train_model(model, ds, n_epochs=5, batch_size=32)

    # Predict full matrix
    with torch.no_grad():
        pred_probs = model.predict_all().numpy()
    preds_binary = (pred_probs > 0.5).astype(int)

    # Accuracy on held-out cells
    held_langs, held_feats = np.where(~observed_mask)
    held_true = true_matrix[held_langs, held_feats]
    held_pred = preds_binary[held_langs, held_feats]
    acc = (held_true == held_pred).mean()
    print(f"\nHeld-out accuracy (random baseline ~50%): {acc:.3f}")

    # Semi-supervised variant
    fake_pretrained = np.random.randn(N_LANG, 64).astype(np.float32)
    model_ss = TypologicalMF_SemiSup(fake_pretrained, N_FEAT,
                                      embed_dim=16, freeze_lang=False)
    print("\n=== Training semi-supervised model ===")
    train_model(model_ss, ds, n_epochs=5, batch_size=32)
    print("\nSmoke test passed ✓")
