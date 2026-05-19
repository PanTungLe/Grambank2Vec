"""
A Probabilistic Generative Model of Linguistic Typology
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
                feat_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns predicted probabilities and the batch L2 penalty.

        The L2 penalty is computed ONLY over the embeddings accessed
        in this batch.  This is critical for Adam: global weight_decay
        or global L2 applies a decay gradient to every embedding at
        every step, but most embeddings are absent from any given batch.
        Adam's adaptive rate turns that sparse decay into constant-rate
        sign-gradient descent, crushing all embeddings to zero.

        By regularising only the batch embeddings, unseen embeddings
        receive zero gradient and stay at their current values.
        """
        lang_emb = self.lang_embeddings(lang_idx)   # (B, d)
        feat_emb = self.feat_embeddings(feat_idx)    # (B, d)
        logits = (lang_emb * feat_emb).sum(dim=-1)   # dot product → (B,)
        preds = torch.sigmoid(logits)

        # Per-batch L2: mean over the batch so the scale is independent
        # of batch size and comparable to the mean BCE
        batch_l2 = (lang_emb.pow(2).sum() + feat_emb.pow(2).sum()) / lang_idx.shape[0]

        return preds, batch_l2

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
        preds = torch.sigmoid(logits)

        # L2 handled by AdamW(weight_decay=l2_reg) in train_model.
        return preds, torch.tensor(0.0)  # placeholder keeps (preds, l2) API

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
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    l2_reg: float = 0.1,       # corresponds to Gaussian prior σ²=10
    device: str = "cpu",
    patience: int = 5,
) -> Tuple[list, int]:
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
    patience : int – early stopping: stop if loss does not improve by >0.001
                     for this many consecutive epochs

    Returns
    -------
    losses : list of float – average BCE loss per epoch
    epochs_run : int – number of epochs actually run (may be < n_epochs
                 if early stopping was triggered)
    """
    model = model.to(device)
    loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, drop_last=False)

    # Use AdamW so weight decay is applied multiplicatively (decoupled from
    # gradient normalization).  Plain Adam adds l2_reg to the gradient, which
    # Adam then normalizes to ±lr — cancelling the BCE signal at equal strength
    # regardless of l2_reg magnitude.  AdamW applies the decay after the Adam
    # step, preserving the correct magnitude relationship.
    # Filter to trainable parameters only (skips frozen pretrained embeddings)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=l2_reg)
    criterion = nn.BCELoss()

    losses = []
    best_loss = float('inf')
    epochs_no_improve = 0
    epochs_run = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        model.train()
        for lang_idx, feat_idx, vals in loader:
            lang_idx = lang_idx.to(device)
            feat_idx = feat_idx.to(device)
            vals = vals.to(device)

            preds, _batch_l2 = model(lang_idx, feat_idx)
            bce = criterion(preds, vals)

            optimizer.zero_grad()
            bce.backward()
            optimizer.step()

            epoch_loss += bce.item()
            n_batches += 1

        avg = epoch_loss / n_batches
        losses.append(avg)
        epochs_run = epoch + 1
        print(f"  Epoch {epoch+1}/{n_epochs}  loss={avg:.4f}")

        if avg < best_loss - 0.001:
            best_loss = avg
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    return losses, epochs_run


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
    losses, epochs_run = train_model(model, ds, n_epochs=5, batch_size=32)

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
    losses, epochs_run = train_model(model_ss, ds, n_epochs=5, batch_size=32)
    print("\nSmoke test passed ✓")
"""\nA Probabilistic Generative Model of Linguistic Typology\n=========================================================\nPyTorch implementation of the core matrix factorisation model and\nsemi-supervised extension from Bjerva et al. (NAACL 2019).\n\nThe model explains a binary matrix of typological parameters via\nexponential-family matrix factorisation:\n\n    p(π_i^ℓ | λ^ℓ) = sigmoid(e_{π_i}^T  λ^ℓ)\n\nwhere λ^ℓ is a language embedding and e_{π_i} is a parameter embedding.\n"""\n\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import Dataset, DataLoader\nimport numpy as np\nfrom typing import Optional, Tuple\n\n\n# ---------------------------------------------------------------------------\n# 1. Dataset\n# ---------------------------------------------------------------------------\n\nclass WALSDataset(Dataset):\n    """\n    Wraps the binarised WALS matrix for PyTorch training.\n\n    Each sample is a (language_index, feature_index, value) triple\n    drawn from the *observed* cells of the matrix.\n\n    Parameters\n    ----------\n    lang_indices : np.ndarray of shape (N,)\n        Language index for each observed cell.\n    feat_indices : np.ndarray of shape (N,)\n        Feature index for each observed cell.\n    values : np.ndarray of shape (N,)\n        Binary value (0 or 1) for each observed cell.\n    """\n\n    def __init__(self, lang_indices: np.ndarray,\n                 feat_indices: np.ndarray,\n                 values: np.ndarray):\n        self.lang_indices = torch.LongTensor(lang_indices)\n        self.feat_indices = torch.LongTensor(feat_indices)\n        self.values = torch.FloatTensor(values)\n\n    def __len__(self):\n        return len(self.values)\n\n    def __getitem__(self, idx):\n        return (self.lang_indices[idx],\n                self.feat_indices[idx],\n                self.values[idx])\n\n\n# ---------------------------------------------------------------------------\n# 2. Core Model  (Section 3 of the paper)\n# ---------------------------------------------------------------------------\n\nclass TypologicalMF(nn.Module):\n    """\n    Exponential-family matrix factorisation for typological features.\n\n    Each language ℓ has a learned embedding  λ^ℓ ∈ ℝ^d.\n    Each (binarised) feature i has an embedding   e_{π_i} ∈ ℝ^d.\n\n    The probability that feature i is 'on' for language ℓ is:\n        p(π_i^ℓ = 1) = σ(e_{π_i}^T  λ^ℓ)\n\n    A Gaussian prior N(0, σ²I) on the language embeddings is realised\n    via L2 regularisation during training.\n\n    Parameters\n    ----------\n    n_languages : int\n        Total number of languages.\n    n_features : int\n        Total number of *binarised* features.\n    embed_dim : int\n        Dimensionality d of the embeddings.\n    """\n\n    def __init__(self, n_languages: int, n_features: int,\n                 embed_dim: int = 64):\n        super().__init__()\n        self.lang_embeddings = nn.Embedding(n_languages, embed_dim)\n        self.feat_embeddings = nn.Embedding(n_features, embed_dim)\n\n        # Initialise close to zero (standard practice for MF)\n        nn.init.normal_(self.lang_embeddings.weight, std=0.01)\n        nn.init.normal_(self.feat_embeddings.weight, std=0.01)\n\n    def forward(self, lang_idx: torch.Tensor,\n                feat_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:\n        """\n        Returns predicted probabilities and the batch L2 penalty.\n\n        The L2 penalty is computed ONLY over the embeddings accessed\n        in this batch.  This is critical for Adam: global weight_decay\n        or global L2 applies a decay gradient to every embedding at\n        every step, but most embeddings are absent from any given batch.\n        Adam's adaptive rate turns that sparse decay into constant-rate\n        sign-gradient descent, crushing all embeddings to zero.\n\n        By regularising only the batch embeddings, unseen embeddings\n        receive zero gradient and stay at their current values.\n        """\n        lang_emb = self.lang_embeddings(lang_idx)   # (B, d)\n        feat_emb = self.feat_embeddings(feat_idx)    # (B, d)\n        logits = (lang_emb * feat_emb).sum(dim=-1)   # dot product → (B,)\n        preds = torch.sigmoid(logits)\n\n        # Per-batch L2: mean over the batch so the scale is independent\n        # of batch size and comparable to the mean BCE\n        batch_l2 = (lang_emb.pow(2).sum() + feat_emb.pow(2).sum()) / lang_idx.shape[0]\n\n        return preds, batch_l2\n\n    def predict_all(self) -> torch.Tensor:\n        """\n        Returns the full (n_languages × n_features) probability matrix.\n        Useful at evaluation time.\n        """\n        # (L, d) @ (d, F) → (L, F)\n        logits = self.lang_embeddings.weight @ self.feat_embeddings.weight.T\n        return torch.sigmoid(logits)\n\n\n# ---------------------------------------------------------------------------\n# 3. Semi-supervised Model  (Section 4 of the paper)\n# ---------------------------------------------------------------------------\n\nclass TypologicalMF_SemiSup(nn.Module):\n    """\n    Semi-supervised extension that replaces learned language embeddings\n    with *fixed* pre-trained language embeddings (e.g. from a char-LM).\n\n    Because the language representations are frozen, the likelihood is\n    convex in the parameter (feature) embeddings.\n\n    Parameters\n    ----------\n    pretrained_lang_embs : np.ndarray of shape (n_languages, d_pretrained)\n        Pre-trained language embeddings (e.g. from Östling & Tiedemann 2017).\n    n_features : int\n        Total number of binarised features.\n    embed_dim : int\n        Dimensionality of the feature embeddings. If it differs from\n        d_pretrained, a linear projection is applied.\n    freeze_lang : bool\n        If True (default), language embeddings are NOT updated.\n        Set to False to fine-tune them (paper reports this works better\n        when some in-branch training data is available).\n    """\n\n    def __init__(self, pretrained_lang_embs: np.ndarray,\n                 n_features: int,\n                 embed_dim: int = 64,\n                 freeze_lang: bool = False):\n        super().__init__()\n        n_languages, d_pre = pretrained_lang_embs.shape\n\n        # Language embeddings initialised from pre-trained vectors\n        self.lang_embeddings = nn.Embedding(n_languages, d_pre)\n        self.lang_embeddings.weight = nn.Parameter(\n            torch.FloatTensor(pretrained_lang_embs),\n            requires_grad=not freeze_lang\n        )\n\n        # Optional projection if dimensions don't match\n        if d_pre != embed_dim:\n            self.proj = nn.Linear(d_pre, embed_dim, bias=False)\n        else:\n            self.proj = nn.Identity()\n\n        self.feat_embeddings = nn.Embedding(n_features, embed_dim)\n        nn.init.normal_(self.feat_embeddings.weight, std=0.01)\n\n    def forward(self, lang_idx, feat_idx):\n        lang_emb = self.proj(self.lang_embeddings(lang_idx))\n        feat_emb = self.feat_embeddings(feat_idx)\n        logits = (lang_emb * feat_emb).sum(dim=-1)\n        preds = torch.sigmoid(logits)\n\n        # L2 handled by AdamW(weight_decay=l2_reg) in train_model.\n        return preds, torch.tensor(0.0)  # placeholder keeps (preds, l2) API\n\n    def predict_all(self):\n        all_lang = self.proj(self.lang_embeddings.weight)\n        logits = all_lang @ self.feat_embeddings.weight.T\n        return torch.sigmoid(logits)\n\n\n# ---------------------------------------------------------------------------\n# 4. Training Loop\n# ---------------------------------------------------------------------------\n\ndef train_model(\n    model: nn.Module,\n    train_dataset: WALSDataset,\n    n_epochs: int = 30,\n    batch_size: int = 64,\n    lr: float = 1e-3,\n    l2_reg: float = 0.1,       # corresponds to Gaussian prior σ²=10\n    device: str = \"cpu\",\n    patience: int = 5,\n) -> Tuple[list, int]:\n    \"\"\"\n    Train the matrix factorisation model.\n\n    Parameters\n    ----------\n    model : TypologicalMF or TypologicalMF_SemiSup\n    train_dataset : WALSDataset with observed (lang, feat, val) triples\n    n_epochs : int\n    batch_size : int\n    lr : float – learning rate for Adam\n    l2_reg : float – L2 regularisation weight (= 1/σ² from the prior)\n    device : str\n    patience : int – early stopping: stop if loss does not improve by >0.001\n                     for this many consecutive epochs\n\n    Returns\n    -------\n    losses : list of float – average BCE loss per epoch\n    epochs_run : int – number of epochs actually run (may be < n_epochs\n                 if early stopping was triggered)\n    \"\"\"\n    model = model.to(device)\n    loader = DataLoader(train_dataset, batch_size=batch_size,\n                        shuffle=True, drop_last=False)\n\n    # Use AdamW so weight decay is applied multiplicatively (decoupled from\n    # gradient normalization).  Plain Adam adds l2_reg to the gradient, which\n    # Adam then normalizes to ±lr — cancelling the BCE signal at equal strength\n    # regardless of l2_reg magnitude.  AdamW applies the decay after the Adam\n    # step, preserving the correct magnitude relationship.\n    # Filter to trainable parameters only (skips frozen pretrained embeddings)\n    trainable_params = [p for p in model.parameters() if p.requires_grad]\n    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=l2_reg)\n    criterion = nn.BCELoss()\n\n    losses = []\n    best_loss = float('inf')\n    epochs_no_improve = 0\n    epochs_run = 0\n\n    for epoch in range(n_epochs):\n        epoch_loss = 0.0\n        n_batches = 0\n        model.train()\n        for lang_idx, feat_idx, vals in loader:\n            lang_idx = lang_idx.to(device)\n            feat_idx = feat_idx.to(device)\n            vals = vals.to(device)\n\n            preds, _batch_l2 = model(lang_idx, feat_idx)\n            bce = criterion(preds, vals)\n\n            optimizer.zero_grad()\n            bce.backward()\n            optimizer.step()\n\n            epoch_loss += bce.item()\n            n_batches += 1\n\n        avg = epoch_loss / n_batches\n        losses.append(avg)\n        epochs_run = epoch + 1\n        print(f\"  Epoch {epoch+1}/{n_epochs}  loss={avg:.4f}\")\n\n        if avg < best_loss - 0.001:\n            best_loss = avg\n            epochs_no_improve = 0\n        else:\n            epochs_no_improve += 1\n        if epochs_no_improve >= patience:\n            print(f\"  Early stopping at epoch {epoch+1}\")\n            break\n\n    return losses, epochs_run\n\n\n# ---------------------------------------------------------------------------\n# 5. Quick Smoke Test\n# ---------------------------------------------------------------------------\n\nif __name__ == \"__main__\":\n    # Tiny synthetic example to verify shapes & training loop\n    np.random.seed(42)\n    N_LANG, N_FEAT = 50, 30\n    # Generate a random sparse binary matrix\n    true_matrix = (np.random.rand(N_LANG, N_FEAT) > 0.5).astype(float)\n\n    # Observe 70 % of cells\n    observed_mask = np.random.rand(N_LANG, N_FEAT) < 0.7\n    langs, feats = np.where(observed_mask)\n    vals = true_matrix[langs, feats]\n\n    ds = WALSDataset(langs, feats, vals)\n    model = TypologicalMF(N_LANG, N_FEAT, embed_dim=16)\n    print(\"=== Training core model ===")\n    losses, epochs_run = train_model(model, ds, n_epochs=5, batch_size=32)\n\n    # Predict full matrix\n    with torch.no_grad():\n        pred_probs = model.predict_all().numpy()\n    preds_binary = (pred_probs > 0.5).astype(int)\n\n    # Accuracy on held-out cells\n    held_langs, held_feats = np.where(~observed_mask)\n    held_true = true_matrix[held_langs, held_feats]\n    held_pred = preds_binary[held_langs, held_feats]\n    acc = (held_true == held_pred).mean()\n    print(f\"\\nHeld-out accuracy (random baseline ~50%): {acc:.3f}\")\n\n    # Semi-supervised variant\n    fake_pretrained = np.random.randn(N_LANG, 64).astype(np.float32)\n    model_ss = TypologicalMF_SemiSup(fake_pretrained, N_FEAT,\n                                      embed_dim=16, freeze_lang=False)\n    print(\"\\n=== Training semi-supervised model ===")\n    losses, epochs_run = train_model(model_ss, ds, n_epochs=5, batch_size=32)\n    print(\"\\nSmoke test passed ✓\")\n
