"""
Learned Feature-Value Embeddings for Typological Matrix Factorisation
======================================================================
An alternative to the binary one-hot encoding in Bjerva et al. (2019).

Instead of expanding multi-valued features into independent binary columns,
each feature VALUE gets a dense, trainable embedding.  Prediction for a
given (language, feature) pair is a softmax over the dot products of the
language embedding with every possible value embedding for that feature.

Key differences from model.py (binary baseline):
  1. Feature values have LEARNED GEOMETRY — SOV and OVS can be nearby.
  2. Prediction is natively CATEGORICAL via softmax (mutual exclusivity
     is built into the architecture, not patched on with argmax).
  3. Information FLOWS BETWEEN VALUES of the same feature through shared
     language embeddings and through the embedding space geometry.

Architecture
------------
Given language ℓ and feature f with possible values {v_1, ..., v_K}:

    score_k  = e_{v_k}^T  λ_ℓ           for each value k
    P(v_k | ℓ, f) = softmax(score_1, ..., score_K)

where λ_ℓ ∈ ℝ^d is a language embedding and e_{v_k} ∈ ℝ^d is a value
embedding.  Each (feature, value) pair has a globally unique embedding.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Dataset  — categorical triples instead of binary triples
# ---------------------------------------------------------------------------

class CategoricalTypDataset(Dataset):
    """
    Wraps the CATEGORICAL feature matrix for PyTorch training.

    Each sample is a (language_index, feature_index, value_index) triple
    drawn from the *observed* cells of the original (non-binarised) matrix.

    Parameters
    ----------
    lang_indices : np.ndarray of shape (N,)
    feat_indices : np.ndarray of shape (N,)
        Index into the list of original features (NOT binary columns).
    value_indices : np.ndarray of shape (N,)
        Index of the observed value within that feature's value set.
    """

    def __init__(self, lang_indices: np.ndarray,
                 feat_indices: np.ndarray,
                 value_indices: np.ndarray):
        self.lang_indices = torch.LongTensor(lang_indices)
        self.feat_indices = torch.LongTensor(feat_indices)
        self.value_indices = torch.LongTensor(value_indices)

    def __len__(self):
        return len(self.value_indices)

    def __getitem__(self, idx):
        return (self.lang_indices[idx],
                self.feat_indices[idx],
                self.value_indices[idx])


# ---------------------------------------------------------------------------
# 2. Data Preparation  — categorical encoding (NO binarisation)
# ---------------------------------------------------------------------------

def prepare_categorical(
    df, feature_cols: list,
) -> Tuple[np.ndarray, list, Dict[int, List[int]], Dict[int, List[str]]]:
    """
    Prepare a categorical feature matrix — NO one-hot expansion.

    Each cell contains the local value index (0, 1, 2, ...) or -1 for
    missing.  Also builds the global value ID mapping that the model needs.

    Parameters
    ----------
    df : pd.DataFrame with feature columns
    feature_cols : list of column names (e.g. ["feat_81A", "feat_26A", ...])

    Returns
    -------
    cat_matrix : np.ndarray (n_langs, n_features), int, -1 = missing
    kept_feature_names : list of str — the original feature column names kept
    feat_to_global_ids : dict  feat_idx → list of global value IDs
        Maps each feature index to the global embedding indices of its values.
        e.g. {0: [0,1,2,3,4,5,6], 1: [7,8,9,10,11], ...}
    feat_to_value_names : dict  feat_idx → list of value name strings
        Same order as feat_to_global_ids.
    """
    kept_feature_names = []
    feat_to_global_ids: Dict[int, List[int]] = {}
    feat_to_value_names: Dict[int, List[str]] = {}

    columns = []
    global_id = 0
    feat_idx = 0

    for col in feature_cols:
        unique_vals = sorted(df[col].dropna().unique())
        n_vals = len(unique_vals)

        if n_vals <= 1:
            continue  # no discriminative information

        val_to_local = {v: i for i, v in enumerate(unique_vals)}

        # Build column: local value index or -1
        cat_col = np.full(len(df), -1, dtype=int)
        observed = df[col].notna()
        for row_idx in range(len(df)):
            if observed.iloc[row_idx]:
                cat_col[row_idx] = val_to_local[df[col].iloc[row_idx]]

        columns.append(cat_col)
        kept_feature_names.append(col)

        # Global IDs for this feature's values
        global_ids = list(range(global_id, global_id + n_vals))
        feat_to_global_ids[feat_idx] = global_ids
        feat_to_value_names[feat_idx] = unique_vals

        global_id += n_vals
        feat_idx += 1

    cat_matrix = np.column_stack(columns)  # (n_langs, n_features)
    total_values = global_id

    print(f"Categorical matrix: {cat_matrix.shape[0]} langs × "
          f"{cat_matrix.shape[1]} features, "
          f"{total_values} total unique values across all features")

    return cat_matrix, kept_feature_names, feat_to_global_ids, feat_to_value_names


# ---------------------------------------------------------------------------
# 3. Core Model — learned value embeddings with softmax
# ---------------------------------------------------------------------------

class TypologicalMF_Learned(nn.Module):
    """
    Matrix factorisation with learned feature-value embeddings.

    Instead of binary columns, each (feature, value) pair has a dense
    embedding.  Prediction is softmax over value scores for a given feature.

    Parameters
    ----------
    n_languages : int
    n_total_values : int
        Total number of unique (feature, value) pairs across all features.
        This is the size of the global value embedding table.
    feat_to_global_ids : dict
        Maps feature index → list of global value embedding indices.
    embed_dim : int
        Dimensionality d of all embeddings.
    """

    def __init__(self, n_languages: int, n_total_values: int,
                 feat_to_global_ids: Dict[int, List[int]],
                 embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.feat_to_global_ids = feat_to_global_ids

        self.lang_embeddings = nn.Embedding(n_languages, embed_dim)
        self.value_embeddings = nn.Embedding(n_total_values, embed_dim)

        nn.init.normal_(self.lang_embeddings.weight, std=0.01)
        nn.init.normal_(self.value_embeddings.weight, std=0.01)

        # Pre-compute padded tensors for efficient batched softmax.
        # For each feature, we need the list of global value IDs.
        # Since features have different numbers of values, we pad to
        # the maximum and use a mask.
        max_n_vals = max(len(gids) for gids in feat_to_global_ids.values())
        n_features = len(feat_to_global_ids)

        # (n_features, max_n_vals) — padded global IDs
        padded_ids = torch.zeros(n_features, max_n_vals, dtype=torch.long)
        # (n_features, max_n_vals) — True where valid, False where padded
        valid_mask = torch.zeros(n_features, max_n_vals, dtype=torch.bool)

        for fidx, gids in feat_to_global_ids.items():
            for j, gid in enumerate(gids):
                padded_ids[fidx, j] = gid
                valid_mask[fidx, j] = True

        self.register_buffer("padded_ids", padded_ids)
        self.register_buffer("valid_mask", valid_mask)

    def forward(self, lang_idx: torch.Tensor, feat_idx: torch.Tensor,
                value_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-probabilities for the observed (lang, feat, value) triples.

        Parameters
        ----------
        lang_idx : (B,) language indices
        feat_idx : (B,) feature indices (original, NOT binary columns)
        value_idx : (B,) local value indices (0, 1, ..., K_f - 1)

        Returns
        -------
        log_probs : (B,) log P(true_value | language, feature)
        batch_l2 : scalar, L2 penalty on accessed embeddings
        """
        B = lang_idx.shape[0]
        lang_emb = self.lang_embeddings(lang_idx)  # (B, d)

        # For each sample in the batch, get the global IDs and mask
        # for the feature's value set
        batch_padded = self.padded_ids[feat_idx]    # (B, max_n_vals)
        batch_mask = self.valid_mask[feat_idx]       # (B, max_n_vals)

        # Look up ALL value embeddings for each sample's feature
        val_embs = self.value_embeddings(batch_padded)  # (B, max_n_vals, d)

        # Scores: dot product of language embedding with each value embedding
        # (B, 1, d) @ (B, d, max_n_vals) → (B, 1, max_n_vals) → (B, max_n_vals)
        scores = torch.bmm(lang_emb.unsqueeze(1),
                           val_embs.transpose(1, 2)).squeeze(1)

        # Mask out padded positions with -inf before softmax
        scores = scores.masked_fill(~batch_mask, float("-inf"))

        # Log-softmax over value dimension
        log_probs_all = F.log_softmax(scores, dim=-1)   # (B, max_n_vals)

        # Extract the log-prob of the TRUE value for each sample
        log_probs = log_probs_all.gather(1, value_idx.unsqueeze(1)).squeeze(1)

        # L2 regularisation ONLY on batch-accessed embeddings.
        # Critical for Adam: global weight_decay applies a decay gradient
        # to every embedding at every step, but most embeddings are absent
        # from any given batch.  Adam's adaptive denominator turns that
        # sparse decay into constant-rate sign-gradient descent, crushing
        # all embeddings to zero.  See model.py for detailed explanation.
        #
        # We penalise the language embeddings AND the value embeddings
        # that were actually looked up in this batch.
        unique_global_ids = batch_padded[batch_mask].unique()
        accessed_val_embs = self.value_embeddings(unique_global_ids)
        batch_l2 = (lang_emb.pow(2).sum()
                    + accessed_val_embs.pow(2).sum()) / B

        return log_probs, batch_l2

    def predict_feature(self, lang_idx: torch.Tensor,
                        feat_idx: int) -> torch.Tensor:
        """
        Predict value probabilities for a set of languages on one feature.

        Parameters
        ----------
        lang_idx : (N,) language indices
        feat_idx : int, the feature index

        Returns
        -------
        probs : (N, K_f) probabilities over values
        """
        lang_emb = self.lang_embeddings(lang_idx)  # (N, d)
        gids = self.feat_to_global_ids[feat_idx]
        gids_t = torch.tensor(gids, device=lang_emb.device)
        val_embs = self.value_embeddings(gids_t)    # (K_f, d)

        scores = lang_emb @ val_embs.T               # (N, K_f)
        return F.softmax(scores, dim=-1)

    def predict_all_features(self) -> Dict[int, torch.Tensor]:
        """
        Predict value probabilities for ALL languages on ALL features.

        Returns
        -------
        predictions : dict  feat_idx → (n_languages, K_f) probability tensor
        """
        all_lang_emb = self.lang_embeddings.weight    # (L, d)
        preds = {}
        for fidx, gids in self.feat_to_global_ids.items():
            gids_t = torch.tensor(gids, device=all_lang_emb.device)
            val_embs = self.value_embeddings(gids_t)  # (K_f, d)
            scores = all_lang_emb @ val_embs.T         # (L, K_f)
            preds[fidx] = F.softmax(scores, dim=-1)
        return preds

    def get_value_embeddings_for_feature(self, feat_idx: int) -> np.ndarray:
        """
        Extract the learned value embeddings for a feature (for analysis).

        Returns
        -------
        embs : (K_f, d) numpy array
        """
        gids = self.feat_to_global_ids[feat_idx]
        gids_t = torch.tensor(gids, device=self.value_embeddings.weight.device)
        return self.value_embeddings(gids_t).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 4. Training Loop  — cross-entropy instead of BCE
# ---------------------------------------------------------------------------

def train_model_learned(
    model: TypologicalMF_Learned,
    train_dataset: CategoricalTypDataset,
    n_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    l2_reg: float = 0.1,
    device: str = "cpu",
) -> list:
    """
    Train the learned-embedding model with categorical cross-entropy.

    The loss is:  -mean(log P(true_value | lang, feat)) + l2_reg * L2

    Parameters
    ----------
    model : TypologicalMF_Learned
    train_dataset : CategoricalTypDataset
    n_epochs, batch_size, lr, l2_reg, device : same as model.train_model

    Returns
    -------
    losses : list of float — average NLL per epoch
    """
    model = model.to(device)
    loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        model.train()

        for lang_idx, feat_idx, val_idx in loader:
            lang_idx = lang_idx.to(device)
            feat_idx = feat_idx.to(device)
            val_idx = val_idx.to(device)

            log_probs, batch_l2 = model(lang_idx, feat_idx, val_idx)

            # Negative log-likelihood (cross-entropy)
            nll = -log_probs.mean()
            loss = nll + (l2_reg / 2.0) * batch_l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += nll.item()
            n_batches += 1

        avg = epoch_loss / n_batches
        losses.append(avg)
        print(f"  Epoch {epoch+1}/{n_epochs}  NLL={avg:.4f}")

    return losses


# ---------------------------------------------------------------------------
# 5. Evaluation  — direct categorical, no decode step needed
# ---------------------------------------------------------------------------

def evaluate_learned(
    model: TypologicalMF_Learned,
    eval_langs: np.ndarray,
    eval_feats: np.ndarray,
    eval_vals: np.ndarray,
    feat_to_value_names: Dict[int, List[str]],
    device: str = "cpu",
) -> float:
    """
    Predict held-out features and compute micro-F1.

    Because the model is natively categorical, no argmax-over-binary-columns
    decoding is needed — we just take the argmax of the softmax directly.

    Parameters
    ----------
    eval_langs : (N,) language indices
    eval_feats : (N,) feature indices (original features)
    eval_vals  : (N,) true local value indices
    feat_to_value_names : maps feat_idx → list of value name strings

    Returns
    -------
    f1 : float — micro-averaged F1 score
    """
    from sklearn.metrics import f1_score

    model.eval()
    with torch.no_grad():
        all_preds = model.predict_all_features()

    y_true, y_pred = [], []
    for i in range(len(eval_langs)):
        li = int(eval_langs[i])
        fi = int(eval_feats[i])
        true_vi = int(eval_vals[i])

        probs = all_preds[fi][li].cpu().numpy()  # (K_f,)
        pred_vi = int(np.argmax(probs))

        value_names = feat_to_value_names[fi]
        y_true.append(value_names[true_vi])
        y_pred.append(value_names[pred_vi])

    if len(y_true) == 0:
        return 0.0
    return f1_score(y_true, y_pred, average="micro")


# ---------------------------------------------------------------------------
# 6. Branch-based splitting — categorical version
# ---------------------------------------------------------------------------

def split_by_branch_categorical(
    df,
    cat_matrix: np.ndarray,
    target_branch: str,
    in_branch_train_frac: float = 0.20,
    rng=None,
) -> Tuple[CategoricalTypDataset, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/eval splits on the CATEGORICAL matrix.

    Same protocol as the binary version:
    1. All observed cells for out-of-branch languages → train
    2. In-branch: 80% of observed features for eval, rest for training
       (in_branch_train_frac of total observed features go to train)

    No need to worry about keeping binary sub-columns together — each
    feature is already a single column.

    Returns
    -------
    train_ds : CategoricalTypDataset
    eval_langs, eval_feats, eval_vals : arrays for evaluation
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_langs, n_feats = cat_matrix.shape
    in_branch_mask = (df["genus"] == target_branch).values

    # Out-of-branch: all observed cells go to training
    out_langs, out_feats, out_vals = [], [], []
    for i in range(n_langs):
        if in_branch_mask[i]:
            continue
        observed = np.where(cat_matrix[i] >= 0)[0]
        out_langs.append(np.full(len(observed), i))
        out_feats.append(observed)
        out_vals.append(cat_matrix[i, observed])

    # In-branch: split by feature per language
    in_train_l, in_train_f, in_train_v = [], [], []
    eval_l, eval_f, eval_v = [], [], []

    for i in np.where(in_branch_mask)[0]:
        observed_feats = np.where(cat_matrix[i] >= 0)[0].tolist()
        rng.shuffle(observed_feats)
        n_obs = len(observed_feats)
        n_eval = int(np.ceil(n_obs * 0.80))
        eval_feats_i = observed_feats[:n_eval]
        train_pool = observed_feats[n_eval:]

        n_use = min(round(n_obs * in_branch_train_frac), len(train_pool))
        train_feats_i = train_pool[:n_use]

        for fi in eval_feats_i:
            eval_l.append(i)
            eval_f.append(fi)
            eval_v.append(cat_matrix[i, fi])

        for fi in train_feats_i:
            in_train_l.append(i)
            in_train_f.append(fi)
            in_train_v.append(cat_matrix[i, fi])

    all_train_l = np.concatenate(out_langs + [np.array(in_train_l, dtype=int)])
    all_train_f = np.concatenate(out_feats + [np.array(in_train_f, dtype=int)])
    all_train_v = np.concatenate(out_vals + [np.array(in_train_v, dtype=int)])

    train_ds = CategoricalTypDataset(all_train_l, all_train_f, all_train_v)

    return (train_ds,
            np.array(eval_l, dtype=int),
            np.array(eval_f, dtype=int),
            np.array(eval_v, dtype=int))


# ---------------------------------------------------------------------------
# 7. Majority Baseline (categorical)
# ---------------------------------------------------------------------------

def majority_baseline_categorical(
    train_ds: CategoricalTypDataset,
    eval_langs: np.ndarray,
    eval_feats: np.ndarray,
    eval_vals: np.ndarray,
    feat_to_value_names: Dict[int, List[str]],
) -> float:
    """
    Most-frequent-class baseline for categorical features.

    For each feature, predict the most common value from training data.
    """
    from collections import defaultdict
    from sklearn.metrics import f1_score as _f1_score

    # Count occurrences of each (feat, value) in training
    feat_val_counts: dict = defaultdict(lambda: defaultdict(int))
    for k in range(len(train_ds)):
        _, fi, vi = train_ds[k]
        feat_val_counts[int(fi)][int(vi)] += 1

    y_true, y_pred = [], []
    for i in range(len(eval_langs)):
        li = int(eval_langs[i])
        fi = int(eval_feats[i])
        vi = int(eval_vals[i])

        value_names = feat_to_value_names[fi]
        true_label = value_names[vi]

        counts = feat_val_counts.get(fi, {})
        if counts:
            pred_vi = max(counts, key=counts.get)
            pred_label = value_names[pred_vi]
        else:
            pred_label = value_names[0]

        y_true.append(true_label)
        y_pred.append(pred_label)

    if len(y_true) == 0:
        return 0.0
    return _f1_score(y_true, y_pred, average="micro")


# ---------------------------------------------------------------------------
# 8. Full Experiment Runner
# ---------------------------------------------------------------------------

def run_experiments_learned(
    df,
    cat_matrix: np.ndarray,
    feat_to_global_ids: Dict[int, List[int]],
    feat_to_value_names: Dict[int, List[str]],
    in_branch_fracs: list = [0.0, 0.01, 0.05, 0.10, 0.20],
    n_repeats: int = 5,
    embed_dim: int = 64,
    n_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    l2_reg: float = 0.1,
    device: str = "cpu",
    min_branch_size: int = 5,
    only_branches: list = None,
):
    """
    Run the full experiment suite using the learned-embedding model.

    Same protocol as evaluation_pipeline.run_experiments, but using
    categorical data and the softmax model.

    Returns a DataFrame with columns:
        [branch, macroarea, in_branch_frac, repeat, model, f1]
    """
    import pandas as pd

    n_langs = cat_matrix.shape[0]
    n_total_values = max(max(gids) for gids in feat_to_global_ids.values()) + 1

    branch_counts = df["genus"].value_counts()
    qualifying = branch_counts[branch_counts >= min_branch_size].index.tolist()
    if only_branches:
        qualifying = [b for b in qualifying if b in only_branches]
    print(f"\nQualifying branches (≥{min_branch_size} languages): "
          f"{len(qualifying)}")

    rows = []

    for branch in qualifying:
        macroarea = df.loc[df["genus"] == branch, "macroarea"].iloc[0]
        print(f"\n{'='*60}")
        print(f"Branch: {branch}  (macroarea: {macroarea})")
        print(f"{'='*60}")

        for frac in in_branch_fracs:
            for rep in range(n_repeats):
                rng = np.random.default_rng(
                    seed=rep * 1000 + hash(branch) % 10000)

                train_ds, ev_l, ev_f, ev_v = split_by_branch_categorical(
                    df, cat_matrix, target_branch=branch,
                    in_branch_train_frac=frac, rng=rng)

                if len(ev_v) == 0:
                    continue

                # --- Majority baseline ---
                f1_freq = majority_baseline_categorical(
                    train_ds, ev_l, ev_f, ev_v,
                    feat_to_value_names)
                rows.append({
                    "branch": branch, "macroarea": macroarea,
                    "in_branch_frac": frac, "repeat": rep,
                    "model": "Freq", "f1": f1_freq,
                })

                # --- Learned Embedding Model ---
                model = TypologicalMF_Learned(
                    n_langs, n_total_values, feat_to_global_ids, embed_dim)

                print(f"\n  [Learned] branch={branch}, frac={frac}, "
                      f"rep={rep+1}")
                train_model_learned(
                    model, train_ds, n_epochs=n_epochs,
                    batch_size=batch_size, lr=lr,
                    l2_reg=l2_reg, device=device)

                f1 = evaluate_learned(
                    model, ev_l, ev_f, ev_v,
                    feat_to_value_names, device)

                rows.append({
                    "branch": branch, "macroarea": macroarea,
                    "in_branch_frac": frac, "repeat": rep,
                    "model": "Learned", "f1": f1
                })
                print(f"    → F1 = {f1:.4f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from scipy.spatial.distance import cosine as cosine_dist

    np.random.seed(42)
    torch.manual_seed(42)

    # Synthetic data: 50 languages, 10 features with varying cardinality
    N_LANG = 50
    feature_cards = [7, 5, 3, 2, 4, 6, 2, 3, 5, 4]  # values per feature
    N_FEAT = len(feature_cards)

    # Build categorical matrix
    cat_matrix = np.full((N_LANG, N_FEAT), -1, dtype=int)
    for fi, card in enumerate(feature_cards):
        for li in range(N_LANG):
            if np.random.rand() < 0.7:  # 70% observed
                cat_matrix[li, fi] = np.random.randint(0, card)

    # Build feat_to_global_ids and feat_to_value_names
    feat_to_global_ids = {}
    feat_to_value_names = {}
    gid = 0
    for fi, card in enumerate(feature_cards):
        feat_to_global_ids[fi] = list(range(gid, gid + card))
        feat_to_value_names[fi] = [f"val_{fi}_{v}" for v in range(card)]
        gid += card

    n_total = gid
    print(f"Synthetic data: {N_LANG} languages, {N_FEAT} features, "
          f"{n_total} total values")

    # Create dataset from observed cells
    ls, fs, vs = [], [], []
    for li in range(N_LANG):
        for fi in range(N_FEAT):
            if cat_matrix[li, fi] >= 0:
                ls.append(li)
                fs.append(fi)
                vs.append(cat_matrix[li, fi])

    # Hold out 30% for testing
    indices = np.arange(len(ls))
    np.random.shuffle(indices)
    split = int(0.7 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    train_ds = CategoricalTypDataset(
        np.array(ls)[train_idx],
        np.array(fs)[train_idx],
        np.array(vs)[train_idx])

    model = TypologicalMF_Learned(N_LANG, n_total, feat_to_global_ids,
                                   embed_dim=32)

    print("\n=== Training learned-embedding model ===")
    train_model_learned(model, train_ds, n_epochs=10, batch_size=32)

    # Evaluate on held-out
    f1 = evaluate_learned(
        model,
        np.array(ls)[test_idx],
        np.array(fs)[test_idx],
        np.array(vs)[test_idx],
        feat_to_value_names)
    print(f"\nHeld-out F1 (random baseline varies by cardinality): {f1:.3f}")

    # Print value embedding geometry (pairwise cosine distances)
    for fi in range(N_FEAT):
        if feature_cards[fi] >= 3:
            print(f"\n=== Value embedding geometry for feature {fi} "
                  f"({feature_cards[fi]} values) ===")
            embs = model.get_value_embeddings_for_feature(fi)
            names = feat_to_value_names[fi]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    d = cosine_dist(embs[i], embs[j])
                    print(f"  {names[i]} <-> {names[j]}: "
                          f"cosine distance = {d:.3f}")
            break  # show one feature as example

    # Test split_by_branch_categorical with synthetic genus column
    print("\n=== Testing split_by_branch_categorical ===")
    df_synth = pd.DataFrame({
        "wals_code": [f"lang_{i}" for i in range(N_LANG)],
        "name": [f"Language {i}" for i in range(N_LANG)],
        "genus": ["GroupA"] * 10 + ["GroupB"] * 40,
        "family": ["FamX"] * N_LANG,
        "macroarea": ["Eurasia"] * N_LANG,
    })
    train_ds2, ev_l, ev_f, ev_v = split_by_branch_categorical(
        df_synth, cat_matrix, "GroupA", in_branch_train_frac=0.10,
        rng=np.random.default_rng(0))
    print(f"Train samples: {len(train_ds2)}, Eval samples: {len(ev_l)}")

    model2 = TypologicalMF_Learned(N_LANG, n_total, feat_to_global_ids,
                                    embed_dim=32)
    train_model_learned(model2, train_ds2, n_epochs=5, batch_size=32)
    f1_split = evaluate_learned(model2, ev_l, ev_f, ev_v, feat_to_value_names)
    print(f"Split eval F1: {f1_split:.3f}")

    print("\nSmoke test passed.")
