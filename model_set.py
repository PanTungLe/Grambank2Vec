"""
SetEncoder: set-conditioned categorical value predictor for typological features.
Zero-shot by construction — no language embedding table.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Tuple


class SetEncoder(nn.Module):
    """
    Set-conditioned encoder for typological feature prediction.

    Architecture:
      1. For each observed pair (f_i, v_i):
           pair_emb_i = feat_emb(f_i) + val_emb(global_id(v_i))   # (d,)
      2. Context encoding:
           context_vec = mean(pair_emb_i for i in context)          # (d,)
           z = context_mlp(context_vec)                              # (d,)
      3. Predict target feature f_j:
           val_embs_j = val_emb(feat_to_global_ids[j])              # (K_j, d)
           logits_j   = z @ val_embs_j.T                            # (K_j,)
           P(v_j | f_j, context) = softmax(logits_j)

    Parameters
    ----------
    n_features : int
    n_total_values : int
    feat_to_global_ids : dict  feat_idx -> list of global value IDs
    embed_dim : int
    hidden_dim : int
    """

    def __init__(self, n_features: int, n_total_values: int,
                 feat_to_global_ids: Dict[int, List[int]],
                 embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.feat_to_global_ids = feat_to_global_ids
        self.embed_dim = embed_dim

        self.feat_embeddings = nn.Embedding(n_features, embed_dim)
        self.val_embeddings  = nn.Embedding(n_total_values, embed_dim)
        self.context_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.normal_(self.feat_embeddings.weight, std=0.01)
        nn.init.normal_(self.val_embeddings.weight,  std=0.01)

    def encode_context(self, context_feat_indices: List[int],
                       context_global_val_ids: List[int],
                       device) -> torch.Tensor:
        """
        Encode a context set into a language representation z.

        Parameters
        ----------
        context_feat_indices  : list of int, length K_obs
        context_global_val_ids: list of int, length K_obs (global value IDs)
        device : str or torch.device

        Returns
        -------
        z : Tensor (1, d)
        """
        if len(context_feat_indices) == 0:
            return torch.zeros(1, self.embed_dim, device=device)

        fi = torch.tensor(context_feat_indices, dtype=torch.long, device=device)
        vi = torch.tensor(context_global_val_ids, dtype=torch.long, device=device)

        feat_embs = self.feat_embeddings(fi)   # (K, d)
        val_embs  = self.val_embeddings(vi)    # (K, d)
        pair_embs = feat_embs + val_embs       # (K, d)

        context_vec = pair_embs.mean(0, keepdim=True)  # (1, d)
        z = self.context_mlp(context_vec)              # (1, d)
        return z

    def predict_feature(self, z: torch.Tensor, feat_idx: int,
                        device) -> torch.Tensor:
        """
        Predict distribution over values of feature feat_idx given z.

        Returns
        -------
        probs : Tensor (K_f,) — softmax probabilities
        """
        gids = self.feat_to_global_ids[feat_idx]
        gids_t = torch.tensor(gids, dtype=torch.long, device=device)
        val_embs = self.val_embeddings(gids_t)   # (K_f, d)
        logits = (z @ val_embs.T).squeeze(0)     # (K_f,)
        return F.softmax(logits, dim=0)

    def forward(self, context_feat_indices: List[int],
                context_global_val_ids: List[int],
                target_feat_indices: List[int],
                target_val_local_indices: List[int],
                device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Parameters
        ----------
        context_feat_indices    : list of int — context feature indices
        context_global_val_ids  : list of int — global val IDs for context
        target_feat_indices     : list of int — target feature indices
        target_val_local_indices: list of int — correct local value index per target
        device : str or torch.device

        Returns
        -------
        loss : scalar Tensor — mean cross-entropy over targets
        l2   : scalar Tensor — L2 on accessed embeddings only (Adam-safe)
        """
        z = self.encode_context(context_feat_indices,
                                context_global_val_ids, device)

        if len(target_feat_indices) == 0:
            return (torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device))

        total_nll = torch.tensor(0.0, device=device)
        accessed_embs = []

        for fi, vi_local in zip(target_feat_indices, target_val_local_indices):
            gids = self.feat_to_global_ids[fi]
            gids_t = torch.tensor(gids, dtype=torch.long, device=device)
            val_embs = self.val_embeddings(gids_t)   # (K_f, d)
            logits = (z @ val_embs.T).squeeze(0)     # (K_f,)
            log_probs = F.log_softmax(logits, dim=0)
            total_nll -= log_probs[vi_local]
            accessed_embs.append(val_embs)

        loss = total_nll / len(target_feat_indices)

        # Adam-safe L2: only accessed embeddings
        all_accessed = torch.cat(accessed_embs, dim=0)   # (total_K, d)
        if len(context_feat_indices) > 0:
            fi_t = torch.tensor(context_feat_indices,
                                dtype=torch.long, device=device)
            vi_t = torch.tensor(context_global_val_ids,
                                dtype=torch.long, device=device)
            ctx_embs = torch.cat([self.feat_embeddings(fi_t),
                                  self.val_embeddings(vi_t)], dim=0)
            all_accessed = torch.cat([all_accessed, ctx_embs], dim=0)
        l2 = all_accessed.pow(2).mean()

        return loss, l2


class SetEncoderDataset(Dataset):
    """
    Training dataset for SetEncoder.

    For each language, randomly splits observed features into context and
    targets at getitem time. This gives diverse training signal across epochs.

    Parameters
    ----------
    cat_matrix : np.ndarray (n_langs, n_features), int, -1 = missing
    feat_to_global_ids : dict  feat_idx -> list of global value IDs
    min_context : int
    min_targets : int
    """

    def __init__(self, cat_matrix: np.ndarray,
                 feat_to_global_ids: Dict[int, List[int]],
                 min_context: int = 1, min_targets: int = 1):
        self.cat_matrix = cat_matrix
        self.feat_to_global_ids = feat_to_global_ids
        self.min_context = min_context
        self.min_targets = min_targets

        self.valid_langs = [
            i for i in range(cat_matrix.shape[0])
            if int((cat_matrix[i] >= 0).sum()) >= min_context + min_targets
        ]

    def __len__(self) -> int:
        return len(self.valid_langs)

    def __getitem__(self, idx: int) -> dict:
        lang_idx = self.valid_langs[idx]
        observed_feats = np.where(self.cat_matrix[lang_idx] >= 0)[0].tolist()

        max_ctx = len(observed_feats) - self.min_targets
        ctx_size = int(np.random.randint(self.min_context, max_ctx + 1))

        perm = np.random.permutation(len(observed_feats))
        ctx_feats = [observed_feats[i] for i in perm[:ctx_size]]
        tgt_feats = [observed_feats[i] for i in perm[ctx_size:]]

        ctx_gvals = [
            self.feat_to_global_ids[fi][int(self.cat_matrix[lang_idx, fi])]
            for fi in ctx_feats
        ]
        tgt_locals = [int(self.cat_matrix[lang_idx, fi]) for fi in tgt_feats]

        return {
            "lang_idx":   lang_idx,
            "ctx_feats":  ctx_feats,
            "ctx_gvals":  ctx_gvals,
            "tgt_feats":  tgt_feats,
            "tgt_locals": tgt_locals,
        }


def train_set_encoder(model: SetEncoder, dataset: SetEncoderDataset,
                      n_epochs: int = 200, lr: float = 1e-3,
                      l2_reg: float = 0.1, device: str = "cpu",
                      seed: int = 42) -> List[float]:
    """
    Train SetEncoder. Iterates over dataset indices directly (no DataLoader)
    to handle ragged context/target lists. Fresh random split per language
    per epoch.

    Returns list of mean epoch losses.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(len(dataset)).tolist()

        for idx in perm:
            sample = dataset[idx]
            if not sample["tgt_feats"]:
                continue

            loss, l2 = model(
                sample["ctx_feats"],
                sample["ctx_gvals"],
                sample["tgt_feats"],
                sample["tgt_locals"],
                device,
            )
            total = loss + (l2_reg / 2.0) * l2

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  loss={avg:.4f}")

    return losses
