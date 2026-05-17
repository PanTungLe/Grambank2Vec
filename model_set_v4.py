"""
CrossAttentionSetEncoder: per-target cross-attention replaces mean pooling,
letting the model attend differently for each target feature.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from model_set import SetEncoderDataset   # reuse dataset


class CrossAttentionSetEncoder(nn.Module):
    """
    Cross-attention SetEncoder: for each target feature f_j the model
    attends over the observed context using f_j's embedding as a query.

    Architecture:
      pair_emb_i = feat_emb(f_i) + val_emb(global_id(v_i))  [observed]
      query_j    = feat_emb(f_j)
      attn_j     = softmax(query_j @ pair_embs.T / sqrt(d))
      ctx_j      = attn_j @ pair_embs
      z_j        = context_mlp(ctx_j)
      logits_j   = z_j @ val_emb(feat_to_global_ids[j]).T
      P(v_j|f_j,O) = softmax(logits_j)

    Key difference from v3 SetEncoder: z_j is DIFFERENT per target feature.
    The model learns which observed features are most predictive of each
    typological property being inferred.
    """

    def __init__(self, n_features: int, n_total_values: int,
                 feat_to_global_ids: Dict[int, List[int]],
                 embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.feat_to_global_ids = feat_to_global_ids
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5

        self.feat_embeddings = nn.Embedding(n_features, embed_dim)
        self.val_embeddings  = nn.Embedding(n_total_values, embed_dim)
        self.context_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.normal_(self.feat_embeddings.weight, std=0.01)
        nn.init.normal_(self.val_embeddings.weight,  std=0.01)

    def _compute_pair_embs(self, ctx_feat_indices: List[int],
                           ctx_gvals: List[int],
                           device) -> torch.Tensor:
        """Returns (K_ctx, d) pair embeddings, or None if context is empty."""
        if not ctx_feat_indices:
            return None
        fi = torch.tensor(ctx_feat_indices, dtype=torch.long, device=device)
        vi = torch.tensor(ctx_gvals,        dtype=torch.long, device=device)
        return self.feat_embeddings(fi) + self.val_embeddings(vi)

    def logits_for_feature(self, ctx_feat_indices: List[int],
                           ctx_gvals: List[int],
                           target_feat_idx: int,
                           device) -> torch.Tensor:
        """
        Compute raw logits (before softmax, before temperature) for a
        single target feature given context.

        Returns
        -------
        logits : Tensor (K_f,)
        """
        pair_embs = self._compute_pair_embs(ctx_feat_indices, ctx_gvals, device)
        query = self.feat_embeddings(
            torch.tensor([target_feat_idx], dtype=torch.long, device=device))
                                                    # (1, d)
        if pair_embs is None:
            ctx_vec = torch.zeros(1, self.embed_dim, device=device)
        else:
            scores  = (query @ pair_embs.T) * self.scale   # (1, K_ctx)
            attn    = torch.softmax(scores, dim=-1)
            ctx_vec = attn @ pair_embs                      # (1, d)

        z = self.context_mlp(ctx_vec)                       # (1, d)

        gids     = self.feat_to_global_ids[target_feat_idx]
        gids_t   = torch.tensor(gids, dtype=torch.long, device=device)
        val_embs = self.val_embeddings(gids_t)              # (K_f, d)
        return (z @ val_embs.T).squeeze(0)                  # (K_f,)

    def predict_feature(self, ctx_feat_indices: List[int],
                        ctx_gvals: List[int],
                        target_feat_idx: int,
                        device,
                        temperature: float = 1.0) -> torch.Tensor:
        """
        Returns log-softmax probabilities with temperature scaling.
        Temperature is applied to LOGITS before softmax.
        """
        logits = self.logits_for_feature(
            ctx_feat_indices, ctx_gvals, target_feat_idx, device)
        return F.log_softmax(logits / temperature, dim=0)

    def forward(self, ctx_feat_indices: List[int],
                ctx_gvals: List[int],
                target_feat_indices: List[int],
                target_val_local_indices: List[int],
                device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass. Vectorized cross-attention over all targets.

        Returns
        -------
        loss : scalar Tensor — mean cross-entropy over targets
        l2   : scalar Tensor — L2 on accessed embeddings only (Adam-safe)
        """
        if not target_feat_indices:
            return (torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device))

        pair_embs = self._compute_pair_embs(ctx_feat_indices, ctx_gvals, device)

        tgt_fi_t    = torch.tensor(target_feat_indices, dtype=torch.long, device=device)
        tgt_queries = self.feat_embeddings(tgt_fi_t)        # (K_tgt, d)

        if pair_embs is not None:
            scores   = (tgt_queries @ pair_embs.T) * self.scale  # (K_tgt, K_ctx)
            attn     = torch.softmax(scores, dim=-1)
            ctx_vecs = attn @ pair_embs                           # (K_tgt, d)
        else:
            ctx_vecs = torch.zeros(
                len(target_feat_indices), self.embed_dim, device=device)

        z_targets = self.context_mlp(ctx_vecs)              # (K_tgt, d)

        total_nll    = torch.tensor(0.0, device=device)
        accessed_val = []

        for i, (fi, vi_local) in enumerate(zip(target_feat_indices,
                                               target_val_local_indices)):
            gids   = self.feat_to_global_ids[fi]
            gids_t = torch.tensor(gids, dtype=torch.long, device=device)
            ve     = self.val_embeddings(gids_t)                  # (K_fi, d)
            logits = (z_targets[i:i+1] @ ve.T).squeeze(0)        # (K_fi,)
            lp     = F.log_softmax(logits, dim=0)
            total_nll -= lp[vi_local]
            accessed_val.append(ve)

        loss = total_nll / len(target_feat_indices)

        # Adam-safe L2: only accessed embeddings
        parts = accessed_val + [tgt_queries]
        if pair_embs is not None:
            parts.append(pair_embs)
        all_acc = torch.cat([p.reshape(-1, self.embed_dim) for p in parts], dim=0)
        l2 = all_acc.pow(2).mean()

        return loss, l2


def train_cross_attention(model: CrossAttentionSetEncoder,
                          dataset: SetEncoderDataset,
                          n_epochs: int = 200, lr: float = 1e-3,
                          l2_reg: float = 0.1, device: str = "cpu",
                          seed: int = 42) -> List[float]:
    """
    Training loop: identical structure to train_set_encoder in model_set.py.
    Iterate over dataset indices directly — ragged context/target lists.
    Fresh random split per language per epoch via SetEncoderDataset.__getitem__.
    Adam with weight_decay=0; per-batch L2 over accessed embeddings only.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for idx in torch.randperm(len(dataset)).tolist():
            sample = dataset[idx]
            if not sample["tgt_feats"]:
                continue
            loss, l2 = model(
                sample["ctx_feats"], sample["ctx_gvals"],
                sample["tgt_feats"], sample["tgt_locals"],
                device,
            )
            total = loss + (l2_reg / 2.0) * l2
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  loss={avg:.4f}")

    return losses
