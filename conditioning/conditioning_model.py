"""
Conditioning models for RQ4: language embeddings conditioned on URIEL+
geographic and phylogenetic vectors.

Constraint 3 (enforced by caller via uriel_plus_loader):
  ONLY geo (URIEL+ data[2], 299-dim) and phylo (URIEL+ data[0], 3718-dim)
  are permitted. NEVER use data[1] (typological/features.npz).

Architecture (both models):
  conditioned_lang_emb = base_lang_emb + cond_proj(concat(geo_vec, phylo_vec))

cond_proj is a single bias-free linear layer (no nonlinearity), initialised
to zero so training starts from the same place as the unconditioned baseline.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. URIEL+ vector loading and alignment
# ──────────────────────────────────────────────────────────────────────────────

def load_uriel_vectors_for_lang2id(
    parquet_path: str,
    lang2id: Dict[str, int],
    geo_dim: int = 299,
    phylo_dim: int = 3718,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load URIEL+ geo and phylo vectors from parquet and align to lang2id order.

    Languages absent from the parquet receive zero vectors (no conditioning
    signal). If the parquet has a has_geo column (new single-file format),
    coverage is reported based on that flag; otherwise it is inferred from
    parquet membership. The returned arrays are indexed identically to lang2id.

    Returns
    -------
    geo_matrix   : (n_langs, geo_dim)   float32
    phylo_matrix : (n_langs, phylo_dim) float32
    meta         : dict with coverage statistics
    """
    import pyarrow.parquet as pq

    n_langs = max(lang2id.values()) + 1
    geo_matrix = np.zeros((n_langs, geo_dim), dtype=np.float32)
    phylo_matrix = np.zeros((n_langs, phylo_dim), dtype=np.float32)

    table = pq.read_table(parquet_path)
    glottocodes = table.column("glottocode").to_pylist()
    geo_rows = table.column("geo_vec").to_pylist()
    phylo_rows = table.column("phylo_vec").to_pylist()

    # Read has_geo flag if present (new single-file parquet format).
    if "has_geo" in table.schema.names:
        has_geo_flags = table.column("has_geo").to_pylist()
        gc_to_has_geo: Dict[str, bool] = dict(zip(glottocodes, has_geo_flags))
    else:
        gc_to_has_geo = {}

    gc_to_row: Dict[str, int] = {gc: i for i, gc in enumerate(glottocodes)}

    found, missing = 0, 0
    for gc, lang_idx in lang2id.items():
        if gc in gc_to_row:
            row_i = gc_to_row[gc]
            geo_matrix[lang_idx] = geo_rows[row_i]
            phylo_matrix[lang_idx] = phylo_rows[row_i]
            # Count as "found" only if has_geo=True (or no flag column present)
            if gc_to_has_geo.get(gc, True):
                found += 1
            else:
                missing += 1
        else:
            missing += 1

    meta = {
        "n_langs": n_langs,
        "found": found,
        "missing": missing,
        "pct_found": 100.0 * found / n_langs if n_langs else 0.0,
        "geo_dim": geo_dim,
        "phylo_dim": phylo_dim,
        "parquet_path": str(parquet_path),
    }
    log.info(
        "URIEL+ alignment: %d/%d (%.1f%%) with real data; "
        "%d get zero vectors.",
        found, n_langs, meta["pct_found"], missing,
    )
    return geo_matrix, phylo_matrix, meta


# ──────────────────────────────────────────────────────────────────────────────
# 2. Conditioned Learned Model (categorical softmax)
# ──────────────────────────────────────────────────────────────────────────────

class TypologicalMF_Conditioned(nn.Module):
    """
    TypologicalMF_Learned augmented with URIEL+ geo+phylo conditioning.

    The conditioning offset (default cond_mode='both') is:
        offset = cond_proj( concat(geo_vec, phylo_vec) )   [shape (d,)]

    cond_proj is a single linear layer, no nonlinearity, bias=False.

    Parameters
    ----------
    n_languages : int
    n_total_values : int
        Total number of (feature, value) pairs across all features.
    feat_to_global_ids : dict
        Maps feature index → list of global value embedding indices.
    geo_matrix : np.ndarray, shape (n_languages, geo_dim)
        URIEL+ geographic vectors aligned to lang2id order.
    phylo_matrix : np.ndarray, shape (n_languages, phylo_dim)
        URIEL+ phylogenetic vectors aligned to lang2id order.
    embed_dim : int
    cond_mode : str
        'both'       — residual from concat(geo, phylo) [default, zero-init]
        'geo_only'   — residual from concat(geo, zeros) [phylo channel zeroed]
        'phylo_only' — residual from concat(zeros, phylo) [geo channel zeroed]
        'init_only'  — conditioning applied once at init to seed lang_emb,
                       cond_proj then frozen (normal-init weights, not zero)
    """

    def __init__(
        self,
        n_languages: int,
        n_total_values: int,
        feat_to_global_ids: Dict[int, List[int]],
        geo_matrix: np.ndarray,
        phylo_matrix: np.ndarray,
        embed_dim: int = 64,
        cond_mode: str = "both",
    ) -> None:
        super().__init__()
        assert cond_mode in ("both", "geo_only", "phylo_only", "init_only"), \
            f"Unknown cond_mode: {cond_mode!r}"
        self.embed_dim = embed_dim
        self.cond_mode = cond_mode
        self.feat_to_global_ids = feat_to_global_ids

        geo_dim = geo_matrix.shape[1]
        phylo_dim = phylo_matrix.shape[1]

        self.lang_embeddings = nn.Embedding(n_languages, embed_dim)
        self.value_embeddings = nn.Embedding(n_total_values, embed_dim)

        self.cond_proj = nn.Linear(geo_dim + phylo_dim, embed_dim, bias=False)

        # Fixed URIEL+ buffers (not trained — registered as buffers so they
        # move to the correct device with .to(device)).
        self.register_buffer(
            "geo_vecs", torch.tensor(geo_matrix, dtype=torch.float32))
        self.register_buffer(
            "phylo_vecs", torch.tensor(phylo_matrix, dtype=torch.float32))

        nn.init.normal_(self.lang_embeddings.weight, std=0.01)
        nn.init.normal_(self.value_embeddings.weight, std=0.01)

        if cond_mode == "init_only":
            # Seed lang_emb from geo+phylo once, then freeze cond_proj.
            nn.init.normal_(self.cond_proj.weight, std=0.01)
            with torch.no_grad():
                concat = torch.cat([self.geo_vecs, self.phylo_vecs], dim=-1)
                offsets = self.cond_proj(concat)           # (n_langs, d)
                self.lang_embeddings.weight.data.add_(offsets)
            for p in self.cond_proj.parameters():
                p.requires_grad = False
        else:
            # 'both', 'geo_only', 'phylo_only': zero-init → no-op at start.
            nn.init.zeros_(self.cond_proj.weight)
            # Pre-concatenate the (masked) conditioning input so _conditioned_lang_emb
            # only needs a single index op + linear instead of cat + zeros_like per batch.
            geo_t   = self.geo_vecs
            phylo_t = self.phylo_vecs
            if cond_mode == "geo_only":
                phylo_t = torch.zeros_like(phylo_t)
            elif cond_mode == "phylo_only":
                geo_t = torch.zeros_like(geo_t)
            self.register_buffer("cond_input", torch.cat([geo_t, phylo_t], dim=1))

        # Pre-compute padded value-ID tables for efficient batched softmax.
        max_n_vals = max(len(gids) for gids in feat_to_global_ids.values())
        n_features = len(feat_to_global_ids)
        padded_ids = torch.zeros(n_features, max_n_vals, dtype=torch.long)
        valid_mask = torch.zeros(n_features, max_n_vals, dtype=torch.bool)
        for fidx, gids in feat_to_global_ids.items():
            for j, gid in enumerate(gids):
                padded_ids[fidx, j] = gid
                valid_mask[fidx, j] = True
        self.register_buffer("padded_ids", padded_ids)
        self.register_buffer("valid_mask", valid_mask)

    # ------------------------------------------------------------------
    def _conditioned_lang_emb(self, lang_idx: torch.Tensor) -> torch.Tensor:
        """Return language embedding, optionally with URIEL+ conditioning offset."""
        lang_emb = self.lang_embeddings(lang_idx)                 # (B, d)
        if self.cond_mode == "init_only":
            # Conditioning was baked into lang_emb.weight at init; no runtime offset.
            return lang_emb
        # cond_input is pre-concatenated (and zero-masked for geo/phylo_only modes)
        # — one index op replaces two separate lookups + cat + zeros_like per batch.
        return lang_emb + self.cond_proj(self.cond_input[lang_idx])

    # ------------------------------------------------------------------
    def forward(
        self,
        lang_idx: torch.Tensor,
        feat_idx: torch.Tensor,
        value_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log P(true_value | lang, feat) and per-batch L2 penalty.

        Returns
        -------
        log_probs : (B,)
        batch_l2  : scalar
        """
        B = lang_idx.shape[0]
        lang_emb = self._conditioned_lang_emb(lang_idx)            # (B, d)

        batch_padded = self.padded_ids[feat_idx]                    # (B, max_n_vals)
        batch_mask = self.valid_mask[feat_idx]                      # (B, max_n_vals)
        val_embs = self.value_embeddings(batch_padded)              # (B, max_n_vals, d)

        scores = torch.bmm(
            lang_emb.unsqueeze(1),
            val_embs.transpose(1, 2),
        ).squeeze(1)
        scores = scores.masked_fill(~batch_mask, float("-inf"))
        log_probs = F.log_softmax(scores, dim=-1).gather(
            1, value_idx.unsqueeze(1)
        ).squeeze(1)

        # Per-batch L2 on accessed embeddings + conditioning weights.
        # Adam(weight_decay=0) is used; global decay would harm sparse embs.
        unique_global_ids = batch_padded[batch_mask].unique()
        accessed_val_embs = self.value_embeddings(unique_global_ids)
        batch_l2 = (
            lang_emb.pow(2).sum()
            + accessed_val_embs.pow(2).sum()
            + self.cond_proj.weight.pow(2).sum()
        ) / B

        return log_probs, batch_l2

    # ------------------------------------------------------------------
    def predict_feature(
        self, lang_idx: torch.Tensor, feat_idx: int
    ) -> torch.Tensor:
        """Return (N, K_f) value probabilities for given languages on one feature."""
        lang_emb = self._conditioned_lang_emb(lang_idx)
        gids_t = torch.tensor(
            self.feat_to_global_ids[feat_idx], device=lang_emb.device)
        val_embs = self.value_embeddings(gids_t)
        return F.softmax(lang_emb @ val_embs.T, dim=-1)

    def predict_all_features(self) -> Dict[int, torch.Tensor]:
        """Return {feat_idx: (n_langs, K_f) probabilities} for all features."""
        all_idx = torch.arange(
            self.lang_embeddings.num_embeddings,
            device=self.geo_vecs.device,
        )
        all_emb = self._conditioned_lang_emb(all_idx)              # (L, d)
        preds: Dict[int, torch.Tensor] = {}
        for fidx, gids in self.feat_to_global_ids.items():
            gids_t = torch.tensor(gids, device=all_emb.device)
            val_embs = self.value_embeddings(gids_t)
            preds[fidx] = F.softmax(all_emb @ val_embs.T, dim=-1)
        return preds


# ──────────────────────────────────────────────────────────────────────────────
# 3. Conditioned TCF Model (binary sigmoid)
# ──────────────────────────────────────────────────────────────────────────────

class TypologicalMF_TCF_Conditioned(nn.Module):
    """
    Binary TCF model (TypologicalMF from model.py) with URIEL+ conditioning.

    Uses the same conditioning architecture as TypologicalMF_Conditioned:
        conditioned_emb = lang_emb + cond_proj(concat(geo, phylo))

    Forward signature matches the binary TCF model:
        forward(lang_idx, feat_idx) → (preds, batch_l2)
    where preds = sigmoid(dot(conditioned_lang_emb, feat_emb)).

    Parameters
    ----------
    n_languages : int
    n_features : int
        Number of *binarised* feature columns (NOT original features).
    geo_matrix : np.ndarray, shape (n_languages, geo_dim)
    phylo_matrix : np.ndarray, shape (n_languages, phylo_dim)
    embed_dim : int
    cond_mode : str
        'both' | 'geo_only' | 'phylo_only' | 'init_only'  (see Learned variant)
    """

    def __init__(
        self,
        n_languages: int,
        n_features: int,
        geo_matrix: np.ndarray,
        phylo_matrix: np.ndarray,
        embed_dim: int = 64,
        cond_mode: str = "both",
    ) -> None:
        super().__init__()
        assert cond_mode in ("both", "geo_only", "phylo_only", "init_only"), \
            f"Unknown cond_mode: {cond_mode!r}"
        self.embed_dim = embed_dim
        self.cond_mode = cond_mode

        geo_dim = geo_matrix.shape[1]
        phylo_dim = phylo_matrix.shape[1]

        self.lang_embeddings = nn.Embedding(n_languages, embed_dim)
        self.feat_embeddings = nn.Embedding(n_features, embed_dim)
        self.cond_proj = nn.Linear(geo_dim + phylo_dim, embed_dim, bias=False)

        self.register_buffer(
            "geo_vecs", torch.tensor(geo_matrix, dtype=torch.float32))
        self.register_buffer(
            "phylo_vecs", torch.tensor(phylo_matrix, dtype=torch.float32))

        nn.init.normal_(self.lang_embeddings.weight, std=0.01)
        nn.init.normal_(self.feat_embeddings.weight, std=0.01)

        if cond_mode == "init_only":
            nn.init.normal_(self.cond_proj.weight, std=0.01)
            with torch.no_grad():
                concat = torch.cat([self.geo_vecs, self.phylo_vecs], dim=-1)
                offsets = self.cond_proj(concat)
                self.lang_embeddings.weight.data.add_(offsets)
            for p in self.cond_proj.parameters():
                p.requires_grad = False
        else:
            nn.init.zeros_(self.cond_proj.weight)
            geo_t   = self.geo_vecs
            phylo_t = self.phylo_vecs
            if cond_mode == "geo_only":
                phylo_t = torch.zeros_like(phylo_t)
            elif cond_mode == "phylo_only":
                geo_t = torch.zeros_like(geo_t)
            self.register_buffer("cond_input", torch.cat([geo_t, phylo_t], dim=1))

    # ------------------------------------------------------------------
    def _conditioned_lang_emb(self, lang_idx: torch.Tensor) -> torch.Tensor:
        lang_emb = self.lang_embeddings(lang_idx)
        if self.cond_mode == "init_only":
            return lang_emb
        return lang_emb + self.cond_proj(self.cond_input[lang_idx])

    # ------------------------------------------------------------------
    def forward(
        self,
        lang_idx: torch.Tensor,
        feat_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        preds    : (B,) sigmoid probabilities
        batch_l2 : scalar
        """
        lang_emb = self._conditioned_lang_emb(lang_idx)     # (B, d)
        feat_emb = self.feat_embeddings(feat_idx)             # (B, d)
        logits = (lang_emb * feat_emb).sum(dim=-1)
        preds = torch.sigmoid(logits)

        B = lang_idx.shape[0]
        batch_l2 = (
            lang_emb.pow(2).sum()
            + feat_emb.pow(2).sum()
            + self.cond_proj.weight.pow(2).sum()
        ) / B

        return preds, batch_l2

    def predict_all(self) -> torch.Tensor:
        """Return the full (n_languages × n_features) probability matrix."""
        all_idx = torch.arange(
            self.lang_embeddings.num_embeddings,
            device=self.geo_vecs.device,
        )
        all_emb = self._conditioned_lang_emb(all_idx)         # (L, d)
        logits = all_emb @ self.feat_embeddings.weight.T       # (L, F)
        return torch.sigmoid(logits)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Training loop (shared by both conditioned models)
# ──────────────────────────────────────────────────────────────────────────────

def train_conditioned(
    model: nn.Module,
    train_dataset,
    n_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    l2_reg: float = 0.1,
    device: str = "cpu",
    patience: int = 0,
) -> List[float]:
    """
    Train a conditioned model (Learned or TCF).

    Works with:
    - TypologicalMF_Conditioned  (dataset yields lang, feat, value triples)
    - TypologicalMF_TCF_Conditioned (dataset yields lang, feat, label pairs)

    Adam(weight_decay=0) per task spec.  Per-batch L2 is handled inside
    model.forward().

    Parameters
    ----------
    model : TypologicalMF_Conditioned or TypologicalMF_TCF_Conditioned
    train_dataset : CategoricalTypDataset or TypDataset
    n_epochs : int
    batch_size : int
    lr : float
    l2_reg : float
    device : str
    patience : int, 0 = no early stopping

    Returns
    -------
    losses : list of float — average NLL per epoch
    """
    model = model.to(device)
    is_categorical = hasattr(model, "value_embeddings")

    # JIT-compile the forward+backward graph for faster repeated passes.
    # Skipped for short runs (n_epochs < 10) where compile overhead > savings.
    if n_epochs >= 10 and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass  # eager fallback (compile unavailable or model not compilable)

    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    losses: List[float] = []
    best_loss = float("inf")
    no_improve = 0

    for epoch in range(n_epochs):
        epoch_nll = 0.0
        n_batches = 0
        model.train()

        for batch in loader:
            if is_categorical:
                lang_idx, feat_idx, val_idx = (t.to(device) for t in batch)
                log_probs, batch_l2 = model(lang_idx, feat_idx, val_idx)
                nll = -log_probs.mean()
            else:
                lang_idx, feat_idx, labels = (t.to(device) for t in batch)
                preds, batch_l2 = model(lang_idx, feat_idx)
                nll = F.binary_cross_entropy(preds, labels.float())

            loss = nll + (l2_reg / 2.0) * batch_l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_nll += nll.item()
            n_batches += 1

        avg = epoch_nll / n_batches
        losses.append(avg)

        if patience > 0:
            if avg < best_loss - 1e-6:
                best_loss = avg
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.info("Early stopping at epoch %d (patience=%d).",
                             epoch + 1, patience)
                    break

    return losses


# ──────────────────────────────────────────────────────────────────────────────
# 5. Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model_learned import CategoricalTypDataset

    torch.manual_seed(42)
    np.random.seed(42)

    N_LANG = 40
    GEO_DIM = 12
    PHYLO_DIM = 20
    EMBED_DIM = 16
    feature_cards = [3, 2, 4, 5]
    feat_to_global_ids: Dict[int, List[int]] = {}
    gid = 0
    for fi, card in enumerate(feature_cards):
        feat_to_global_ids[fi] = list(range(gid, gid + card))
        gid += card
    n_total_values = gid

    geo_mat = np.random.randn(N_LANG, GEO_DIM).astype(np.float32)
    phylo_mat = np.random.randn(N_LANG, PHYLO_DIM).astype(np.float32)

    cat_matrix = np.full((N_LANG, len(feature_cards)), -1, dtype=int)
    for fi, card in enumerate(feature_cards):
        for li in range(N_LANG):
            if np.random.rand() < 0.7:
                cat_matrix[li, fi] = np.random.randint(0, card)

    ls, fs, vs = zip(*[
        (li, fi, cat_matrix[li, fi])
        for li in range(N_LANG)
        for fi in range(len(feature_cards))
        if cat_matrix[li, fi] >= 0
    ])
    ds = CategoricalTypDataset(
        np.array(ls), np.array(fs), np.array(vs))

    # ── Test TypologicalMF_Conditioned ──
    model_c = TypologicalMF_Conditioned(
        N_LANG, n_total_values, feat_to_global_ids,
        geo_mat, phylo_mat, EMBED_DIM)

    # Conditioning should start as no-op
    with torch.no_grad():
        test_idx = torch.arange(5)
        base = model_c.lang_embeddings(test_idx)
        cond = model_c._conditioned_lang_emb(test_idx)
        assert torch.allclose(base, cond), "cond_proj initialised to non-zero!"

    losses_c = train_conditioned(model_c, ds, n_epochs=5, batch_size=16)
    assert len(losses_c) == 5 and all(np.isfinite(l) for l in losses_c)

    preds_c = model_c.predict_all_features()
    assert len(preds_c) == len(feature_cards)
    print(f"TypologicalMF_Conditioned smoke test passed.  "
          f"Final NLL={losses_c[-1]:.4f}")

    # ── Test TypologicalMF_TCF_Conditioned ──
    N_BINARY_FEATS = sum(feature_cards)
    model_tcf = TypologicalMF_TCF_Conditioned(
        N_LANG, N_BINARY_FEATS, geo_mat, phylo_mat, EMBED_DIM)

    with torch.no_grad():
        base_tcf = model_tcf.lang_embeddings(test_idx)
        cond_tcf = model_tcf._conditioned_lang_emb(test_idx)
        assert torch.allclose(base_tcf, cond_tcf), \
            "TCF cond_proj initialised to non-zero!"

    # synthetic binary dataset
    from torch.utils.data import TensorDataset
    bl = torch.randint(0, N_LANG, (200,))
    bf = torch.randint(0, N_BINARY_FEATS, (200,))
    by = torch.randint(0, 2, (200,)).float()
    tcf_ds = TensorDataset(bl, bf, by)

    losses_tcf = train_conditioned(model_tcf, tcf_ds, n_epochs=5, batch_size=16)
    assert len(losses_tcf) == 5 and all(np.isfinite(l) for l in losses_tcf)

    probs = model_tcf.predict_all()
    assert probs.shape == (N_LANG, N_BINARY_FEATS)
    print(f"TypologicalMF_TCF_Conditioned smoke test passed.  "
          f"Final BCE={losses_tcf[-1]:.4f}")

    print("\nAll smoke tests passed.")
