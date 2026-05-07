"""
Tests for canonical/conditioning_model.py (Phase 7 — RQ4)

Covers:
  - TypologicalMF_Conditioned: init, no-op start, forward shapes, predict_*
  - TypologicalMF_TCF_Conditioned: same
  - load_uriel_vectors_for_lang2id: alignment correctness, missing fallback
  - train_conditioned: loss is finite, conditioning weights move off zero
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from canonical.conditioning_model import (
    TypologicalMF_Conditioned,
    TypologicalMF_TCF_Conditioned,
    load_uriel_vectors_for_lang2id,
    train_conditioned,
)
from model_learned import CategoricalTypDataset


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

N_LANG = 30
GEO_DIM = 8
PHYLO_DIM = 12
EMBED_DIM = 16
FEATURE_CARDS = [3, 2, 4]


def _make_feat_global_ids():
    feat_to_global_ids: Dict[int, List[int]] = {}
    gid = 0
    for fi, card in enumerate(FEATURE_CARDS):
        feat_to_global_ids[fi] = list(range(gid, gid + card))
        gid += card
    return feat_to_global_ids, gid


def _make_geo_phylo():
    rng = np.random.default_rng(0)
    geo = rng.standard_normal((N_LANG, GEO_DIM)).astype(np.float32)
    phylo = rng.standard_normal((N_LANG, PHYLO_DIM)).astype(np.float32)
    return geo, phylo


def _make_cat_dataset(feat_to_global_ids):
    rng = np.random.default_rng(1)
    n_feats = len(FEATURE_CARDS)
    cat = np.full((N_LANG, n_feats), -1, dtype=int)
    for fi, card in enumerate(FEATURE_CARDS):
        mask = rng.random(N_LANG) < 0.7
        cat[mask, fi] = rng.integers(0, card, mask.sum())
    ls, fs, vs = zip(*[
        (li, fi, cat[li, fi])
        for li in range(N_LANG) for fi in range(n_feats)
        if cat[li, fi] >= 0
    ])
    return CategoricalTypDataset(np.array(ls), np.array(fs), np.array(vs))


def _make_tcf_dataset():
    rng = np.random.default_rng(2)
    n_binary = sum(FEATURE_CARDS)
    bl = torch.randint(0, N_LANG, (200,))
    bf = torch.randint(0, n_binary, (200,))
    by = torch.from_numpy(rng.integers(0, 2, 200).astype(np.float32))
    return TensorDataset(bl, bf, by), n_binary


# ──────────────────────────────────────────────────────────────────────────────
# TypologicalMF_Conditioned
# ──────────────────────────────────────────────────────────────────────────────

class TestTypologicalMFConditioned:

    def _model(self):
        feat_to_global_ids, n_total = _make_feat_global_ids()
        geo, phylo = _make_geo_phylo()
        return TypologicalMF_Conditioned(
            N_LANG, n_total, feat_to_global_ids, geo, phylo, EMBED_DIM)

    def test_instantiation(self):
        m = self._model()
        assert m.lang_embeddings.weight.shape == (N_LANG, EMBED_DIM)
        assert m.cond_proj.weight.shape == (EMBED_DIM, GEO_DIM + PHYLO_DIM)

    def test_cond_proj_initialised_zero(self):
        m = self._model()
        assert torch.all(m.cond_proj.weight == 0), \
            "cond_proj must start at zero (conditioning is a no-op at init)"

    def test_conditioning_is_noop_at_init(self):
        m = self._model()
        with torch.no_grad():
            idx = torch.arange(5)
            base = m.lang_embeddings(idx)
            conditioned = m._conditioned_lang_emb(idx)
        assert torch.allclose(base, conditioned), \
            "Conditioned embedding must equal base embedding when cond_proj=0"

    def test_forward_shapes(self):
        m = self._model()
        B = 8
        lang_idx = torch.randint(0, N_LANG, (B,))
        feat_idx = torch.randint(0, len(FEATURE_CARDS), (B,))
        val_idx = torch.zeros(B, dtype=torch.long)
        log_probs, batch_l2 = m(lang_idx, feat_idx, val_idx)
        assert log_probs.shape == (B,)
        assert batch_l2.shape == ()

    def test_log_probs_non_positive(self):
        m = self._model()
        B = 16
        lang_idx = torch.randint(0, N_LANG, (B,))
        feat_idx = torch.randint(0, len(FEATURE_CARDS), (B,))
        val_idx = torch.zeros(B, dtype=torch.long)
        log_probs, _ = m(lang_idx, feat_idx, val_idx)
        assert (log_probs <= 0).all(), "log-softmax must be ≤ 0"

    def test_predict_feature_shape(self):
        m = self._model()
        lang_idx = torch.arange(N_LANG)
        probs = m.predict_feature(lang_idx, feat_idx=0)
        assert probs.shape == (N_LANG, FEATURE_CARDS[0])
        assert torch.allclose(probs.sum(dim=-1), torch.ones(N_LANG), atol=1e-5)

    def test_predict_all_features_shape(self):
        m = self._model()
        preds = m.predict_all_features()
        assert len(preds) == len(FEATURE_CARDS)
        for fi, card in enumerate(FEATURE_CARDS):
            assert preds[fi].shape == (N_LANG, card)

    def test_training_updates_conditioning(self):
        feat_to_global_ids, n_total = _make_feat_global_ids()
        geo, phylo = _make_geo_phylo()
        m = TypologicalMF_Conditioned(
            N_LANG, n_total, feat_to_global_ids, geo, phylo, EMBED_DIM)
        ds = _make_cat_dataset(feat_to_global_ids)

        before = m.cond_proj.weight.detach().clone()
        train_conditioned(m, ds, n_epochs=3, batch_size=16, lr=1e-2)
        after = m.cond_proj.weight.detach()

        assert not torch.allclose(before, after), \
            "cond_proj.weight must move from zero during training"

    def test_training_losses_finite(self):
        feat_to_global_ids, n_total = _make_feat_global_ids()
        geo, phylo = _make_geo_phylo()
        m = TypologicalMF_Conditioned(
            N_LANG, n_total, feat_to_global_ids, geo, phylo, EMBED_DIM)
        ds = _make_cat_dataset(feat_to_global_ids)
        losses = train_conditioned(m, ds, n_epochs=3, batch_size=16)
        assert all(np.isfinite(l) for l in losses), \
            f"Non-finite loss encountered: {losses}"

    def test_geo_phylo_buffers_not_trainable(self):
        m = self._model()
        param_names = {n for n, _ in m.named_parameters()}
        assert "geo_vecs" not in param_names
        assert "phylo_vecs" not in param_names


# ──────────────────────────────────────────────────────────────────────────────
# TypologicalMF_TCF_Conditioned
# ──────────────────────────────────────────────────────────────────────────────

class TestTypologicalMFTCFConditioned:

    def _model(self):
        n_binary = sum(FEATURE_CARDS)
        geo, phylo = _make_geo_phylo()
        return TypologicalMF_TCF_Conditioned(
            N_LANG, n_binary, geo, phylo, EMBED_DIM)

    def test_instantiation(self):
        m = self._model()
        assert m.lang_embeddings.weight.shape == (N_LANG, EMBED_DIM)
        assert m.feat_embeddings.weight.shape == (sum(FEATURE_CARDS), EMBED_DIM)
        assert m.cond_proj.weight.shape == (EMBED_DIM, GEO_DIM + PHYLO_DIM)

    def test_cond_proj_initialised_zero(self):
        m = self._model()
        assert torch.all(m.cond_proj.weight == 0)

    def test_conditioning_is_noop_at_init(self):
        m = self._model()
        with torch.no_grad():
            idx = torch.arange(5)
            base = m.lang_embeddings(idx)
            conditioned = m._conditioned_lang_emb(idx)
        assert torch.allclose(base, conditioned)

    def test_forward_shapes(self):
        m = self._model()
        B = 8
        n_binary = sum(FEATURE_CARDS)
        lang_idx = torch.randint(0, N_LANG, (B,))
        feat_idx = torch.randint(0, n_binary, (B,))
        preds, batch_l2 = m(lang_idx, feat_idx)
        assert preds.shape == (B,)
        assert batch_l2.shape == ()

    def test_preds_in_01(self):
        m = self._model()
        B = 20
        n_binary = sum(FEATURE_CARDS)
        lang_idx = torch.randint(0, N_LANG, (B,))
        feat_idx = torch.randint(0, n_binary, (B,))
        preds, _ = m(lang_idx, feat_idx)
        assert ((preds >= 0) & (preds <= 1)).all()

    def test_predict_all_shape(self):
        m = self._model()
        probs = m.predict_all()
        assert probs.shape == (N_LANG, sum(FEATURE_CARDS))

    def test_training_updates_conditioning(self):
        n_binary = sum(FEATURE_CARDS)
        geo, phylo = _make_geo_phylo()
        m = TypologicalMF_TCF_Conditioned(
            N_LANG, n_binary, geo, phylo, EMBED_DIM)
        ds, _ = _make_tcf_dataset()

        before = m.cond_proj.weight.detach().clone()
        train_conditioned(m, ds, n_epochs=3, batch_size=16, lr=1e-2)
        after = m.cond_proj.weight.detach()

        assert not torch.allclose(before, after)

    def test_training_losses_finite(self):
        m = self._model()
        ds, _ = _make_tcf_dataset()
        losses = train_conditioned(m, ds, n_epochs=3, batch_size=16)
        assert all(np.isfinite(l) for l in losses)


# ──────────────────────────────────────────────────────────────────────────────
# load_uriel_vectors_for_lang2id
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadUrielVectors:

    def _make_parquet(self, tmpdir, glottocodes, geo_dim, phylo_dim, rng=None):
        import pyarrow as pa
        import pyarrow.parquet as pq

        if rng is None:
            rng = np.random.default_rng(99)
        geo_rows = [rng.standard_normal(geo_dim).astype(np.float32).tolist()
                    for _ in glottocodes]
        phylo_rows = [rng.standard_normal(phylo_dim).astype(np.float32).tolist()
                      for _ in glottocodes]
        table = pa.table({
            "glottocode": pa.array(glottocodes, type=pa.string()),
            "geo_vec": pa.array(geo_rows, type=pa.list_(pa.float32())),
            "phylo_vec": pa.array(phylo_rows, type=pa.list_(pa.float32())),
        })
        path = str(Path(tmpdir) / "vectors.parquet")
        pq.write_table(table, path)
        return path, geo_rows, phylo_rows

    def test_alignment_correct(self, tmp_path):
        gcs = ["aaaa1234", "bbbb5678", "cccc9012"]
        lang2id = {"aaaa1234": 2, "bbbb5678": 0, "cccc9012": 1}
        pq_path, geo_rows, phylo_rows = self._make_parquet(
            tmp_path, gcs, GEO_DIM, PHYLO_DIM)

        geo_mat, phylo_mat, meta = load_uriel_vectors_for_lang2id(
            pq_path, lang2id, GEO_DIM, PHYLO_DIM)

        # lang2id["aaaa1234"] = 2 → row 2 of geo_mat should be geo_rows[0]
        np.testing.assert_array_almost_equal(
            geo_mat[2], geo_rows[0], decimal=5)
        np.testing.assert_array_almost_equal(
            geo_mat[0], geo_rows[1], decimal=5)
        np.testing.assert_array_almost_equal(
            geo_mat[1], geo_rows[2], decimal=5)

        assert meta["found"] == 3
        assert meta["missing"] == 0

    def test_missing_gets_zero_vector(self, tmp_path):
        gcs = ["aaaa1234"]
        lang2id = {"aaaa1234": 0, "xxxx9999": 1}
        pq_path, _, _ = self._make_parquet(tmp_path, gcs, GEO_DIM, PHYLO_DIM)

        geo_mat, phylo_mat, meta = load_uriel_vectors_for_lang2id(
            pq_path, lang2id, GEO_DIM, PHYLO_DIM)

        assert meta["missing"] == 1
        np.testing.assert_array_equal(geo_mat[1], np.zeros(GEO_DIM))
        np.testing.assert_array_equal(phylo_mat[1], np.zeros(PHYLO_DIM))

    def test_output_shapes(self, tmp_path):
        n = 10
        gcs = [f"lang{i:04d}" for i in range(n)]
        lang2id = {gc: i for i, gc in enumerate(gcs)}
        pq_path, _, _ = self._make_parquet(
            tmp_path, gcs, GEO_DIM, PHYLO_DIM)

        geo_mat, phylo_mat, meta = load_uriel_vectors_for_lang2id(
            pq_path, lang2id, GEO_DIM, PHYLO_DIM)

        assert geo_mat.shape == (n, GEO_DIM)
        assert phylo_mat.shape == (n, PHYLO_DIM)
        assert meta["pct_found"] == 100.0
