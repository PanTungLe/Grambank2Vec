"""
Integration tests for the spatiophylogenetic basis wiring into conditioning_pipeline.

Uses a 20-language toy dataset with synthetic geography and classification,
builds the eigenvector basis, and verifies forward passes of both model classes
produce finite losses — no real CLDF data or trained checkpoints required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from canonical.spatiophylo_basis import (
    build_spatiophylo_basis,
    phylo_covariance_from_classification,
)

N_LANGS  = 20
N_FEATS  = 6
N_VALUES = 3  # values per feature (for Learned model)


# ── Toy data fixtures ───────────────────────────────────────────────────────────

def _make_toy_inputs():
    rng = np.random.default_rng(0)
    gcs = [f"lang{i:03d}" for i in range(N_LANGS)]
    lang2id = {gc: i for i, gc in enumerate(gcs)}

    # Three geographic clusters
    centres = np.array([[10.0, 20.0], [-15.0, 130.0], [50.0, -80.0]])
    coords = {
        gc: tuple(centres[i % 3] + rng.normal(0, 4, size=2))
        for i, gc in enumerate(gcs)
    }

    # Three families, each with two genera
    classification = {
        gc: [
            f"fam{i % 3}",
            f"fam{i % 3}.gen{(i // 3) % 2}",
            gc,
        ]
        for i, gc in enumerate(gcs)
    }

    phylo_cov = phylo_covariance_from_classification(classification, gcs)
    phylo_E, spatial_E, meta = build_spatiophylo_basis(
        lang2id, coords, phylo_cov, gcs, var_target=0.90)

    return lang2id, phylo_E, spatial_E, meta


# ── Tests ───────────────────────────────────────────────────────────────────────

class TestSpatiophyloBasisShape:

    def test_output_shapes_aligned_to_lang2id(self):
        lang2id, phylo_E, spatial_E, meta = _make_toy_inputs()
        assert phylo_E.shape[0] == N_LANGS
        assert spatial_E.shape[0] == N_LANGS
        assert phylo_E.dtype == np.float32
        assert spatial_E.dtype == np.float32

    def test_meta_coverage_fields(self):
        lang2id, phylo_E, spatial_E, meta = _make_toy_inputs()
        assert 0.0 <= meta["phylo_coverage"] <= 1.0
        assert 0.0 <= meta["spatial_coverage"] <= 1.0
        assert meta["phylo_dim"] == phylo_E.shape[1]
        assert meta["spatial_dim"] == spatial_E.shape[1]

    def test_full_coverage(self):
        lang2id, phylo_E, spatial_E, meta = _make_toy_inputs()
        # All languages have coords and are in the tree → expect full coverage
        assert meta["phylo_coverage"] == pytest.approx(1.0)
        assert meta["spatial_coverage"] == pytest.approx(1.0)


class TestLearnedModelForwardPass:
    """TypologicalMF_Conditioned with spatiophylo matrices → finite loss."""

    def test_forward_pass_both_mode(self):
        from canonical.conditioning_model import (
            TypologicalMF_Conditioned,
            train_conditioned,
        )
        lang2id, phylo_E, spatial_E, _ = _make_toy_inputs()

        # Build a minimal feat_to_global_ids: N_FEATS features, N_VALUES each
        feat_to_global_ids = {
            fi: list(range(fi * N_VALUES, (fi + 1) * N_VALUES))
            for fi in range(N_FEATS)
        }
        n_total_values = N_FEATS * N_VALUES

        model = TypologicalMF_Conditioned(
            N_LANGS, n_total_values, feat_to_global_ids,
            geo_matrix=spatial_E,
            phylo_matrix=phylo_E,
            embed_dim=16,
            cond_mode="both",
        )

        # Synthetic training triples: one (lang, feat, value) per cell
        rng = np.random.default_rng(1)
        n_obs = N_LANGS * N_FEATS
        langs  = np.repeat(np.arange(N_LANGS), N_FEATS)
        feats  = np.tile(np.arange(N_FEATS), N_LANGS)
        values = rng.integers(0, N_VALUES, size=n_obs)

        from torch.utils.data import TensorDataset
        ds = TensorDataset(
            torch.LongTensor(langs),
            torch.LongTensor(feats),
            torch.LongTensor(values),
        )

        # One epoch of training must not error and must produce finite loss
        train_conditioned(model, ds, n_epochs=1, batch_size=32,
                          lr=1e-3, l2_reg=0.01, device="cpu", patience=0)

        model.eval()
        with torch.no_grad():
            emb = model.lang_embeddings.weight
        assert torch.isfinite(emb).all(), "lang embeddings contain non-finite values"

    def test_geo_only_mode(self):
        from canonical.conditioning_model import TypologicalMF_Conditioned
        lang2id, phylo_E, spatial_E, _ = _make_toy_inputs()
        feat_to_global_ids = {fi: [fi] for fi in range(N_FEATS)}
        model = TypologicalMF_Conditioned(
            N_LANGS, N_FEATS, feat_to_global_ids,
            geo_matrix=spatial_E,
            phylo_matrix=phylo_E,
            embed_dim=16,
            cond_mode="geo_only",
        )
        with torch.no_grad():
            emb = model.lang_embeddings.weight
        assert emb.shape == (N_LANGS, 16)

    def test_phylo_only_mode(self):
        from canonical.conditioning_model import TypologicalMF_Conditioned
        lang2id, phylo_E, spatial_E, _ = _make_toy_inputs()
        feat_to_global_ids = {fi: [fi] for fi in range(N_FEATS)}
        model = TypologicalMF_Conditioned(
            N_LANGS, N_FEATS, feat_to_global_ids,
            geo_matrix=spatial_E,
            phylo_matrix=phylo_E,
            embed_dim=16,
            cond_mode="phylo_only",
        )
        with torch.no_grad():
            emb = model.lang_embeddings.weight
        assert emb.shape == (N_LANGS, 16)


class TestTCFModelForwardPass:
    """TypologicalMF_TCF_Conditioned with spatiophylo matrices → finite loss."""

    def test_forward_pass_both_mode(self):
        from canonical.conditioning_model import (
            TypologicalMF_TCF_Conditioned,
            train_conditioned,
        )
        lang2id, phylo_E, spatial_E, _ = _make_toy_inputs()

        n_binary = N_FEATS * N_VALUES
        model = TypologicalMF_TCF_Conditioned(
            N_LANGS, n_binary,
            geo_matrix=spatial_E,
            phylo_matrix=phylo_E,
            embed_dim=16,
            cond_mode="both",
        )

        # Synthetic binary triples (observed = 0 or 1)
        rng = np.random.default_rng(2)
        n_obs = N_LANGS * n_binary
        langs  = np.repeat(np.arange(N_LANGS), n_binary)
        feats  = np.tile(np.arange(n_binary), N_LANGS)
        values = rng.integers(0, 2, size=n_obs).astype(np.float32)

        from torch.utils.data import TensorDataset
        ds = TensorDataset(
            torch.LongTensor(langs),
            torch.LongTensor(feats),
            torch.FloatTensor(values),
        )

        train_conditioned(model, ds, n_epochs=1, batch_size=32,
                          lr=1e-3, l2_reg=0.01, device="cpu", patience=0)

        model.eval()
        with torch.no_grad():
            emb = model.lang_embeddings.weight
        assert torch.isfinite(emb).all(), "lang embeddings contain non-finite values"

    def test_cond_mode_attribute_stored(self):
        from canonical.conditioning_model import TypologicalMF_TCF_Conditioned
        lang2id, phylo_E, spatial_E, _ = _make_toy_inputs()
        for mode in ("both", "geo_only", "phylo_only", "init_only"):
            m = TypologicalMF_TCF_Conditioned(
                N_LANGS, N_FEATS, spatial_E, phylo_E,
                embed_dim=16, cond_mode=mode)
            assert m.cond_mode == mode
