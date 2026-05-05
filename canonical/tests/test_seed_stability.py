"""
Unit tests for canonical/seed_stability.py helpers.

All tests use synthetic data; no checkpoint files or CLDF repos are required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CANONICAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CANONICAL_DIR))

from seed_stability import (
    jaccard,
    pairwise_jaccard_top_k,
    silhouette_stability,
    greenberg_stability,
    procrustes_stability,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic checkpoints
# ---------------------------------------------------------------------------

def make_ckpt(seed: int, n_lang: int = 20, n_fv: int = 12, d: int = 4,
              database: str = "wals", architecture: str = "learned") -> dict:
    rng = np.random.default_rng(seed)
    lang_emb = rng.standard_normal((n_lang, d))
    fv_emb = rng.standard_normal((n_fv, d))
    # Three features with 4 values each
    fv2id = {f"F{fi}=v{vi}": fi * 4 + vi
             for fi in range(3) for vi in range(4)}
    return {
        "database": database,
        "architecture": architecture,
        "lang_emb": lang_emb,
        "fv_emb": fv_emb,
        "fv2id": fv2id,
        "lang2id": {f"gc{i:04d}": i for i in range(n_lang)},
        "config": {},
    }


def make_clustered_ckpt(seed: int, n_lang: int = 24, d: int = 8,
                        n_fv_per_feat: int = 3, n_feats: int = 4) -> dict:
    """Embeddings where same-feature values cluster tightly."""
    rng = np.random.default_rng(seed)
    fv2id = {}
    fv_emb_rows = []
    for fi in range(n_feats):
        center = rng.standard_normal(d) * 10
        for vi in range(n_fv_per_feat):
            fv2id[f"F{fi}=v{vi}"] = len(fv_emb_rows)
            fv_emb_rows.append(center + rng.standard_normal(d) * 0.05)
    fv_emb = np.stack(fv_emb_rows)
    lang_emb = rng.standard_normal((n_lang, d))
    return {
        "database": "wals",
        "architecture": "learned",
        "lang_emb": lang_emb,
        "fv_emb": fv_emb,
        "fv2id": fv2id,
        "lang2id": {f"gc{i:04d}": i for i in range(n_lang)},
        "config": {},
    }


# ---------------------------------------------------------------------------
# jaccard()
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_identical_sets(self):
        assert jaccard({"a", "b", "c"}, {"a", "b", "c"}) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        assert jaccard({"a"}, {"b"}) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # |{a,b} ∩ {b,c}| / |{a,b,c}| = 1/3
        assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_empty_sets(self):
        assert jaccard(set(), set()) == pytest.approx(1.0)

    def test_one_empty(self):
        assert jaccard({"a"}, set()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# pairwise_jaccard_top_k()
# ---------------------------------------------------------------------------

class TestPairwiseJaccard:
    def test_identical_checkpoints_jaccard_one(self):
        ckpt = make_ckpt(0, n_fv=12)
        targets = ["F0=v0", "F1=v0"]
        result = pairwise_jaccard_top_k([ckpt, ckpt], targets, top_k=3)
        for tgt, stats in result.items():
            assert stats["mean_jaccard"] == pytest.approx(1.0)

    def test_missing_target_skipped(self):
        ckpt1 = make_ckpt(0, n_fv=12)
        ckpt2 = make_ckpt(1, n_fv=12)
        # Manually insert a target that does NOT exist in either fv2id
        result = pairwise_jaccard_top_k([ckpt1, ckpt2], ["MISSING=x"], top_k=3)
        assert "MISSING=x" not in result

    def test_n_pairs_correct(self):
        ckpts = [make_ckpt(i, n_fv=12) for i in range(4)]
        targets = ["F0=v0"]
        result = pairwise_jaccard_top_k(ckpts, targets, top_k=3)
        # C(4,2) = 6 pairs
        assert result["F0=v0"]["n_pairs"] == 6

    def test_jaccard_in_range(self):
        ckpts = [make_ckpt(i, n_fv=12) for i in range(3)]
        targets = [f"F{fi}=v{vi}" for fi in range(3) for vi in range(4)]
        result = pairwise_jaccard_top_k(ckpts, targets, top_k=3)
        for tgt, stats in result.items():
            assert 0.0 <= stats["mean_jaccard"] <= 1.0


# ---------------------------------------------------------------------------
# silhouette_stability()
# ---------------------------------------------------------------------------

class TestSilhouetteStability:
    def test_returns_mean_std(self):
        ckpts = [make_clustered_ckpt(i) for i in range(3)]
        result = silhouette_stability(ckpts)
        assert "mean" in result and "std" in result
        assert result["mean"] is not None

    def test_per_seed_length(self):
        ckpts = [make_clustered_ckpt(i) for i in range(5)]
        result = silhouette_stability(ckpts)
        assert len(result["per_seed"]) == 5

    def test_stable_clusters_positive_mean(self):
        ckpts = [make_clustered_ckpt(i) for i in range(4)]
        result = silhouette_stability(ckpts)
        assert result["mean"] > 0.0


# ---------------------------------------------------------------------------
# greenberg_stability()
# ---------------------------------------------------------------------------

class TestGreenbergStability:
    def _make_greenberg_pairs(self):
        return [{
            "name": "TestPair",
            "a_pos": "F0=v0", "a_neg": "F0=v1",
            "b_pos": "F1=v0", "b_neg": "F1=v1",
        }]

    def test_residual_stats_returned(self):
        ckpts = [make_ckpt(i, n_fv=12) for i in range(3)]
        pairs = self._make_greenberg_pairs()
        result = greenberg_stability(ckpts, pairs, n_baseline=20, seed=0)
        assert len(result) == 1
        entry = result[0]
        assert "residual_mean" in entry
        assert "p_value_mean" in entry

    def test_missing_target_skipped(self):
        ckpts = [make_ckpt(i, n_fv=12) for i in range(2)]
        pairs = [{"name": "Skip",
                  "a_pos": "MISSING=x", "a_neg": "F0=v0",
                  "b_pos": "F0=v1", "b_neg": "F1=v0"}]
        result = greenberg_stability(ckpts, pairs, n_baseline=10, seed=0)
        assert result[0].get("skipped_seeds") is not None or result[0]["n_seeds"] == 0

    def test_n_seeds_matches(self):
        k = 4
        ckpts = [make_ckpt(i, n_fv=12) for i in range(k)]
        pairs = self._make_greenberg_pairs()
        result = greenberg_stability(ckpts, pairs, n_baseline=10, seed=0)
        assert result[0]["n_seeds"] == k


# ---------------------------------------------------------------------------
# procrustes_stability()
# ---------------------------------------------------------------------------

class TestProcrustesStability:
    def test_identical_checkpoints_zero_disparity(self):
        ckpt = make_ckpt(0, n_lang=30)
        result = procrustes_stability([ckpt, ckpt, ckpt])
        assert result["mean_disparity"] == pytest.approx(0.0, abs=1e-6)

    def test_n_pairs_correct(self):
        k = 5
        ckpts = [make_ckpt(i, n_lang=20) for i in range(k)]
        result = procrustes_stability(ckpts)
        assert result["n_pairs"] == k * (k - 1) // 2  # C(5,2) = 10

    def test_disparity_non_negative(self):
        ckpts = [make_ckpt(i, n_lang=25) for i in range(3)]
        result = procrustes_stability(ckpts)
        assert result["min_disparity"] >= 0.0

    def test_matrix_shape(self):
        k = 4
        ckpts = [make_ckpt(i, n_lang=20) for i in range(k)]
        result = procrustes_stability(ckpts)
        mat = np.array(result["disparity_matrix"])
        assert mat.shape == (k, k)

    def test_matrix_symmetric(self):
        k = 3
        ckpts = [make_ckpt(i, n_lang=20) for i in range(k)]
        result = procrustes_stability(ckpts)
        mat = np.array(result["disparity_matrix"])
        assert np.allclose(mat, mat.T, atol=1e-10)

    def test_different_seeds_positive_disparity(self):
        ckpts = [make_ckpt(i, n_lang=20) for i in range(3)]
        result = procrustes_stability(ckpts)
        # Distinct random matrices should not be identical
        assert result["mean_disparity"] > 0.0
