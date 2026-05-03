"""
Unit tests for canonical/compare_databases.py helpers.

All tests use synthetic embeddings; no CLDF repos are required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CANONICAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CANONICAL_DIR))

from compare_databases import (
    build_shared_index,
    procrustes_align,
    permutation_test_procrustes,
    cca_correlations,
    family_preservation_probe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_embeddings(n: int, d: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


# ---------------------------------------------------------------------------
# Step 1 — build_shared_index
# ---------------------------------------------------------------------------

class TestBuildSharedIndex:
    def test_intersection_correct(self):
        a = {"lang0001": 0, "lang0002": 1, "lang0003": 2}
        b = {"lang0002": 0, "lang0003": 1, "lang0004": 2}
        shared, idx_a, idx_b = build_shared_index(a, b)
        assert shared == ["lang0002", "lang0003"]
        assert list(idx_a) == [1, 2]
        assert list(idx_b) == [0, 1]

    def test_empty_intersection(self):
        a = {"x": 0}
        b = {"y": 0}
        shared, idx_a, idx_b = build_shared_index(a, b)
        assert shared == []
        assert len(idx_a) == 0
        assert len(idx_b) == 0

    def test_identical_dicts(self):
        a = {"a": 0, "b": 1, "c": 2}
        shared, idx_a, idx_b = build_shared_index(a, a)
        assert shared == ["a", "b", "c"]
        assert np.array_equal(idx_a, idx_b)

    def test_sorted_output(self):
        a = {"z": 0, "a": 1, "m": 2}
        b = {"m": 0, "z": 1, "a": 2}
        shared, _, _ = build_shared_index(a, b)
        assert shared == sorted(shared)

    def test_index_dtypes(self):
        a = {"g1": 0, "g2": 1}
        b = {"g1": 3, "g2": 4}
        _, idx_a, idx_b = build_shared_index(a, b)
        assert idx_a.dtype == np.int64
        assert idx_b.dtype == np.int64


# ---------------------------------------------------------------------------
# Step 2 — procrustes_align
# ---------------------------------------------------------------------------

class TestProcrustesAlign:
    def test_identical_matrices_zero_disparity(self):
        A = make_embeddings(20, 8)
        disparity, _, _ = procrustes_align(A, A.copy())
        assert disparity == pytest.approx(0.0, abs=1e-6)

    def test_disparity_non_negative(self):
        A = make_embeddings(20, 8, seed=1)
        B = make_embeddings(20, 8, seed=2)
        disparity, _, _ = procrustes_align(A, B)
        assert disparity >= 0.0

    def test_disparity_bounded(self):
        # Standardised Procrustes disparity is in [0, 2].
        A = make_embeddings(30, 16, seed=3)
        B = make_embeddings(30, 16, seed=4)
        d, _, _ = procrustes_align(A, B)
        assert 0.0 <= d <= 2.0 + 1e-9

    def test_rotation_does_not_increase_disparity(self):
        # Rotating B by a random orthogonal matrix should not change disparity.
        rng = np.random.default_rng(5)
        A = make_embeddings(20, 8)
        B = make_embeddings(20, 8, seed=6)
        Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
        d_orig, _, _ = procrustes_align(A, B)
        d_rot, _, _ = procrustes_align(A, B @ Q)
        assert d_orig == pytest.approx(d_rot, abs=1e-6)

    def test_aligned_matrix_shape(self):
        A = make_embeddings(15, 6)
        B = make_embeddings(15, 6, seed=7)
        _, A_std, B_aligned = procrustes_align(A, B)
        assert A_std.shape == A.shape
        assert B_aligned.shape == B.shape


# ---------------------------------------------------------------------------
# Step 3 — permutation_test_procrustes
# ---------------------------------------------------------------------------

class TestPermutationTest:
    def test_p_value_in_range(self):
        A = make_embeddings(30, 8)
        B = make_embeddings(30, 8, seed=1)
        _, null, p = permutation_test_procrustes(A, B, n_perm=50, seed=0)
        assert 0.0 <= p <= 1.0

    def test_null_length(self):
        A = make_embeddings(20, 8)
        B = make_embeddings(20, 8, seed=2)
        _, null, _ = permutation_test_procrustes(A, B, n_perm=100, seed=0)
        assert len(null) == 100

    def test_identical_matrices_low_p(self):
        # Two identical matrices → observed disparity ≈ 0 → very low p.
        A = make_embeddings(40, 16)
        _, _, p = permutation_test_procrustes(A, A.copy(), n_perm=200, seed=7)
        assert p < 0.05

    def test_reproducible(self):
        A = make_embeddings(20, 8)
        B = make_embeddings(20, 8, seed=3)
        _, null1, p1 = permutation_test_procrustes(A, B, n_perm=50, seed=99)
        _, null2, p2 = permutation_test_procrustes(A, B, n_perm=50, seed=99)
        assert np.array_equal(null1, null2)
        assert p1 == p2


# ---------------------------------------------------------------------------
# Step 4 — cca_correlations
# ---------------------------------------------------------------------------

class TestCCACorrelations:
    def test_returns_correct_n_components(self):
        A = make_embeddings(50, 8)
        B = make_embeddings(50, 8, seed=1)
        corrs = cca_correlations(A, B, n_components=5)
        assert len(corrs) == 5

    def test_capped_at_min_dim(self):
        # d=4, request 10 → should return 4
        A = make_embeddings(50, 4)
        B = make_embeddings(50, 6, seed=2)
        corrs = cca_correlations(A, B, n_components=10)
        assert len(corrs) == 4

    def test_correlations_in_range(self):
        A = make_embeddings(60, 8)
        B = make_embeddings(60, 8, seed=3)
        corrs = cca_correlations(A, B, n_components=4)
        for c in corrs:
            assert -1.0 - 1e-6 <= c <= 1.0 + 1e-6

    def test_perfect_linear_map_high_correlation(self):
        # B = A @ R for a random orthogonal R → CCA should give perfect corr.
        rng = np.random.default_rng(42)
        A = rng.standard_normal((80, 8))
        Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
        B = A @ Q
        corrs = cca_correlations(A, B, n_components=3)
        for c in corrs:
            assert abs(c) > 0.9, f"Expected high correlation, got {c}"


# ---------------------------------------------------------------------------
# Step 5 — family_preservation_probe
# ---------------------------------------------------------------------------

class TestFamilyPreservationProbe:
    def _make_family_data(self, n_langs=40, n_fams=4, d=8, seed=0):
        """
        Create embeddings where same-family languages cluster tightly.
        Returns (glottocodes, emb_a, emb_b, family_map).
        """
        rng = np.random.default_rng(seed)
        glottocodes = [f"gc{i:04d}" for i in range(n_langs)]
        families = [f"F{i % n_fams}" for i in range(n_langs)]
        family_map = dict(zip(glottocodes, families))

        centers = rng.standard_normal((n_fams, d)) * 10
        emb_a = np.array([
            centers[i % n_fams] + rng.standard_normal(d) * 0.05
            for i in range(n_langs)
        ])
        emb_b = np.array([
            centers[i % n_fams] + rng.standard_normal(d) * 0.05
            for i in range(n_langs)
        ])
        return glottocodes, emb_a, emb_b, family_map

    def test_perfect_clustering_high_score(self):
        gc, emb_a, emb_b, fmap = self._make_family_data()
        result = family_preservation_probe(
            gc, emb_a, emb_b, fmap, top_k=5, n_baseline=50, seed=0)
        assert result["score_a"] > 0.5
        assert result["score_b"] > 0.5

    def test_low_p_for_clustered_data(self):
        gc, emb_a, emb_b, fmap = self._make_family_data(n_langs=40, n_fams=4, d=8)
        result = family_preservation_probe(
            gc, emb_a, emb_b, fmap, top_k=5, n_baseline=100, seed=1)
        assert result["p_value_a"] < 0.05

    def test_returns_note_for_no_families(self):
        gc = ["g0", "g1", "g2"]
        emb = make_embeddings(3, 4)
        fmap = {"g0": "F0"}  # only one member → no valid language
        result = family_preservation_probe(gc, emb, emb, fmap,
                                           top_k=2, n_baseline=10, seed=0)
        assert "note" in result

    def test_output_keys(self):
        gc, emb_a, emb_b, fmap = self._make_family_data(n_langs=20, n_fams=2)
        result = family_preservation_probe(
            gc, emb_a, emb_b, fmap, top_k=3, n_baseline=20, seed=0)
        for key in ("n_valid_languages", "n_families", "score_a", "score_b",
                    "baseline_mean", "baseline_p95", "p_value_a", "p_value_b"):
            assert key in result, f"missing key: {key}"
