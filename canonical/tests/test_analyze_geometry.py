"""
Unit tests for canonical/analyze_geometry.py helpers.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CANONICAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CANONICAL_DIR))

from analyze_geometry import (
    cosine_normalise,
    feature_id_of,
    resolve_fv_id,
    probe_a_nearest_neighbours,
    probe_b_silhouette,
    probe_c_greenberg,
    _residual,
)


class TestResolveFvId:
    def test_direct_hit(self):
        assert resolve_fv_id({"81A=SOV": 5, "81A=SVO": 6}, "81A=SOV") == 5

    def test_tcf_fallback_eq1_to_bare(self):
        # T-CF: bare 'GB071' represents 'GB071=1' (the presence column)
        assert resolve_fv_id({"GB071": 12, "GB022": 13}, "GB071=1") == 12

    def test_tcf_eq0_returns_none(self):
        # T-CF has no row for the absence value
        assert resolve_fv_id({"GB071": 12}, "GB071=0") is None

    def test_unknown_label(self):
        assert resolve_fv_id({"GB071": 12}, "GB999=1") is None

    def test_eq1_prefers_literal_over_fallback(self):
        # If both 'X=1' and 'X' are present, the literal wins
        assert resolve_fv_id({"X=1": 10, "X": 99}, "X=1") == 10


class TestCosineNormalise:
    def test_unit_norm(self):
        M = np.random.RandomState(0).randn(5, 8)
        N = cosine_normalise(M)
        norms = np.linalg.norm(N, axis=1)
        assert np.allclose(norms, 1.0)

    def test_zero_row(self):
        M = np.array([[0., 0.], [3., 4.]])
        N = cosine_normalise(M)
        assert np.allclose(N[0], 0.0)
        assert np.allclose(np.linalg.norm(N[1]), 1.0)


class TestFeatureIdOf:
    def test_strips_value(self):
        assert feature_id_of("81A=SOV") == "81A"

    def test_no_equals_returns_self(self):
        # 2-valued T-CF features have no "=value" suffix
        assert feature_id_of("GB070") == "GB070"


class TestProbeA:
    def test_returns_top_k(self):
        rng = np.random.default_rng(0)
        M = rng.standard_normal((20, 8))
        fv2id = {f"F{i}=v": i for i in range(20)}
        nn = probe_a_nearest_neighbours(M, fv2id, ["F0=v", "F1=v"], top_k=5)
        assert set(nn.keys()) == {"F0=v", "F1=v"}
        for tgt, nbrs in nn.items():
            assert len(nbrs) == 5
            for r in nbrs:
                assert r["neighbour"] != tgt

    def test_skips_missing(self):
        M = np.zeros((3, 4))
        fv2id = {"A=x": 0, "B=y": 1, "C=z": 2}
        nn = probe_a_nearest_neighbours(M, fv2id, ["A=x", "MISSING=q"], 2)
        assert "A=x" in nn
        assert "MISSING=q" not in nn

    def test_neighbours_sorted_descending(self):
        rng = np.random.default_rng(42)
        M = rng.standard_normal((10, 8))
        fv2id = {f"F{i}=v": i for i in range(10)}
        nn = probe_a_nearest_neighbours(M, fv2id, ["F0=v"], top_k=5)
        sims = [r["cosine_sim"] for r in nn["F0=v"]]
        assert sims == sorted(sims, reverse=True)


class TestProbeB:
    def test_perfect_clusters_high_silhouette(self):
        # Build embeddings where same-feature values are tightly grouped
        # in clear clusters far from each other → silhouette near +1.
        rng = np.random.default_rng(0)
        M_list = []
        labels = []
        for fi in range(3):
            center = rng.standard_normal(8) * 10
            for vi in range(4):
                M_list.append(center + rng.standard_normal(8) * 0.01)
                labels.append(f"F{fi}=v{vi}")
        M = np.stack(M_list)
        fv2id = {lab: i for i, lab in enumerate(labels)}
        result = probe_b_silhouette(M, fv2id)
        assert result["silhouette_global"] > 0.5
        assert result["n_features_used"] == 3

    def test_singleton_features_excluded(self):
        # Feature G has only one value → must be excluded from silhouette.
        # Need ≥2 multi-value features for silhouette to be defined.
        M = np.random.RandomState(0).randn(7, 4)
        fv2id = {
            "F=a": 0, "F=b": 1, "F=c": 2,
            "G=only": 3,
            "H=x": 4, "H=y": 5, "H=z": 6,
        }
        result = probe_b_silhouette(M, fv2id)
        assert "G" not in result["per_feature"]
        assert "F" in result["per_feature"]
        assert "H" in result["per_feature"]


class TestProbeC:
    def test_residual_zero_for_identical_difference(self):
        # If (a_pos - a_neg) == (b_pos - b_neg), residual is zero
        emb = np.array([
            [1., 0.], [0., 0.],   # a_pos - a_neg = (1, 0)
            [3., 0.], [2., 0.],   # b_pos - b_neg = (1, 0)
        ])
        r = _residual(emb, 0, 1, 2, 3)
        assert r == pytest.approx(0.0)

    def test_residual_known_value(self):
        emb = np.array([
            [1., 0.], [0., 0.],   # diff_a = (1, 0)
            [0., 0.], [0., 1.],   # diff_b = (0, -1)
        ])
        # (1, 0) - (0, -1) = (1, 1) → norm = sqrt(2)
        r = _residual(emb, 0, 1, 2, 3)
        assert r == pytest.approx(np.sqrt(2))

    def test_skipped_when_target_missing(self):
        emb = np.zeros((3, 2))
        fv2id = {"A=1": 0, "A=0": 1, "B=1": 2}
        pairs = [{
            "name": "T", "a_pos": "A=1", "a_neg": "A=0",
            "b_pos": "B=1", "b_neg": "MISSING",
        }]
        out = probe_c_greenberg(emb, fv2id, pairs, n_baseline=10, seed=0)
        assert "skipped" in out[0]

    def test_p_value_in_range(self):
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((20, 8))
        fv2id = {f"X={i}": i for i in range(20)}
        pairs = [{
            "name": "rand", "a_pos": "X=0", "a_neg": "X=1",
            "b_pos": "X=2", "b_neg": "X=3",
        }]
        out = probe_c_greenberg(emb, fv2id, pairs, n_baseline=200, seed=42)
        assert 0.0 <= out[0]["empirical_p_value"] <= 1.0
        assert out[0]["baseline_p5"] <= out[0]["baseline_median"]
