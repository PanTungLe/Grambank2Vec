"""
Tests for canonical/uriel_plus_loader.py.

No URIEL+ installation or network access required — all heavy ops are patched.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from canonical.uriel_plus_loader import run_loader

GEO_DIM   = 4
PHYLO_DIM = 6

_FOUND_GCS   = ["aari1239", "abkh1244"]
_MISSING_GCS = ["xxxx9999"]
_ALL_GCS     = _FOUND_GCS + _MISSING_GCS


def _fake_geo():
    return {gc: np.ones(GEO_DIM, dtype=np.float32) * 0.5 for gc in _FOUND_GCS}


def _fake_phylo():
    return {gc: np.ones(PHYLO_DIM, dtype=np.float32) * 0.3 for gc in _FOUND_GCS}


def _run_loader_patched(tmp_path: Path, missing: list[str] | None = None) -> Path:
    """
    Call run_loader with _collect_glottocodes, _clone_or_verify,
    _load_uriel_plus, and _extract_raw_vectors all patched.

    Returns the path to the saved parquet.
    """
    if missing is None:
        missing = _MISSING_GCS
    found = [gc for gc in _ALL_GCS if gc not in missing]
    geo   = {gc: np.ones(GEO_DIM,   dtype=np.float32) * 0.5 for gc in found}
    phylo = {gc: np.ones(PHYLO_DIM, dtype=np.float32) * 0.3 for gc in found}

    with (
        patch("canonical.uriel_plus_loader._collect_glottocodes",
              return_value=_ALL_GCS),
        patch("canonical.uriel_plus_loader._clone_or_verify"),
        patch("canonical.uriel_plus_loader._load_uriel_plus",
              return_value=None),
        patch("canonical.uriel_plus_loader._extract_raw_vectors",
              return_value=(geo, phylo, found, missing)),
    ):
        run_loader(output_dir=str(tmp_path), urielplus_dir="/fake",
                   smoke=False, seed=42)

    return tmp_path / "uriel_plus_vectors.parquet"


# ──────────────────────────────────────────────────────────────────────────────
# Schema and zero-vector tests
# ──────────────────────────────────────────────────────────────────────────────

class TestZeroVectorFallback:

    def test_parquet_schema_has_required_columns(self, tmp_path):
        """Output parquet must have exactly glottocode, geo_vec, phylo_vec, has_geo, has_phylo."""
        out = _run_loader_patched(tmp_path)
        schema_names = set(pq.read_table(str(out)).schema.names)
        assert schema_names == {"glottocode", "geo_vec", "phylo_vec",
                                "has_geo", "has_phylo"}

    def test_missing_language_gets_zero_geo_vector(self, tmp_path):
        """Languages absent from URIEL+ must get an all-zero geo vector."""
        out = _run_loader_patched(tmp_path)
        df  = pq.read_table(str(out)).to_pandas()
        row = df[df["glottocode"] == "xxxx9999"].iloc[0]
        assert np.allclose(np.array(row["geo_vec"]), 0.0), \
            "Missing language must receive a zero geo vector"

    def test_missing_language_gets_zero_phylo_vector(self, tmp_path):
        """Languages absent from URIEL+ must get an all-zero phylo vector."""
        out = _run_loader_patched(tmp_path)
        df  = pq.read_table(str(out)).to_pandas()
        row = df[df["glottocode"] == "xxxx9999"].iloc[0]
        assert np.allclose(np.array(row["phylo_vec"]), 0.0), \
            "Missing language must receive a zero phylo vector"

    def test_missing_language_has_geo_false(self, tmp_path):
        """Languages absent from URIEL+ must have has_geo=False."""
        out = _run_loader_patched(tmp_path)
        df  = pq.read_table(str(out)).to_pandas()
        row = df[df["glottocode"] == "xxxx9999"].iloc[0]
        assert row["has_geo"]   is False or row["has_geo"]   == False
        assert row["has_phylo"] is False or row["has_phylo"] == False

    def test_found_language_has_geo_true(self, tmp_path):
        """Languages present in URIEL+ must have has_geo=True."""
        out = _run_loader_patched(tmp_path)
        df  = pq.read_table(str(out)).to_pandas()
        for gc in _FOUND_GCS:
            row = df[df["glottocode"] == gc].iloc[0]
            assert row["has_geo"]   is True or row["has_geo"]   == True
            assert row["has_phylo"] is True or row["has_phylo"] == True

    def test_found_language_vector_nonzero(self, tmp_path):
        """Languages present in URIEL+ must keep their extracted vectors."""
        out = _run_loader_patched(tmp_path)
        df  = pq.read_table(str(out)).to_pandas()
        row = df[df["glottocode"] == _FOUND_GCS[0]].iloc[0]
        assert not np.allclose(np.array(row["geo_vec"]), 0.0), \
            "Found language vector must not be zeroed out"

    def test_all_glottocodes_present_in_output(self, tmp_path):
        """Every input Glottocode (found and missing) must appear in the parquet."""
        out = _run_loader_patched(tmp_path)
        df  = pq.read_table(str(out)).to_pandas()
        assert set(df["glottocode"].tolist()) == set(_ALL_GCS)

    def test_summary_counts_are_correct(self, tmp_path):
        """run_loader must return correct n_found / n_missing summary stats."""
        with (
            patch("canonical.uriel_plus_loader._collect_glottocodes",
                  return_value=_ALL_GCS),
            patch("canonical.uriel_plus_loader._clone_or_verify"),
            patch("canonical.uriel_plus_loader._load_uriel_plus",
                  return_value=None),
            patch("canonical.uriel_plus_loader._extract_raw_vectors",
                  return_value=(_fake_geo(), _fake_phylo(),
                                _FOUND_GCS, _MISSING_GCS)),
        ):
            summary = run_loader(output_dir=str(tmp_path),
                                 urielplus_dir="/fake", smoke=False, seed=42)

        assert summary["n_found"]   == len(_FOUND_GCS)
        assert summary["n_missing"] == len(_MISSING_GCS)
        assert summary["n_glottocodes"] == len(_ALL_GCS)


# ──────────────────────────────────────────────────────────────────────────────
# Model training on zero-vector languages
# ──────────────────────────────────────────────────────────────────────────────

class TestModelTrainsOnZeroVectors:

    def test_conditioned_model_trains_without_crashing(self, tmp_path):
        """
        TypologicalMF_Conditioned must train successfully even when half the
        languages have zero geo/phylo vectors (simulating missing URIEL+ data).
        """
        import torch
        from torch.utils.data import TensorDataset
        from canonical.conditioning_model import (
            TypologicalMF_Conditioned,
            train_conditioned,
        )

        n_langs   = 10
        n_feats   = 3
        embed_dim = 8

        # First 5 languages have zero vectors (missing from URIEL+)
        geo_matrix   = np.zeros((n_langs, GEO_DIM),   dtype=np.float32)
        phylo_matrix = np.zeros((n_langs, PHYLO_DIM), dtype=np.float32)
        geo_matrix[5:]   = np.random.default_rng(0).random((5, GEO_DIM)).astype(np.float32)
        phylo_matrix[5:] = np.random.default_rng(1).random((5, PHYLO_DIM)).astype(np.float32)

        feat_to_global_ids = {i: [i] for i in range(n_feats)}

        model = TypologicalMF_Conditioned(
            n_langs, n_feats, feat_to_global_ids,
            geo_matrix, phylo_matrix, embed_dim=embed_dim,
        )

        lang_idx = torch.arange(n_langs).repeat(n_feats)
        feat_idx = torch.arange(n_feats).repeat_interleave(n_langs)
        vals     = torch.zeros(n_langs * n_feats, dtype=torch.long)
        train_ds = TensorDataset(lang_idx, feat_idx, vals)

        # Must not raise
        train_conditioned(model, train_ds, n_epochs=2, batch_size=16, lr=1e-3)

    def test_conditioned_model_produces_finite_outputs(self, tmp_path):
        """Forward pass on a zero-vector language must not produce NaN/Inf."""
        import torch
        from canonical.conditioning_model import TypologicalMF_Conditioned

        n_langs   = 4
        n_feats   = 2
        embed_dim = 8

        # All languages have zero vectors
        geo_matrix   = np.zeros((n_langs, GEO_DIM),   dtype=np.float32)
        phylo_matrix = np.zeros((n_langs, PHYLO_DIM), dtype=np.float32)

        feat_to_global_ids = {i: [i] for i in range(n_feats)}

        model = TypologicalMF_Conditioned(
            n_langs, n_feats, feat_to_global_ids,
            geo_matrix, phylo_matrix, embed_dim=embed_dim,
        )
        model.eval()

        lang_idx  = torch.tensor([0, 1, 2, 3])
        feat_idx  = torch.tensor([0, 1, 0, 1])
        value_idx = torch.zeros(4, dtype=torch.long)  # each feature has 1 value (idx 0)
        with torch.no_grad():
            logits, _ = model(lang_idx, feat_idx, value_idx)

        assert torch.isfinite(logits).all(), \
            "Forward pass on zero-vector languages must produce finite logits"
