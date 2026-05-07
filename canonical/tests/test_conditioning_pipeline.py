"""
Tests for canonical/conditioning_pipeline.py — conditioning_variants flag.

Heavy dependencies (load_data, load_uriel_vectors_for_lang2id, _run_one) are
patched so no CLDF data or model training is required.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from canonical.conditioning_pipeline import parse_args, run_pipeline

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

N_LANGS   = 20
GEO_DIM   = 8
PHYLO_DIM = 12
N_FEATS   = 4


def _fake_df() -> pd.DataFrame:
    return pd.DataFrame({
        "glottocode": [f"lang{i:04d}" for i in range(N_LANGS)],
        "genus":     ["BranchA"] * 10 + ["BranchB"] * 10,
        "family":    ["FamA"]    * 10 + ["FamB"]    * 10,
        "macroarea": ["Africa"]  * N_LANGS,
    })


def _fake_data_dict() -> dict:
    cat = np.zeros((N_LANGS, N_FEATS), dtype=int)
    return {
        "cat_matrix":          cat,
        "kept_feature_names":  [f"F{i}" for i in range(N_FEATS)],
        "feat_to_global_ids":  {i: [i] for i in range(N_FEATS)},
        "feat_to_value_names": {i: {0: "val0"} for i in range(N_FEATS)},
    }


def _fake_uriel_meta() -> dict:
    return {"pct_found": 100.0, "found": N_LANGS, "n_langs": N_LANGS}


def _run_pipeline_patched(tmp_path: Path, variants=None) -> pd.DataFrame:
    """
    Build minimal args and call run_pipeline with all heavy ops patched out.

    Parameters
    ----------
    tmp_path : base directory for outputs (test-isolated)
    variants : list[str] | None — passed to --conditioning_variants;
               None uses argparse default (['both'])
    """
    # Create a stub parquet so the existence check in run_pipeline passes.
    uriel_dir = tmp_path / "uriel"
    uriel_dir.mkdir(parents=True, exist_ok=True)
    (uriel_dir / "uriel_plus_vectors_familymean.parquet").touch()

    argv = [
        "--database",       "grambank",
        "--architecture",   "learned",
        "--data_path",      str(tmp_path),
        "--imputation",     "familymean",
        "--uriel_dir",      str(uriel_dir),
        "--out_dir",        str(tmp_path / "results"),
        "--seed",           "42",
        "--smoke",
        "--smoke_n_langs",  "0",   # skip language subsetting
    ]
    if variants is not None:
        argv += ["--conditioning_variants"] + variants

    args = parse_args(argv)

    fake_df     = _fake_df()
    fake_lang2id = {gc: i for i, gc in enumerate(fake_df["glottocode"])}

    with (
        patch(
            "canonical.conditioning_pipeline.load_data",
            return_value=(fake_df, _fake_data_dict(), fake_lang2id, {}),
        ),
        patch(
            "canonical.conditioning_pipeline.load_uriel_vectors_for_lang2id",
            return_value=(
                np.zeros((N_LANGS, GEO_DIM),   dtype=np.float32),
                np.zeros((N_LANGS, PHYLO_DIM), dtype=np.float32),
                _fake_uriel_meta(),
            ),
        ),
        patch(
            "canonical.conditioning_pipeline._run_one",
            return_value=(0.5, 0.6),
        ),
    ):
        return run_pipeline(args)


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestConditioningVariants:

    def test_three_variants_yield_triple_row_count(self, tmp_path):
        """--conditioning_variants geo phylo both must produce 3× rows vs default."""
        result_default = _run_pipeline_patched(tmp_path / "default")
        result_triple  = _run_pipeline_patched(
            tmp_path / "triple", variants=["geo", "phylo", "both"])

        assert len(result_triple) == 3 * len(result_default), (
            f"Expected 3× rows: got {len(result_triple)} vs {len(result_default)}"
        )

    def test_conditioning_column_present_in_default(self, tmp_path):
        """Every row must carry a 'conditioning' column even in the default run."""
        result = _run_pipeline_patched(tmp_path)
        assert "conditioning" in result.columns
        assert (result["conditioning"] == "both").all()

    def test_conditioning_values_match_variants(self, tmp_path):
        """Each row's 'conditioning' value must match the requested variant."""
        result = _run_pipeline_patched(
            tmp_path, variants=["geo", "phylo", "both"])
        assert set(result["conditioning"].unique()) == {"geo", "phylo", "both"}

    def test_both_model_types_present_per_variant(self, tmp_path):
        """baseline and conditioned rows must exist for every variant."""
        result = _run_pipeline_patched(
            tmp_path, variants=["geo", "phylo", "both"])
        for variant in ("geo", "phylo", "both"):
            subset = result[result["conditioning"] == variant]
            assert set(subset["model_type"].unique()) == {"baseline", "conditioned"}, \
                f"Missing model_type rows for variant={variant!r}"

    def test_single_variant_row_count_unchanged(self, tmp_path):
        """Specifying one variant explicitly must give the same count as the default."""
        result_default  = _run_pipeline_patched(tmp_path / "a")
        result_explicit = _run_pipeline_patched(tmp_path / "b", variants=["both"])
        assert len(result_explicit) == len(result_default)


# ──────────────────────────────────────────────────────────────────────────────
# Spec-numbered tests (points 1-4) + TCF interface verification
# ──────────────────────────────────────────────────────────────────────────────

class TestConditioningVariantsSpec:
    """
    Explicit tests mirroring the four numbered spec assertions plus a check
    that TypologicalMF_TCF_Conditioned exposes the same cond_mode interface.
    """

    def test_spec_1_default_no_flag_produces_n_rows(self, tmp_path):
        """Spec 1 — Default (no flag) produces N rows (N > 0)."""
        result = _run_pipeline_patched(tmp_path)
        assert len(result) > 0, "Default run must produce at least one row"

    def test_spec_2_explicit_both_produces_n_rows_with_both_column(self, tmp_path):
        """Spec 2 — --conditioning_variants both produces N rows with conditioning='both'."""
        result_default  = _run_pipeline_patched(tmp_path / "default")
        result_explicit = _run_pipeline_patched(tmp_path / "explicit", variants=["both"])
        assert len(result_explicit) == len(result_default), (
            f"Explicit 'both' must produce same row count as default: "
            f"{len(result_explicit)} vs {len(result_default)}"
        )
        assert "conditioning" in result_explicit.columns
        assert (result_explicit["conditioning"] == "both").all(), (
            "Every row must carry conditioning='both' when only 'both' is requested"
        )

    def test_spec_3_geo_phylo_both_produces_3n_rows_one_per_variant(self, tmp_path):
        """Spec 3 — --conditioning_variants geo phylo both produces 3N rows, one per variant."""
        n_result     = _run_pipeline_patched(tmp_path / "n",  variants=["both"])
        tri_result   = _run_pipeline_patched(tmp_path / "3n", variants=["geo", "phylo", "both"])
        n = len(n_result)
        assert len(tri_result) == 3 * n, (
            f"Expected 3×{n}={3*n} rows, got {len(tri_result)}"
        )
        # Each variant accounts for exactly N rows
        for v in ("geo", "phylo", "both"):
            n_v = len(tri_result[tri_result["conditioning"] == v])
            assert n_v == n, (
                f"Expected {n} rows for conditioning={v!r}, got {n_v}"
            )

    def test_spec_4_conditioning_column_matches_variant_used(self, tmp_path):
        """Spec 4 — The 'conditioning' column value matches the variant used."""
        cases = [
            (["geo"],                  {"geo"}),
            (["phylo"],                {"phylo"}),
            (["both"],                 {"both"}),
            (["geo", "phylo", "both"], {"geo", "phylo", "both"}),
        ]
        for i, (variants, expected) in enumerate(cases):
            result = _run_pipeline_patched(tmp_path / f"case{i}", variants=variants)
            got = set(result["conditioning"].unique())
            assert got == expected, (
                f"variants={variants}: expected conditioning values {expected}, got {got}"
            )

    def test_tcf_conditioned_class_exposes_cond_mode(self):
        """
        TypologicalMF_TCF_Conditioned must accept the same cond_mode kwarg as
        TypologicalMF_Conditioned, even though the pipeline's Tier-1 runs use
        the Learned architecture.
        """
        import inspect
        from canonical.conditioning_model import TypologicalMF_TCF_Conditioned

        sig = inspect.signature(TypologicalMF_TCF_Conditioned.__init__)
        assert "cond_mode" in sig.parameters, (
            "TypologicalMF_TCF_Conditioned.__init__ must accept a 'cond_mode' kwarg"
        )

        # All four valid mode strings must be accepted without raising.
        geo   = np.zeros((4, GEO_DIM),   dtype=np.float32)
        phylo = np.zeros((4, PHYLO_DIM), dtype=np.float32)
        for mode in ("both", "geo_only", "phylo_only", "init_only"):
            m = TypologicalMF_TCF_Conditioned(4, 3, geo, phylo,
                                              embed_dim=8, cond_mode=mode)
            assert m.cond_mode == mode, (
                f"Expected m.cond_mode={mode!r}, got {m.cond_mode!r}"
            )
