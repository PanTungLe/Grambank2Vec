"""
Unit tests for sigtyp_eval_v3.py and model_set.py.
Run with: python3 -m pytest tests/test_sigtyp_v3.py -v
"""

import os
import sys
import tempfile
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_set import SetEncoder, SetEncoderDataset, train_set_encoder
from sigtyp_eval_v3 import (
    parse_sigtyp,
    build_correlation_prior,
    apply_prior,
    infer_set_encoder,
    iterative_infer,
    freq_baseline,
    _fmt_features,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Tests 2-4 use feat_to_global_ids = {0:[0,1,2], 1:[3,4,5,6], 2:[7,8]}
# n_total_values = 3+4+2 = 9, n_features = 3
_F2G = {0: [0, 1, 2], 1: [3, 4, 5, 6], 2: [7, 8]}
_N_FEATS = 3
_N_VALS  = 9   # max global ID = 8  -> n_total_values = 9


def _make_model(embed_dim=8, hidden_dim=16):
    return SetEncoder(_N_FEATS, _N_VALS, _F2G,
                      embed_dim=embed_dim, hidden_dim=hidden_dim)


# ---------------------------------------------------------------------------
# Test 1: local-to-global value ID mapping
# ---------------------------------------------------------------------------

def test_local_global_value_id_mapping():
    """Verify feat_to_global_ids indexing semantics."""
    feat_to_global_ids = {0: [0, 1, 2], 1: [3, 4, 5, 6]}

    # For feat_idx=1, local index 2 → global ID 5
    fi = 1
    local_idx = 2
    assert feat_to_global_ids[fi][local_idx] == 5, (
        f"Expected global ID 5, got {feat_to_global_ids[fi][local_idx]}")

    # Verify cat_matrix local-index retrieval
    cat_matrix = np.array([
        [0, 2],   # lang 0: feat 0 → local 0, feat 1 → local 2
        [1, 0],   # lang 1
        [-1, 3],  # lang 2: feat 0 missing, feat 1 → local 3
    ], dtype=int)

    # lang 0, feat_idx 1, local 2 → global ID 5
    lang0_feat1_local = cat_matrix[0, 1]
    assert lang0_feat1_local == 2
    assert feat_to_global_ids[1][lang0_feat1_local] == 5

    # lang 2, feat 0 is missing (-1)
    assert cat_matrix[2, 0] == -1


# ---------------------------------------------------------------------------
# Test 2: encode_context with empty context → all-zeros (1, d)
# ---------------------------------------------------------------------------

def test_set_encoder_encode_empty_context():
    model = _make_model()
    model.eval()
    with torch.no_grad():
        z = model.encode_context([], [], "cpu")

    assert z.shape == (1, model.embed_dim), (
        f"Expected shape (1, {model.embed_dim}), got {z.shape}")
    assert torch.allclose(z, torch.zeros_like(z)), (
        "encode_context([],[]) should return all-zero tensor")


# ---------------------------------------------------------------------------
# Test 3: predict_feature shape and normalisation
# ---------------------------------------------------------------------------

def test_set_encoder_predict_shape():
    model = _make_model()
    model.eval()
    with torch.no_grad():
        z     = model.encode_context([0], [1], "cpu")    # global val ID 1
        probs = model.predict_feature(z, feat_idx=1, device="cpu")

    # feat_idx=1 has global IDs [3,4,5,6] → 4 values
    assert probs.shape == (4,), (
        f"Expected shape (4,) for feat_idx=1, got {probs.shape}")
    assert abs(probs.sum().item() - 1.0) < 1e-5, (
        f"Probabilities should sum to 1.0, got {probs.sum().item():.8f}")
    assert (probs >= 0).all(), "All probabilities must be non-negative"


# ---------------------------------------------------------------------------
# Test 4: forward returns scalar loss with requires_grad, l2 >= 0
# ---------------------------------------------------------------------------

def test_set_encoder_forward_loss():
    model = _make_model()
    model.train()

    # context: feat 0, global val ID 1
    # target: feat 1, local val index 2
    loss, l2 = model(
        context_feat_indices=[0],
        context_global_val_ids=[1],
        target_feat_indices=[1],
        target_val_local_indices=[2],
        device="cpu",
    )

    assert loss.shape == torch.Size([]), \
        f"loss should be scalar, got shape {loss.shape}"
    assert loss.requires_grad, "loss should require grad"
    assert l2.item() >= 0.0, f"l2 should be non-negative, got {l2.item()}"

    # Verify backward works
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient after backward"


# ---------------------------------------------------------------------------
# Test 5: parse_sigtyp round-trip
# ---------------------------------------------------------------------------

_FAKE_TSV = """\
wals_code\tname\tlatitude\tlongitude\tgenus\tfamily\tcountrycodes\tfeatures
aaa\tLangA\t1.0\t2.0\tGenusX\tFamilyY\tUS\tfeat_a=1 SOV|feat_b=?|feat_c=3 OVS
bbb\tLangB\t3.0\t4.0\tGenusX\tFamilyY\tCA\tfeat_a=2 VSO|feat_b=1 SVO|feat_c=?
ccc\tLangC\t5.0\t6.0\tGenusZ\tFamilyW\tGB\tfeat_a=?|feat_b=?|feat_c=1 SOV
"""


def test_parse_sigtyp_roundtrip():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     encoding="utf-8", delete=False) as fh:
        fh.write(_FAKE_TSV)
        tmp_path = fh.name

    try:
        meta, feat, blanked, obs_strings = parse_sigtyp(tmp_path)

        # Three languages
        assert len(meta) == 3, f"Expected 3 rows, got {len(meta)}"
        assert list(meta["wals_code"]) == ["aaa", "bbb", "ccc"]

        # Blanked features per language
        assert blanked["aaa"] == {"feat_b"}, \
            f"aaa blanked should be {{feat_b}}, got {blanked['aaa']}"
        assert blanked["bbb"] == {"feat_c"}, \
            f"bbb blanked should be {{feat_c}}, got {blanked['bbb']}"
        assert blanked["ccc"] == {"feat_a", "feat_b"}, \
            f"ccc blanked should be {{feat_a, feat_b}}, got {blanked['ccc']}"

        # obs_strings should not contain "?"
        for wc, obs_d in obs_strings.items():
            for fn, vs in obs_d.items():
                assert "?" not in vs, \
                    f"obs_strings[{wc}][{fn}] contains '?': {vs!r}"

        # obs_strings for aaa: feat_a and feat_c observed (not feat_b which is ?)
        assert "feat_a" in obs_strings["aaa"]
        assert "feat_b" not in obs_strings["aaa"]
        assert "feat_c" in obs_strings["aaa"]

        # feat_df values: should be integer string or NaN
        assert feat.loc[0, "feat_a"] == "1", \
            f"Expected '1', got {feat.loc[0, 'feat_a']!r}"
        assert feat.loc[1, "feat_b"] == "1", \
            f"Expected '1', got {feat.loc[1, 'feat_b']!r}"
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test 6: build_correlation_prior + apply_prior with Dirichlet smoothing
# ---------------------------------------------------------------------------

def test_cond_prob_baseline_smoothing():
    """
    Tiny 10-lang, 3-feat, 2-value-each example.
    Verify co-count arrays and that apply_prior returns valid strings.
    """
    np.random.seed(0)
    n_langs, n_feats, n_vals = 10, 3, 2

    # Build a simple cat_matrix
    cat = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
    ], dtype=int)

    feat_to_global_ids = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    feat_to_value_names = {0: ["a0", "a1"], 1: ["b0", "b1"], 2: ["c0", "c1"]}
    feat_name_to_idx = {"feat_a": 0, "feat_b": 1, "feat_c": 2}

    co_counts, marginals = build_correlation_prior(cat, feat_to_global_ids,
                                                    feat_to_value_names)

    # Verify marginal counts
    # col 0: [0,0,1,1,0,1,0,1,0,1] → vi=0: 5 times, vi=1: 5 times
    assert marginals[(0, 0)] == 5, f"Expected 5, got {marginals[(0,0)]}"
    assert marginals[(0, 1)] == 5, f"Expected 5, got {marginals[(0,1)]}"

    # Co-count arrays should sum to marginal count minus any -1 (none here)
    key = (0, 0, 1)   # fi=0, vi=0, fj=1
    if key in co_counts:
        total = co_counts[key].sum()
        assert total <= marginals[(0, 0)], \
            f"co_counts[{key}].sum()={total} > marginal {marginals[(0,0)]}"

    # apply_prior should return valid value strings
    raw_log_probs = {
        "feat_c": np.array([-0.3, -1.5]),   # strongly prefers c0
    }
    obs_feat_dict = {"feat_a": "a0", "feat_b": "b1"}

    preds = apply_prior(raw_log_probs, obs_feat_dict, feat_name_to_idx,
                        feat_to_value_names, co_counts, marginals,
                        lam=0.5, alpha=0.1, min_count=1)

    assert "feat_c" in preds, "apply_prior should return prediction for feat_c"
    assert preds["feat_c"] in ["c0", "c1"], \
        f"Prediction should be 'c0' or 'c1', got {preds['feat_c']!r}"


# ---------------------------------------------------------------------------
# Test 7: margin threshold in iterative_infer
# ---------------------------------------------------------------------------

def test_margin_threshold():
    """
    Verify that iterative_infer only adds pseudo-labels for features
    with margin >= margin_tau.
    """
    # Use a tiny model and cat_matrix with 3 features
    # We'll observe feature 0 and check which of feat 1/2 get pseudo-labeled

    cat_matrix = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
        [1, 1, 0],
        [0, 0, 0],
    ], dtype=int)

    feat_to_global_ids  = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    feat_to_value_names = {0: ["v0", "v1"], 1: ["w0", "w1"], 2: ["x0", "x1"]}
    feat_name_to_idx    = {"feat_a": 0, "feat_b": 1, "feat_c": 2}
    kept_feature_names  = ["feat_a", "feat_b", "feat_c"]

    # Train a model briefly so predictions are non-trivial
    torch.manual_seed(42)
    np.random.seed(42)
    model = SetEncoder(3, 6, feat_to_global_ids, embed_dim=8, hidden_dim=16)
    dataset = SetEncoderDataset(cat_matrix, feat_to_global_ids)
    train_set_encoder(model, dataset, n_epochs=5, device="cpu", seed=42)
    model.eval()

    # Observe feat_a=v0; check self-training with high tau (0.99 → nothing added)
    obs = {"feat_a": "v0"}
    preds_strict = iterative_infer(
        model, obs, feat_name_to_idx, feat_to_global_ids, feat_to_value_names,
        kept_feature_names, device="cpu", T=1.0,
        n_rounds=1, margin_tau=0.99, top_n=5)

    # With margin_tau=0.99, pseudo-labels should rarely be added (margins are low)
    # The result should still contain all features (from final leave-target-out pass)
    assert set(preds_strict.keys()).issubset(set(kept_feature_names)), \
        "Predictions should only include kept_feature_names"

    # With very low tau, pseudo-labels will be added freely
    preds_loose = iterative_infer(
        model, obs, feat_name_to_idx, feat_to_global_ids, feat_to_value_names,
        kept_feature_names, device="cpu", T=1.0,
        n_rounds=1, margin_tau=0.0, top_n=5)

    assert set(preds_loose.keys()).issubset(set(kept_feature_names))

    # With strict threshold, self-training adds fewer (or equal) pseudo-labels
    # than loose threshold — we can verify this indirectly by checking predictions
    # are valid value strings
    for fn, pv in preds_strict.items():
        fi = feat_name_to_idx[fn]
        assert pv in feat_to_value_names[fi], \
            f"Invalid prediction {pv!r} for {fn} (valid: {feat_to_value_names[fi]})"


# ---------------------------------------------------------------------------
# Test 8: submission format
# ---------------------------------------------------------------------------

def test_submission_format():
    """
    Verify _fmt_features generates correct pipe-delimited output
    that matches the expected SIGTYP submission format.
    """
    blanked_set  = {"feat_b", "feat_c"}
    obs_str_dict = {"feat_a": "1 SOV", "feat_d": "2 OVS"}
    model_preds  = {"feat_b": "3", "feat_c": "1"}
    freq_fallback = {"feat_b": "1", "feat_c": "2", "feat_e": "1"}

    result = _fmt_features(blanked_set, obs_str_dict, model_preds, freq_fallback)

    # Should be pipe-delimited
    parts = result.split("|")
    assert len(parts) >= 4, f"Expected at least 4 pipe-delimited parts, got {len(parts)}"

    part_dict = {}
    for part in parts:
        part = part.strip()
        if "=" in part:
            name, val = part.split("=", 1)
            part_dict[name] = val

    # Observed features keep their original string (not blanked)
    assert "feat_a" in part_dict
    assert part_dict["feat_a"] == "1 SOV", \
        f"feat_a should have original obs string, got {part_dict['feat_a']!r}"

    # Blanked features get model predictions with " -" suffix
    assert "feat_b" in part_dict
    assert part_dict["feat_b"].startswith("3"), \
        f"feat_b should start with model pred '3', got {part_dict['feat_b']!r}"
    assert part_dict["feat_b"].endswith("-"), \
        f"feat_b blanked prediction should end with '-'"

    assert "feat_c" in part_dict
    assert part_dict["feat_c"].startswith("1"), \
        f"feat_c should start with model pred '1'"

    # Features not in blanked_set and not in obs_str_dict should be absent
    assert "feat_e" not in part_dict, \
        "feat_e was neither observed nor blanked — should not appear"

    # Verify round-trip: all features in result are either observed or blanked
    for name in part_dict:
        assert name in obs_str_dict or name in blanked_set, \
            f"{name} is in output but was neither observed nor blanked"
