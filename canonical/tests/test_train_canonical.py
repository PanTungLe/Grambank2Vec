"""
Synthetic-data smoke tests for canonical/train_canonical.py.

These tests do NOT require CLDF repos; they bypass the data-loading step
and feed synthetic data directly into the training loop helpers.

All tests verify:
  - training loop runs without error
  - artifacts are dumped with correct shapes
  - bit-identical outputs under the same seed (reproducibility gate)
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# --- ensure repo root and canonical/ are importable ---
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CANONICAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CANONICAL_DIR))

import pandas as pd

from model import TypologicalMF, WALSDataset
from model_learned import TypologicalMF_Learned, CategoricalTypDataset
from utils import (
    seed_everything,
    build_all_triples_binary,
    build_all_triples_categorical,
)
from train_canonical import (
    train_epoch,
    compute_val_loss,
    get_emb_norms,
    dump_artifacts,
    build_lang2id,
    parse_args,
)


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

N_LANG, N_BINARY_FEAT = 30, 20
N_ORIG_FEAT = 10
FEATURE_CARDS = [7, 5, 3, 2, 4, 6, 2, 3, 5, 4]
EMBED_DIM = 8
SEED = 42


def make_binary_data():
    """Return (binary_matrix, binary_col_names, feature_groups, feature_value_names)."""
    rng = np.random.default_rng(0)
    bm = rng.choice([0.0, 1.0, np.nan], size=(N_LANG, N_BINARY_FEAT),
                    p=[0.4, 0.4, 0.2])
    col_names = [f"feat_F{i}" for i in range(N_BINARY_FEAT)]
    feature_groups = {f"feat_F{i}": [i] for i in range(N_BINARY_FEAT)}
    feature_value_names = {f"feat_F{i}": [f"v0_{i}", f"v1_{i}"]
                           for i in range(N_BINARY_FEAT)}
    return bm, col_names, feature_groups, feature_value_names


def make_categorical_data():
    """Return (cat_matrix, kept_feature_names, feat_to_global_ids, feat_to_value_names)."""
    rng = np.random.default_rng(0)
    n_feat = len(FEATURE_CARDS)
    cat = np.full((N_LANG, n_feat), -1, dtype=np.int64)
    for fi, card in enumerate(FEATURE_CARDS):
        for li in range(N_LANG):
            if rng.random() < 0.75:
                cat[li, fi] = rng.integers(0, card)

    kept = [f"feat_F{fi}" for fi in range(n_feat)]
    feat_to_global_ids: dict[int, list[int]] = {}
    feat_to_value_names: dict[int, list[str]] = {}
    gid = 0
    for fi, card in enumerate(FEATURE_CARDS):
        feat_to_global_ids[fi] = list(range(gid, gid + card))
        feat_to_value_names[fi] = [f"val_{fi}_{v}" for v in range(card)]
        gid += card

    return cat, kept, feat_to_global_ids, feat_to_value_names


def run_tcf_training(out_dir: str, seed: int = SEED) -> None:
    """Full TCF training run into out_dir with the given seed."""
    seed_everything(seed)
    bm, col_names, fg, fvn = make_binary_data()
    all_l, all_f, all_v = build_all_triples_binary(bm)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(all_l))
    rng.shuffle(idx)
    n_val = max(1, len(idx) // 20)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = WALSDataset(all_l[train_idx], all_f[train_idx],
                           all_v[train_idx].astype(np.float32))
    val_ds = WALSDataset(all_l[val_idx], all_f[val_idx],
                         all_v[val_idx].astype(np.float32))

    from torch.utils.data import DataLoader
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = TypologicalMF(N_LANG, N_BINARY_FEAT, EMBED_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

    best_val = float("inf")
    for _ in range(5):
        train_epoch(model, train_loader, opt, 0.1, "tcf", "cpu")
        vl = compute_val_loss(model, val_loader, "tcf", "cpu")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))

    model.load_state_dict(torch.load(os.path.join(out_dir, "model_best.pt"),
                                     map_location="cpu"))

    df = pd.DataFrame({"wals_code": [f"L{i}" for i in range(N_LANG)]})

    class FakeArgs:
        architecture = "tcf"
        database = "wals"
        data_path = "/fake"

    dump_artifacts(model, FakeArgs(), {
        "binary_col_names": col_names,
        "feature_groups": fg,
        "feature_value_names": fvn,
    }, {f"gc{i}": i for i in range(N_LANG)},
       {f"L{i}": i for i in range(N_LANG)}, out_dir)


def run_learned_training(out_dir: str, seed: int = SEED) -> None:
    """Full Learned training run into out_dir with the given seed."""
    seed_everything(seed)
    cat, kept, f2gids, f2vnames = make_categorical_data()
    all_l, all_f, all_v = build_all_triples_categorical(cat)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(all_l))
    rng.shuffle(idx)
    n_val = max(1, len(idx) // 20)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = CategoricalTypDataset(
        all_l[train_idx], all_f[train_idx], all_v[train_idx])
    val_ds = CategoricalTypDataset(
        all_l[val_idx], all_f[val_idx], all_v[val_idx])

    from torch.utils.data import DataLoader
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    n_total_values = max(max(gids) for gids in f2gids.values()) + 1
    model = TypologicalMF_Learned(N_LANG, n_total_values, f2gids, EMBED_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

    best_val = float("inf")
    for _ in range(5):
        train_epoch(model, train_loader, opt, 0.1, "learned", "cpu")
        vl = compute_val_loss(model, val_loader, "learned", "cpu")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))

    model.load_state_dict(torch.load(os.path.join(out_dir, "model_best.pt"),
                                     map_location="cpu"))

    class FakeArgs:
        architecture = "learned"
        database = "wals"
        data_path = "/fake"

    dump_artifacts(model, FakeArgs(), {
        "kept_feature_names": kept,
        "feat_to_global_ids": f2gids,
        "feat_to_value_names": f2vnames,
    }, {f"gc{i}": i for i in range(N_LANG)},
       {f"gc{i}": i for i in range(N_LANG)}, out_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_build_all_triples_binary(self):
        bm = np.array([[0., 1., np.nan], [1., np.nan, 0.]])
        l, f, v = build_all_triples_binary(bm)
        assert len(l) == 4
        assert set(zip(l.tolist(), f.tolist())) == {(0,0),(0,1),(1,0),(1,2)}
        assert v.dtype == np.float32

    def test_build_all_triples_categorical(self):
        cm = np.array([[0, 1, -1], [2, -1, 0]])
        l, f, v = build_all_triples_categorical(cm)
        assert len(l) == 4
        assert v.dtype == np.int64

    def test_get_emb_norms_tcf(self):
        model = TypologicalMF(10, 5, 8)
        lang_n, fv_n = get_emb_norms(model, "tcf")
        assert lang_n > 0
        assert fv_n > 0

    def test_get_emb_norms_learned(self):
        f2g = {0: [0, 1, 2], 1: [3, 4]}
        model = TypologicalMF_Learned(10, 5, f2g, 8)
        lang_n, fv_n = get_emb_norms(model, "learned")
        assert lang_n > 0
        assert fv_n > 0


class TestTCFTrainingArtifacts:
    def test_artifacts_created(self, tmp_path):
        run_tcf_training(str(tmp_path))
        assert (tmp_path / "lang_embeddings.npy").exists()
        assert (tmp_path / "binarycol_embeddings.npy").exists()
        assert (tmp_path / "binarycol2id.json").exists()
        assert (tmp_path / "feat2values.json").exists()
        assert (tmp_path / "lang2id.json").exists()

    def test_embedding_shapes(self, tmp_path):
        run_tcf_training(str(tmp_path))
        lang_emb = np.load(tmp_path / "lang_embeddings.npy")
        bin_emb = np.load(tmp_path / "binarycol_embeddings.npy")
        assert lang_emb.shape == (N_LANG, EMBED_DIM)
        assert bin_emb.shape == (N_BINARY_FEAT, EMBED_DIM)

    def test_binarycol2id_keys_stripped(self, tmp_path):
        run_tcf_training(str(tmp_path))
        b2id = json.loads((tmp_path / "binarycol2id.json").read_text())
        for key in b2id:
            assert not key.startswith("feat_"), \
                f"binarycol2id key still has 'feat_' prefix: {key}"

    def test_reproducibility(self, tmp_path):
        d1 = tmp_path / "run1"
        d2 = tmp_path / "run2"
        d1.mkdir(); d2.mkdir()
        run_tcf_training(str(d1), seed=42)
        run_tcf_training(str(d2), seed=42)
        e1 = np.load(d1 / "lang_embeddings.npy")
        e2 = np.load(d2 / "lang_embeddings.npy")
        assert np.array_equal(e1, e2), \
            "Same seed produced different embeddings — seeding is broken!"


class TestLearnedTrainingArtifacts:
    def test_artifacts_created(self, tmp_path):
        run_learned_training(str(tmp_path))
        assert (tmp_path / "lang_embeddings.npy").exists()
        assert (tmp_path / "featvalue_embeddings.npy").exists()
        assert (tmp_path / "featvalue2id.json").exists()
        assert (tmp_path / "feat2values.json").exists()
        assert (tmp_path / "lang2id.json").exists()

    def test_embedding_shapes(self, tmp_path):
        run_learned_training(str(tmp_path))
        lang_emb = np.load(tmp_path / "lang_embeddings.npy")
        fv_emb = np.load(tmp_path / "featvalue_embeddings.npy")
        n_total_values = sum(FEATURE_CARDS)
        assert lang_emb.shape == (N_LANG, EMBED_DIM)
        assert fv_emb.shape == (n_total_values, EMBED_DIM)

    def test_featvalue2id_format(self, tmp_path):
        run_learned_training(str(tmp_path))
        fv2id = json.loads((tmp_path / "featvalue2id.json").read_text())
        for key in fv2id:
            assert "=" in key, \
                f"featvalue2id key missing '=' separator: {key}"
            assert not key.startswith("feat_"), \
                f"featvalue2id key still has 'feat_' prefix: {key}"

    def test_feat2values_populated(self, tmp_path):
        run_learned_training(str(tmp_path))
        f2v = json.loads((tmp_path / "feat2values.json").read_text())
        assert len(f2v) == len(FEATURE_CARDS)
        for k, vals in f2v.items():
            assert isinstance(vals, list) and len(vals) > 0

    def test_reproducibility(self, tmp_path):
        d1 = tmp_path / "run1"
        d2 = tmp_path / "run2"
        d1.mkdir(); d2.mkdir()
        run_learned_training(str(d1), seed=42)
        run_learned_training(str(d2), seed=42)
        e1 = np.load(d1 / "lang_embeddings.npy")
        e2 = np.load(d2 / "lang_embeddings.npy")
        assert np.array_equal(e1, e2), \
            "Same seed produced different embeddings — seeding is broken!"
        fv1 = np.load(d1 / "featvalue_embeddings.npy")
        fv2 = np.load(d2 / "featvalue_embeddings.npy")
        assert np.array_equal(fv1, fv2), \
            "Same seed produced different value embeddings — seeding is broken!"


class TestDifferentSeedsDiffer:
    def test_different_seeds_differ_tcf(self, tmp_path):
        d42 = tmp_path / "s42"; d43 = tmp_path / "s43"
        d42.mkdir(); d43.mkdir()
        run_tcf_training(str(d42), seed=42)
        run_tcf_training(str(d43), seed=43)
        e42 = np.load(d42 / "lang_embeddings.npy")
        e43 = np.load(d43 / "lang_embeddings.npy")
        assert not np.array_equal(e42, e43), \
            "Different seeds produced identical embeddings — something is wrong!"

    def test_different_seeds_differ_learned(self, tmp_path):
        d42 = tmp_path / "s42"; d43 = tmp_path / "s43"
        d42.mkdir(); d43.mkdir()
        run_learned_training(str(d42), seed=42)
        run_learned_training(str(d43), seed=43)
        e42 = np.load(d42 / "lang_embeddings.npy")
        e43 = np.load(d43 / "lang_embeddings.npy")
        assert not np.array_equal(e42, e43), \
            "Different seeds produced identical embeddings — something is wrong!"
