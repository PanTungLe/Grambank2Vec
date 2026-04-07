#!/usr/bin/env python3
"""
Tests for the Bjerva et al. (2019) replication pipeline.

Tests use synthetic data to verify all components work correctly
without requiring external data downloads.

Run: python -m pytest test_pipeline.py -v
  or: python test_pipeline.py
"""

import json
import os
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from model import TypologicalMF, TypologicalMF_SemiSup, WALSDataset, train_model
from char_lm import CharLSTM_LM, MultilingualCharDataset, build_vocab, train_charlm
from data_preparation import (
    binarise_wals,
    binarise_features,
    load_grambank_cldf,
    load_glottolog_genus_map,
    _extract_glottolog_genus,
    is_latin_cyrillic_greek,
    match_wals_to_bible,
    align_embeddings,
    _extract_iso_from_filename,
    load_ebible_texts,
    load_parabible_texts,
    load_bible_texts,
)
from evaluation_pipeline import (
    split_by_branch,
    decode_and_evaluate,
    majority_baseline,
    knn_baseline,
    run_experiments,
    summarise_results,
)


# ============================================================================
# Fixtures
# ============================================================================

def make_synthetic_wals(n_langs=60, n_features=8, seed=42):
    """Create a synthetic WALS-like DataFrame and binary matrix."""
    rng = np.random.default_rng(seed)

    # Create languages with genus/family structure
    genera = ["Romance", "Germanic", "Slavic", "Bantu",
              "Semitic", "Turkic"]
    families = ["Indo-European", "Indo-European", "Indo-European",
                "Niger-Congo", "Afro-Asiatic", "Turkic"]
    macroareas = ["Eurasia", "Eurasia", "Eurasia",
                  "Africa", "Africa", "Eurasia"]

    langs_per_genus = n_langs // len(genera)
    rows = []
    for gi, genus in enumerate(genera):
        for li in range(langs_per_genus):
            lang_idx = gi * langs_per_genus + li
            row = {
                "wals_code": f"l{lang_idx:03d}",
                "name": f"Language_{lang_idx}",
                "genus": genus,
                "family": families[gi],
                "macroarea": macroareas[gi],
                "iso639p3code": f"x{lang_idx:02d}",
            }
            # Generate feature values with genus-based patterns
            for fi in range(n_features):
                n_vals = rng.integers(2, 5)
                # Languages in the same genus tend to have similar values
                base_probs = rng.dirichlet(np.ones(n_vals) * 0.5)
                # Add genus-specific bias
                genus_bias = (gi + fi) % n_vals
                base_probs[genus_bias] += 0.5
                base_probs /= base_probs.sum()

                val = rng.choice([f"val{v}" for v in range(n_vals)],
                                 p=base_probs)
                # 20% chance of missing
                if rng.random() < 0.2:
                    val = np.nan
                row[f"feat_{fi}"] = val
            rows.append(row)

    df = pd.DataFrame(rows)
    feature_cols = [f"feat_{fi}" for fi in range(n_features)]

    return df, feature_cols


def make_synthetic_bible_texts(n_langs=10, n_tokens=60000, seed=42):
    """Create synthetic Bible-like texts for testing."""
    rng = np.random.default_rng(seed)
    texts = {}
    # Use Latin characters
    vocab = list("abcdefghijklmnopqrstuvwxyz ")
    for i in range(n_langs):
        lang_id = f"x{i:02d}"
        chars = rng.choice(vocab, size=n_tokens * 5)
        text = "".join(chars)
        texts[lang_id] = text
    return texts


# ============================================================================
# Model Tests
# ============================================================================

class TestModels:
    def test_typological_mf_shapes(self):
        model = TypologicalMF(50, 30, embed_dim=16)
        lang_idx = torch.tensor([0, 1, 2])
        feat_idx = torch.tensor([5, 10, 15])
        probs, batch_l2 = model(lang_idx, feat_idx)
        assert probs.shape == (3,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_typological_mf_predict_all(self):
        model = TypologicalMF(10, 20, embed_dim=8)
        with torch.no_grad():
            full = model.predict_all()
        assert full.shape == (10, 20)
        assert (full >= 0).all() and (full <= 1).all()

    def test_semisup_model(self):
        embs = np.random.randn(10, 32).astype(np.float32)
        model = TypologicalMF_SemiSup(embs, 20, embed_dim=16)
        lang_idx = torch.tensor([0, 1])
        feat_idx = torch.tensor([5, 10])
        probs, batch_l2 = model(lang_idx, feat_idx)
        assert probs.shape == (2,)

    def test_semisup_predict_all(self):
        embs = np.random.randn(10, 32).astype(np.float32)
        model = TypologicalMF_SemiSup(embs, 20, embed_dim=16)
        with torch.no_grad():
            full = model.predict_all()
        assert full.shape == (10, 20)

    def test_training_loop(self):
        ds = WALSDataset(
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 1, 0, 1, 0, 1]),
            np.array([1, 0, 0, 1, 1, 0], dtype=float),
        )
        model = TypologicalMF(3, 2, embed_dim=4)
        losses = train_model(model, ds, n_epochs=3, batch_size=2)
        assert len(losses) == 3
        assert all(isinstance(l, float) for l in losses)

    def test_semisup_training(self):
        embs = np.random.randn(3, 8).astype(np.float32)
        ds = WALSDataset(
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 1, 0, 1, 0, 1]),
            np.array([1, 0, 0, 1, 1, 0], dtype=float),
        )
        model = TypologicalMF_SemiSup(embs, 2, embed_dim=4)
        losses = train_model(model, ds, n_epochs=3, batch_size=2)
        assert len(losses) == 3


# ============================================================================
# Data Preparation Tests
# ============================================================================

class TestDataPreparation:
    def test_binarise_wals(self):
        df, feature_cols = make_synthetic_wals(n_langs=20, n_features=3)
        bmat, names, groups, _val_names = binarise_wals(df, feature_cols)

        assert bmat.shape[0] == len(df)
        assert bmat.shape[1] == len(names)
        assert len(groups) == 3
        # Check binary values
        valid = bmat[~np.isnan(bmat)]
        assert set(valid.tolist()).issubset({0.0, 1.0})

    def test_binarise_feature_groups(self):
        df, feature_cols = make_synthetic_wals(n_langs=20, n_features=3)
        bmat, names, groups, _val_names = binarise_wals(df, feature_cols)

        # Each feature should have at least 2 binary columns
        for feat, indices in groups.items():
            assert len(indices) >= 1

        # All indices should be covered
        all_indices = []
        for indices in groups.values():
            all_indices.extend(indices)
        assert sorted(all_indices) == list(range(bmat.shape[1]))

    def test_is_latin_cyrillic_greek(self):
        assert is_latin_cyrillic_greek("Hello world this is English text")
        assert is_latin_cyrillic_greek("Привет мир это русский текст")
        assert not is_latin_cyrillic_greek("")
        assert not is_latin_cyrillic_greek("ab")

    def test_extract_iso_from_filename(self):
        assert _extract_iso_from_filename("eng.txt") == "eng"
        assert _extract_iso_from_filename("eng-web.txt") == "eng"
        assert _extract_iso_from_filename("/path/to/fra.txt") == "fra"
        assert _extract_iso_from_filename("spa-rvr.txt") == "spa"
        # eBible naming convention
        assert _extract_iso_from_filename("eng-eng_kjv.txt") == "eng"
        assert _extract_iso_from_filename("spa-spaRV1909.txt") == "spa"

    def test_match_wals_to_bible_iso(self):
        df = pd.DataFrame({
            "wals_code": ["abc", "def", "ghi"],
            "name": ["Lang A", "Lang B", "Lang C"],
            "genus": ["G1", "G2", "G3"],
            "iso639p3code": ["eng", "fra", "deu"],
        })
        bible_ids = ["eng", "fra", "spa"]
        matched = match_wals_to_bible(df, bible_ids)
        assert len(matched) == 2  # eng and fra match
        assert set(matched["bible_lang_id"]) == {"eng", "fra"}

    def test_match_wals_to_bible_manual(self):
        df = pd.DataFrame({
            "wals_code": ["abc", "def"],
            "name": ["Lang A", "Lang B"],
            "genus": ["G1", "G2"],
        })
        bible_ids = ["eng", "fra"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                          delete=False) as f:
            f.write("wals_code,bible_lang_id\n")
            f.write("abc,eng\n")
            f.write("def,fra\n")
            mapping_path = f.name

        try:
            matched = match_wals_to_bible(df, bible_ids,
                                           manual_mapping_csv=mapping_path)
            assert len(matched) == 2
        finally:
            os.unlink(mapping_path)

    def test_align_embeddings(self):
        df = pd.DataFrame({
            "wals_code": ["l0", "l1", "l2", "l3"],
            "name": ["A", "B", "C", "D"],
        })
        matched_df = pd.DataFrame({
            "wals_code": ["l1", "l3"],
            "bible_lang_id": ["eng", "fra"],
        })
        embs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        dtype=np.float32)
        lang2idx = {"eng": 0, "fra": 1, "deu": 2}

        aligned = align_embeddings(embs, lang2idx, df, matched_df)
        assert aligned.shape == (4, 3)
        np.testing.assert_array_equal(aligned[0], [0, 0, 0])  # no match
        np.testing.assert_array_equal(aligned[1], [1, 2, 3])  # eng
        np.testing.assert_array_equal(aligned[2], [0, 0, 0])  # no match
        np.testing.assert_array_equal(aligned[3], [4, 5, 6])  # fra

    def test_load_ebible_texts(self):
        """Test loading from a directory with plain verse-per-line files (eBible format)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake eBible-style Bible text file (plain text, one verse per line)
            lines = [" ".join(["word"] * 100) for _ in range(600)]
            # Insert some blank lines and <range> tokens like real eBible data
            lines.insert(50, "")
            lines.insert(100, "<range>")
            with open(os.path.join(tmpdir, "eng-engKJV.txt"), "w") as f:
                f.write("\n".join(lines))

            texts = load_ebible_texts(tmpdir, script_filter=False,
                                      min_tokens=100)
            assert "eng" in texts
            assert len(texts["eng"].split()) >= 100
            # Ensure <range> tokens are not in the text
            assert "<range>" not in texts["eng"]

    def test_load_parabible_texts_legacy(self):
        """Test loading from a directory with TAB-separated files (legacy ParaBible)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake Bible text file
            text = "\n".join(
                f"4000{i:04d}\t" + " ".join(["word"] * 100)
                for i in range(600)
            )
            with open(os.path.join(tmpdir, "eng.txt"), "w") as f:
                f.write(text)

            texts = load_parabible_texts(tmpdir, script_filter=False,
                                          min_tokens=100)
            assert "eng" in texts
            assert len(texts["eng"].split()) >= 100

    def test_load_bible_texts_both(self):
        """Test loading from both eBible and ParaBible, merging results."""
        with tempfile.TemporaryDirectory() as ebible_dir, \
             tempfile.TemporaryDirectory() as para_dir:
            # eBible: eng (shorter) + fra
            eng_lines = [" ".join(["hello"] * 100) for _ in range(600)]
            with open(os.path.join(ebible_dir, "eng-engKJV.txt"), "w") as f:
                f.write("\n".join(eng_lines))
            fra_lines = [" ".join(["bonjour"] * 100) for _ in range(600)]
            with open(os.path.join(ebible_dir, "fra-fraLSG.txt"), "w") as f:
                f.write("\n".join(fra_lines))

            # ParaBible: eng (longer) + spa
            eng_lines_long = [" ".join(["hello"] * 100) for _ in range(900)]
            eng_text = "\n".join(
                f"4000{i:04d}\t" + eng_lines_long[i]
                for i in range(len(eng_lines_long))
            )
            with open(os.path.join(para_dir, "eng.txt"), "w") as f:
                f.write(eng_text)
            spa_lines = [" ".join(["hola"] * 100) for _ in range(600)]
            spa_text = "\n".join(
                f"4000{i:04d}\t" + spa_lines[i]
                for i in range(len(spa_lines))
            )
            with open(os.path.join(para_dir, "spa.txt"), "w") as f:
                f.write(spa_text)

            texts = load_bible_texts(
                bible_source="both",
                ebible_dir=ebible_dir,
                parabible_dir=para_dir,
                script_filter=False,
                min_tokens=100,
            )
            # Should have eng, fra, spa
            assert "eng" in texts
            assert "fra" in texts
            assert "spa" in texts
            # eng should come from ParaBible (longer)
            assert len(texts["eng"].split()) >= 90000

    def test_load_bible_texts_single_source(self):
        """Test load_bible_texts with a single source selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lines = [" ".join(["word"] * 100) for _ in range(600)]
            with open(os.path.join(tmpdir, "eng-engKJV.txt"), "w") as f:
                f.write("\n".join(lines))

            # ebible only
            texts = load_bible_texts(
                bible_source="ebible",
                ebible_dir=tmpdir,
                script_filter=False,
                min_tokens=100,
            )
            assert "eng" in texts

            # parabible only
            texts = load_bible_texts(
                bible_source="parabible",
                parabible_dir=tmpdir,
                script_filter=False,
                min_tokens=100,
            )
            assert "eng" in texts


# ============================================================================
# Char-LM Tests
# ============================================================================

class TestCharLM:
    def test_build_vocab(self):
        texts = {"eng": "hello world " * 1000, "fra": "bonjour monde " * 1000}
        vocab = build_vocab(texts, min_freq=2, max_vocab=100)
        assert "<PAD>" in vocab
        assert "<UNK>" in vocab
        assert "h" in vocab

    def test_charlm_model_forward(self):
        model = CharLSTM_LM(n_chars=50, n_langs=5,
                            char_emb_dim=16, lang_emb_dim=8,
                            hidden_dim=32, n_layers=2)
        lang_idx = torch.tensor([0, 1])
        char_seq = torch.randint(0, 50, (2, 20))
        logits = model(lang_idx, char_seq)
        assert logits.shape == (2, 20, 50)

    def test_charlm_get_embeddings(self):
        model = CharLSTM_LM(n_chars=50, n_langs=5,
                            char_emb_dim=16, lang_emb_dim=8,
                            hidden_dim=32, n_layers=2)
        embs = model.get_lang_embeddings()
        assert embs.shape == (5, 8)

    def test_train_charlm(self):
        texts = {
            "eng": "the quick brown fox jumps over the lazy dog " * 2000,
            "fra": "le rapide renard brun saute par dessus le chien " * 2000,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            embs = train_charlm(
                texts,
                output_dir=tmpdir,
                char_emb_dim=16,
                lang_emb_dim=8,
                hidden_dim=32,
                n_layers=1,
                seq_len=50,
                samples_per_lang=20,
                batch_size=4,
                max_epochs=2,
                patience=2,
                device="cpu",
            )
            assert embs.shape == (2, 8)
            assert os.path.exists(os.path.join(tmpdir, "lang_embeddings.npy"))
            assert os.path.exists(os.path.join(tmpdir, "lang2idx.json"))


# ============================================================================
# Evaluation Pipeline Tests
# ============================================================================

class TestEvaluation:
    def test_split_by_branch(self):
        df, feature_cols = make_synthetic_wals(n_langs=60, n_features=5)
        bmat, names, groups, _val_names = binarise_wals(df, feature_cols)

        train_ds, eval_langs, eval_feats, eval_vals = split_by_branch(
            df, bmat, groups,
            target_branch="Romance",
            in_branch_train_frac=0.10,
        )

        assert len(train_ds) > 0
        assert len(eval_langs) > 0
        assert len(eval_feats) == len(eval_vals)

        # Eval should only contain in-branch languages
        romance_indices = set(
            np.where((df["genus"] == "Romance").values)[0])
        assert all(li in romance_indices for li in eval_langs)

    def test_split_fraction_uses_total_observed(self):
        """Bug 1 fix: in_branch_train_frac should apply to n_obs, not pool size."""
        df, feature_cols = make_synthetic_wals(n_langs=60, n_features=8)
        bmat, names, groups, _val_names = binarise_wals(df, feature_cols)

        # With frac=0.10, each in-branch language should get ~10% of its
        # observed features for training (not 10% of the tiny 20% pool).
        train_ds, eval_langs, eval_feats, eval_vals = split_by_branch(
            df, bmat, groups,
            target_branch="Romance",
            in_branch_train_frac=0.10,
        )

        # Count in-branch training cells — should be non-zero
        romance_indices = set(
            np.where((df["genus"] == "Romance").values)[0])
        in_branch_train_count = 0
        for i in range(len(train_ds)):
            li, _, _ = train_ds[i]
            if int(li) in romance_indices:
                in_branch_train_count += 1
        assert in_branch_train_count > 0, (
            "Bug 1 regression: no in-branch training data at frac=0.10")

    def test_split_no_leakage(self):
        """Verify no feature-level information leaks between train and eval."""
        df, feature_cols = make_synthetic_wals(n_langs=60, n_features=5)
        bmat, names, groups, _val_names = binarise_wals(df, feature_cols)

        train_ds, eval_langs, eval_feats, eval_vals = split_by_branch(
            df, bmat, groups,
            target_branch="Romance",
            in_branch_train_frac=0.20,
        )

        # Build sets of (lang, orig_feat) pairs in each split
        train_pairs = set()
        for i in range(len(train_ds)):
            li, fi, _ = train_ds[i]
            li, fi = int(li), int(fi)
            for orig_feat, bins in groups.items():
                if fi in bins:
                    train_pairs.add((li, orig_feat))
                    break

        eval_pairs = set()
        for i in range(len(eval_langs)):
            li, fi = int(eval_langs[i]), int(eval_feats[i])
            for orig_feat, bins in groups.items():
                if fi in bins:
                    eval_pairs.add((li, orig_feat))
                    break

        # No overlap between train and eval (lang, orig_feat) pairs
        overlap = train_pairs & eval_pairs
        assert len(overlap) == 0, f"Leakage detected: {len(overlap)} pairs"

    def test_decode_and_evaluate(self):
        df, feature_cols = make_synthetic_wals(n_langs=60, n_features=5)
        bmat, names, groups, val_names = binarise_wals(df, feature_cols)

        train_ds, eval_langs, eval_feats, eval_vals = split_by_branch(
            df, bmat, groups,
            target_branch="Romance",
            in_branch_train_frac=0.10,
        )

        model = TypologicalMF(len(df), len(names), embed_dim=8)
        train_model(model, train_ds, n_epochs=3, batch_size=32)

        f1 = decode_and_evaluate(
            model, eval_langs, eval_feats, eval_vals,
            groups, val_names, df, "Romance")
        assert 0.0 <= f1 <= 1.0

    def test_majority_baseline(self):
        df, feature_cols = make_synthetic_wals(n_langs=60, n_features=5)
        bmat, names, groups, val_names = binarise_wals(df, feature_cols)

        train_ds, eval_langs, eval_feats, eval_vals = split_by_branch(
            df, bmat, groups,
            target_branch="Romance",
            in_branch_train_frac=0.10,
        )

        f1 = majority_baseline(
            train_ds, eval_langs, eval_feats, eval_vals,
            groups, val_names, bmat.shape[1])
        assert 0.0 <= f1 <= 1.0

    def test_knn_baseline(self):
        df, feature_cols = make_synthetic_wals(n_langs=60, n_features=5)
        bmat, names, groups, val_names = binarise_wals(df, feature_cols)
        embs = np.random.randn(len(df), 16).astype(np.float32)

        train_ds, eval_langs, eval_feats, eval_vals = split_by_branch(
            df, bmat, groups,
            target_branch="Romance",
            in_branch_train_frac=0.10,
        )

        f1 = knn_baseline(
            embs, train_ds, eval_langs, eval_feats, eval_vals,
            groups, val_names, df, "Romance", bmat.shape[1], k=1)
        assert 0.0 <= f1 <= 1.0

    def test_run_experiments_small(self):
        """Full experiment on small synthetic data."""
        df, feature_cols = make_synthetic_wals(n_langs=60, n_features=5)
        bmat, names, groups, val_names = binarise_wals(df, feature_cols)

        results = run_experiments(
            df, bmat, groups,
            feature_value_names=val_names,
            in_branch_fracs=[0.0, 0.10],
            n_repeats=1,
            embed_dim=8,
            n_epochs=2,
            batch_size=32,
            min_branch_size=5,
        )

        assert len(results) > 0
        assert "branch" in results.columns
        assert "model" in results.columns
        assert "f1" in results.columns

        summary = summarise_results(results)
        assert len(summary) > 0
        assert "mean_f1" in summary.columns

    def test_run_experiments_with_pretrained(self):
        """Full experiment with pretrained embeddings."""
        df, feature_cols = make_synthetic_wals(n_langs=60, n_features=5)
        bmat, names, groups, val_names = binarise_wals(df, feature_cols)
        embs = np.random.randn(len(df), 16).astype(np.float32)

        results = run_experiments(
            df, bmat, groups,
            feature_value_names=val_names,
            in_branch_fracs=[0.0],
            n_repeats=1,
            embed_dim=8,
            n_epochs=2,
            batch_size=32,
            pretrained_embs=embs,
            min_branch_size=5,
        )

        models_tested = set(results["model"])
        assert "T-CF" in models_tested
        assert "SemiSup" in models_tested
        assert "Freq" in models_tested
        assert "KNN" in models_tested

    def test_bible_wals_intersection_filter(self):
        """Bug 2 fix: only languages with Bible data should be kept."""
        df, feature_cols = make_synthetic_wals(n_langs=60, n_features=5)
        bmat, names, groups, _val_names = binarise_wals(df, feature_cols)

        # Simulate pretrained embeddings where only first 30 langs have data
        embs = np.zeros((60, 16), dtype=np.float32)
        embs[:30] = np.random.randn(30, 16).astype(np.float32)

        # Apply the Bible filter (same logic as run_all.py Step 3b)
        has_bible = np.any(embs != 0, axis=1)
        assert has_bible.sum() == 30

        df_filtered = df[has_bible].reset_index(drop=True)
        bmat_filtered = bmat[has_bible]
        embs_filtered = embs[has_bible]

        bmat_filtered, _, groups_filtered, _vn = binarise_wals(
            df_filtered, feature_cols)

        assert len(df_filtered) == 30
        assert bmat_filtered.shape[0] == 30
        assert embs_filtered.shape[0] == 30


# ============================================================================
# Integration test
# ============================================================================

class TestIntegration:
    def test_full_pipeline_synthetic(self):
        """Test the full pipeline with synthetic data (no external deps)."""
        df, feature_cols = make_synthetic_wals(n_langs=60, n_features=5)
        bmat, names, groups, val_names = binarise_wals(df, feature_cols)

        # Create synthetic Bible texts matching some WALS languages
        bible_texts = {}
        for i in range(20):
            iso = f"x{i:02d}"
            bible_texts[iso] = "the quick brown fox " * 15000

        # Match
        matched = match_wals_to_bible(df, list(bible_texts.keys()))
        assert len(matched) > 0

        # Train char-LM
        with tempfile.TemporaryDirectory() as tmpdir:
            embs = train_charlm(
                bible_texts,
                output_dir=tmpdir,
                char_emb_dim=16,
                lang_emb_dim=8,
                hidden_dim=32,
                n_layers=1,
                seq_len=50,
                samples_per_lang=10,
                batch_size=4,
                max_epochs=2,
                patience=2,
            )

            with open(os.path.join(tmpdir, "lang2idx.json")) as f:
                lang2idx = json.load(f)

            # Align
            aligned = align_embeddings(embs, lang2idx, df, matched)
            assert aligned.shape[0] == len(df)

            # Run experiments
            results = run_experiments(
                df, bmat, groups,
                feature_value_names=val_names,
                in_branch_fracs=[0.0, 0.10],
                n_repeats=1,
                embed_dim=8,
                n_epochs=2,
                batch_size=32,
                pretrained_embs=aligned,
                min_branch_size=5,
            )

            summary = summarise_results(results)
            assert len(summary) > 0

            models_tested = set(results["model"])
            assert "T-CF" in models_tested
            assert "SemiSup" in models_tested
            assert "Freq" in models_tested
            assert "KNN" in models_tested


# ============================================================================
# Grambank Tests
# ============================================================================

def make_synthetic_grambank_repo(tmpdir, n_langs=80, n_features=10, seed=42):
    """
    Create a synthetic Grambank-like CLDF repo for testing.

    Returns the path to the repo root.
    """
    rng = np.random.default_rng(seed)
    cldf_dir = os.path.join(tmpdir, "cldf")
    os.makedirs(cldf_dir, exist_ok=True)

    # --- languages.csv ---
    families = ["Atlantic-Congo", "Austronesian", "Indo-European",
                "Sino-Tibetan", "Uto-Aztecan"]
    macroareas = ["Africa", "Papunesia", "Eurasia", "Eurasia",
                  "North America"]
    lang_rows = []
    for i in range(n_langs):
        fam_idx = i % len(families)
        lang_rows.append({
            "ID": f"abcd{i:04d}",
            "Name": f"Language_{i}",
            "Macroarea": macroareas[fam_idx],
            "Latitude": rng.uniform(-60, 60),
            "Longitude": rng.uniform(-180, 180),
            "Glottocode": f"abcd{i:04d}",
            "Family": families[fam_idx],
        })
    pd.DataFrame(lang_rows).to_csv(
        os.path.join(cldf_dir, "languages.csv"), index=False)

    # --- parameters.csv ---
    param_rows = []
    for fi in range(n_features):
        param_rows.append({
            "ID": f"GB{fi:03d}",
            "Name": f"Feature {fi}",
            "Description": f"Test feature {fi}",
        })
    pd.DataFrame(param_rows).to_csv(
        os.path.join(cldf_dir, "parameters.csv"), index=False)

    # --- codes.csv and values.csv ---
    # Most features are binary (0/1), a few have 3 values
    code_rows = []
    value_rows = []
    vid = 0
    feature_cardinalities = []
    for fi in range(n_features):
        pid = f"GB{fi:03d}"
        # First 7 features are binary, rest have 3 values
        if fi < 7:
            n_vals = 2
            val_labels = ["no", "yes"]
        else:
            n_vals = 3
            val_labels = ["absent", "present", "both"]
        feature_cardinalities.append(n_vals)

        for vi in range(n_vals):
            code_rows.append({
                "ID": f"{pid}-{vi}",
                "Parameter_ID": pid,
                "Name": val_labels[vi],
            })

        # Generate values for each language
        for li in range(n_langs):
            # 15% missing
            if rng.random() < 0.15:
                continue
            val = rng.integers(0, n_vals)
            value_rows.append({
                "ID": f"v{vid}",
                "Language_ID": f"abcd{li:04d}",
                "Parameter_ID": pid,
                "Value": str(val),
                "Code_ID": f"{pid}-{val}",
            })
            vid += 1

    # Inject some "?" values (uncertain/not determinable) for testing
    # These should be filtered out by load_grambank_cldf
    for li in range(min(5, n_langs)):
        value_rows.append({
            "ID": f"v{vid}",
            "Language_ID": f"abcd{li:04d}",
            "Parameter_ID": "GB000",
            "Value": "?",
            "Code_ID": "",
        })
        vid += 1

    pd.DataFrame(code_rows).to_csv(
        os.path.join(cldf_dir, "codes.csv"), index=False)
    pd.DataFrame(value_rows).to_csv(
        os.path.join(cldf_dir, "values.csv"), index=False)

    return tmpdir


class TestGrambank:
    def test_extract_glottolog_genus(self):
        """Test Glottolog classification path parsing."""
        assert _extract_glottolog_genus(
            "Indo-European/Germanic/West Germanic", level=1) == "Germanic"
        assert _extract_glottolog_genus(
            "Indo-European/Germanic/West Germanic", level=2) == "West Germanic"
        assert _extract_glottolog_genus(
            "Indo-European/Germanic", level=0) == "Indo-European"
        # Not deep enough — returns deepest available
        assert _extract_glottolog_genus(
            "Indo-European", level=2) == "Indo-European"
        assert _extract_glottolog_genus("", level=1) is None
        assert _extract_glottolog_genus(None, level=1) is None

    def test_load_grambank_cldf_family_genus(self):
        """Test loading Grambank with Family-based genus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = make_synthetic_grambank_repo(tmpdir, n_langs=50,
                                                     n_features=8)
            df, feature_cols = load_grambank_cldf(
                repo_dir,
                min_lang_features=1,
                min_feature_value_langs=1,
                genus_source="family",
            )

            assert len(df) > 0
            assert len(feature_cols) > 0
            assert "genus" in df.columns
            assert "glottocode" in df.columns
            assert "family" in df.columns
            # All feature columns should start with feat_GB
            assert all(c.startswith("feat_GB") for c in feature_cols)
            # Genus should equal family when using family source
            assert (df["genus"] == df["family"]).all()

    def test_grambank_binarise(self):
        """Test that binarise_features works correctly on Grambank data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = make_synthetic_grambank_repo(tmpdir, n_langs=50,
                                                     n_features=8)
            df, feature_cols = load_grambank_cldf(
                repo_dir,
                min_lang_features=1,
                min_feature_value_langs=1,
                genus_source="family",
            )

            bmat, names, groups, value_names = binarise_features(
                df, feature_cols)

            assert bmat.shape[0] == len(df)
            assert bmat.shape[1] == len(names)
            # Binary features should produce single columns,
            # 3-valued features should produce 3 columns
            valid = bmat[~np.isnan(bmat)]
            assert set(valid.tolist()).issubset({0.0, 1.0})

    def test_grambank_branch_split(self):
        """Test branch-based splitting on Grambank data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = make_synthetic_grambank_repo(tmpdir, n_langs=80,
                                                     n_features=8)
            df, feature_cols = load_grambank_cldf(
                repo_dir,
                min_lang_features=1,
                min_feature_value_langs=1,
                genus_source="family",
            )

            bmat, names, groups, value_names = binarise_features(
                df, feature_cols)

            # Pick a family with enough languages
            branch_counts = df["genus"].value_counts()
            target = branch_counts[branch_counts >= 5].index[0]

            train_ds, eval_langs, eval_feats, eval_vals = split_by_branch(
                df, bmat, groups,
                target_branch=target,
                in_branch_train_frac=0.10,
            )

            assert len(train_ds) > 0
            assert len(eval_langs) > 0
            # Eval should only contain in-branch languages
            in_branch = set(np.where((df["genus"] == target).values)[0])
            assert all(li in in_branch for li in eval_langs)

    def test_binarise_features_alias(self):
        """Test that binarise_wals is an alias for binarise_features."""
        assert binarise_wals is binarise_features

    def test_grambank_values_are_categorical(self):
        """Verify Grambank values are treated as categorical labels, not numeric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = make_synthetic_grambank_repo(tmpdir, n_langs=30,
                                                     n_features=5)
            df, feature_cols = load_grambank_cldf(
                repo_dir,
                min_lang_features=1,
                min_feature_value_langs=1,
                genus_source="family",
            )

            # Values should be strings (human-readable labels), not numeric
            for col in feature_cols:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    assert all(isinstance(v, str) for v in non_null), \
                        f"Feature {col} has non-string values"

    def test_grambank_with_glottolog_genus_missing(self):
        """Test fallback when glottolog requested but not provided."""
        import warnings
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = make_synthetic_grambank_repo(tmpdir, n_langs=30,
                                                     n_features=5)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                df, feature_cols = load_grambank_cldf(
                    repo_dir,
                    min_lang_features=1,
                    min_feature_value_langs=1,
                    genus_source="glottolog",
                    glottolog_repo_dir=None,
                )
                # Should warn and fall back to family
                assert any("glottolog" in str(warning.message).lower()
                           for warning in w)
            assert "genus" in df.columns

    def test_grambank_question_mark_filter(self):
        """Verify that '?' values are filtered out before pivot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = make_synthetic_grambank_repo(tmpdir, n_langs=30,
                                                     n_features=5)
            df, feature_cols = load_grambank_cldf(
                repo_dir,
                min_lang_features=1,
                min_feature_value_langs=1,
                genus_source="family",
            )

            # No feature column should contain "value_?" or "?" as a value
            for col in feature_cols:
                non_null = df[col].dropna()
                vals = set(non_null.tolist())
                assert "value_?" not in vals, \
                    f"Feature {col} contains 'value_?'"
                assert "?" not in vals, \
                    f"Feature {col} contains '?'"

    def test_grambank_uncertain_label_filter(self):
        """Verify that 'Not known' / 'Uncertain' labels are filtered out."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cldf_dir = os.path.join(tmpdir, "cldf")
            os.makedirs(cldf_dir, exist_ok=True)

            # Minimal languages.csv
            pd.DataFrame([{
                "ID": "abcd0001", "Name": "Lang1",
                "Macroarea": "Eurasia", "Family": "TestFam",
            }, {
                "ID": "abcd0002", "Name": "Lang2",
                "Macroarea": "Eurasia", "Family": "TestFam",
            }]).to_csv(os.path.join(cldf_dir, "languages.csv"), index=False)

            # parameters.csv
            pd.DataFrame([{
                "ID": "GB001", "Name": "Feat1", "Description": "test",
            }]).to_csv(os.path.join(cldf_dir, "parameters.csv"), index=False)

            # codes.csv — map "?" code to "Not known"
            pd.DataFrame([
                {"ID": "GB001-0", "Parameter_ID": "GB001", "Name": "no"},
                {"ID": "GB001-1", "Parameter_ID": "GB001", "Name": "yes"},
                {"ID": "GB001-?", "Parameter_ID": "GB001",
                 "Name": "Not known"},
            ]).to_csv(os.path.join(cldf_dir, "codes.csv"), index=False)

            # values.csv — one real value and one "?" with "Not known" label
            pd.DataFrame([
                {"ID": "v1", "Language_ID": "abcd0001",
                 "Parameter_ID": "GB001", "Value": "1",
                 "Code_ID": "GB001-1"},
                {"ID": "v2", "Language_ID": "abcd0002",
                 "Parameter_ID": "GB001", "Value": "?",
                 "Code_ID": "GB001-?"},
                {"ID": "v3", "Language_ID": "abcd0001",
                 "Parameter_ID": "GB001", "Value": "0",
                 "Code_ID": "GB001-0"},
            ]).to_csv(os.path.join(cldf_dir, "values.csv"), index=False)

            df, feature_cols = load_grambank_cldf(
                tmpdir,
                min_lang_features=1,
                min_feature_value_langs=1,
                genus_source="family",
            )

            # The "Not known" value should have been filtered out
            for col in feature_cols:
                non_null = df[col].dropna()
                for v in non_null:
                    assert "not known" not in str(v).lower(), \
                        f"'Not known' value leaked through in {col}"

    def test_glottolog_genus_map_with_classification(self):
        """Test load_glottolog_genus_map with a Classification column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cldf_dir = os.path.join(tmpdir, "cldf")
            os.makedirs(cldf_dir, exist_ok=True)
            pd.DataFrame([
                {"ID": "germ1234", "Name": "German",
                 "Classification": "Indo-European/Germanic/West Germanic"},
                {"ID": "fren1234", "Name": "French",
                 "Classification": "Indo-European/Romance/Gallo-Romance"},
                {"ID": "mand1234", "Name": "Mandarin",
                 "Classification": "Sino-Tibetan/Sinitic"},
            ]).to_csv(os.path.join(cldf_dir, "languages.csv"), index=False)

            genus_map = load_glottolog_genus_map(tmpdir, genus_level=1)
            assert genus_map["germ1234"] == "Germanic"
            assert genus_map["fren1234"] == "Romance"
            assert genus_map["mand1234"] == "Sinitic"

    def test_glottolog_genus_map_with_parent_id(self):
        """Test load_glottolog_genus_map hierarchy reconstruction from Parent_ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cldf_dir = os.path.join(tmpdir, "cldf")
            os.makedirs(cldf_dir, exist_ok=True)
            # Simulate a Glottolog CSV with Family_ID + Parent_ID but no
            # Classification column
            pd.DataFrame([
                {"ID": "indo1234", "Name": "Indo-European",
                 "Family_ID": "", "Parent_ID": ""},
                {"ID": "germ1235", "Name": "Germanic",
                 "Family_ID": "indo1234", "Parent_ID": "indo1234"},
                {"ID": "west1236", "Name": "West Germanic",
                 "Family_ID": "indo1234", "Parent_ID": "germ1235"},
                {"ID": "germ1237", "Name": "German",
                 "Family_ID": "indo1234", "Parent_ID": "west1236"},
            ]).to_csv(os.path.join(cldf_dir, "languages.csv"), index=False)

            genus_map = load_glottolog_genus_map(tmpdir, genus_level=1)
            # German: path is [Indo-European, Germanic, West Germanic, German]
            # level=1 should give "Germanic"
            assert genus_map["germ1237"] == "Germanic"

    def test_glottolog_genus_map_family_fallback(self):
        """Test load_glottolog_genus_map falls back to Family column."""
        import warnings
        with tempfile.TemporaryDirectory() as tmpdir:
            cldf_dir = os.path.join(tmpdir, "cldf")
            os.makedirs(cldf_dir, exist_ok=True)
            # CSV with only ID, Name, Family — no Classification or Parent_ID
            pd.DataFrame([
                {"ID": "germ1234", "Name": "German",
                 "Family": "Indo-European"},
                {"ID": "mand1234", "Name": "Mandarin",
                 "Family": "Sino-Tibetan"},
            ]).to_csv(os.path.join(cldf_dir, "languages.csv"), index=False)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                genus_map = load_glottolog_genus_map(tmpdir, genus_level=1)
                # Should warn about falling back
                assert any("family" in str(warning.message).lower()
                           for warning in w)
            assert genus_map["germ1234"] == "Indo-European"
            assert genus_map["mand1234"] == "Sino-Tibetan"


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
