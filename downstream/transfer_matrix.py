"""
downstream/transfer_matrix.py
==============================
Build the full N_src × N_tgt POS transfer performance matrix.

For each source language s:
  1. Load train split for s
  2. Train a biLSTM POS tagger on s (with source-language vocabulary)
  3. For each target language t ≠ s:
       a. Load test split for t
       b. Evaluate trained tagger zero-shot on t
       c. Record token_accuracy

Output: transfer_matrix.csv  (rows = source, cols = target, values = accuracy)

Design notes
------------
- Uses a SHARED TAG VOCABULARY across all languages (universal POS = UPOS).
  Since all UD treebanks use the same 17-tag UPOS scheme, this is valid.
- Uses UNK for all target-language words (source vocabulary is small for
  low-resource languages, so most target words hit UNK anyway).
- Characters transfer more robustly than words — the char CNN gives the
  tagger some morphological generalisation.
- Models are NOT saved by default (disk-intensive).  Use --save_models
  to save for debugging.

Usage
-----
    python downstream/transfer_matrix.py \\
        --manifest downstream/ud_data/manifest.json \\
        --out_dir downstream/transfer \\
        --device cuda \\
        --n_epochs 30 \\
        --max_train_sents 5000 \\
        --max_test_sents 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from downstream.ud_data import load_pos_dataset
from downstream.tagger import (
    Vocab, build_vocabs, train_tagger, evaluate_tagger,
    save_tagger, load_tagger,
)


# ---------------------------------------------------------------------------
# Universal POS tag set (17 UPOS tags, same across all UD treebanks)
# ---------------------------------------------------------------------------

UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ",
    "SYM", "VERB", "X",
]


def build_universal_tag_vocab() -> Vocab:
    """
    Build a fixed tag vocabulary from the 17 universal POS tags.
    This is shared across ALL source and target languages.
    """
    tag_vocab = Vocab()
    tag_vocab.add("<PAD>")  # index 0
    for tag in UPOS_TAGS:
        tag_vocab.add(tag)
    tag_vocab.freeze()
    return tag_vocab


# ---------------------------------------------------------------------------
# Transfer matrix builder
# ---------------------------------------------------------------------------

def build_transfer_matrix(
    manifest: Dict[str, Dict[str, str]],
    out_dir: str,
    device: str = "cpu",
    n_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 5,
    max_train_sents: Optional[int] = 5000,
    max_test_sents:  Optional[int] = 500,
    save_models: bool = False,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Build the N_src × N_tgt transfer performance matrix.

    Parameters
    ----------
    manifest        {glottocode -> {split -> local_path}}
    out_dir         output directory
    max_train_sents cap training sentences per source language (for speed)
    max_test_sents  cap test sentences per target language (for speed)
    save_models     if True, save each source tagger to disk
    overwrite       if True, re-run even if result already exists

    Returns
    -------
    df  pd.DataFrame with index=source_glottocode, columns=target_glottocode,
        values=token_accuracy
    """
    os.makedirs(out_dir, exist_ok=True)
    matrix_path = os.path.join(out_dir, "transfer_matrix.csv")

    if os.path.exists(matrix_path) and not overwrite:
        print(f"Loading existing transfer matrix from {matrix_path}")
        return pd.read_csv(matrix_path, index_col=0)

    # Filter to languages with both train and test splits
    eligible = {
        glot: paths for glot, paths in manifest.items()
        if "train" in paths and "test" in paths
    }
    glottocodes = sorted(eligible.keys())
    n = len(glottocodes)
    print(f"\n{n} languages with train+test splits found in manifest")
    print(f"Transfer matrix size: {n}×{n} = {n*n} cells "
          f"({n*(n-1)} non-diagonal)")

    # Universal tag vocab
    tag_vocab = build_universal_tag_vocab()

    # Pre-load all test sets
    print("\nPre-loading test sets...")
    test_data: Dict[str, tuple] = {}
    for glot in glottocodes:
        w, t = load_pos_dataset(eligible[glot]["test"],
                                max_sentences=max_test_sents)
        if w:
            test_data[glot] = (w, t)
            print(f"  {glot}: {len(w)} test sentences")

    # Initialise result matrix with NaN
    matrix = pd.DataFrame(
        np.nan,
        index=glottocodes,
        columns=glottocodes,
    )

    # Per-source partial result file for resumability
    partial_path = os.path.join(out_dir, "transfer_matrix_partial.csv")
    if os.path.exists(partial_path) and not overwrite:
        partial = pd.read_csv(partial_path, index_col=0)
        # Merge existing results
        for src in partial.index:
            if src in matrix.index:
                for tgt in partial.columns:
                    if tgt in matrix.columns:
                        v = partial.loc[src, tgt]
                        if not np.isnan(v):
                            matrix.loc[src, tgt] = v
        print(f"Resumed from partial results: "
              f"{(~matrix.isna()).sum().sum()} cells already computed")

    # Main loop: train one source tagger, evaluate on all targets
    for src_i, src in enumerate(glottocodes):
        if not matrix.loc[src].isna().any():
            print(f"\n[{src_i+1}/{n}] {src}: all results cached, skipping")
            continue

        print(f"\n[{src_i+1}/{n}] Training on source: {src}")
        t_start = time.time()

        # Load source training data
        w_train, t_train = load_pos_dataset(
            eligible[src]["train"], max_sentences=max_train_sents)
        if len(w_train) < 10:
            print(f"  WARNING: only {len(w_train)} training sentences, skipping")
            continue

        print(f"  {len(w_train)} training sentences")

        # Build source vocabulary (word+char); tag vocab is universal
        word_vocab, char_vocab, _ = build_vocabs(w_train, t_train)

        # Optionally use source test as dev for early stopping
        w_dev, t_dev = None, None
        src_test = test_data.get(src)
        if src_test:
            w_dev, t_dev = src_test

        # Train
        model, word_vocab, char_vocab, _ = train_tagger(
            w_train, t_train,
            word_vocab=word_vocab,
            char_vocab=char_vocab,
            tag_vocab=tag_vocab,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            patience=patience,
            word_seqs_dev=w_dev,
            tag_seqs_dev=t_dev,
            verbose=True,
        )

        if save_models:
            model_dir = os.path.join(out_dir, "models", src)
            save_tagger(model, word_vocab, char_vocab, tag_vocab, model_dir)

        # Evaluate on all targets
        print(f"  Evaluating on {len(test_data)} target languages...")
        for tgt, (w_test, t_test) in test_data.items():
            if not np.isnan(matrix.loc[src, tgt]):
                continue

            metrics = evaluate_tagger(
                model, w_test, t_test,
                word_vocab, char_vocab, tag_vocab,
                device=device,
            )
            acc = metrics["token_accuracy"]
            matrix.loc[src, tgt] = acc

            if tgt == src:
                tag = "(same)"
            else:
                tag = ""
            print(f"    → {tgt}: {acc:.4f} {tag}")

        elapsed = time.time() - t_start
        print(f"  Done in {elapsed:.1f}s")

        # Save partial results after each source language
        matrix.to_csv(partial_path)

    # Save final matrix
    matrix.to_csv(matrix_path)
    print(f"\nTransfer matrix saved to: {matrix_path}")
    print(f"Shape: {matrix.shape}")
    print(f"Non-NaN cells: {(~matrix.isna()).sum().sum()}")

    # Print summary stats
    off_diag = matrix.values[~np.eye(n, dtype=bool)].flatten()
    off_diag = off_diag[~np.isnan(off_diag)]
    print(f"\nOff-diagonal accuracy: mean={off_diag.mean():.4f}  "
          f"std={off_diag.std():.4f}  "
          f"min={off_diag.min():.4f}  max={off_diag.max():.4f}")

    return matrix


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build POS transfer matrix for typological transfer experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True,
                   help="Path to manifest.json from ud_data.py")
    p.add_argument("--out_dir", required=True,
                   help="Output directory for transfer matrix and models")
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "cuda", "mps"],
                   help="PyTorch device")
    p.add_argument("--n_epochs", type=int, default=20,
                   help="Max training epochs per source tagger")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience")
    p.add_argument("--max_train_sents", type=int, default=5000,
                   help="Cap training sentences per language (0=no cap)")
    p.add_argument("--max_test_sents", type=int, default=500,
                   help="Cap test sentences per language (0=no cap)")
    p.add_argument("--save_models", action="store_true",
                   help="Save trained model checkpoints to disk")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Run on 3 languages only (smoke test)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.smoke:
        # Keep only 3 languages
        smoke_langs = ["stan1293", "nucl1301", "tami1289"]
        manifest = {k: v for k, v in manifest.items() if k in smoke_langs}
        print(f"SMOKE TEST: using {list(manifest.keys())}")

    max_train = args.max_train_sents if args.max_train_sents > 0 else None
    max_test  = args.max_test_sents  if args.max_test_sents  > 0 else None

    matrix = build_transfer_matrix(
        manifest=manifest,
        out_dir=args.out_dir,
        device=args.device,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        max_train_sents=max_train,
        max_test_sents=max_test,
        save_models=args.save_models,
        overwrite=args.overwrite,
    )
    print(matrix)


if __name__ == "__main__":
    main()
