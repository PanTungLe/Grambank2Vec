#!/usr/bin/env python3
"""
End-to-End Experiment Runner
==============================
Replicates the full pipeline from Bjerva et al. (NAACL 2019):
  "A Probabilistic Generative Model of Linguistic Typology"

Steps:
  1. Clone WALS and ParaBible repos (if not already present)
  2. Download and extract ParaBible corpus texts
  3. Load and binarise WALS data
  4. Match WALS languages to Bible translations
  5. Train character-level LM to get language embeddings
  6. Align embeddings with WALS language order
  7. Run all experiments (T-CF, SemiSup, baselines)
  8. Print summary table (Table 1 from the paper)

Usage:
    python run_all.py --output_dir experiments/
    python run_all.py --wals_repo /path/to/wals --parabible_dir /path/to/texts
    python run_all.py --skip_charlm  # Skip char-LM training (T-CF only)
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from data_preparation import (
    load_wals_cldf,
    binarise_wals,
    load_parabible_texts,
    match_wals_to_bible,
    align_embeddings,
)
from char_lm import train_charlm
from evaluation_pipeline import run_experiments, summarise_results


def clone_repo(url: str, target_dir: str) -> str:
    """Clone a git repo if it doesn't exist yet."""
    if os.path.isdir(target_dir):
        print(f"  Repository already exists: {target_dir}")
        return target_dir
    print(f"  Cloning {url} → {target_dir}")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, target_dir],
        check=True,
    )
    return target_dir


def download_parabible_corpus(parabible_repo: str, output_dir: str) -> str:
    """
    Download the ParaBible corpus zip and extract text files.

    The ParaBible project downloads its corpus from a remote server.
    This function replicates that download step without requiring Docker.
    """
    corpus_dir = os.path.join(output_dir, "corpus-txt")
    if os.path.isdir(corpus_dir) and len(os.listdir(corpus_dir)) > 10:
        print(f"  Corpus already extracted: {corpus_dir}")
        return corpus_dir

    os.makedirs(output_dir, exist_ok=True)

    # Try to download the corpus zip directly
    zip_url = "http://91.200.84.6/parabible-data/full_pb_corpus.zip"
    zip_path = os.path.join(output_dir, "full_pb_corpus.zip")

    if not os.path.exists(zip_path):
        print(f"  Downloading corpus from {zip_url}")
        print("  (This may take a while for ~1800 translations)")
        try:
            import requests
            resp = requests.get(zip_url, stream=True, timeout=300)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  Downloaded to {zip_path}")
        except Exception as e:
            print(f"  WARNING: Could not download corpus: {e}")
            print("  You can manually place Bible text files in:")
            print(f"    {corpus_dir}")
            print("  (One .txt file per language, TAB-separated verse_id\\ttext)")
            os.makedirs(corpus_dir, exist_ok=True)
            return corpus_dir

    # Extract
    print(f"  Extracting {zip_path}")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    print(f"  Extracted to {output_dir}")

    # The zip might extract to a subdirectory
    if not os.path.isdir(corpus_dir):
        # Look for the actual extracted directory
        for entry in os.listdir(output_dir):
            candidate = os.path.join(output_dir, entry)
            if os.path.isdir(candidate) and entry != "corpus-txt":
                # Check if it has .txt files
                txt_files = [f for f in os.listdir(candidate)
                             if f.endswith(".txt")]
                if len(txt_files) > 10:
                    os.rename(candidate, corpus_dir)
                    break

    if not os.path.isdir(corpus_dir):
        os.makedirs(corpus_dir, exist_ok=True)

    return corpus_dir


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end replication of Bjerva et al. (NAACL 2019)")

    # Data source paths
    parser.add_argument("--wals_repo", type=str, default=None,
                        help="Path to cloned cldf-datasets/wals repo "
                             "(will be cloned if not provided)")
    parser.add_argument("--parabible_dir", type=str, default=None,
                        help="Path to directory with Bible text files "
                             "(will be downloaded if not provided)")
    parser.add_argument("--manual_mapping", type=str, default=None,
                        help="CSV file mapping wals_code to bible_lang_id")

    # Output
    parser.add_argument("--output_dir", type=str, default="experiments",
                        help="Directory for all outputs")

    # Pipeline control
    parser.add_argument("--skip_charlm", action="store_true",
                        help="Skip char-LM training (run T-CF only)")
    parser.add_argument("--pretrained_embs", type=str, default=None,
                        help="Path to pre-existing aligned embeddings .npy")

    # Char-LM hyperparameters
    parser.add_argument("--charlm_hidden_dim", type=int, default=1024)
    parser.add_argument("--charlm_lang_emb_dim", type=int, default=64)
    parser.add_argument("--charlm_max_epochs", type=int, default=30)
    parser.add_argument("--charlm_patience", type=int, default=5)
    parser.add_argument("--charlm_samples_per_lang", type=int, default=1000)

    # Experiment hyperparameters
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=0.1)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--min_branch_size", type=int, default=4)

    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ================================================================
    # Step 1: Get data repositories
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Data Acquisition")
    print("=" * 60)

    # WALS
    if args.wals_repo is None:
        args.wals_repo = os.path.join(args.output_dir, "wals")
        clone_repo("https://github.com/cldf-datasets/wals.git",
                    args.wals_repo)
    print(f"WALS repo: {args.wals_repo}")

    # ParaBible
    if not args.skip_charlm and args.pretrained_embs is None:
        if args.parabible_dir is None:
            parabible_repo = os.path.join(args.output_dir, "parabible")
            clone_repo("https://github.com/LingConLab/parabible.git",
                        parabible_repo)
            # Download and extract corpus texts
            corpus_dir = download_parabible_corpus(
                parabible_repo,
                os.path.join(args.output_dir, "parabible_corpus"))
            args.parabible_dir = corpus_dir
        print(f"ParaBible texts: {args.parabible_dir}")

    # ================================================================
    # Step 2: Load and prepare WALS data
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Load and Binarise WALS")
    print("=" * 60)

    df, feature_cols = load_wals_cldf(args.wals_repo)
    binary_matrix, bin_names, feature_groups = binarise_wals(df, feature_cols)

    # Save prepared data
    wals_csv = os.path.join(args.output_dir, "wals_prepared.csv")
    df.to_csv(wals_csv, index=False)
    print(f"Saved: {wals_csv}")

    # ================================================================
    # Step 3: Train char-LM and get language embeddings
    # ================================================================
    pretrained_embs = None
    charlm_lang2idx = None

    if args.pretrained_embs:
        print("\n" + "=" * 60)
        print("STEP 3: Loading Pre-trained Embeddings")
        print("=" * 60)
        pretrained_embs = np.load(args.pretrained_embs).astype(np.float32)
        print(f"Loaded: {pretrained_embs.shape}")

        if pretrained_embs.shape[0] != len(df):
            print(f"WARNING: Shape mismatch ({pretrained_embs.shape[0]} vs "
                  f"{len(df)} WALS languages). Padding/truncating.")
            if pretrained_embs.shape[0] < len(df):
                padded = np.zeros((len(df), pretrained_embs.shape[1]),
                                  dtype=np.float32)
                padded[:pretrained_embs.shape[0]] = pretrained_embs
                pretrained_embs = padded
            else:
                pretrained_embs = pretrained_embs[:len(df)]

    elif not args.skip_charlm and args.parabible_dir:
        print("\n" + "=" * 60)
        print("STEP 3: Character-level LM Training")
        print("=" * 60)

        # 3a. Load Bible texts
        bible_texts = load_parabible_texts(
            args.parabible_dir,
            script_filter=True,
            min_tokens=50000,
        )

        if len(bible_texts) == 0:
            print("WARNING: No Bible texts loaded. Running T-CF only.")
        else:
            # 3b. Match WALS languages to Bible translations
            matched_df = match_wals_to_bible(
                df, list(bible_texts.keys()),
                manual_mapping_csv=args.manual_mapping,
            )

            if len(matched_df) < 10:
                print("WARNING: Too few matches. Running T-CF only.")
            else:
                # 3c. Train char-LM
                charlm_dir = os.path.join(args.output_dir, "charlm_output")
                raw_embs = train_charlm(
                    bible_texts,
                    output_dir=charlm_dir,
                    hidden_dim=args.charlm_hidden_dim,
                    lang_emb_dim=args.charlm_lang_emb_dim,
                    samples_per_lang=args.charlm_samples_per_lang,
                    max_epochs=args.charlm_max_epochs,
                    patience=args.charlm_patience,
                    device=args.device,
                )

                # 3d. Load lang2idx mapping
                lang2idx_path = os.path.join(charlm_dir, "lang2idx.json")
                with open(lang2idx_path) as f:
                    charlm_lang2idx = json.load(f)

                # 3e. Align embeddings to WALS order
                pretrained_embs = align_embeddings(
                    raw_embs, charlm_lang2idx, df, matched_df)

                # Save aligned embeddings
                aligned_path = os.path.join(
                    args.output_dir, "lang_embeddings_aligned.npy")
                np.save(aligned_path, pretrained_embs)
                print(f"Saved aligned embeddings: {aligned_path}")

    elif args.skip_charlm:
        print("\n" + "=" * 60)
        print("STEP 3: Skipped (--skip_charlm)")
        print("=" * 60)

    # ================================================================
    # Step 4: Run experiments
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Running Experiments")
    print("=" * 60)

    results = run_experiments(
        df, binary_matrix, feature_groups,
        in_branch_fracs=[0.0, 0.01, 0.05, 0.10, 0.20],
        n_repeats=args.n_repeats,
        embed_dim=args.embed_dim,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2_reg=args.l2_reg,
        pretrained_embs=pretrained_embs,
        device=args.device,
        min_branch_size=args.min_branch_size,
    )

    # ================================================================
    # Step 5: Save and display results
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Results")
    print("=" * 60)

    results_csv = os.path.join(args.output_dir, "results.csv")
    results.to_csv(results_csv, index=False)
    print(f"Detailed results saved to {results_csv}")

    summary = summarise_results(results)
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS (cf. Table 1 in the paper)")
    print("=" * 60)
    print(summary.to_string(index=False))

    # Also save summary
    summary_csv = os.path.join(args.output_dir, "summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to {summary_csv}")

    # Print expected vs actual comparison
    expected = {
        0.0: {"T-CF": 0.40},
        0.01: {"T-CF": 0.46},
        0.05: {"T-CF": 0.66},
        0.10: {"T-CF": 0.78},
        0.20: {"T-CF": 0.88},
    }
    if pretrained_embs is not None:
        expected[0.0]["SemiSup"] = 0.39
        expected[0.01]["SemiSup"] = 0.53
        expected[0.05]["SemiSup"] = 0.76
        expected[0.10]["SemiSup"] = 0.90
        expected[0.20]["SemiSup"] = 0.98

    print("\n" + "=" * 60)
    print("COMPARISON WITH PAPER (approximate expected values)")
    print("=" * 60)
    print(f"{'Frac':<8} {'Model':<10} {'Got':<10} {'Expected':<10}")
    print("-" * 38)
    for _, row in summary.iterrows():
        frac = row["in_branch_frac"]
        model = row["model"]
        got = row["mean_f1"]
        exp = expected.get(frac, {}).get(model, "N/A")
        if isinstance(exp, float):
            print(f"{frac:<8.2f} {model:<10} {got:<10.4f} {exp:<10.2f}")
        else:
            print(f"{frac:<8.2f} {model:<10} {got:<10.4f} {exp:<10}")


if __name__ == "__main__":
    main()
