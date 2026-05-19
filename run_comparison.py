#!/usr/bin/env python3
"""
Unified entry point for running the binary baseline (T-CF) vs.
learned feature-value embedding comparison on WALS or Grambank.

Usage
-----
    # WALS (original Bjerva et al. replication):
    python run_comparison.py --database wals --wals_repo /path/to/wals

    # Grambank with Glottolog genus mapping (recommended):
    python run_comparison.py --database grambank \\
        --grambank_repo /path/to/grambank \\
        --glottolog_repo /path/to/glottolog-cldf

    # Grambank with Family-based genus (no Glottolog needed, coarser):
    python run_comparison.py --database grambank \\
        --grambank_repo /path/to/grambank \\
        --genus_source family

Grambank-specific notes
-----------------------
Grambank features are MOSTLY binary (yes=1, no=0), with some 3+ valued
features. Implications for each model:

  Binary baseline (T-CF): Most features produce single binary columns
  (not one-hot expansions), so the binary matrix is closer to the
  original data than for WALS.

  Learned embedding model: Binary features get two value embeddings each.
  The model's advantage over the baseline comes primarily from:
    (a) The shared value embedding space allowing cross-feature learning
    (b) Features with 3+ values getting proper categorical treatment
    (c) The softmax ensuring mutual exclusivity (minor for binary features)
"""

import argparse
import sys

import numpy as np
import pandas as pd

from data_preparation import load_wals_cldf, load_grambank_cldf, binarise_features
from compare_models import run_comparison, summarise


def main():
    parser = argparse.ArgumentParser(
        description="Compare binary vs. learned-embedding models on "
                    "WALS or Grambank",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Database selection ---
    parser.add_argument("--database", type=str, default="wals",
                        choices=["wals", "grambank"],
                        help="Typological database to use (default: wals)")
    parser.add_argument("--wals_repo", type=str, default=None,
                        help="Path to cloned cldf-datasets/wals repo "
                             "(required for --database wals)")
    parser.add_argument("--grambank_repo", type=str, default=None,
                        help="Path to cloned glottobank/grambank repo "
                             "(required for --database grambank)")
    parser.add_argument("--glottolog_repo", type=str, default=None,
                        help="Path to cloned glottolog-cldf repo. "
                             "Provides accurate genus-level groupings for "
                             "Grambank (recommended).")
    parser.add_argument("--genus_source", type=str, default="glottolog",
                        choices=["glottolog", "family"],
                        help="How to derive genus groupings for Grambank: "
                             "'glottolog' uses Glottolog classification "
                             "(requires --glottolog_repo), 'family' uses "
                             "Grambank Family column (coarser). "
                             "Default: glottolog.")

    # --- Model hyperparameters ---
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=0.1)
    parser.add_argument("--n_repeats", type=int, default=5)

    # --- Evaluation ---
    parser.add_argument("--min_branch_size", type=int, default=5,
                        help="Skip branches with fewer languages (default: 5)")
    parser.add_argument("--branches", type=str, nargs="+", default=None,
                        help="Only evaluate specific branches")

    # --- Output ---
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_csv", type=str,
                        default="comparison_results.csv")

    args = parser.parse_args()

    # --- Load data ---
    if args.database == "grambank":
        if args.grambank_repo is None:
            parser.error("--grambank_repo is required when --database=grambank")
        df, feature_cols = load_grambank_cldf(
            args.grambank_repo,
            genus_source=args.genus_source,
            glottolog_repo_dir=args.glottolog_repo,
        )

        print(f"DEBUG: df shape = {df.shape}")
        print(f"DEBUG: feature_cols count = {len(feature_cols)}")
        print(f"DEBUG: sample feature values:")
        for col in feature_cols[:5]:
            print(f"  {col}: {df[col].value_counts().to_dict()}")

    else:
        if args.wals_repo is None:
            parser.error("--wals_repo is required when --database=wals")
        df, feature_cols = load_wals_cldf(args.wals_repo)

    # --- Run comparison ---
    results = run_comparison(
        df=df,
        feature_cols=feature_cols,
        embed_dim=args.embed_dim,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2_reg=args.l2_reg,
        n_repeats=args.n_repeats,
        device=args.device,
        min_branch_size=args.min_branch_size,
        only_branches=args.branches,
    )

    # --- Save and display ---
    results.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")

    summary, pivot = summarise(results)
    print("\n" + "=" * 70)
    print(f"COMPARISON — {args.database.upper()}: "
          f"Binary (T-CF) vs. Learned Embeddings")
    print("=" * 70)
    print("\nDetailed:")
    print(summary.to_string(index=False))
    print("\nSide-by-side (mean F1):")
    print(pivot.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
