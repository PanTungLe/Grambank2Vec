"""
Compare Binary Baseline vs. Learned Feature-Value Embeddings
==============================================================
Runs both models on identical train/eval splits for a fair comparison.

Usage:
    python compare_models.py --wals_repo /path/to/wals
    python compare_models.py --database grambank --grambank_repo /path/to/grambank

This produces a single CSV with columns:
    [branch, macroarea, in_branch_frac, repeat, model, f1]
where 'model' is one of: "Freq", "T-CF" (binary baseline), "Learned".
"""

import argparse
from typing import Optional
import numpy as np
import pandas as pd

from data_preparation import load_wals_cldf, load_grambank_cldf, binarise_features
from model import TypologicalMF, WALSDataset, train_model
from model_learned import (
    TypologicalMF_Learned, CategoricalTypDataset,
    prepare_categorical, train_model_learned,
    evaluate_learned, split_by_branch_categorical,
)
from evaluation_pipeline import (
    split_by_branch, decode_and_evaluate, majority_baseline,
)


def run_comparison(
    df: pd.DataFrame = None,
    feature_cols: list = None,
    wals_repo_dir: str = None,
    in_branch_fracs: list = [0.0, 0.01, 0.05, 0.10, 0.20],
    n_repeats: int = 5,
    embed_dim: int = 64,
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    l2_reg: float = 6.4,
    device: str = "cpu",
    min_branch_size: int = 5,
    only_branches: list = None,
    pretrained_embs: Optional[np.ndarray] = None,
    bible_lang_mask: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run both models on the same data and same splits.

    Accepts either a pre-loaded (df, feature_cols) or a wals_repo_dir
    for backwards compatibility.
    """

    # ================================================================
    # 1. Load data once, prepare BOTH representations
    # ================================================================
    if df is None:
        if wals_repo_dir is None:
            raise ValueError("Either (df, feature_cols) or wals_repo_dir "
                             "must be provided")
        df, feature_cols = load_wals_cldf(wals_repo_dir)

    # Bible ∩ WALS filter (Fix 1)
    bible_filtered = False
    if bible_lang_mask or pretrained_embs is not None:
        if bible_lang_mask:
            has_bible = np.load(bible_lang_mask).astype(bool)
        else:
            has_bible = np.any(pretrained_embs != 0, axis=1)
        n_before = len(df)
        df = df[has_bible].reset_index(drop=True)
        if pretrained_embs is not None:
            pretrained_embs = pretrained_embs[has_bible]
        print(f"Bible ∩ WALS filter: {n_before} → {len(df)} languages")
        bible_filtered = True
    else:
        print("WARNING: No Bible filter applied — results not comparable "
              "to Bjerva et al. Table 1. "
              "Labelling as 'full WALS (no Bible filter)'.")

    # Binary representation (Bjerva et al. baseline)
    binary_matrix, bin_names, feature_groups, feature_value_names = \
        binarise_features(df, feature_cols)

    # Categorical representation (learned embeddings)
    cat_matrix, kept_feat_names, feat_to_global_ids, feat_to_value_names = \
        prepare_categorical(df, feature_cols)

    n_langs = len(df)
    n_bfeat = binary_matrix.shape[1]
    n_total_values = max(max(gids) for gids in feat_to_global_ids.values()) + 1

    # ================================================================
    # 2. Identify qualifying branches
    # ================================================================
    branch_counts = df["genus"].value_counts()
    qualifying = branch_counts[branch_counts >= min_branch_size].index.tolist()
    if only_branches:
        qualifying = [b for b in qualifying if b in only_branches]
    print(f"\nQualifying branches (≥{min_branch_size} languages): "
          f"{len(qualifying)}")
    branch_list = sorted(qualifying)
    branch_to_int = {b: i for i, b in enumerate(branch_list)}
    print(f"Binary features: {n_bfeat}")
    print(f"Categorical features: {cat_matrix.shape[1]}, "
          f"total values: {n_total_values}")

    # ================================================================
    # 3. Run experiments with IDENTICAL random seeds
    # ================================================================
    rows = []

    for branch in qualifying:
        macroarea = df.loc[df["genus"] == branch, "macroarea"].iloc[0]
        print(f"\n{'='*60}")
        print(f"Branch: {branch}  (macroarea: {macroarea})")
        print(f"{'='*60}")

        for frac in in_branch_fracs:
            for rep in range(n_repeats):
                # Same seed for both models → same split (deterministic, PYTHONHASHSEED-independent)
                seed = rep * 1000 + branch_to_int[branch] * 37
                rng_bin = np.random.default_rng(seed=seed)
                rng_cat = np.random.default_rng(seed=seed)

                # ------------------------------------------------
                # Binary split (for baseline + T-CF)
                # ------------------------------------------------
                train_ds_bin, eval_l_bin, eval_f_bin, eval_v_bin = \
                    split_by_branch(
                        df, binary_matrix, feature_groups,
                        target_branch=branch,
                        in_branch_train_frac=frac,
                        rng=rng_bin)

                if len(eval_v_bin) == 0:
                    continue

                # --- Majority baseline ---
                f1_freq = majority_baseline(
                    train_ds_bin, eval_l_bin, eval_f_bin, eval_v_bin,
                    feature_groups, feature_value_names, n_bfeat)
                rows.append({
                    "branch": branch, "macroarea": macroarea,
                    "in_branch_frac": frac, "repeat": rep,
                    "model": "Freq", "f1": f1_freq
                })

                # --- T-CF (binary baseline) ---
                model_bin = TypologicalMF(n_langs, n_bfeat, embed_dim)
                print(f"\n  [T-CF] branch={branch}, frac={frac}, "
                      f"rep={rep+1}")
                _losses, _epochs = train_model(model_bin, train_ds_bin, n_epochs=n_epochs,
                                               batch_size=batch_size, lr=lr,
                                               l2_reg=l2_reg, device=device)

                f1_tcf = decode_and_evaluate(
                    model_bin, eval_l_bin, eval_f_bin, eval_v_bin,
                    feature_groups, feature_value_names, df, branch,
                    device)
                rows.append({
                    "branch": branch, "macroarea": macroarea,
                    "in_branch_frac": frac, "repeat": rep,
                    "model": "T-CF", "f1": f1_tcf
                })
                print(f"    → T-CF F1 = {f1_tcf:.4f}")

                # ------------------------------------------------
                # Categorical split (for learned model)
                # ------------------------------------------------
                train_ds_cat, eval_l_cat, eval_f_cat, eval_v_cat = \
                    split_by_branch_categorical(
                        df, cat_matrix, target_branch=branch,
                        in_branch_train_frac=frac,
                        rng=rng_cat)

                if len(eval_v_cat) == 0:
                    continue

                # --- Learned embedding model ---
                model_lrn = TypologicalMF_Learned(
                    n_langs, n_total_values, feat_to_global_ids, embed_dim)
                print(f"  [Learned] branch={branch}, frac={frac}, "
                      f"rep={rep+1}")
                train_model_learned(
                    model_lrn, train_ds_cat, n_epochs=n_epochs,
                    batch_size=batch_size, lr=lr,
                    l2_reg=l2_reg, device=device)

                f1_lrn = evaluate_learned(
                    model_lrn, eval_l_cat, eval_f_cat, eval_v_cat,
                    feat_to_value_names, device)
                rows.append({
                    "branch": branch, "macroarea": macroarea,
                    "in_branch_frac": frac, "repeat": rep,
                    "model": "Learned", "f1": f1_lrn
                })
                print(f"    → Learned F1 = {f1_lrn:.4f}")

    result_df = pd.DataFrame(rows)
    result_df["filter_label"] = ("Bible∩WALS" if bible_filtered
                                  else "full WALS (no Bible filter)")
    return result_df


def summarise(results_df: pd.DataFrame) -> pd.DataFrame:
    """Produce comparison table: mean F1 by model and in-branch fraction."""
    summary = (results_df
               .groupby(["in_branch_frac", "model"])["f1"]
               .agg(["mean", "std", "count"])
               .reset_index())
    summary.columns = ["in_branch_frac", "model", "mean_f1", "std_f1", "n"]

    # Pivot for side-by-side comparison
    pivot = summary.pivot(index="in_branch_frac", columns="model",
                          values="mean_f1")
    return summary, pivot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare binary vs. learned-embedding models")
    parser.add_argument("--database", type=str, default="wals",
                        choices=["wals", "grambank"],
                        help="Typological database to use (default: wals)")
    parser.add_argument("--wals_repo", type=str, default=None,
                        help="Path to cloned cldf-datasets/wals repo")
    parser.add_argument("--grambank_repo", type=str, default=None,
                        help="Path to cloned glottobank/grambank repo")
    parser.add_argument("--glottolog_repo", type=str, default=None,
                        help="Path to cloned glottolog-cldf repo "
                             "(for Grambank genus mapping)")
    parser.add_argument("--genus_source", type=str, default="glottolog",
                        choices=["glottolog", "family"],
                        help="Genus source for Grambank (default: glottolog)")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=6.4,
                        help="L2 coeff. Default 6.4 = 0.1 * batch_size(64) to match "
                             "Bjerva et al. weight_decay=0.1 effective magnitude.")
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--min_branch_size", type=int, default=5)
    parser.add_argument("--branches", type=str, nargs="+", default=None,
                        help="Only evaluate specific branches")
    parser.add_argument("--bible_lang_mask", type=str, default=None,
                        help="Path to .npy boolean mask for Bible ∩ WALS "
                             "filtering. If not provided, runs on full WALS.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_csv", type=str,
                        default="comparison_results.csv")
    args = parser.parse_args()

    # Load data based on selected database
    if args.database == "grambank":
        if args.grambank_repo is None:
            parser.error("--grambank_repo is required when --database=grambank")
        df, feature_cols = load_grambank_cldf(
            args.grambank_repo,
            genus_source=args.genus_source,
            glottolog_repo_dir=args.glottolog_repo,
        )
    else:
        if args.wals_repo is None:
            parser.error("--wals_repo is required when --database=wals")
        df, feature_cols = load_wals_cldf(args.wals_repo)

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
        bible_lang_mask=args.bible_lang_mask,
    )

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
