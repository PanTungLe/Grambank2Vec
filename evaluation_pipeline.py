"""
Evaluation Pipeline for Typological Collaborative Filtering
=============================================================
Implements the full experiment workflow from Bjerva et al. (NAACL 2019):

1. Load and binarise WALS (from cldf-datasets/wals GitHub repo)
2. Split by language branch (genus)
3. For each branch, hold out 80% of feature-language pairs for evaluation
4. Train on out-of-branch data + varying % of in-branch data
5. Predict held-out features and compute micro-F1

Usage
-----
    # Core model only (from WALS CLDF repo):
    python evaluation_pipeline.py --wals_repo /path/to/wals

    # With pre-trained embeddings from ParaBible char-LM:
    python evaluation_pipeline.py \\
        --wals_repo /path/to/wals \\
        --pretrained_embs charlm_output/lang_embeddings.npy
"""

import argparse
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from model import TypologicalMF, TypologicalMF_SemiSup, WALSDataset, train_model
from data_preparation import load_wals_cldf, binarise_wals


# ============================================================================
# 2.  Branch-based Train / Eval Split
# ============================================================================

def split_by_branch(
    df: pd.DataFrame,
    binary_matrix: np.ndarray,
    feature_groups: dict,
    target_branch: str,
    in_branch_train_frac: float = 0.20,
    rng: np.random.Generator = None,
) -> Tuple[WALSDataset, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train and eval splits following the paper's protocol.

    1. All observed cells for languages OUTSIDE target_branch → train.
    2. For languages INSIDE target_branch:
       a. Randomly select 80 % of observed *original-feature–language*
          pairs for evaluation.
       b. From the remaining 20 %, use `in_branch_train_frac` for training
          (this corresponds to the paper's 0–20 % in-branch training).

    IMPORTANT: when splitting, all binary columns belonging to the same
    original feature for a given language must go to the same split.

    Returns
    -------
    train_ds : WALSDataset
    eval_langs : np.ndarray  – language indices for each eval cell
    eval_feats : np.ndarray  – binary feature indices for each eval cell
    eval_vals  : np.ndarray  – ground-truth binary values for eval cells
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_langs, n_bfeat = binary_matrix.shape
    in_branch_mask = (df["genus"] == target_branch).values

    # ---- Out-of-branch: everything observed goes to training ----
    out_langs, out_feats = [], []
    out_vals = []
    for i in range(n_langs):
        if in_branch_mask[i]:
            continue
        observed = np.where(~np.isnan(binary_matrix[i]))[0]
        out_langs.append(np.full(len(observed), i))
        out_feats.append(observed)
        out_vals.append(binary_matrix[i, observed])

    # ---- In-branch: split by original feature per language ----
    in_train_langs, in_train_feats, in_train_vals = [], [], []
    eval_langs_l, eval_feats_l, eval_vals_l = [], [], []

    in_branch_indices = np.where(in_branch_mask)[0]
    for i in in_branch_indices:
        # Collect original features that are observed for this language
        observed_orig_feats = []
        for orig_feat, bin_indices in feature_groups.items():
            # Check if at least one binary col is observed (not NaN)
            if not np.isnan(binary_matrix[i, bin_indices]).all():
                observed_orig_feats.append(orig_feat)

        rng.shuffle(observed_orig_feats)
        n_obs = len(observed_orig_feats)
        # 80 % for evaluation pool, 20 % candidate for training
        n_eval = int(np.ceil(n_obs * 0.80))
        eval_orig = observed_orig_feats[:n_eval]
        train_candidate = observed_orig_feats[n_eval:]

        # From training candidates, use `in_branch_train_frac`
        n_use = max(int(len(train_candidate) * in_branch_train_frac), 0)
        train_orig = train_candidate[:n_use]

        # Map original features back to binary column indices
        for orig_feat in eval_orig:
            for bi in feature_groups[orig_feat]:
                if not np.isnan(binary_matrix[i, bi]):
                    eval_langs_l.append(i)
                    eval_feats_l.append(bi)
                    eval_vals_l.append(binary_matrix[i, bi])

        for orig_feat in train_orig:
            for bi in feature_groups[orig_feat]:
                if not np.isnan(binary_matrix[i, bi]):
                    in_train_langs.append(i)
                    in_train_feats.append(bi)
                    in_train_vals.append(binary_matrix[i, bi])

    # Combine out-of-branch + in-branch training data
    all_train_langs = np.concatenate(out_langs + [np.array(in_train_langs, dtype=int)])
    all_train_feats = np.concatenate(out_feats + [np.array(in_train_feats, dtype=int)])
    all_train_vals  = np.concatenate(out_vals  + [np.array(in_train_vals, dtype=float)])

    train_ds = WALSDataset(all_train_langs, all_train_feats, all_train_vals)

    eval_langs = np.array(eval_langs_l, dtype=int)
    eval_feats = np.array(eval_feats_l, dtype=int)
    eval_vals  = np.array(eval_vals_l, dtype=float)

    return train_ds, eval_langs, eval_feats, eval_vals


# ============================================================================
# 3.  Evaluation Helpers
# ============================================================================

def decode_and_evaluate(
    model: torch.nn.Module,
    eval_langs: np.ndarray,
    eval_feats: np.ndarray,
    eval_vals: np.ndarray,
    feature_groups: dict,
    df: pd.DataFrame,
    target_branch: str,
    device: str = "cpu",
) -> float:
    """
    Predict held-out features and compute micro-F1 on the
    *original* (non-binarised) features.

    The decoding step: for each original multi-valued feature, pick
    the value whose binary indicator has the highest predicted probability.

    Returns
    -------
    f1 : float – micro-averaged F1 score
    """
    model.eval()
    with torch.no_grad():
        prob_matrix = model.predict_all().cpu().numpy()   # (L, F_bin)

    in_branch_indices = set(np.where((df["genus"] == target_branch).values)[0])

    y_true_orig, y_pred_orig = [], []

    # Group eval cells by (language, original_feature)
    cell_map = defaultdict(dict)  # (lang_idx, orig_feat) → {bin_idx: true_val}
    for k in range(len(eval_langs)):
        li, bi, val = int(eval_langs[k]), int(eval_feats[k]), eval_vals[k]
        # Find which original feature this binary index belongs to
        for orig_feat, bin_indices in feature_groups.items():
            if bi in bin_indices:
                cell_map[(li, orig_feat)][bi] = val
                break

    for (li, orig_feat), bi_vals in cell_map.items():
        bin_indices = feature_groups[orig_feat]
        # True label: which binary index is 1?
        true_label = None
        for bi, v in bi_vals.items():
            if v == 1.0:
                true_label = bi
                break
        if true_label is None:
            continue  # skip if no positive label (shouldn't happen)

        # Predicted label: argmax of predicted probs over the group
        pred_probs = [prob_matrix[li, bi] for bi in bin_indices]
        pred_label = bin_indices[int(np.argmax(pred_probs))]

        y_true_orig.append(true_label)
        y_pred_orig.append(pred_label)

    if len(y_true_orig) == 0:
        warnings.warn(f"No eval samples for branch '{target_branch}'")
        return 0.0

    f1 = f1_score(y_true_orig, y_pred_orig, average="micro")
    return f1


# ============================================================================
# 4.  Full Experiment Runner
# ============================================================================

def run_experiments(
    df: pd.DataFrame,
    binary_matrix: np.ndarray,
    feature_groups: dict,
    in_branch_fracs: list = [0.0, 0.01, 0.05, 0.10, 0.20],
    n_repeats: int = 5,
    embed_dim: int = 64,
    n_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    l2_reg: float = 0.1,
    pretrained_embs: Optional[np.ndarray] = None,
    device: str = "cpu",
    min_branch_size: int = 4,
) -> pd.DataFrame:
    """
    Run the full set of experiments across all qualifying branches.

    Parameters
    ----------
    df : pd.DataFrame with 'genus' column
    binary_matrix : binarised WALS matrix
    feature_groups : mapping original feat → list of binary indices
    in_branch_fracs : list of in-branch training fractions to test
    n_repeats : how many random repetitions per (branch, fraction)
    embed_dim : dimensionality of embeddings
    pretrained_embs : (n_langs, d) array of pre-trained language embeddings;
                      if provided, also runs the semi-supervised model
    device : 'cpu' or 'cuda'
    min_branch_size : skip branches with fewer languages

    Returns
    -------
    results_df : pd.DataFrame with columns
        [branch, macroarea, in_branch_frac, repeat, model, f1]
    """
    n_langs, n_bfeat = binary_matrix.shape

    # Identify qualifying branches
    branch_counts = df["genus"].value_counts()
    qualifying = branch_counts[branch_counts >= min_branch_size].index.tolist()
    print(f"\nQualifying branches (≥{min_branch_size} languages): "
          f"{len(qualifying)}")

    rows = []

    for branch in qualifying:
        macroarea = df.loc[df["genus"] == branch, "macroarea"].iloc[0]
        print(f"\n{'='*60}")
        print(f"Branch: {branch}  (macroarea: {macroarea})")
        print(f"{'='*60}")

        for frac in in_branch_fracs:
            for rep in range(n_repeats):
                rng = np.random.default_rng(seed=rep * 1000 + hash(branch) % 10000)

                # --- Split ---
                train_ds, eval_langs, eval_feats, eval_vals = split_by_branch(
                    df, binary_matrix, feature_groups,
                    target_branch=branch,
                    in_branch_train_frac=frac,
                    rng=rng,
                )

                if len(eval_vals) == 0:
                    continue

                # --- T-CF (core model) ---
                model = TypologicalMF(n_langs, n_bfeat, embed_dim)
                print(f"\n  [T-CF] branch={branch}, frac={frac}, rep={rep+1}")
                train_model(model, train_ds, n_epochs=n_epochs,
                            batch_size=batch_size, lr=lr,
                            l2_reg=l2_reg, device=device)

                f1_tcf = decode_and_evaluate(
                    model, eval_langs, eval_feats, eval_vals,
                    feature_groups, df, branch, device)
                rows.append({
                    "branch": branch, "macroarea": macroarea,
                    "in_branch_frac": frac, "repeat": rep,
                    "model": "T-CF", "f1": f1_tcf
                })
                print(f"    → F1 = {f1_tcf:.4f}")

                # --- SemiSup (if pretrained embeddings provided) ---
                if pretrained_embs is not None:
                    model_ss = TypologicalMF_SemiSup(
                        pretrained_embs, n_bfeat,
                        embed_dim=embed_dim, freeze_lang=False)
                    print(f"  [SemiSup] branch={branch}, frac={frac}, "
                          f"rep={rep+1}")
                    train_model(model_ss, train_ds, n_epochs=n_epochs,
                                batch_size=batch_size, lr=lr,
                                l2_reg=l2_reg, device=device)

                    f1_ss = decode_and_evaluate(
                        model_ss, eval_langs, eval_feats, eval_vals,
                        feature_groups, df, branch, device)
                    rows.append({
                        "branch": branch, "macroarea": macroarea,
                        "in_branch_frac": frac, "repeat": rep,
                        "model": "SemiSup", "f1": f1_ss
                    })
                    print(f"    → F1 = {f1_ss:.4f}")

    results_df = pd.DataFrame(rows)
    return results_df


def summarise_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce Table 1 from the paper: mean and std of F1 grouped by
    in-branch fraction and model.
    """
    summary = (results_df
               .groupby(["in_branch_frac", "model"])["f1"]
               .agg(["mean", "std"])
               .reset_index())
    summary.columns = ["in_branch_frac", "model", "mean_f1", "std_f1"]
    return summary


# ============================================================================
# 5.  Main  —  Putting It All Together
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Replicate typological collaborative filtering experiments")
    parser.add_argument("--wals_repo", type=str, required=True,
                        help="Path to cloned cldf-datasets/wals repo")
    parser.add_argument("--pretrained_embs", type=str, default=None,
                        help="Path to .npy file with pre-trained language "
                             "embeddings (n_langs × d). Row order must "
                             "match the WALS languages after filtering.")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=0.1)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_csv", type=str, default="results.csv")
    args = parser.parse_args()

    # 1. Load and binarise WALS from CLDF repo
    df, feature_cols = load_wals_cldf(args.wals_repo)
    binary_matrix, bin_names, feature_groups = binarise_wals(df, feature_cols)

    # 2. Optionally load pre-trained language embeddings
    pretrained = None
    if args.pretrained_embs:
        pretrained = np.load(args.pretrained_embs).astype(np.float32)
        assert pretrained.shape[0] == len(df), (
            f"Pretrained embeddings have {pretrained.shape[0]} rows but "
            f"WALS has {len(df)} languages after filtering."
        )
        print(f"Loaded pretrained embeddings: {pretrained.shape}")

    # 3. Run experiments
    results = run_experiments(
        df, binary_matrix, feature_groups,
        in_branch_fracs=[0.0, 0.01, 0.05, 0.10, 0.20],
        n_repeats=args.n_repeats,
        embed_dim=args.embed_dim,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2_reg=args.l2_reg,
        pretrained_embs=pretrained,
        device=args.device,
    )

    # 4. Save and display results
    results.to_csv(args.output_csv, index=False)
    print(f"\nDetailed results saved to {args.output_csv}")

    summary = summarise_results(results)
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS (cf. Table 1 in the paper)")
    print("=" * 60)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
