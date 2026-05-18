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
from data_preparation import (
    load_wals_cldf, load_grambank_cldf, binarise_features, binarise_wals,
    align_embeddings,
)


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
        train_pool = observed_orig_feats[n_eval:]

        # From training pool, use `in_branch_train_frac` of total observed
        n_use = min(round(n_obs * in_branch_train_frac), len(train_pool))
        train_orig = train_pool[:n_use]

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
    feature_value_names: dict,
    df: pd.DataFrame,
    target_branch: str,
    device: str = "cpu",
) -> float:
    """
    Predict held-out features and compute micro-F1 on the
    *original* (non-binarised) features.

    Decoding:
    - For multi-valued features (≥3 values, one-hot encoded):
      pick the value whose binary indicator has the highest
      predicted probability (argmax).
    - For 2-valued features (single binary column):
      predict value_1 if p > 0.5, else value_0.

    Returns
    -------
    f1 : float – micro-averaged F1 score
    """
    model.eval()
    with torch.no_grad():
        prob_matrix = model.predict_all().cpu().numpy()   # (L, F_bin)

    y_true_orig, y_pred_orig = [], []

    # Group eval cells by (language, original_feature)
    cell_map = defaultdict(dict)  # (lang_idx, orig_feat) → {bin_idx: true_val}
    for k in range(len(eval_langs)):
        li, bi, val = int(eval_langs[k]), int(eval_feats[k]), eval_vals[k]
        for orig_feat, bin_indices in feature_groups.items():
            if bi in bin_indices:
                cell_map[(li, orig_feat)][bi] = val
                break

    for (li, orig_feat), bi_vals in cell_map.items():
        bin_indices = feature_groups[orig_feat]
        value_names = feature_value_names[orig_feat]

        if len(bin_indices) == 1:
            # 2-valued feature: single binary column
            bi = bin_indices[0]
            true_bin_val = bi_vals[bi]
            true_label = value_names[int(true_bin_val)]
            pred_p = prob_matrix[li, bi]
            pred_label = value_names[1] if pred_p > 0.5 else value_names[0]
        else:
            # Multi-valued feature (≥3): one-hot encoded
            # True label: value name of the column that is 1
            true_label = None
            for j, bi in enumerate(bin_indices):
                if bi_vals.get(bi, 0.0) == 1.0:
                    true_label = value_names[j]
                    break
            if true_label is None:
                continue  # skip if no positive label

            # Predicted label: argmax
            pred_probs = [prob_matrix[li, bi] for bi in bin_indices]
            pred_label = value_names[int(np.argmax(pred_probs))]

        y_true_orig.append(true_label)
        y_pred_orig.append(pred_label)

    if len(y_true_orig) == 0:
        warnings.warn(f"No eval samples for branch '{target_branch}'")
        return 0.0

    f1 = f1_score(y_true_orig, y_pred_orig, average="micro")
    return f1


# ============================================================================
# 3b.  Baseline Methods
# ============================================================================

def majority_baseline(
    train_ds: WALSDataset,
    eval_langs: np.ndarray,
    eval_feats: np.ndarray,
    eval_vals: np.ndarray,
    feature_groups: dict,
    feature_value_names: dict,
    n_binary_feats: int,
) -> float:
    """
    Most-frequent-class baseline (Freq. in Table 1).

    For each original feature, predict the most common value from training.
    """
    # Count how often each binary feature is 1 in training
    feat_counts = np.zeros(n_binary_feats)
    feat_totals = np.zeros(n_binary_feats)
    for i in range(len(train_ds)):
        _, fi, vi = train_ds[i]
        fi = int(fi)
        feat_counts[fi] += float(vi)
        feat_totals[fi] += 1

    y_true, y_pred = [], []
    cell_map = defaultdict(dict)
    for k in range(len(eval_langs)):
        li, bi, val = int(eval_langs[k]), int(eval_feats[k]), eval_vals[k]
        for orig_feat, bin_indices in feature_groups.items():
            if bi in bin_indices:
                cell_map[(li, orig_feat)][bi] = val
                break

    for (li, orig_feat), bi_vals in cell_map.items():
        bin_indices = feature_groups[orig_feat]
        value_names = feature_value_names[orig_feat]

        if len(bin_indices) == 1:
            # 2-valued feature
            bi = bin_indices[0]
            true_label = value_names[int(bi_vals[bi])]
            # Predict majority: if count_1 > count_0, predict value_1
            n1 = feat_counts[bi]
            n0 = feat_totals[bi] - feat_counts[bi]
            pred_label = value_names[1] if n1 >= n0 else value_names[0]
        else:
            # Multi-valued feature
            true_label = None
            for j, bi in enumerate(bin_indices):
                if bi_vals.get(bi, 0.0) == 1.0:
                    true_label = value_names[j]
                    break
            if true_label is None:
                continue
            counts = [feat_counts[bi] for bi in bin_indices]
            pred_label = value_names[int(np.argmax(counts))]

        y_true.append(true_label)
        y_pred.append(pred_label)

    if len(y_true) == 0:
        return 0.0
    return f1_score(y_true, y_pred, average="micro")


def knn_baseline(
    pretrained_embs: np.ndarray,
    train_ds: WALSDataset,
    eval_langs: np.ndarray,
    eval_feats: np.ndarray,
    eval_vals: np.ndarray,
    feature_groups: dict,
    feature_value_names: dict,
    df: pd.DataFrame,
    target_branch: str,
    n_binary_feats: int,
    k: int = 1,
) -> float:
    """
    KNN baseline using pre-trained embeddings (Individual pred. in Table 1).

    For each in-branch language, find the k nearest training languages
    by embedding distance and predict the most common value among them.
    """
    in_branch_mask = (df["genus"] == target_branch).values
    in_branch_indices = set(np.where(in_branch_mask)[0])
    out_branch_indices = np.where(~in_branch_mask)[0]

    if len(out_branch_indices) == 0 or pretrained_embs is None:
        return 0.0

    # Build per-feature training data from out-of-branch languages
    train_feat_vals = defaultdict(dict)  # feat_idx -> {lang_idx: val}
    for i in range(len(train_ds)):
        li, fi, vi = train_ds[i]
        li, fi = int(li), int(fi)
        train_feat_vals[fi][li] = float(vi)

    # Group eval cells
    cell_map = defaultdict(dict)
    for k_idx in range(len(eval_langs)):
        li, bi, val = int(eval_langs[k_idx]), int(eval_feats[k_idx]), eval_vals[k_idx]
        for orig_feat, bin_indices in feature_groups.items():
            if bi in bin_indices:
                cell_map[(li, orig_feat)][bi] = val
                break

    y_true, y_pred = [], []

    for (li, orig_feat), bi_vals in cell_map.items():
        bin_indices = feature_groups[orig_feat]
        value_names = feature_value_names[orig_feat]

        if len(bin_indices) == 1:
            # 2-valued feature
            bi = bin_indices[0]
            true_label = value_names[int(bi_vals[bi])]
        else:
            # Multi-valued feature
            true_label = None
            for j, bi in enumerate(bin_indices):
                if bi_vals.get(bi, 0.0) == 1.0:
                    true_label = value_names[j]
                    break
            if true_label is None:
                continue

        # Find k nearest training languages that have this feature observed
        train_langs_with_feat = []
        for bi in bin_indices:
            for tl in train_feat_vals.get(bi, {}):
                if tl not in in_branch_indices:
                    train_langs_with_feat.append(tl)
        train_langs_with_feat = list(set(train_langs_with_feat))

        if not train_langs_with_feat:
            y_true.append(true_label)
            y_pred.append(value_names[0])
            continue

        # Compute distances
        query_emb = pretrained_embs[li]
        dists = []
        for tl in train_langs_with_feat:
            d = np.linalg.norm(query_emb - pretrained_embs[tl])
            dists.append((d, tl))
        dists.sort()
        neighbors = [tl for _, tl in dists[:k]]

        if len(bin_indices) == 1:
            # 2-valued: vote on binary value, decode to label
            bi = bin_indices[0]
            vote_sum = sum(train_feat_vals.get(bi, {}).get(tl, 0.0) for tl in neighbors)
            pred_label = value_names[1] if vote_sum > len(neighbors) / 2 else value_names[0]
        else:
            # Multi-valued: vote on each column
            votes = np.zeros(len(bin_indices))
            for tl in neighbors:
                for j, bi in enumerate(bin_indices):
                    votes[j] += train_feat_vals.get(bi, {}).get(tl, 0.0)
            pred_label = value_names[int(np.argmax(votes))]

        y_true.append(true_label)
        y_pred.append(pred_label)

    if len(y_true) == 0:
        return 0.0
    return f1_score(y_true, y_pred, average="micro")


# ============================================================================
# 4.  Full Experiment Runner
# ============================================================================

def run_experiments(
    df: pd.DataFrame,
    binary_matrix: np.ndarray,
    feature_groups: dict,
    feature_value_names: dict = None,
    in_branch_fracs: list = [0.0, 0.01, 0.05, 0.10, 0.20],
    n_repeats: int = 5,
    embed_dim: int = 64,
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    l2_reg: float = 6.4,
    pretrained_embs: Optional[np.ndarray] = None,
    device: str = "cpu",
    min_branch_size: int = 5,
    only_branches: Optional[list] = None,
    freeze_lang: bool = False,
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
    if only_branches:
        qualifying = [b for b in qualifying if b in only_branches]
    print(f"\nQualifying branches (≥{min_branch_size} languages): "
          f"{len(qualifying)}")

    branch_list = sorted(qualifying)
    branch_to_int = {b: i for i, b in enumerate(branch_list)}

    rows = []

    for branch in qualifying:
        macroarea = df.loc[df["genus"] == branch, "macroarea"].iloc[0]
        print(f"\n{'='*60}")
        print(f"Branch: {branch}  (macroarea: {macroarea})")
        print(f"{'='*60}")

        for frac in in_branch_fracs:
            for rep in range(n_repeats):
                rng = np.random.default_rng(seed=rep * 1000 + branch_to_int[branch] * 37)

                # --- Split ---
                train_ds, eval_langs, eval_feats, eval_vals = split_by_branch(
                    df, binary_matrix, feature_groups,
                    target_branch=branch,
                    in_branch_train_frac=frac,
                    rng=rng,
                )

                if len(eval_vals) == 0:
                    continue

                n_bfeat_local = binary_matrix.shape[1]

                # --- Majority baseline ---
                f1_freq = majority_baseline(
                    train_ds, eval_langs, eval_feats, eval_vals,
                    feature_groups, feature_value_names, n_bfeat_local)
                rows.append({
                    "branch": branch, "macroarea": macroarea,
                    "in_branch_frac": frac, "repeat": rep,
                    "model": "Freq", "f1": f1_freq
                })

                # --- KNN baseline (if pretrained embeddings provided) ---
                if pretrained_embs is not None:
                    f1_knn = knn_baseline(
                        pretrained_embs, train_ds,
                        eval_langs, eval_feats, eval_vals,
                        feature_groups, feature_value_names,
                        df, branch, n_bfeat_local, k=1)
                    rows.append({
                        "branch": branch, "macroarea": macroarea,
                        "in_branch_frac": frac, "repeat": rep,
                        "model": "KNN", "f1": f1_knn
                    })

                # --- T-CF (core model) ---
                model = TypologicalMF(n_langs, n_bfeat, embed_dim)
                print(f"\n  [T-CF] branch={branch}, frac={frac}, rep={rep+1}")
                _losses, _epochs = train_model(model, train_ds, n_epochs=n_epochs,
                                               batch_size=batch_size, lr=lr,
                                               l2_reg=l2_reg, device=device)

                f1_tcf = decode_and_evaluate(
                    model, eval_langs, eval_feats, eval_vals,
                    feature_groups, feature_value_names, df, branch, device)
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
                        embed_dim=embed_dim, freeze_lang=freeze_lang)
                    print(f"  [SemiSup] branch={branch}, frac={frac}, "
                          f"rep={rep+1}")
                    _losses, _epochs = train_model(model_ss, train_ds, n_epochs=n_epochs,
                                                   batch_size=batch_size, lr=lr,
                                                   l2_reg=l2_reg, device=device)

                    f1_ss = decode_and_evaluate(
                        model_ss, eval_langs, eval_feats, eval_vals,
                        feature_groups, feature_value_names, df, branch, device)
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
    parser.add_argument("--pretrained_embs", type=str, default=None,
                        help="Path to .npy file with pre-trained language "
                             "embeddings (n_langs × d). Row order must "
                             "match the languages after filtering.")
    parser.add_argument("--bible_lang_mask", type=str, default=None,
                        help="Path to .npy boolean mask for Bible ∩ database "
                             "filtering. If not provided and --pretrained_embs "
                             "is set, inferred from non-zero embedding rows.")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=6.4,
                        help="L2 coeff. Default 6.4 = 0.1 * batch_size(64) to match "
                             "Bjerva et al. weight_decay=0.1 effective magnitude.")
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--min_branch_size", type=int, default=5,
                        help="Skip branches with fewer languages (paper: >4)")
    parser.add_argument("--branches", type=str, nargs="+", default=None,
                        help="Only evaluate these branches (e.g. --branches Oceanic Slavic)")
    parser.add_argument("--freeze_lang", action="store_true",
                        help="Freeze pretrained language embeddings during "
                             "training. Default is to fine-tune them, as "
                             "reported in Bjerva et al. (2019). Use "
                             "--freeze_lang to freeze them.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_csv", type=str, default="results.csv")
    args = parser.parse_args()

    # 1. Load and binarise typological data
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

    binary_matrix, bin_names, feature_groups, feature_value_names = \
        binarise_features(df, feature_cols)

    # 2. Optionally load pre-trained language embeddings
    pretrained = None
    if args.pretrained_embs:
        pretrained = np.load(args.pretrained_embs).astype(np.float32)
        if pretrained.shape[0] != len(df):
            print(f"WARNING: Pretrained embeddings have {pretrained.shape[0]} "
                  f"rows but database has {len(df)} languages. "
                  f"Assuming embeddings are already aligned or will be padded.")
            if pretrained.shape[0] < len(df):
                padded = np.zeros((len(df), pretrained.shape[1]),
                                  dtype=np.float32)
                padded[:pretrained.shape[0]] = pretrained
                pretrained = padded
            else:
                pretrained = pretrained[:len(df)]
        print(f"Loaded pretrained embeddings: {pretrained.shape}")

    # 2b. Filter to Bible ∩ database intersection (paper §7.1)
    if args.bible_lang_mask or pretrained is not None:
        if args.bible_lang_mask:
            has_bible = np.load(args.bible_lang_mask).astype(bool)
        else:
            has_bible = np.any(pretrained != 0, axis=1)
        n_before = len(df)
        df = df[has_bible].reset_index(drop=True)
        binary_matrix = binary_matrix[has_bible]
        if pretrained is not None:
            pretrained = pretrained[has_bible]
        # Re-binarise to recompute feature_groups with correct indices
        binary_matrix, bin_names, feature_groups, feature_value_names = \
            binarise_features(df, feature_cols)
        print(f"Bible ∩ database filter: {n_before} → {len(df)} languages")
    else:
        print("WARNING: No Bible filter applied — results not comparable "
              "to Bjerva et al. Table 1")

    # 3. Run experiments
    results = run_experiments(
        df, binary_matrix, feature_groups,
        feature_value_names=feature_value_names,
        in_branch_fracs=[0.0, 0.01, 0.05, 0.10, 0.20],
        n_repeats=args.n_repeats,
        embed_dim=args.embed_dim,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2_reg=args.l2_reg,
        pretrained_embs=pretrained,
        device=args.device,
        min_branch_size=args.min_branch_size,
        only_branches=args.branches,
        freeze_lang=args.freeze_lang,
    )

    # 4. Save and display results
    results.to_csv(args.output_csv, index=False)
    print(f"\nDetailed results saved to {args.output_csv}")

    summary = summarise_results(results)
    print("\n" + "=" * 60)
    print(f"AGGREGATE RESULTS — {args.database.upper()}")
    print("=" * 60)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
