"""
downstream/evaluate.py
=======================
Evaluate how well each typological distance matrix predicts
cross-lingual POS transfer performance.

Metrics (all computed per target language, then macro-averaged)
---------------------------------------------------------------
1. Spearman ρ    — rank correlation between predicted source ranking
                   (by typological proximity) and actual ranking
                   (by transfer accuracy).

2. Kendall τ     — alternative rank correlation, more robust to tied
                   ranks (which occur when two sources are equidistant).

3. NDCG@k        — normalised discounted cumulative gain at k=1,3,5.
                   Asks: is the best-performing source language ranked
                   near the top by typological distance?

4. Regret@1      — percentage accuracy lost by choosing the
                   typologically-closest source instead of the
                   oracle-best source.
                   regret = (acc_oracle - acc_chosen) / acc_oracle

5. Pairwise accuracy — given two source candidates, does the distance
                        metric correctly rank which transfers better?
                        Averaged over all pairs for each target.

6. Top-1 / Top-3 selection accuracy — is the oracle-best source
                                       in the top-1 or top-3 predicted?

Usage
-----
    python downstream/evaluate.py \\
        --transfer_matrix downstream/transfer/transfer_matrix.csv \\
        --distances_dir downstream/distances \\
        --out_dir downstream/results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# NDCG implementation
# ---------------------------------------------------------------------------

def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """
    Compute NDCG@k where relevance = actual transfer accuracy (higher = better).

    Parameters
    ----------
    relevance   actual accuracies in the ORDER predicted by the metric
                (i.e. sorted by ascending distance, so best predicted source first)
    k           cutoff
    """
    relevance = relevance[:k]
    if len(relevance) == 0:
        return 0.0
    gains = relevance / np.log2(np.arange(2, len(relevance) + 2))
    dcg = gains.sum()

    ideal = np.sort(relevance)[::-1]
    ideal_gains = ideal / np.log2(np.arange(2, len(ideal) + 2))
    idcg = ideal_gains.sum()

    return float(dcg / idcg) if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-target evaluation
# ---------------------------------------------------------------------------

def evaluate_for_target(
    target: str,
    sources: List[str],
    transfer_row: pd.Series,      # accuracy[source] for this target
    distance_row: pd.Series,      # distance[source] for this target
    k_values: List[int] = (1, 3, 5),
) -> Dict:
    """
    Compute all metrics for a single target language.

    Smaller distance → predicted better source.
    Higher accuracy  → actual better source.
    """
    # Filter to sources that have both distance and accuracy
    valid_sources = [
        s for s in sources
        if s != target
        and s in transfer_row.index
        and s in distance_row.index
        and not np.isnan(transfer_row[s])
        and not np.isnan(distance_row[s])
    ]
    if len(valid_sources) < 2:
        return None

    accs = np.array([transfer_row[s] for s in valid_sources])
    dists = np.array([distance_row[s] for s in valid_sources])

    # Rank by distance (ascending = closer = predicted better)
    pred_order = np.argsort(dists)
    # Rank by accuracy (descending = higher acc = actual better)
    true_order = np.argsort(accs)[::-1]

    # Spearman ρ: correlation between distance rank and (-accuracy) rank
    # Note: distance increases with dissimilarity; accuracy increases with similarity.
    # So a good metric should have negative correlation between dist and acc.
    rho, _ = spearmanr(dists, -accs)  # positive rho = dist predicts acc correctly
    tau, _ = kendalltau(dists, -accs)

    # Relevance in predicted order
    relevance_in_pred_order = accs[pred_order]

    # NDCG@k (normalised by ideal relevance ordering)
    ndcg = {}
    for k in k_values:
        ndcg[f"ndcg@{k}"] = ndcg_at_k(relevance_in_pred_order, k)

    # Regret@1: accuracy loss from choosing top-predicted source
    oracle_acc = accs.max()
    chosen_acc = accs[pred_order[0]]
    regret1 = (oracle_acc - chosen_acc) / max(oracle_acc, 1e-9)

    # Top-k selection accuracy
    oracle_top1 = {valid_sources[true_order[0]]}
    oracle_top3 = set(valid_sources[i] for i in true_order[:3])
    predicted_top1 = {valid_sources[pred_order[0]]}
    predicted_top3 = set(valid_sources[i] for i in pred_order[:3])

    top1_acc = float(len(oracle_top1 & predicted_top1) > 0)
    top3_acc = float(len(oracle_top3 & predicted_top3) > 0)

    # Pairwise accuracy
    n_pairs = n_correct = 0
    for i in range(len(valid_sources)):
        for j in range(i + 1, len(valid_sources)):
            si, sj = valid_sources[i], valid_sources[j]
            # Does the metric correctly identify which source transfers better?
            acc_i, acc_j = accs[i], accs[j]
            if abs(acc_i - acc_j) < 1e-6:
                continue  # tied — skip
            dist_i, dist_j = dists[i], dists[j]
            # Better source = higher accuracy → should have lower distance
            if (acc_i > acc_j) == (dist_i < dist_j):
                n_correct += 1
            n_pairs += 1

    pairwise_acc = n_correct / max(n_pairs, 1)

    return {
        "spearman_rho": float(rho),
        "kendall_tau": float(tau),
        "regret@1": float(regret1),
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "pairwise_acc": pairwise_acc,
        "n_sources": len(valid_sources),
        **ndcg,
    }


# ---------------------------------------------------------------------------
# Full evaluation across all representations
# ---------------------------------------------------------------------------

def evaluate_all(
    transfer_matrix: pd.DataFrame,
    distances_dir: str,
    out_dir: str,
    k_values: List[int] = (1, 3, 5),
) -> pd.DataFrame:
    """
    Evaluate all distance matrices against the transfer matrix.

    Returns
    -------
    summary : pd.DataFrame with one row per representation,
              columns = metric names
    """
    os.makedirs(out_dir, exist_ok=True)

    # Find all distance files
    dist_files = sorted([
        f for f in os.listdir(distances_dir)
        if f.startswith("dist_") and f.endswith(".csv")
    ])
    print(f"\nFound {len(dist_files)} distance matrices to evaluate")

    glottocodes = list(transfer_matrix.index)
    summary_rows = []
    per_target_all = {}

    for dist_file in dist_files:
        repr_name = dist_file[5:-4]  # strip "dist_" and ".csv"
        dist_df = pd.read_csv(
            os.path.join(distances_dir, dist_file), index_col=0)

        print(f"\n  Evaluating: {repr_name}")
        per_target_metrics = {}

        for target in glottocodes:
            if target not in transfer_matrix.index:
                continue
            if target not in dist_df.index:
                continue

            result = evaluate_for_target(
                target=target,
                sources=glottocodes,
                transfer_row=transfer_matrix.loc[target],
                distance_row=dist_df.loc[target],
                k_values=k_values,
            )
            if result is not None:
                per_target_metrics[target] = result

        if not per_target_metrics:
            continue

        # Aggregate across targets
        metric_names = list(next(iter(per_target_metrics.values())).keys())
        row = {"representation": repr_name}
        for metric in metric_names:
            if metric == "n_sources":
                row[metric] = np.mean([v[metric]
                                       for v in per_target_metrics.values()])
            else:
                vals = [v[metric] for v in per_target_metrics.values()]
                row[metric] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals))

        row["n_targets"] = len(per_target_metrics)
        summary_rows.append(row)
        per_target_all[repr_name] = per_target_metrics

        # Print key metrics
        print(f"    Spearman ρ = {row['spearman_rho']:.4f} "
              f"(±{row.get('spearman_rho_std', 0):.4f})")
        print(f"    NDCG@3     = {row.get('ndcg@3', 0):.4f}")
        print(f"    Regret@1   = {row.get('regret@1', 0):.4f}")
        print(f"    Pairwise   = {row.get('pairwise_acc', 0):.4f}")
        print(f"    Top-1      = {row.get('top1_acc', 0):.4f}")

    if not summary_rows:
        print("WARNING: No results computed.")
        return pd.DataFrame()

    summary = pd.DataFrame(summary_rows).set_index("representation")

    # Sort by Spearman ρ descending
    if "spearman_rho" in summary.columns:
        summary = summary.sort_values("spearman_rho", ascending=False)

    # Save
    summary_path = os.path.join(out_dir, "evaluation_summary.csv")
    summary.to_csv(summary_path)
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY (sorted by Spearman ρ)")
    print('='*70)

    display_cols = ["spearman_rho", "ndcg@1", "ndcg@3",
                    "regret@1", "pairwise_acc", "top1_acc", "top3_acc"]
    display_cols = [c for c in display_cols if c in summary.columns]
    print(summary[display_cols].to_string(float_format="{:.4f}".format))
    print(f"\nFull results saved to: {summary_path}")

    # Save per-target breakdown
    for repr_name, per_target in per_target_all.items():
        pt_df = pd.DataFrame(per_target).T
        pt_path = os.path.join(out_dir, f"per_target_{repr_name}.csv")
        pt_df.to_csv(pt_path)

    return summary


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def permutation_test(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> float:
    """
    Approximate permutation test for difference in mean metric values.

    H0: mean(a) = mean(b)
    H1: mean(a) > mean(b)

    Returns p-value.
    """
    rng = np.random.default_rng(seed)
    obs_diff = np.mean(metric_a) - np.mean(metric_b)
    combined = np.concatenate([metric_a, metric_b])
    n = len(metric_a)

    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(combined))
        perm_diff = combined[perm[:n]].mean() - combined[perm[n:]].mean()
        if perm_diff >= obs_diff:
            count += 1
    return (count + 1) / (n_permutations + 1)


def compare_representations(
    per_target_all: Dict[str, Dict],
    repr_a: str,
    repr_b: str,
    metric: str = "spearman_rho",
    out_dir: Optional[str] = None,
) -> Dict:
    """
    Statistical comparison of two representations.
    """
    if repr_a not in per_target_all or repr_b not in per_target_all:
        return {}

    targets_a = per_target_all[repr_a]
    targets_b = per_target_all[repr_b]
    shared = set(targets_a.keys()) & set(targets_b.keys())

    vals_a = np.array([targets_a[t][metric] for t in sorted(shared)])
    vals_b = np.array([targets_b[t][metric] for t in sorted(shared)])

    p_val = permutation_test(vals_a, vals_b)
    result = {
        "repr_a": repr_a,
        "repr_b": repr_b,
        "metric": metric,
        "mean_a": float(vals_a.mean()),
        "mean_b": float(vals_b.mean()),
        "diff": float(vals_a.mean() - vals_b.mean()),
        "p_value": float(p_val),
        "n_targets": len(shared),
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate typological distance matrices against "
                    "POS transfer performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--transfer_matrix", required=True,
                   help="Path to transfer_matrix.csv")
    p.add_argument("--distances_dir", required=True,
                   help="Directory containing dist_*.csv files")
    p.add_argument("--out_dir", required=True,
                   help="Output directory for evaluation results")
    p.add_argument("--k_values", default="1,3,5",
                   help="Comma-separated k values for NDCG@k")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    k_values = [int(k) for k in args.k_values.split(",")]

    transfer_matrix = pd.read_csv(args.transfer_matrix, index_col=0)
    print(f"Transfer matrix: {transfer_matrix.shape}")
    print(f"Off-diagonal valid cells: "
          f"{(~transfer_matrix.isna()).sum().sum() - len(transfer_matrix)}")

    summary = evaluate_all(
        transfer_matrix=transfer_matrix,
        distances_dir=args.distances_dir,
        out_dir=args.out_dir,
        k_values=k_values,
    )


if __name__ == "__main__":
    main()
