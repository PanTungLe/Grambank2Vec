"""
downstream/evaluate.py
=======================
Evaluate how well each typological distance matrix predicts
cross-lingual POS transfer performance.

Metrics (all computed per target language, then macro-averaged)
---------------------------------------------------------------
1. Spearman rho  — rank correlation between predicted source ranking
                   (by typological proximity) and actual ranking
                   (by transfer accuracy).

2. Kendall tau   — alternative rank correlation, more robust to tied ranks.

3. NDCG@k        — normalised discounted cumulative gain at k=1,3,5.
                   DCG@k uses the predicted top-k; IDCG@k uses the true
                   top-k drawn from ALL N candidates (not just top-k).
                   NDCG@1 < 1 unless the metric perfectly finds the best source.

4. Regret@1      — accuracy lost by choosing closest-predicted source
                   instead of oracle-best. regret = (oracle - chosen) / oracle.

5. Pairwise accuracy — pairwise comparison: does distance correctly rank
                        which of two sources transfers better?

6. Top-1 / Top-3 selection accuracy.

Confound diagnostics (optional, requires --treebank_sizes and --script_map)
---------------------------------------------------------------------------
7. Script-controlled Spearman rho — computed only over same-script sources.
   Removes char-CNN character-overlap signal, isolates typological structure.

8. Size-residualised Spearman rho — partial correlation after regressing out
   log(training_size) from both distance and accuracy vectors.

Usage
-----
    python downstream/evaluate.py \\
        --transfer_matrix downstream_results/transfer/transfer_matrix.csv \\
        --distances_dir downstream_results/distances \\
        --out_dir downstream_results/results

    # With confound controls:
    python downstream/evaluate.py \\
        --transfer_matrix downstream_results/transfer/transfer_matrix.csv \\
        --distances_dir downstream_results/distances \\
        --out_dir downstream_results/results \\
        --treebank_sizes downstream_results/ud_data/treebank_sizes.json \\
        --script_map downstream_results/ud_data/script_map.json
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
# NDCG — fixed: ideal drawn from ALL N candidates
# ---------------------------------------------------------------------------

def ndcg_at_k(relevance_in_pred_order: np.ndarray, k: int) -> float:
    """
    Compute NDCG@k.

    Parameters
    ----------
    relevance_in_pred_order
        Full array of N accuracies sorted by predicted rank (ascending distance).
        Must be the FULL N-length array, not pre-truncated to k.
    k
        Cutoff.

    Notes
    -----
    DCG@k  = sum_{i=1}^{k}  rel_i / log2(i+1)
    IDCG@k = sum_{i=1}^{k}  rel*_i / log2(i+1)  (ideal top-k from ALL N)

    NDCG@1 is < 1.0 unless the metric perfectly identifies the best source.
    """
    n = len(relevance_in_pred_order)
    if n == 0 or k == 0:
        return 0.0
    k_eff = min(k, n)

    pred_top_k = relevance_in_pred_order[:k_eff]
    gains = pred_top_k / np.log2(np.arange(2, k_eff + 2))
    dcg = gains.sum()

    # Ideal: best k items from ALL N candidates
    ideal_top_k = np.sort(relevance_in_pred_order)[::-1][:k_eff]
    ideal_gains = ideal_top_k / np.log2(np.arange(2, k_eff + 2))
    idcg = ideal_gains.sum()

    return float(dcg / idcg) if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Confound utilities
# ---------------------------------------------------------------------------

def partial_spearman_residualise(
    x: np.ndarray,
    y: np.ndarray,
    covariate: np.ndarray,
) -> Tuple[float, float]:
    """
    Partial Spearman rho between x and y after removing linear effect of covariate.

    Computes residuals of OLS(x ~ covariate) and OLS(y ~ covariate),
    then Spearman-correlates the residuals.
    """
    def resid(a, b):
        # OLS residual of regressing a on b
        b_ = np.column_stack([np.ones_like(b), b])
        try:
            coef = np.linalg.lstsq(b_, a, rcond=None)[0]
            return a - b_ @ coef
        except Exception:
            return a

    x_resid = resid(x, covariate)
    y_resid = resid(y, covariate)
    rho, pval = spearmanr(x_resid, y_resid)
    return float(rho), float(pval)


# ---------------------------------------------------------------------------
# Per-target evaluation
# ---------------------------------------------------------------------------

def evaluate_for_target(
    target: str,
    sources: List[str],
    transfer_row: pd.Series,
    distance_row: pd.Series,
    k_values: List[int] = (1, 3, 5),
    treebank_sizes: Optional[Dict[str, int]] = None,
    script_map: Optional[Dict[str, str]] = None,
) -> Optional[Dict]:
    """
    Compute all metrics for a single target language.

    Smaller distance -> predicted better source.
    Higher accuracy  -> actual better source.
    """
    valid_sources = [
        s for s in sources
        if s != target
        and s in transfer_row.index
        and s in distance_row.index
        and not np.isnan(float(transfer_row[s]))
        and not np.isnan(float(distance_row[s]))
    ]
    if len(valid_sources) < 2:
        return None

    accs  = np.array([float(transfer_row[s])  for s in valid_sources])
    dists = np.array([float(distance_row[s]) for s in valid_sources])

    # Sort by distance ascending (closest first = predicted best)
    pred_order = np.argsort(dists)
    true_order = np.argsort(accs)[::-1]  # highest accuracy first

    # Spearman rho: positive = dist correctly ranks sources by acc
    rho, rho_p  = spearmanr(dists, -accs)
    tau, tau_p  = kendalltau(dists, -accs)

    # NDCG: relevance in predicted order (full array passed in)
    relevance_in_pred_order = accs[pred_order]
    ndcg = {f"ndcg@{k}": ndcg_at_k(relevance_in_pred_order, k) for k in k_values}

    # Regret@1
    oracle_acc = accs.max()
    chosen_acc = accs[pred_order[0]]
    regret1 = (oracle_acc - chosen_acc) / max(oracle_acc, 1e-9)

    # Top-k selection accuracy
    oracle_top1 = {valid_sources[true_order[0]]}
    oracle_top3 = set(valid_sources[i] for i in true_order[:3])
    top1_acc = float(valid_sources[pred_order[0]] in oracle_top1)
    top3_acc = float(bool(oracle_top3 & {valid_sources[pred_order[i]] for i in range(min(3, len(pred_order)))}))

    # Pairwise accuracy
    n_pairs = n_correct = 0
    for i in range(len(valid_sources)):
        for j in range(i + 1, len(valid_sources)):
            if abs(accs[i] - accs[j]) < 1e-6:
                continue
            if (accs[i] > accs[j]) == (dists[i] < dists[j]):
                n_correct += 1
            n_pairs += 1
    pairwise_acc = n_correct / max(n_pairs, 1)

    result = {
        "spearman_rho":  float(rho),
        "kendall_tau":   float(tau),
        "regret@1":      float(regret1),
        "top1_acc":      top1_acc,
        "top3_acc":      top3_acc,
        "pairwise_acc":  pairwise_acc,
        "n_sources":     len(valid_sources),
        **ndcg,
    }

    # ------------------------------------------------------------------
    # Confound diagnostic 1: script-controlled Spearman
    # Only evaluate over sources sharing the same script as target.
    # Removes char-CNN character-overlap signal.
    # ------------------------------------------------------------------
    if script_map is not None:
        target_script = script_map.get(target)
        same_script_sources = [
            s for s in valid_sources
            if script_map.get(s) == target_script and target_script is not None
        ]
        if len(same_script_sources) >= 3:
            ss_accs  = np.array([float(transfer_row[s])  for s in same_script_sources])
            ss_dists = np.array([float(distance_row[s]) for s in same_script_sources])
            ss_rho, _ = spearmanr(ss_dists, -ss_accs)
            result["spearman_rho_same_script"] = float(ss_rho)
            result["n_same_script_sources"]    = len(same_script_sources)
        else:
            result["spearman_rho_same_script"] = float("nan")
            result["n_same_script_sources"]    = len(same_script_sources)

    # ------------------------------------------------------------------
    # Confound diagnostic 2: size-residualised Spearman
    # Regress log(training_size) out of both distance and accuracy,
    # then correlate residuals.  Controls for "larger treebank = better
    # tagger regardless of target typology" confound.
    # ------------------------------------------------------------------
    if treebank_sizes is not None:
        sizes = np.array([
            np.log1p(treebank_sizes.get(s, 1)) for s in valid_sources
        ])
        rho_resid, _ = partial_spearman_residualise(dists, -accs, sizes)
        result["spearman_rho_size_resid"] = float(rho_resid)

    return result


# ---------------------------------------------------------------------------
# Full evaluation across all representations
# ---------------------------------------------------------------------------

def evaluate_all(
    transfer_matrix: pd.DataFrame,
    distances_dir: str,
    out_dir: str,
    k_values: List[int] = (1, 3, 5),
    treebank_sizes: Optional[Dict[str, int]] = None,
    script_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Evaluate all distance matrices against the transfer matrix.

    Returns
    -------
    summary : pd.DataFrame with one row per representation
    """
    os.makedirs(out_dir, exist_ok=True)

    dist_files = sorted([
        f for f in os.listdir(distances_dir)
        if f.startswith("dist_") and f.endswith(".csv")
    ])
    print(f"\nFound {len(dist_files)} distance matrices to evaluate")

    glottocodes = list(transfer_matrix.index)
    summary_rows = []
    per_target_all = {}

    for dist_file in dist_files:
        repr_name = dist_file[5:-4]
        dist_df = pd.read_csv(
            os.path.join(distances_dir, dist_file), index_col=0)

        print(f"\n  [{repr_name}]")
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
                treebank_sizes=treebank_sizes,
                script_map=script_map,
            )
            if result is not None:
                per_target_metrics[target] = result

        if not per_target_metrics:
            continue

        metric_names = list(next(iter(per_target_metrics.values())).keys())
        row = {"representation": repr_name}
        for metric in metric_names:
            vals = [v[metric] for v in per_target_metrics.values()
                    if not (isinstance(v[metric], float) and np.isnan(v[metric]))]
            if not vals:
                row[metric] = float("nan")
                row[f"{metric}_std"] = float("nan")
            elif metric == "n_sources":
                row[metric] = float(np.mean(vals))
            else:
                row[metric] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals))

        row["n_targets"] = len(per_target_metrics)
        summary_rows.append(row)
        per_target_all[repr_name] = per_target_metrics

        print(f"    Spearman rho          = {row['spearman_rho']:.4f} "
              f"(+/-{row.get('spearman_rho_std', 0):.4f})")
        if "spearman_rho_same_script" in row:
            print(f"    Spearman rho (script) = {row['spearman_rho_same_script']:.4f}")
        if "spearman_rho_size_resid" in row:
            print(f"    Spearman rho (resid)  = {row['spearman_rho_size_resid']:.4f}")
        print(f"    NDCG@1={row.get('ndcg@1',0):.4f}  "
              f"NDCG@3={row.get('ndcg@3',0):.4f}")
        print(f"    Regret@1={row.get('regret@1',0):.4f}  "
              f"Pairwise={row.get('pairwise_acc',0):.4f}  "
              f"Top-1={row.get('top1_acc',0):.4f}")

    if not summary_rows:
        print("WARNING: No results computed.")
        return pd.DataFrame()

    summary = pd.DataFrame(summary_rows).set_index("representation")
    if "spearman_rho" in summary.columns:
        summary = summary.sort_values("spearman_rho", ascending=False)

    summary_path = os.path.join(out_dir, "evaluation_summary.csv")
    summary.to_csv(summary_path)

    print(f"\n{'='*72}")
    print("EVALUATION SUMMARY (sorted by Spearman rho)")
    print('='*72)
    display_cols = (["spearman_rho"]
                    + (["spearman_rho_same_script"] if "spearman_rho_same_script" in summary.columns else [])
                    + (["spearman_rho_size_resid"]  if "spearman_rho_size_resid"  in summary.columns else [])
                    + ["ndcg@1", "ndcg@3", "regret@1", "pairwise_acc", "top1_acc", "top3_acc"])
    display_cols = [c for c in display_cols if c in summary.columns]
    print(summary[display_cols].to_string(float_format="{:.4f}".format))
    print(f"\nFull results saved to: {summary_path}")

    for repr_name, per_target in per_target_all.items():
        pt_df = pd.DataFrame(per_target).T
        pt_df.to_csv(os.path.join(out_dir, f"per_target_{repr_name}.csv"))

    return summary


# ---------------------------------------------------------------------------
# Transfer matrix diagnostics
# ---------------------------------------------------------------------------

def diagnose_transfer_matrix(
    transfer_matrix: pd.DataFrame,
    out_dir: str,
    treebank_sizes: Optional[Dict[str, int]] = None,
    script_map: Optional[Dict[str, str]] = None,
):
    """
    Print diagnostics about the transfer matrix structure.

    These diagnostics directly address the null-result hypothesis:
    if typological signal is absent, is it because script/size dominate?
    """
    os.makedirs(out_dir, exist_ok=True)
    glots = list(transfer_matrix.index)
    N = len(glots)
    mat = transfer_matrix.values.astype(float)

    off_mask = ~np.eye(N, dtype=bool)
    off_diag = mat[off_mask]
    valid = off_diag[~np.isnan(off_diag)]

    print(f"\n{'='*60}")
    print("TRANSFER MATRIX DIAGNOSTICS")
    print('='*60)
    print(f"  Shape: {N}x{N},  valid off-diagonal: {len(valid)}")
    print(f"  Accuracy: mean={valid.mean():.4f}  std={valid.std():.4f}  "
          f"min={valid.min():.4f}  max={valid.max():.4f}")
    print(f"  Diagonal (same-language): "
          f"{np.nanmean(np.diag(mat)):.4f}")

    # Per-source mean accuracy (tagger quality proxy)
    print("\n  Per-source mean transfer accuracy (tagger quality):")
    src_means = []
    for i, glot in enumerate(glots):
        row = mat[i]
        off = row[~np.isnan(row)]
        off = off[np.arange(len(off)) != i] if len(off) > 0 else off
        m = off.mean() if len(off) > 0 else float("nan")
        src_means.append((glot, m))
    src_means.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -1, reverse=True)
    for glot, m in src_means[:10]:
        sz = treebank_sizes.get(glot, "?") if treebank_sizes else "?"
        print(f"    {glot}: {m:.4f}  (train_size={sz})")
    print("    ...")
    for glot, m in src_means[-5:]:
        sz = treebank_sizes.get(glot, "?") if treebank_sizes else "?"
        print(f"    {glot}: {m:.4f}  (train_size={sz})")

    # Correlation between source training size and mean transfer accuracy
    if treebank_sizes:
        src_glots = [g for g, m in src_means if not np.isnan(m)]
        src_accs  = np.array([m for g, m in src_means if not np.isnan(m)])
        sizes     = np.array([np.log1p(treebank_sizes.get(g, 1)) for g in src_glots])
        rho, _ = spearmanr(sizes, src_accs)
        print(f"\n  Spearman(log_size, mean_transfer_acc) = {rho:.4f}")
        print(f"  (if |rho| > 0.4, training size is a major confound)")

    # Script-family analysis
    if script_map:
        print("\n  Within-script vs cross-script transfer:")
        within, cross = [], []
        for i, src in enumerate(glots):
            for j, tgt in enumerate(glots):
                if i == j or np.isnan(mat[i, j]):
                    continue
                if script_map.get(src) == script_map.get(tgt):
                    within.append(mat[i, j])
                else:
                    cross.append(mat[i, j])
        if within and cross:
            print(f"    Within-script: mean={np.mean(within):.4f}  "
                  f"n={len(within)}")
            print(f"    Cross-script:  mean={np.mean(cross):.4f}  "
                  f"n={len(cross)}")
            print(f"    Difference: {np.mean(within) - np.mean(cross):.4f}")
            print(f"    (large difference = script dominates transfer signal)")

    # Save per-target accuracy distribution
    per_tgt = {}
    for j, tgt in enumerate(glots):
        col = mat[:, j]
        col_off = np.array([col[i] for i in range(N) if i != j and not np.isnan(col[i])])
        if len(col_off) > 0:
            per_tgt[tgt] = {
                "mean_acc": float(col_off.mean()),
                "std_acc":  float(col_off.std()),
                "oracle_acc": float(col_off.max()),
                "worst_acc":  float(col_off.min()),
                "acc_range":  float(col_off.max() - col_off.min()),
            }
    per_tgt_df = pd.DataFrame(per_tgt).T.sort_values("acc_range", ascending=False)
    per_tgt_df.to_csv(os.path.join(out_dir, "per_target_accuracy_spread.csv"))
    print(f"\n  Top targets by accuracy range (most discriminative):")
    print(per_tgt_df[["mean_acc", "oracle_acc", "acc_range"]].head(10).to_string(
        float_format="{:.4f}".format))
    print(f"\n  Bottom targets by accuracy range (least discriminative):")
    print(per_tgt_df[["mean_acc", "oracle_acc", "acc_range"]].tail(5).to_string(
        float_format="{:.4f}".format))

    return per_tgt_df


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def permutation_test(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    obs_diff = np.mean(metric_a) - np.mean(metric_b)
    combined = np.concatenate([metric_a, metric_b])
    n = len(metric_a)
    count = sum(
        1 for _ in range(n_permutations)
        if (perm := rng.permutation(len(combined)))[0] is not None
        and combined[perm[:n]].mean() - combined[perm[n:]].mean() >= obs_diff
    )
    return (count + 1) / (n_permutations + 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate typological distance matrices against "
                    "POS transfer performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--transfer_matrix", required=True)
    p.add_argument("--distances_dir",   required=True)
    p.add_argument("--out_dir",         required=True)
    p.add_argument("--k_values", default="1,3,5")
    p.add_argument("--treebank_sizes", default=None,
                   help="JSON {glottocode: n_train_sentences}. "
                        "Enables size-residualised Spearman and size diagnostics.")
    p.add_argument("--script_map", default=None,
                   help="JSON {glottocode: script_family_name}. "
                        "Enables script-controlled Spearman and script diagnostics.")
    p.add_argument("--skip_diagnostics", action="store_true",
                   help="Skip transfer matrix diagnostics (faster)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    k_values = [int(k) for k in args.k_values.split(",")]

    transfer_matrix = pd.read_csv(args.transfer_matrix, index_col=0)
    print(f"Transfer matrix: {transfer_matrix.shape}")

    treebank_sizes = None
    if args.treebank_sizes and os.path.exists(args.treebank_sizes):
        with open(args.treebank_sizes) as f:
            treebank_sizes = json.load(f)
        print(f"Treebank sizes loaded: {len(treebank_sizes)} entries")

    script_map = None
    if args.script_map and os.path.exists(args.script_map):
        with open(args.script_map) as f:
            script_map = json.load(f)
        print(f"Script map loaded: {len(script_map)} entries")

    if not args.skip_diagnostics:
        diagnose_transfer_matrix(
            transfer_matrix, args.out_dir,
            treebank_sizes=treebank_sizes,
            script_map=script_map,
        )

    evaluate_all(
        transfer_matrix=transfer_matrix,
        distances_dir=args.distances_dir,
        out_dir=args.out_dir,
        k_values=k_values,
        treebank_sizes=treebank_sizes,
        script_map=script_map,
    )


if __name__ == "__main__":
    main()
