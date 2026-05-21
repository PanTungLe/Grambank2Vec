"""
downstream/task_weighted_repr.py
==================================
Task-weighted Repr B: instead of uniform averaging across all features,
weight each feature's contribution by its transfer-predictive importance.

Three weighting schemes
-----------------------

1. LOFO-weight (supervised):
   w_f = max(0, importance_f)  where importance_f from leave-one-feature-out.
   Keeps only informative features (positive LOFO importance), zeros noisy ones.
   Requires held-out languages to estimate importances without overfitting.

2. Domain-weight (semi-supervised):
   Upweight features from transfer-relevant domains (WordOrder, Nominal).
   Domain weights derived from domain_analysis output.
   Does not require the transfer matrix for computation.

3. Frequency-weight (unsupervised):
   Weight features by how many languages have observed values in the database.
   Well-attested features may have better-calibrated embeddings.
   No external signal needed.

These are compared against:
   - Repr B (uniform weight, all features)
   - Repr B_masked70 (random 70% drop, best result so far)
   - Repr C (binary profile baseline)

Usage
-----
    python downstream/task_weighted_repr.py \\
        --checkpoint checkpoints/wals_learned_s42 \\
        --lofo_csv downstream_results/feature_importance/wals/lofo_wals.csv \\
        --domain_csv downstream_results/feature_importance/wals/domain_analysis_wals.csv \\
        --out_dir downstream_results/representations/wals_s42_weighted \\
        --glottocodes_file downstream_results/distances/glottocodes.json
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
from scipy.special import softmax as scipy_softmax

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from downstream.representations import load_checkpoint


# ---------------------------------------------------------------------------
# Weighted Repr B
# ---------------------------------------------------------------------------

def compute_repr_B_weighted(
    ckpt: Dict,
    glottocodes: List[str],
    feature_weights: np.ndarray,  # (n_feats,) non-negative weights
) -> Tuple[np.ndarray, List[str]]:
    """
    Weighted Repr B: T_ℓ = Σ_f  w_f · (Σ_v p(v|ℓ,f) · e_{f,v}) / Σ_f w_f

    Parameters
    ----------
    feature_weights : (n_feats,) array of non-negative weights, one per feature
                      in the order of feat_names_sorted.
                      Features with weight=0 are excluded entirely.

    Returns
    -------
    reprs   (N_found, D) float32
    found   list of glottocodes
    """
    lang2id = ckpt["lang2id"]
    lang_embs = ckpt["lang_embs"]
    fv_embs = ckpt["fv_embs"]
    feat_to_global_ids = ckpt["feat_to_global_ids"]
    feat_names = ckpt["feat_names_sorted"]
    n_feats = len(feat_names)

    assert len(feature_weights) == n_feats, \
        f"feature_weights length {len(feature_weights)} != n_feats {n_feats}"

    feat_indices = sorted(feat_to_global_ids.keys())
    weight_sum = feature_weights.sum()
    if weight_sum == 0:
        raise ValueError("All feature weights are zero")

    reprs, found = [], []
    for glot in glottocodes:
        if glot not in lang2id:
            continue
        idx = lang2id[glot]
        lang_emb = lang_embs[idx]

        weighted_sum = np.zeros(lang_embs.shape[1], dtype=np.float64)
        total_weight = 0.0

        for fi in feat_indices:
            w = float(feature_weights[fi])
            if w <= 0:
                continue
            gids = feat_to_global_ids[fi]
            val_embs_f = fv_embs[gids]
            scores = val_embs_f @ lang_emb
            probs = scipy_softmax(scores)
            expected = (probs[:, None] * val_embs_f).sum(0)
            weighted_sum += w * expected
            total_weight += w

        T = (weighted_sum / total_weight).astype(np.float32)
        reprs.append(T)
        found.append(glot)

    if not reprs:
        return np.empty((0, lang_embs.shape[1]), dtype=np.float32), []
    return np.stack(reprs), found


# ---------------------------------------------------------------------------
# Weight derivation functions
# ---------------------------------------------------------------------------

def lofo_weights(
    lofo_csv: str,
    feat_names: List[str],
    strategy: str = "positive_only",
) -> np.ndarray:
    """
    Derive feature weights from LOFO importance scores.

    Strategies
    ----------
    positive_only : w_f = max(0, importance_f)  — exclude noisy features
    rank_based    : w_f = rank(importance_f) / n_feats  — soft ranking
    threshold     : w_f = 1 if importance > mean + std else 0
    """
    df = pd.read_csv(lofo_csv).set_index("feature")
    feat2importance = df["importance"].to_dict()

    weights = np.zeros(len(feat_names), dtype=np.float32)
    for fi, fname in enumerate(feat_names):
        imp = feat2importance.get(fname, 0.0)
        if strategy == "positive_only":
            weights[fi] = max(0.0, float(imp))
        elif strategy == "rank_based":
            weights[fi] = float(imp)  # will rank-normalise below
        elif strategy == "threshold":
            weights[fi] = float(imp)  # will threshold below

    if strategy == "rank_based":
        ranks = np.argsort(np.argsort(weights)) + 1
        weights = ranks.astype(np.float32) / len(weights)
    elif strategy == "threshold":
        mean, std = weights.mean(), weights.std()
        weights = (weights > mean + std).astype(np.float32)

    return weights


def domain_weights(
    domain_csv: str,
    feat_names: List[str],
    db: str,
    min_weight: float = 0.1,
) -> np.ndarray:
    """
    Derive feature weights from domain-level Spearman rho.

    w_f = max(min_weight, domain_rho[domain(f)])
    Features in domains with positive rho get proportionally more weight.
    """
    from downstream.feature_importance import get_feature_domain

    df = pd.read_csv(domain_csv).set_index("domain")
    domain2rho = df["spearman_rho"].to_dict()

    weights = np.zeros(len(feat_names), dtype=np.float32)
    for fi, fname in enumerate(feat_names):
        domain = get_feature_domain(fname, db)
        rho = domain2rho.get(domain, 0.0)
        weights[fi] = max(min_weight, float(rho))

    return weights


def top_k_weights(
    lofo_csv: str,
    feat_names: List[str],
    k: int = 20,
) -> np.ndarray:
    """
    Binary weights: 1 for top-k features by LOFO importance, 0 for rest.
    Tests whether a small curated feature set outperforms masking.
    """
    df = pd.read_csv(lofo_csv).nlargest(k, "importance")
    top_feats = set(df["feature"].tolist())

    weights = np.zeros(len(feat_names), dtype=np.float32)
    for fi, fname in enumerate(feat_names):
        weights[fi] = 1.0 if fname in top_feats else 0.0
    return weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compute task-weighted Repr B for downstream evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--glottocodes_file", required=True,
                   help="JSON list of target glottocodes")
    p.add_argument("--lofo_csv", default=None,
                   help="LOFO importance CSV (from feature_importance.py)")
    p.add_argument("--domain_csv", default=None,
                   help="Domain analysis CSV (from feature_importance.py)")
    p.add_argument("--top_k", type=int, default=20,
                   help="k for top-k binary weighting")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)
    db = ckpt["config"].get("database", "unknown")
    feat_names = ckpt["feat_names_sorted"]
    n_feats = len(feat_names)
    print(f"  DB={db}, {n_feats} features")

    with open(args.glottocodes_file) as f:
        glottocodes = json.load(f)
    print(f"  {len(glottocodes)} target glottocodes")

    variants = {}

    # --- LOFO-weighted variants ---
    if args.lofo_csv and os.path.exists(args.lofo_csv):
        for strategy in ["positive_only", "rank_based", "threshold"]:
            w = lofo_weights(args.lofo_csv, feat_names, strategy)
            n_nonzero = (w > 0).sum()
            print(f"\n[LOFO-{strategy}] {n_nonzero}/{n_feats} features active")
            reprs, found = compute_repr_B_weighted(ckpt, glottocodes, w)
            tag = f"B_lofo_{strategy}"
            variants[tag] = (reprs, found)
            print(f"  Shape: {reprs.shape}")

        # Top-k binary
        for k in [10, 20, 50]:
            if k > n_feats:
                continue
            w = top_k_weights(args.lofo_csv, feat_names, k=k)
            print(f"\n[Top-{k}] {int(w.sum())}/{n_feats} features active")
            reprs, found = compute_repr_B_weighted(ckpt, glottocodes, w)
            tag = f"B_top{k}"
            variants[tag] = (reprs, found)
            print(f"  Shape: {reprs.shape}")

    # --- Domain-weighted variant ---
    if args.domain_csv and os.path.exists(args.domain_csv):
        w = domain_weights(args.domain_csv, feat_names, db)
        n_effective = (w > 0.15).sum()
        print(f"\n[Domain-weighted] {n_effective} features with w>0.15")
        reprs, found = compute_repr_B_weighted(ckpt, glottocodes, w)
        tag = "B_domain_weighted"
        variants[tag] = (reprs, found)
        print(f"  Shape: {reprs.shape}")

    # Save all
    for tag, (reprs, found) in variants.items():
        np.save(os.path.join(args.out_dir, f"repr_{tag}.npy"), reprs)
        with open(os.path.join(args.out_dir, f"repr_{tag}_langs.json"), "w") as f:
            json.dump(found, f)
    print(f"\nSaved {len(variants)} weighted representations to {args.out_dir}")


if __name__ == "__main__":
    main()
