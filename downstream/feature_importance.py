"""
downstream/feature_importance.py
==================================
Identify which typological features carry the most transfer-predictive signal.

Motivation
----------
The masking experiment showed that Repr B under 70% feature masking (Grambank)
outperforms full Repr B. This could mean:
  (a) Most features add noise; a small subset carries the transfer signal
  (b) The specific seed-42 feature subset was a lucky draw
  (c) Feature averaging collapses discriminative signal; subsets preserve it

This module disambiguates (a) from (b/c) by:
  1. Leave-one-feature-out (LOFO): drop one feature at a time from Repr B
     and measure how Spearman rho changes. Features whose removal HURTS
     most are the most important.
  2. Feature category analysis: group features by typological domain
     (word order, morphology, phonology, etc.) and compare domain-level
     Repr B against the full-feature Repr B.
  3. Permutation importance: shuffle one feature's expected embedding
     across languages (break its signal) and measure rho degradation.

Usage
-----
    python downstream/feature_importance.py \\
        --checkpoint checkpoints/grambank_learned_s42 \\
        --transfer_matrix downstream_results/transfer/transfer_matrix.csv \\
        --out_dir downstream_results/feature_importance
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
from scipy.stats import spearmanr
from scipy.special import softmax as scipy_softmax

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from downstream.representations import load_checkpoint
from downstream.distances import cosine_distance_matrix


# ---------------------------------------------------------------------------
# WALS and Grambank feature domain mappings
# ---------------------------------------------------------------------------

# Broad typological domain for each WALS feature chapter
# Source: WALS chapter list (wals.info/chapter)
WALS_FEATURE_DOMAINS = {
    # Phonology (1A-19A)
    **{f"{i}A": "Phonology" for i in range(1, 20)},
    # Morphology (20A-29A)
    **{f"{i}A": "Morphology" for i in range(20, 30)},
    # Nominal (30A-57A)
    **{f"{i}A": "Nominal" for i in range(30, 58)},
    # Verbal (58A-80A)
    **{f"{i}A": "Verbal" for i in range(58, 81)},
    # Word order (81A-97A)
    **{f"{i}A": "WordOrder" for i in range(81, 98)},
    # Simple clauses (98A-121A)
    **{f"{i}A": "SimpleClause" for i in range(98, 122)},
    # Complex sentences (122A-138A)
    **{f"{i}A": "ComplexSentence" for i in range(122, 139)},
    # Lexicon (139A-145A)
    **{f"{i}A": "Lexicon" for i in range(139, 146)},
    # Sign languages (146A-155A)
    **{f"{i}A": "SignLanguage" for i in range(146, 156)},
    # Other (156A+)
    **{f"{i}A": "Other" for i in range(156, 200)},
}

# Grambank feature domains (GB001-GB195)
# Source: Grambank parameter documentation
GRAMBANK_FEATURE_DOMAINS = {}
# GB001-GB010: Morphological complexity
for i in range(1, 11):
    GRAMBANK_FEATURE_DOMAINS[f"GB{i:03d}"] = "Morphology"
# GB011-GB025: Nominal morphology
for i in range(11, 26):
    GRAMBANK_FEATURE_DOMAINS[f"GB{i:03d}"] = "Nominal"
# GB026-GB065: Verbal morphology and TAM
for i in range(26, 66):
    GRAMBANK_FEATURE_DOMAINS[f"GB{i:03d}"] = "Verbal"
# GB066-GB110: Word order
for i in range(66, 111):
    GRAMBANK_FEATURE_DOMAINS[f"GB{i:03d}"] = "WordOrder"
# GB111-GB150: NP structure
for i in range(111, 151):
    GRAMBANK_FEATURE_DOMAINS[f"GB{i:03d}"] = "NominalPhrase"
# GB151-GB195: Clause structure
for i in range(151, 196):
    GRAMBANK_FEATURE_DOMAINS[f"GB{i:03d}"] = "ClauseStructure"


def get_feature_domain(feat_name: str, db: str) -> str:
    """Return broad typological domain for a feature name."""
    if db == "wals":
        return WALS_FEATURE_DOMAINS.get(feat_name, "Other")
    elif db == "grambank":
        return GRAMBANK_FEATURE_DOMAINS.get(feat_name, "Other")
    return "Other"


# ---------------------------------------------------------------------------
# Core: compute Repr B for a feature subset
# ---------------------------------------------------------------------------

def repr_B_subset(
    ckpt: Dict,
    glottocodes: List[str],
    feat_subset_indices: List[int],
) -> np.ndarray:
    """Compute Repr B using only the given feature subset indices."""
    lang2id = ckpt["lang2id"]
    lang_embs = ckpt["lang_embs"]
    fv_embs = ckpt["fv_embs"]
    feat_to_global_ids = ckpt["feat_to_global_ids"]

    result = []
    for glot in glottocodes:
        if glot not in lang2id:
            result.append(np.zeros(lang_embs.shape[1], dtype=np.float32))
            continue
        idx = lang2id[glot]
        lang_emb = lang_embs[idx]

        feat_vecs = []
        for fi in feat_subset_indices:
            gids = feat_to_global_ids[fi]
            val_embs_f = fv_embs[gids]
            scores = val_embs_f @ lang_emb
            probs = scipy_softmax(scores)
            expected = (probs[:, None] * val_embs_f).sum(0)
            feat_vecs.append(expected)

        T = np.stack(feat_vecs).mean(0)
        result.append(T.astype(np.float32))

    return np.stack(result)


def spearman_from_dist(dist_matrix: np.ndarray,
                       transfer_matrix: pd.DataFrame,
                       glottocodes: List[str]) -> float:
    """Compute mean Spearman rho across all targets."""
    N = len(glottocodes)
    rhos = []
    for j, tgt in enumerate(glottocodes):
        if tgt not in transfer_matrix.index:
            continue
        accs  = np.array([float(transfer_matrix.loc[tgt, src])
                          if src in transfer_matrix.columns and
                          not np.isnan(float(transfer_matrix.loc[tgt, src]))
                          else np.nan
                          for src in glottocodes])
        dists = dist_matrix[j]
        valid = ~(np.isnan(accs) | np.isnan(dists))
        valid[j] = False  # exclude self
        if valid.sum() < 3:
            continue
        rho, _ = spearmanr(dists[valid], -accs[valid])
        rhos.append(rho)
    return float(np.mean(rhos)) if rhos else np.nan


# ---------------------------------------------------------------------------
# Leave-one-feature-out importance
# ---------------------------------------------------------------------------

def lofo_importance(
    ckpt: Dict,
    glottocodes: List[str],
    transfer_matrix: pd.DataFrame,
    out_dir: str,
    max_features: int = 50,  # cap for speed; use None for all features
) -> pd.DataFrame:
    """
    Leave-one-feature-out feature importance.

    For each feature f:
      baseline_rho = Spearman(Repr_B_all_features)
      lofo_rho_f   = Spearman(Repr_B_without_f)
      importance_f = baseline_rho - lofo_rho_f
        (positive = removing f hurts = f is informative)
        (negative = removing f helps = f adds noise)
    """
    os.makedirs(out_dir, exist_ok=True)

    feat_names = ckpt["feat_names_sorted"]
    feat_to_global_ids = ckpt["feat_to_global_ids"]
    all_feat_indices = list(range(len(feat_names)))
    db = ckpt["config"].get("database", "unknown")

    # Filter glottocodes to those in both checkpoint and transfer matrix
    valid_glots = [g for g in glottocodes
                   if g in ckpt["lang2id"]
                   and g in transfer_matrix.index]

    print(f"  LOFO analysis: {len(valid_glots)} languages, "
          f"{len(feat_names)} features")

    # Baseline: all features
    B_full = repr_B_subset(ckpt, valid_glots, all_feat_indices)
    D_full = cosine_distance_matrix(B_full)
    baseline_rho = spearman_from_dist(D_full, transfer_matrix, valid_glots)
    print(f"  Baseline (all features): rho = {baseline_rho:.4f}")

    # Cap features for speed
    feat_indices_to_test = all_feat_indices
    if max_features and len(feat_indices_to_test) > max_features:
        # Sample uniformly across feature space
        step = len(feat_indices_to_test) // max_features
        feat_indices_to_test = feat_indices_to_test[::step][:max_features]
        print(f"  Testing {len(feat_indices_to_test)} features (sampled for speed)")

    rows = []
    for i, fi in enumerate(feat_indices_to_test):
        # Remove feature fi
        subset = [f for f in all_feat_indices if f != fi]
        B_sub = repr_B_subset(ckpt, valid_glots, subset)
        D_sub = cosine_distance_matrix(B_sub)
        rho_without = spearman_from_dist(D_sub, transfer_matrix, valid_glots)
        importance = baseline_rho - rho_without

        feat_name = feat_names[fi]
        domain = get_feature_domain(feat_name, db)
        rows.append({
            "feature": feat_name,
            "domain": domain,
            "feat_idx": fi,
            "rho_without": rho_without,
            "importance": importance,  # positive = informative, negative = noisy
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(feat_indices_to_test)}] "
                  f"{feat_name}: importance={importance:+.4f}")

    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    df.to_csv(os.path.join(out_dir, f"lofo_{db}.csv"), index=False)
    print(f"\n  Top 10 most informative features (removal hurts most):")
    print(df[["feature", "domain", "importance"]].head(10).to_string(index=False))
    print(f"\n  Top 10 noisiest features (removal helps most):")
    print(df[["feature", "domain", "importance"]].tail(10).to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Domain-level analysis
# ---------------------------------------------------------------------------

def domain_analysis(
    ckpt: Dict,
    glottocodes: List[str],
    transfer_matrix: pd.DataFrame,
    out_dir: str,
) -> pd.DataFrame:
    """
    Compute Repr B using only features from each typological domain.

    Tests: which feature domain carries the most transfer-predictive signal?
    Word order features should matter most for POS/parsing transfer.
    """
    os.makedirs(out_dir, exist_ok=True)

    feat_names = ckpt["feat_names_sorted"]
    db = ckpt["config"].get("database", "unknown")

    valid_glots = [g for g in glottocodes
                   if g in ckpt["lang2id"]
                   and g in transfer_matrix.index]

    # Group features by domain
    domain_feats: Dict[str, List[int]] = {}
    for fi, fname in enumerate(feat_names):
        domain = get_feature_domain(fname, db)
        domain_feats.setdefault(domain, []).append(fi)

    print(f"  Feature domains: {dict((d, len(v)) for d,v in domain_feats.items())}")

    rows = []
    for domain, feat_indices in sorted(domain_feats.items()):
        if len(feat_indices) < 2:
            continue
        B = repr_B_subset(ckpt, valid_glots, feat_indices)
        D = cosine_distance_matrix(B)
        rho = spearman_from_dist(D, transfer_matrix, valid_glots)
        rows.append({
            "domain": domain,
            "n_features": len(feat_indices),
            "spearman_rho": rho,
        })
        print(f"    {domain:20s}: n={len(feat_indices):3d}  rho={rho:+.4f}")

    df = pd.DataFrame(rows).sort_values("spearman_rho", ascending=False)
    df.to_csv(os.path.join(out_dir, f"domain_analysis_{db}.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Feature importance analysis for typological transfer prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--transfer_matrix", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--glottocodes_file", default=None,
                   help="JSON list of glottocodes (default: all in checkpoint∩transfer)")
    p.add_argument("--max_lofo_features", type=int, default=50,
                   help="Max features for LOFO (0=all, slow)")
    p.add_argument("--skip_lofo", action="store_true")
    p.add_argument("--skip_domain", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)
    db = ckpt["config"].get("database", "unknown")
    print(f"  DB={db}, {len(ckpt['feat_names_sorted'])} features")

    transfer_matrix = pd.read_csv(args.transfer_matrix, index_col=0)
    print(f"Transfer matrix: {transfer_matrix.shape}")

    if args.glottocodes_file:
        with open(args.glottocodes_file) as f:
            glottocodes = json.load(f)
    else:
        glottocodes = [g for g in ckpt["lang2id"]
                       if g in transfer_matrix.index]
    print(f"Shared languages: {len(glottocodes)}")

    if not args.skip_lofo:
        print("\n--- Leave-One-Feature-Out Importance ---")
        max_f = args.max_lofo_features if args.max_lofo_features > 0 else None
        lofo_importance(ckpt, glottocodes, transfer_matrix,
                        args.out_dir, max_features=max_f)

    if not args.skip_domain:
        print("\n--- Domain-Level Analysis ---")
        domain_analysis(ckpt, glottocodes, transfer_matrix, args.out_dir)

    print(f"\nResults saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
