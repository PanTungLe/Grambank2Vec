"""
downstream/task_weighted_repr.py
=================================
Compute task-weighted Repr B variants using LOFO feature importances.

Produces three weighted representations per run:
  repr_B_top{k}.npy             — Repr B restricted to top-k LOFO features
  repr_B_lofo_positive_only.npy — Repr B restricted to features with
                                   positive LOFO importance (removal hurts)
  repr_B_domain_top.npy         — Repr B restricted to features from the
                                   single highest-rho domain

These directly test whether concentrating on LOFO-identified features
boosts the typological transfer signal compared to uniform averaging.

Usage
-----
    python downstream/task_weighted_repr.py \\
        --checkpoint checkpoints/wals_learned_s42 \\
        --out_dir downstream_results/representations/wals_s42_weighted \\
        --glottocodes_file downstream_results/distances/glottocodes.json \\
        --lofo_csv downstream_results/feature_importance/wals/lofo_wals.csv \\
        --domain_csv downstream_results/feature_importance/wals/domain_analysis_wals.csv \\
        --top_k 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from downstream.representations import load_checkpoint, compute_repr_B


def save_repr(out_dir: str, name: str, mat: np.ndarray, langs: List[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"repr_{name}.npy"), mat)
    with open(os.path.join(out_dir, f"repr_{name}_langs.json"), "w") as f:
        json.dump(langs, f)
    print(f"  repr_{name}: {mat.shape}  ({len(langs)} languages)")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Compute task-weighted Repr B variants from LOFO importances",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--glottocodes_file", required=True,
                   help="JSON list of target glottocodes")
    p.add_argument("--lofo_csv", required=True,
                   help="LOFO importance CSV from feature_importance.py")
    p.add_argument("--domain_csv", required=True,
                   help="Domain analysis CSV from feature_importance.py")
    p.add_argument("--top_k", type=int, default=20,
                   help="Number of top LOFO features to use for repr_B_top{k}")
    args = p.parse_args(argv)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)
    feat_names = ckpt["feat_names_sorted"]
    n_feats = len(feat_names)
    print(f"  {n_feats} features, {len(ckpt['lang2id'])} languages")

    with open(args.glottocodes_file) as f:
        glottocodes = json.load(f)
    print(f"  Target glottocodes: {len(glottocodes)}")

    lofo = pd.read_csv(args.lofo_csv)
    domain_df = pd.read_csv(args.domain_csv)

    # Use feat_idx column directly (alphabetical index into feat_names_sorted)
    if "feat_idx" not in lofo.columns:
        # Fall back: look up by feature name
        name2idx = {n: i for i, n in enumerate(feat_names)}
        lofo["feat_idx"] = lofo["feature"].map(name2idx)
        lofo = lofo.dropna(subset=["feat_idx"])
        lofo["feat_idx"] = lofo["feat_idx"].astype(int)

    lofo = lofo.sort_values("importance", ascending=False)

    # --- Repr B: top-k LOFO features ---
    top_k = min(args.top_k, len(lofo))
    top_k_idx = lofo.head(top_k)["feat_idx"].tolist()
    print(f"\nComputing B_top{top_k} ({top_k} features)...")
    B_topk, found = compute_repr_B(ckpt, glottocodes, feature_subset=top_k_idx)
    save_repr(args.out_dir, f"B_top{top_k}", B_topk, found)

    # --- Repr B: LOFO-positive features only ---
    pos_df = lofo[lofo["importance"] > 0]
    pos_idx = pos_df["feat_idx"].tolist()
    print(f"\nComputing B_lofo_positive_only ({len(pos_idx)} features)...")
    B_pos, found = compute_repr_B(ckpt, glottocodes, feature_subset=pos_idx)
    save_repr(args.out_dir, "B_lofo_positive_only", B_pos, found)

    # --- Repr B: top domain features only ---
    best_domain = domain_df.sort_values("spearman_rho", ascending=False).iloc[0]["domain"]
    domain_feats = lofo[lofo["domain"] == best_domain]["feat_idx"].tolist()
    print(f"\nComputing B_domain_top ({len(domain_feats)} features, domain={best_domain})...")
    if domain_feats:
        B_dom, found = compute_repr_B(ckpt, glottocodes, feature_subset=domain_feats)
        save_repr(args.out_dir, f"B_domain_{best_domain.lower()}", B_dom, found)

    # --- Repr B: LOFO-weighted (importance as per-feature weight) ---
    # Only use features with positive importance; weight by raw importance value
    if len(pos_idx) > 0:
        pos_weights = pos_df["importance"].values.astype(np.float32)
        pos_weights /= pos_weights.sum()  # normalise to sum=1

        from scipy.special import softmax as scipy_softmax
        lang2id = ckpt["lang2id"]
        lang_embs = ckpt["lang_embs"]
        fv_embs = ckpt["fv_embs"]
        feat_to_global_ids = ckpt["feat_to_global_ids"]

        print(f"\nComputing B_lofo_weighted ({len(pos_idx)} features, importance-weighted)...")
        reprs_w, found_w = [], []
        for glot in glottocodes:
            if glot not in lang2id:
                continue
            idx = lang2id[glot]
            lang_emb = lang_embs[idx]
            weighted_sum = np.zeros(lang_embs.shape[1], dtype=np.float32)
            for fi, w in zip(pos_idx, pos_weights):
                gids = feat_to_global_ids[fi]
                val_embs_f = fv_embs[gids]
                scores = val_embs_f @ lang_emb
                probs = scipy_softmax(scores)
                expected = (probs[:, None] * val_embs_f).sum(0)
                weighted_sum += w * expected
            reprs_w.append(weighted_sum)
            found_w.append(glot)
        if reprs_w:
            B_w = np.stack(reprs_w, axis=0)
            save_repr(args.out_dir, "B_lofo_weighted", B_w, found_w)

    print(f"\nAll weighted representations saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
