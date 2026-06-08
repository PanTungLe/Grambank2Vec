#!/usr/bin/env python3
"""
Minimal-compute pilot study: 2×2 factorial + probabilistic replication check.

Factor 1 — loss/framing:   Binary BCE (K independent yes/no)
                            vs Categorical CE (K-way softmax)
Factor 2 — vocab:          UFAL (WALS + genus/family/geo-cluster metadata)
                            vs WALS-only value embeddings

Replication check: compare our probabilistic reimplementation to UFAL's
published 0.711 (test macro accuracy) and 0.698 (neural test macro accuracy).

Usage:
    python3 pilot_ablation.py [--data-dir DIR] [--epochs N] [--device cpu]
    Data auto-downloaded from UFAL's public GitHub if not present.
"""

import argparse
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = "/home/user/Grambank2Vec"
sys.path.insert(0, BASE)

from model_ufal import (
    UFALNeural, CategoricalNeural, CategoricalGeomNeural,
    build_ufal_vocab, build_training_data_ufal, build_training_data_wals_only,
    train_ufal_model, train_categorical_model, transductive_finetune_emb,
)
from model_learned import prepare_categorical
from sigtyp_eval_v4 import (
    parse_sigtyp, _strip_to_id,
    _compute_per_lang_acc, _macro_acc,
)


def predict_test_language(model, model_type, new_emb, kept_feature_names,
                          feat_name_to_idx, feat_to_value_names,
                          feat_to_fv_ids_or_gids, device, temperature=1.0):
    """Predict all features from a transductively-finetuned embedding.

    Uses raw dot-product (no cosine norm) matching the fixed training objective.
    """
    results = {}
    model.eval()
    with torch.no_grad():
        for fi, fname in enumerate(kept_feature_names):
            if model_type == 'ufal':
                ids = feat_to_fv_ids_or_gids[fi]
                probs = model.predict_feature_from_emb(new_emb, ids, device)
                pred_local = int(probs.argmax())
            else:
                pred_local, _ = model.predict_from_emb(
                    new_emb, fi, device, temperature)
            results[fname] = feat_to_value_names[fi][pred_local]
    return results


def run_ufal_probabilistic(*args, **kwargs):
    from sigtyp_ablation import run_ufal_probabilistic as _fn
    return _fn(*args, **kwargs)

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

_DATA_BASE = "https://raw.githubusercontent.com/ufal/ST2020/master/data"
_FILES = ["train.csv", "dev.csv", "test_blinded.csv", "test_gold.csv"]


def _ensure_data(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    for fname in _FILES:
        dst = os.path.join(data_dir, fname)
        if not os.path.exists(dst):
            url = f"{_DATA_BASE}/{fname}"
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, dst)
            print(f"    → {dst}")


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _infer_condition(model, model_type, fv_ids_map, test_meta, test_blank,
                     test_obs, feat_name_to_idx, kept_feature_names,
                     feat_to_value_names, embed_dim, device,
                     finetune_steps=50):
    """Transductively fine-tune + predict for every test language."""
    model.eval()
    preds = {}
    for _, row in test_meta.iterrows():
        wc      = row["wals_code"]
        obs_raw = test_obs.get(wc, {})
        obs_fd  = {k: _strip_to_id(v) for k, v in obs_raw.items()
                   if k in feat_name_to_idx}

        new_emb = transductive_finetune_emb(
            model, model_type, obs_fd, feat_name_to_idx, feat_to_value_names,
            fv_ids_map, embed_dim, device,
            n_steps=finetune_steps, lr=0.05, temperature=1.0)

        preds[wc] = predict_test_language(
            model, model_type, new_emb, kept_feature_names,
            feat_name_to_idx, feat_to_value_names, fv_ids_map, device,
            temperature=1.0)
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",  default="/tmp/sigtyp_st2020")
    ap.add_argument("--epochs",    type=int, default=10)
    ap.add_argument("--steps",     type=int, default=500,
                    help="Steps per epoch for UFALNeural (binary)")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--device",    default="cpu")
    ap.add_argument("--seeds",     type=str, default="42,43,44,45,46",
                    help="Comma-separated list of seeds (e.g. 42,43,44,45,46)")
    ap.add_argument("--finetune-steps", type=int, default=50)
    ap.add_argument("--skip-probabilistic", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 70)
    print("SIGTYP pilot: 2×2 factorial + UFAL probabilistic replication")
    print(f"  epochs={args.epochs}, embed_dim={args.embed_dim}, seeds={seeds}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Data
    # -----------------------------------------------------------------------
    print("\n[1] Downloading / loading data ...")
    _ensure_data(args.data_dir)
    d = args.data_dir

    train_meta, train_feat, train_blank, train_obs = parse_sigtyp(
        os.path.join(d, "train.csv"))
    dev_meta,   dev_feat,   dev_blank,   dev_obs   = parse_sigtyp(
        os.path.join(d, "dev.csv"))
    test_meta,  test_feat,  test_blank,  test_obs  = parse_sigtyp(
        os.path.join(d, "test_blinded.csv"))
    gold_meta,  gold_feat,  gold_blank,  gold_obs  = parse_sigtyp(
        os.path.join(d, "test_gold.csv"))

    n_train = len(train_meta)
    print(f"  Train={n_train}, Dev={len(dev_meta)}, Test={len(test_meta)}")

    # Combine train+dev for model training (matches UFAL)
    traindev_feat = pd.concat([train_feat, dev_feat], ignore_index=True)
    traindev_meta = pd.concat([train_meta, dev_meta], ignore_index=True)
    n_langs = len(traindev_meta)

    # -----------------------------------------------------------------------
    # 2. Vocabularies
    # -----------------------------------------------------------------------
    print("\n[2] Building vocabularies ...")
    feature_cols = list(traindev_feat.columns)

    # WALS-only vocab (Factor 2: value embeddings)
    (cat_matrix, kept_feature_names,
     feat_to_global_ids, feat_to_value_names) = prepare_categorical(
        traindev_feat, feature_cols)

    cat_train = cat_matrix[:n_train]
    n_features     = len(kept_feature_names)
    n_total_values = max(max(gids) for gids in feat_to_global_ids.values()) + 1
    feat_name_to_idx = {name: i for i, name in enumerate(kept_feature_names)}
    print(f"  WALS-only vocab: {n_features} features, {n_total_values} value IDs")

    # UFAL vocab: WALS + genus + family + geo-cluster (Factor 2: UFAL baseline)
    (feat_to_fv_ids, metadata_fv_maps,
     n_global_fv_ids, km) = build_ufal_vocab(
        cat_matrix, kept_feature_names, feat_to_value_names,
        traindev_meta, kmeans_clusters=100)  # 100 clusters for speed
    cluster_id_offset = metadata_fv_maps['cluster_id_offset']
    print(f"  UFAL vocab:      {n_global_fv_ids} fv IDs (incl. metadata)")

    # Frequency fallback for unresolvable blanks
    freq_fallback = {}
    for fi, name in enumerate(kept_feature_names):
        col = cat_train[:, fi]
        obs = col[col >= 0]
        if len(obs) == 0:
            freq_fallback[name] = feat_to_value_names[fi][0]
        else:
            cnt = Counter(obs.tolist())
            freq_fallback[name] = feat_to_value_names[fi][
                cnt.most_common(1)[0][0]]

    # -----------------------------------------------------------------------
    # 3. Probabilistic system (replication check — no training needed)
    # -----------------------------------------------------------------------
    # seed_results[condition_label] = list of per-seed macro accuracies
    seed_results = {
        "Binary+UFAL": [],
        "Binary+WALS": [],
        "Categ+UFAL":  [],
        "Categ+WALS":  [],
    }
    results = {}  # condition → macro_acc (last seed, kept for back-compat)

    if not args.skip_probabilistic:
        print("\n[3] Running UFAL probabilistic reimplementation ...")
        t0 = time.time()
        preds_prob = run_ufal_probabilistic(
            train_meta, train_feat, dev_meta, dev_feat,
            test_meta, test_blank, test_obs,
            cat_matrix, kept_feature_names, feat_to_value_names,
            feat_name_to_idx, freq_fallback)
        elapsed = time.time() - t0

        pl_prob = _compute_per_lang_acc(
            preds_prob, test_meta, test_blank, gold_obs, feat_name_to_idx)
        mac_prob = _macro_acc(pl_prob)
        results["probabilistic"] = mac_prob
        print(f"  Probabilistic macro acc = {mac_prob:.4f}  "
              f"[UFAL published: 0.711]  ({elapsed:.1f}s)")

    # -----------------------------------------------------------------------
    # 4. Training positives for binary models
    # -----------------------------------------------------------------------
    print("\n[4] Building training positive pairs ...")
    pos_ufal = build_training_data_ufal(
        cat_matrix, traindev_meta, feat_to_fv_ids,
        metadata_fv_maps, km, cluster_id_offset, feat_to_value_names)
    pos_wals = build_training_data_wals_only(cat_matrix, feat_to_global_ids)
    print(f"  UFAL positives: {len(pos_ufal)}, WALS-only positives: {len(pos_wals)}")

    # -----------------------------------------------------------------------
    # 5. Four neural conditions × N seeds
    # -----------------------------------------------------------------------
    D = args.device
    E = args.embed_dim

    conditions = [
        ("Binary+UFAL",  "binary_ufal"),
        ("Binary+WALS",  "binary_wals"),
        ("Categ+UFAL",   "categ_ufal"),
        ("Categ+WALS",   "categ_wals"),
    ]

    def _run_condition(label, cond_key, seed):
        print(f"\n[5] {label}  seed={seed} ...")
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)

        if cond_key == "binary_ufal":
            model = UFALNeural(n_langs, n_global_fv_ids, E, 0.5)
            train_ufal_model(model, pos_ufal, n_global_fv_ids,
                             n_epochs=args.epochs, steps_per_epoch=args.steps,
                             lr=1e-3, l2_reg=0.1, device=D, seed=seed)
            fv_map = feat_to_fv_ids
            mtype  = 'ufal'

        elif cond_key == "binary_wals":
            model = UFALNeural(n_langs, n_total_values, E, 0.5)
            train_ufal_model(model, pos_wals, n_total_values,
                             n_epochs=args.epochs, steps_per_epoch=args.steps,
                             lr=1e-3, l2_reg=0.1, device=D, seed=seed)
            fv_map = feat_to_global_ids
            mtype  = 'ufal'

        elif cond_key == "categ_ufal":
            model = CategoricalNeural(n_langs, n_global_fv_ids,
                                      feat_to_fv_ids, E, 0.5)
            train_categorical_model(model, cat_matrix,
                                    n_epochs=args.epochs, lr=1e-3,
                                    l2_reg=0.1, device=D, seed=seed)
            fv_map = feat_to_fv_ids
            mtype  = 'categorical'

        else:  # categ_wals
            model = CategoricalGeomNeural(n_langs, n_total_values,
                                          feat_to_global_ids, E, 0.5)
            train_categorical_model(model, cat_matrix,
                                    n_epochs=args.epochs, lr=1e-3,
                                    l2_reg=0.1, device=D, seed=seed)
            fv_map = feat_to_global_ids
            mtype  = 'geom'

        t_train = time.time() - t0
        print(f"  Training done in {t_train:.1f}s, running inference ...")

        preds = _infer_condition(
            model, mtype, fv_map, test_meta, test_blank, test_obs,
            feat_name_to_idx, kept_feature_names, feat_to_value_names,
            E, D, finetune_steps=args.finetune_steps)

        pl  = _compute_per_lang_acc(preds, test_meta, test_blank,
                                     gold_obs, feat_name_to_idx)
        acc = _macro_acc(pl)
        seed_results[label].append(acc)
        results[label] = acc
        total = time.time() - t0
        print(f"  {label} seed={seed}: macro acc = {acc:.4f}  ({total:.1f}s total)")
        return acc

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")
        for label, key in conditions:
            _run_condition(label, key, seed)

    # -----------------------------------------------------------------------
    # 6. Results table  (mean ± std across seeds)
    # -----------------------------------------------------------------------
    def _ms(vals):
        """Return (mean, std) as strings, or ('nan', 'nan') if empty."""
        if not vals:
            return float("nan"), float("nan")
        a = np.array(vals, dtype=float)
        return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  N seeds = {len(seeds)}  ({seeds})")
    print(f"  Epochs  = {args.epochs}, embed_dim = {args.embed_dim}")

    # 2×2 factorial table — mean±std
    bu_m, bu_s = _ms(seed_results["Binary+UFAL"])
    bw_m, bw_s = _ms(seed_results["Binary+WALS"])
    cu_m, cu_s = _ms(seed_results["Categ+UFAL"])
    cw_m, cw_s = _ms(seed_results["Categ+WALS"])

    def _fmt(m, s):
        return f"{m:.4f}±{s:.4f}"

    col_w = 18
    print(f"\n  2×2 Factorial Table (macro accuracy, test set, mean±std):\n")
    print(f"  {'':20s}  {'UFAL vocab':>{col_w}}  {'WALS-only vocab':>{col_w}}")
    print(f"  {'':20s}  {'(WALS+metadata)':>{col_w}}  {'(no metadata)':>{col_w}}")
    print(f"  {'-'*60}")
    print(f"  {'Binary BCE':20s}  {_fmt(bu_m, bu_s):>{col_w}}  {_fmt(bw_m, bw_s):>{col_w}}")
    print(f"  {'Categorical CE':20s}  {_fmt(cu_m, cu_s):>{col_w}}  {_fmt(cw_m, cw_s):>{col_w}}")
    print()

    # Per-seed raw numbers
    print("  Per-seed breakdown:")
    for s, bu, bw, cu, cw in zip(
            seeds,
            seed_results["Binary+UFAL"],
            seed_results["Binary+WALS"],
            seed_results["Categ+UFAL"],
            seed_results["Categ+WALS"]):
        print(f"    seed={s}  BinUFAL={bu:.4f}  BinWALS={bw:.4f}  "
              f"CatUFAL={cu:.4f}  CatWALS={cw:.4f}")
    print()

    # Main effects (per-seed, then averaged)
    f1_per = [((cu - bu) + (cw - bw)) / 2
              for bu, bw, cu, cw in zip(
                  seed_results["Binary+UFAL"], seed_results["Binary+WALS"],
                  seed_results["Categ+UFAL"],  seed_results["Categ+WALS"])]
    f2_per = [((bw - bu) + (cw - cu)) / 2
              for bu, bw, cu, cw in zip(
                  seed_results["Binary+UFAL"], seed_results["Binary+WALS"],
                  seed_results["Categ+UFAL"],  seed_results["Categ+WALS"])]
    f1_m, f1_s = _ms(f1_per)
    f2_m, f2_s = _ms(f2_per)
    print(f"  Factor 1 (Categorical − Binary):   {f1_m:+.4f} ± {f1_s:.4f}")
    print(f"  Factor 2 (WALS-only − UFAL vocab): {f2_m:+.4f} ± {f2_s:.4f}")
    print()

    # Probabilistic comparison
    if "probabilistic" in results:
        mac_prob = results["probabilistic"]
        print(f"  Probabilistic reimpl:  {mac_prob:.4f}")
        print(f"  UFAL published (probabilistic test): 0.7110")
        print(f"  UFAL published (neural test):        0.6980")
        print(f"  Δ vs UFAL probabilistic: {mac_prob - 0.711:+.4f}")
        print()

    print("  Note: pilot uses 10 epochs vs UFAL's 200; expect ~30–50% of")
    print("  full-training gap to close with more compute.")
    print("=" * 70)


if __name__ == "__main__":
    main()
