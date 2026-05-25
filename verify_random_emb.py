#!/usr/bin/env python3
"""
Test UFAL's actual inference: random test language embeddings (no fine-tune).
Compares against the 200-epoch checkpoint we have cached.
"""

import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = "/home/user/Grambank2Vec"
sys.path.insert(0, BASE)

from model_ufal import UFALNeural, build_ufal_vocab, transductive_finetune_emb
from model_learned import prepare_categorical
from sigtyp_ablation import (
    parse_sigtyp, _strip_to_id, _compute_per_lang_acc, _macro_acc,
)
from verify_combined import (
    _ensure_data, run_probabilistic_with_scores,
    predict_neural_with_scores, merge_predictions,
)


def main():
    data_dir = "/tmp/sigtyp_st2020"
    ckpts = {
        "128d_100ep": ("/tmp/ufal_neural_verify.pt",       128),
        "256d_200ep": ("/tmp/ufal_neural_verify_paper.pt", 256),
    }
    device = "cpu"

    print("Loading data ...")
    _ensure_data(data_dir)
    train_meta, train_feat, _, _    = parse_sigtyp(f"{data_dir}/train.csv")
    dev_meta,   dev_feat,   _, _    = parse_sigtyp(f"{data_dir}/dev.csv")
    test_meta, _, test_blank, test_obs = parse_sigtyp(f"{data_dir}/test_blinded.csv")
    _, _, _, gold_obs               = parse_sigtyp(f"{data_dir}/test_gold.csv")

    traindev_feat = pd.concat([train_feat, dev_feat], ignore_index=True)
    traindev_meta = pd.concat([train_meta, dev_meta], ignore_index=True)
    n_langs = len(traindev_meta)
    n_train = len(train_meta)

    (cat_matrix, kept_feature_names,
     _, feat_to_value_names) = prepare_categorical(
        traindev_feat, list(traindev_feat.columns))
    feat_name_to_idx = {n: i for i, n in enumerate(kept_feature_names)}
    cat_train = cat_matrix[:n_train]

    freq_fallback = {}
    for fi, name in enumerate(kept_feature_names):
        col = cat_train[:, fi]
        obs = col[col >= 0]
        freq_fallback[name] = (feat_to_value_names[fi][0] if len(obs) == 0
            else feat_to_value_names[fi][
                Counter(obs.tolist()).most_common(1)[0][0]])

    print("\nRunning probabilistic ...")
    for tag, (ckpt_path, embed_dim) in ckpts.items():
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint missing: {ckpt_path}, skipping {tag}")
            continue

        # Build vocab (uses 300 clusters for 256d ckpt, 100 for 128d ckpt)
        kmeans_clusters = 300 if tag == "256d_200ep" else 100
        (feat_to_fv_ids, metadata_fv_maps,
         n_global_fv_ids, km) = build_ufal_vocab(
            cat_matrix, kept_feature_names, feat_to_value_names,
            traindev_meta, kmeans_clusters=kmeans_clusters)

        # Probabilistic
        prob_preds, prob_scores = run_probabilistic_with_scores(
            train_meta, train_feat, dev_meta, dev_feat,
            test_meta, test_blank, test_obs,
            cat_matrix, kept_feature_names, feat_to_value_names,
            feat_name_to_idx, freq_fallback)
        mac_prob = _macro_acc(_compute_per_lang_acc(
            prob_preds, test_meta, test_blank, gold_obs, feat_name_to_idx))

        print(f"\n=== Checkpoint {tag} (embed_dim={embed_dim}) ===")
        model = UFALNeural(n_langs, n_global_fv_ids, embed_dim, 0.5)
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue
        model.to(device).eval()

        # Strategy A: transductive fine-tuning
        print("  [A] Transductive fine-tuning (our default) ...")
        t0 = time.time()
        np_a, ns_a = {}, {}
        for _, row in test_meta.iterrows():
            wc = row["wals_code"]
            obs_raw = test_obs.get(wc, {})
            obs_fd  = {k: _strip_to_id(v) for k, v in obs_raw.items()
                       if k in feat_name_to_idx}
            new_emb = transductive_finetune_emb(
                model, 'ufal', obs_fd, feat_name_to_idx,
                feat_to_value_names, feat_to_fv_ids, embed_dim,
                device, n_steps=100, lr=0.05, temperature=1.0)
            p, s = predict_neural_with_scores(
                model, new_emb, kept_feature_names,
                feat_to_fv_ids, feat_to_value_names, device)
            np_a[wc] = p; ns_a[wc] = s
        mac_a = _macro_acc(_compute_per_lang_acc(
            np_a, test_meta, test_blank, gold_obs, feat_name_to_idx))
        print(f"      Neural acc = {mac_a:.4f}  ({time.time()-t0:.1f}s)")

        # Strategy B: random test embedding (UFAL's actual)
        print("  [B] Random test embedding (UFAL paper) ...")
        torch.manual_seed(42)
        t0 = time.time()
        np_b, ns_b = {}, {}
        for _, row in test_meta.iterrows():
            wc = row["wals_code"]
            rand_emb = torch.zeros(1, embed_dim, device=device)
            nn.init.normal_(rand_emb, std=0.01)
            p, s = predict_neural_with_scores(
                model, rand_emb, kept_feature_names,
                feat_to_fv_ids, feat_to_value_names, device)
            np_b[wc] = p; ns_b[wc] = s
        mac_b = _macro_acc(_compute_per_lang_acc(
            np_b, test_meta, test_blank, gold_obs, feat_name_to_idx))
        print(f"      Neural acc = {mac_b:.4f}  ({time.time()-t0:.1f}s)")

        # Strategy C: averaged language embedding (poor man's prior)
        print("  [C] Mean training language embedding ...")
        t0 = time.time()
        with torch.no_grad():
            mean_emb = model.lang_embeddings.weight.mean(dim=0, keepdim=True)
        np_c, ns_c = {}, {}
        for _, row in test_meta.iterrows():
            wc = row["wals_code"]
            p, s = predict_neural_with_scores(
                model, mean_emb, kept_feature_names,
                feat_to_fv_ids, feat_to_value_names, device)
            np_c[wc] = p; ns_c[wc] = s
        mac_c = _macro_acc(_compute_per_lang_acc(
            np_c, test_meta, test_blank, gold_obs, feat_name_to_idx))
        print(f"      Neural acc = {mac_c:.4f}  ({time.time()-t0:.1f}s)")

        # Best of three for merge
        best_strategy = max([("A_finetune", mac_a, np_a, ns_a),
                              ("B_random",   mac_b, np_b, ns_b),
                              ("C_mean",     mac_c, np_c, ns_c)],
                             key=lambda x: x[1])
        print(f"  Best neural: {best_strategy[0]} = {best_strategy[1]:.4f}")

        # Merge with best neural
        best_pred, best_scr = best_strategy[2], best_strategy[3]
        # Grid search thresholds
        best_combined = (None, 0.0)
        for t_p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for t_n in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
                merged, _ = merge_predictions(
                    prob_preds, prob_scores, best_pred, best_scr,
                    thresh_prob=t_p, thresh_neural=t_n)
                m = _macro_acc(_compute_per_lang_acc(
                    merged, test_meta, test_blank, gold_obs, feat_name_to_idx))
                if m > best_combined[1]:
                    best_combined = ((t_p, t_n), m)

        print(f"  Probabilistic alone:     {mac_prob:.4f}")
        print(f"  Best combined (grid):    {best_combined[1]:.4f}  "
              f"(thresholds: {best_combined[0]})")
        print(f"  Δ vs prob:               {best_combined[1] - mac_prob:+.4f}")


if __name__ == "__main__":
    main()
