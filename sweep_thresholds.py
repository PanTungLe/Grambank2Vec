#!/usr/bin/env python3
"""
Threshold sweep on cached UFAL combined predictions.
Reuses /tmp/ufal_neural_verify.pt checkpoint to avoid retraining.
"""

import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

BASE = "/home/user/Grambank2Vec"
sys.path.insert(0, BASE)

from model_ufal import (
    UFALNeural, build_ufal_vocab, transductive_finetune_emb,
)
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
    ckpt     = "/tmp/ufal_neural_verify.pt"
    device   = "cpu"
    embed_dim = 128
    clusters  = 100

    print("Loading data ...")
    _ensure_data(data_dir)
    train_meta, train_feat, _, _    = parse_sigtyp(f"{data_dir}/train.csv")
    dev_meta,   dev_feat,   _, _    = parse_sigtyp(f"{data_dir}/dev.csv")
    test_meta,  _, test_blank, test_obs = parse_sigtyp(f"{data_dir}/test_blinded.csv")
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

    (feat_to_fv_ids, metadata_fv_maps,
     n_global_fv_ids, km) = build_ufal_vocab(
        cat_matrix, kept_feature_names, feat_to_value_names,
        traindev_meta, kmeans_clusters=clusters)

    freq_fallback = {}
    for fi, name in enumerate(kept_feature_names):
        col = cat_train[:, fi]
        obs = col[col >= 0]
        freq_fallback[name] = (feat_to_value_names[fi][0] if len(obs) == 0
            else feat_to_value_names[fi][
                Counter(obs.tolist()).most_common(1)[0][0]])

    print("\nRunning probabilistic ...")
    prob_preds, prob_scores = run_probabilistic_with_scores(
        train_meta, train_feat, dev_meta, dev_feat,
        test_meta, test_blank, test_obs,
        cat_matrix, kept_feature_names, feat_to_value_names,
        feat_name_to_idx, freq_fallback)
    mac_prob = _macro_acc(_compute_per_lang_acc(
        prob_preds, test_meta, test_blank, gold_obs, feat_name_to_idx))

    print(f"\nLoading neural checkpoint {ckpt} ...")
    model = UFALNeural(n_langs, n_global_fv_ids, embed_dim, 0.5)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    model.to(device).eval()

    print("Neural inference ...")
    t0 = time.time()
    neural_preds, neural_scores = {}, {}
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
        neural_preds[wc]  = p
        neural_scores[wc] = s
    mac_neural = _macro_acc(_compute_per_lang_acc(
        neural_preds, test_meta, test_blank, gold_obs, feat_name_to_idx))
    print(f"  Done in {time.time()-t0:.1f}s")

    # Diagnose neural score distribution
    all_neural = []
    for wc in neural_scores:
        all_neural.extend(neural_scores[wc].values())
    all_prob = []
    for wc in prob_scores:
        all_prob.extend(prob_scores[wc].values())
    all_neural = np.array(all_neural)
    all_prob   = np.array(all_prob)
    print(f"\nScore distributions:")
    print(f"  Neural scores: mean={all_neural.mean():.3f}, "
          f"median={np.median(all_neural):.3f}, "
          f"q10={np.percentile(all_neural,10):.3f}, "
          f"q90={np.percentile(all_neural,90):.3f}")
    print(f"  Prob   scores: mean={all_prob.mean():.3f}, "
          f"median={np.median(all_prob):.3f}, "
          f"q10={np.percentile(all_prob,10):.3f}, "
          f"q90={np.percentile(all_prob,90):.3f}")

    print(f"\nBaselines:")
    print(f"  Probabilistic alone:  {mac_prob:.4f}")
    print(f"  Neural alone:         {mac_neural:.4f}")
    print(f"  UFAL paper target:    0.755 (dev combined)")

    print(f"\nThreshold sweep (use prob if score_p > t_p AND score_n < t_n):")
    print(f"  {'t_p':>5} {'t_n':>5} | {'combined':>9} | {'use_prob':>9}")
    best = ((0.0, 1.0), mac_prob, (0, 0))
    grid_p = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    grid_n = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
              0.90, 0.95, 1.00]
    for t_p in grid_p:
        for t_n in grid_n:
            merged, (n_p, n_n, n_dis) = merge_predictions(
                prob_preds, prob_scores, neural_preds, neural_scores,
                thresh_prob=t_p, thresh_neural=t_n)
            mac = _macro_acc(_compute_per_lang_acc(
                merged, test_meta, test_blank, gold_obs, feat_name_to_idx))
            tag = "  *" if mac > best[1] else ""
            if mac > best[1]:
                best = ((t_p, t_n), mac, (n_p, n_n))
            if mac >= mac_prob - 0.002 or (t_p in (0.0, 0.5) and t_n in (0.5, 0.65, 1.0)):
                print(f"  {t_p:>5.2f} {t_n:>5.2f} | {mac:>9.4f} | "
                      f"{n_p:>5}/{n_p+n_n:<5d}{tag}")

    print(f"\nBest threshold: t_p={best[0][0]}, t_n={best[0][1]}")
    print(f"  Combined acc: {best[1]:.4f}")
    print(f"  Probabilistic used: {best[2][0]} / {sum(best[2])}")
    print(f"  vs probabilistic alone: {best[1] - mac_prob:+.4f}")
    print(f"  vs UFAL paper 0.755:    {best[1] - 0.755:+.4f}")


if __name__ == "__main__":
    main()
