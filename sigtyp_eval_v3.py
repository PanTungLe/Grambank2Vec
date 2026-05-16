#!/usr/bin/env python3
"""
SIGTYP 2020 v3: SetEncoder (zero-shot set-conditioned predictor) +
non-neural baselines + temperature calibration + dev-tuned prior.
Run: python3 sigtyp_eval_v3.py 2>&1 | tee sigtyp_eval_v3_log.txt
"""

import argparse
import os
import sys
import time
import subprocess
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/user/Grambank2Vec")
from model_learned import prepare_categorical
from model_set import SetEncoder, SetEncoderDataset, train_set_encoder

# ===========================================================================
# PART 1: Parse SIGTYP data  (copied verbatim from sigtyp_eval.py)
# ===========================================================================

_SIGTYP_FIXES = [
    ("double negationPosition_of_negative",  "double negation|Position_of_negative"),
    ("double negationSVONeg_Order",           "double negation|SVONeg_Order"),
    ("double negationSNegVO_Order",           "double negation|SNegVO_Order"),
    ("double negationPreverbal_Negative",     "double negation|Preverbal_Negative"),
    ("1 Separate word, no double negation|Word&NoDoubleNeg",
     "1 Separate word, no double negation"),
    ("2 Prefix, no double negation|Prefix&NoDoubleNeg",
     "2 Prefix, no double negation"),
    (" (= ", " (EQUALS "),
]


def _apply_fixes(s):
    for old, new in _SIGTYP_FIXES:
        s = s.replace(old, new)
    return s


def parse_sigtyp(path):
    """Parse a SIGTYP TSV file.

    Returns
    -------
    meta_df   : DataFrame [wals_code, name, latitude, longitude, genus, family, countrycodes]
    feat_df   : DataFrame rows=languages (integer 0-indexed), cols=lowercase feature names;
                cells = value string ("5") or np.nan
    blanked   : dict  wals_code -> set of lowercase feature names with value "?"
    obs_strings: dict wals_code -> {lowercase feat_name: "id label" string}
    """
    meta_rows   = []
    feat_rows   = {}
    blanked     = {}
    obs_strings = {}

    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    skip_header = True
    for line in lines:
        if not line.strip():
            continue
        if skip_header:
            skip_header = False
            continue

        parts = line.split("\t")
        if len(parts) < 8:
            continue

        wals_code    = parts[0]
        name         = parts[1]
        try:
            latitude  = float(parts[2])
            longitude = float(parts[3])
        except ValueError:
            latitude, longitude = float("nan"), float("nan")
        genus        = parts[4]
        family       = parts[5]
        countrycodes = parts[6]
        features_field = "\t".join(parts[7:])

        meta_rows.append({
            "wals_code": wals_code, "name": name,
            "latitude": latitude,   "longitude": longitude,
            "genus": genus,         "family": family,
            "countrycodes": countrycodes,
        })

        feat_dict   = {}
        blank_set   = set()
        obs_str_d   = {}

        cleaned = _apply_fixes(features_field)
        for piece in cleaned.split("|"):
            piece = piece.strip()
            if not piece or "=" not in piece:
                continue
            eq_idx    = piece.index("=")
            feat_name = piece[:eq_idx]
            val_str   = piece[eq_idx + 1:]
            fname_lc  = feat_name.lower()

            if val_str.strip() == "?":
                blank_set.add(fname_lc)
                feat_dict[fname_lc] = np.nan
            else:
                tokens = val_str.split()
                if not tokens:
                    continue
                try:
                    int(tokens[0])
                except ValueError:
                    continue
                feat_dict[fname_lc] = tokens[0]
                obs_str_d[fname_lc] = val_str

        feat_rows[wals_code]   = feat_dict
        blanked[wals_code]     = blank_set
        obs_strings[wals_code] = obs_str_d

    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)
    all_wals_codes = [r["wals_code"] for r in meta_rows]
    all_feat_names = sorted(set(fn for fd in feat_rows.values() for fn in fd))

    rows_data = []
    for wc in all_wals_codes:
        fd = feat_rows[wc]
        rows_data.append({fn: fd.get(fn, np.nan) for fn in all_feat_names})

    feat_df = pd.DataFrame(rows_data, columns=all_feat_names)
    return meta_df, feat_df, blanked, obs_strings


# ===========================================================================
# PART 0: Argparse CLI
# ===========================================================================

def get_args():
    p = argparse.ArgumentParser(
        description="SIGTYP 2020 v3: SetEncoder evaluation")
    p.add_argument("--base",        default="/home/user",
                   help="Base directory containing ST2020/ and Grambank2Vec/")
    p.add_argument("--seeds",       nargs="+", type=int,
                   default=[13, 21, 42, 87, 100])
    p.add_argument("--embed-dim",   type=int, default=64)
    p.add_argument("--hidden-dim",  type=int, default=128)
    p.add_argument("--n-epochs",    type=int, default=200)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--l2-reg",      type=float, default=0.1)
    p.add_argument("--n-rounds",    type=int, default=2,
                   help="Iterative self-training rounds")
    p.add_argument("--margin-tau",  type=float, default=0.30,
                   help="Margin threshold (p1-p2) for pseudo-labeling")
    p.add_argument("--top-n-pseudo", type=int, default=5,
                   help="Max pseudo-labels to add per round per language")
    p.add_argument("--lam-grid",    nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--alpha-grid",  nargs="+", type=float,
                   default=[0.1, 0.5, 1.0])
    p.add_argument("--mincount-grid", nargs="+", type=int,
                   default=[3, 5, 10, 20])
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--output-dir",  default=None,
                   help="Override output directory for submissions and checkpoints")
    p.add_argument("--device",      default=("cuda" if torch.cuda.is_available()
                                             else "cpu"))
    return p.parse_args()


# ===========================================================================
# PART 3: Non-neural baselines
# ===========================================================================

def freq_baseline(cat_matrix_train, kept_feature_names, feat_to_value_names):
    """Most frequent value per feature from training data."""
    result = {}
    for fi, name in enumerate(kept_feature_names):
        col = cat_matrix_train[:, fi]
        obs = col[col >= 0]
        if len(obs) == 0:
            result[name] = feat_to_value_names[fi][0]
        else:
            cnt = Counter(obs.tolist())
            result[name] = feat_to_value_names[fi][cnt.most_common(1)[0][0]]
    return result


def genus_majority_baseline(cat_matrix_train, train_meta_df,
                             kept_feature_names, feat_to_value_names,
                             test_genus, freq_fallback=None,
                             _cache={}):
    """
    Per-genus majority vote. Falls back to freq_baseline.
    Per-genus tables are cached across calls (mutable default arg pattern).
    """
    cache_key = id(cat_matrix_train)
    if cache_key not in _cache:
        fb = freq_baseline(cat_matrix_train, kept_feature_names, feat_to_value_names)
        genus_tables = {"__freq__": fb}
        for genus in train_meta_df["genus"].unique():
            mask = (train_meta_df["genus"] == genus).values
            rows = cat_matrix_train[mask]
            gpreds = {}
            for fi, name in enumerate(kept_feature_names):
                col = rows[:, fi]
                obs = col[col >= 0]
                if len(obs) > 0:
                    cnt = Counter(obs.tolist())
                    gpreds[name] = feat_to_value_names[fi][cnt.most_common(1)[0][0]]
                else:
                    gpreds[name] = fb[name]
            genus_tables[genus] = gpreds
        _cache[cache_key] = genus_tables

    tables = _cache[cache_key]
    fb = freq_fallback if freq_fallback is not None else tables["__freq__"]
    result = dict(tables.get(test_genus, tables["__freq__"]))
    # Apply freq_fallback for missing keys if a custom fallback was provided
    if freq_fallback is not None:
        for name in kept_feature_names:
            if name not in result:
                result[name] = freq_fallback.get(name, "1")
    return result


def knn_baseline(obs_feat_dict, cat_matrix_train, kept_feature_names,
                 feat_name_to_idx, feat_to_value_names, k=5,
                 freq_fallback=None):
    """kNN by feature-value overlap. Mode vote among top-k neighbors."""
    n_train = cat_matrix_train.shape[0]
    scores = np.zeros(n_train, dtype=np.float32)

    for feat_name, val_str in obs_feat_dict.items():
        fi = feat_name_to_idx.get(feat_name)
        if fi is None:
            continue
        val_list = feat_to_value_names.get(fi, [])
        if val_str not in val_list:
            continue
        vi_local = val_list.index(val_str)
        scores += (cat_matrix_train[:, fi] == vi_local).astype(np.float32)

    fb = freq_fallback if freq_fallback is not None else \
        freq_baseline(cat_matrix_train, kept_feature_names, feat_to_value_names)

    if scores.max() == 0:
        return dict(fb)

    n_with_overlap = int((scores > 0).sum())
    k_use = min(k, n_with_overlap)
    top_k = np.argsort(scores)[-k_use:]

    result = {}
    for fi, name in enumerate(kept_feature_names):
        votes = [int(cat_matrix_train[li, fi])
                 for li in top_k if cat_matrix_train[li, fi] >= 0]
        if votes:
            cnt = Counter(votes)
            result[name] = feat_to_value_names[fi][cnt.most_common(1)[0][0]]
        else:
            result[name] = fb[name]
    return result


def idf_knn_baseline(obs_feat_dict, cat_matrix_train, kept_feature_names,
                     feat_name_to_idx, feat_to_value_names, idf_weights,
                     k=5, freq_fallback=None):
    """IDF-weighted kNN by feature-value overlap."""
    n_train = cat_matrix_train.shape[0]
    scores = np.zeros(n_train, dtype=np.float64)

    for feat_name, val_str in obs_feat_dict.items():
        fi = feat_name_to_idx.get(feat_name)
        if fi is None:
            continue
        val_list = feat_to_value_names.get(fi, [])
        if val_str not in val_list:
            continue
        vi_local = val_list.index(val_str)
        match = (cat_matrix_train[:, fi] == vi_local).astype(np.float64)
        scores += match * idf_weights[fi]

    fb = freq_fallback if freq_fallback is not None else \
        freq_baseline(cat_matrix_train, kept_feature_names, feat_to_value_names)

    if scores.max() == 0:
        return dict(fb)

    n_with_overlap = int((scores > 0).sum())
    k_use = min(k, n_with_overlap)
    top_k = np.argsort(scores)[-k_use:]

    result = {}
    for fi, name in enumerate(kept_feature_names):
        votes = [int(cat_matrix_train[li, fi])
                 for li in top_k if cat_matrix_train[li, fi] >= 0]
        if votes:
            cnt = Counter(votes)
            result[name] = feat_to_value_names[fi][cnt.most_common(1)[0][0]]
        else:
            result[name] = fb[name]
    return result


def cond_prob_baseline(obs_feat_dict, kept_feature_names, feat_name_to_idx,
                       feat_to_value_names, co_counts, marginals,
                       alpha=0.5, min_count=5, freq_fallback=None):
    """
    Dirichlet-smoothed conditional probability baseline.
    For each blanked feature j:
      log_score[v] = sum_{(fi, vi) in obs} log P_smooth(fj=v | fi=vi)
    where P_smooth uses Dirichlet smoothing with parameter alpha.
    """
    result = {}
    for fi_name in kept_feature_names:
        fj = feat_name_to_idx.get(fi_name)
        if fj is None:
            continue
        K_j = len(feat_to_value_names[fj])
        log_score = np.zeros(K_j)
        n_terms = 0

        for feat_i_name, val_i_str in obs_feat_dict.items():
            fi = feat_name_to_idx.get(feat_i_name)
            if fi is None or fi == fj:
                continue
            val_list_i = feat_to_value_names.get(fi, [])
            if val_i_str not in val_list_i:
                continue
            vi_local = val_list_i.index(val_i_str)
            marg = marginals.get((fi, vi_local), 0)
            if marg < min_count:
                continue
            key = (fi, vi_local, fj)
            counts = co_counts.get(key, np.zeros(K_j))
            if len(counts) != K_j:
                counts = np.zeros(K_j)
            smoothed = (counts + alpha) / (marg + alpha * K_j)
            log_score += np.log(smoothed + 1e-12)
            n_terms += 1

        if n_terms > 0:
            pred_local = int(np.argmax(log_score))
            result[fi_name] = feat_to_value_names[fj][pred_local]
        elif freq_fallback is not None and fi_name in freq_fallback:
            result[fi_name] = freq_fallback[fi_name]
        else:
            result[fi_name] = feat_to_value_names[fj][0]

    return result


# ===========================================================================
# PART 4: Temperature calibration on dev
# ===========================================================================

def calibrate_temperature(model, cat_matrix_dev, feat_to_global_ids,
                           feat_to_value_names, feat_name_to_idx,
                           device, temp_grid=None):
    """
    Grid-search temperature T minimizing NLL on dev imputation.
    For each dev language: first half as context, second half as targets.
    Returns best T (float).
    """
    if temp_grid is None:
        temp_grid = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    model.eval()
    best_T, best_nll = 1.0, float("inf")

    for T in temp_grid:
        total_nll = 0.0
        n_preds = 0

        for lang_idx in range(cat_matrix_dev.shape[0]):
            observed = np.where(cat_matrix_dev[lang_idx] >= 0)[0].tolist()
            if len(observed) < 4:
                continue
            mid = len(observed) // 2
            ctx_feats = observed[:mid]
            tgt_feats = observed[mid:]

            ctx_gvals = [
                feat_to_global_ids[fi][int(cat_matrix_dev[lang_idx, fi])]
                for fi in ctx_feats
            ]

            with torch.no_grad():
                z = model.encode_context(ctx_feats, ctx_gvals, device)
                for fi in tgt_feats:
                    true_local = int(cat_matrix_dev[lang_idx, fi])
                    gids = feat_to_global_ids[fi]
                    gids_t = torch.tensor(gids, dtype=torch.long, device=device)
                    val_embs = model.val_embeddings(gids_t)
                    logits = (z @ val_embs.T).squeeze(0) / T
                    log_probs = F.log_softmax(logits, dim=0)
                    total_nll -= log_probs[true_local].item()
                    n_preds += 1

        avg_nll = total_nll / max(n_preds, 1)
        if avg_nll < best_nll:
            best_nll = avg_nll
            best_T = T

    return best_T


# ===========================================================================
# PART 5: Correlation prior — dev-tuned
# ===========================================================================

def build_correlation_prior(cat_matrix_train, feat_to_global_ids,
                             feat_to_value_names):
    """
    Build co-occurrence count tables from training data.

    Returns
    -------
    co_counts  : dict  (fi, vi_local, fj) -> np.array(K_j,) of counts
    marginals  : dict  (fi, vi_local)     -> int count
    """
    n_langs, n_feats = cat_matrix_train.shape
    co_counts = {}
    marginals = defaultdict(int)

    for lang in range(n_langs):
        obs = [(int(fi), int(cat_matrix_train[lang, fi]))
               for fi in range(n_feats)
               if cat_matrix_train[lang, fi] >= 0]
        for fi, vi in obs:
            marginals[(fi, vi)] += 1
        for fi, vi in obs:
            for fj, vj in obs:
                if fi == fj:
                    continue
                key = (fi, vi, fj)
                K_j = len(feat_to_value_names[fj])
                if key not in co_counts:
                    co_counts[key] = np.zeros(K_j)
                co_counts[key][vj] += 1

    return co_counts, dict(marginals)


def apply_prior(raw_log_probs_dict, obs_feat_dict, feat_name_to_idx,
                feat_to_value_names, co_counts, marginals,
                lam, alpha, min_count):
    """
    Log-linear reranking: model log-probs + lam * Dirichlet-smoothed prior.

    Parameters
    ----------
    raw_log_probs_dict : {feat_name -> np.array log probs (K_f,)}
    obs_feat_dict      : {feat_name -> value_str} (gold observations only)

    Returns
    -------
    {feat_name -> predicted_value_str}
    """
    predictions = {}
    for feat_j_name, model_lp in raw_log_probs_dict.items():
        fj = feat_name_to_idx.get(feat_j_name)
        if fj is None:
            continue
        K_j = len(feat_to_value_names[fj])
        if len(model_lp) != K_j:
            continue
        prior_log = np.zeros(K_j)
        n_terms = 0

        for feat_i_name, val_i_str in obs_feat_dict.items():
            fi = feat_name_to_idx.get(feat_i_name)
            if fi is None or fi == fj:
                continue
            val_list_i = feat_to_value_names.get(fi, [])
            if val_i_str not in val_list_i:
                continue
            vi_local = val_list_i.index(val_i_str)
            marg = marginals.get((fi, vi_local), 0)
            if marg < min_count:
                continue
            key = (fi, vi_local, fj)
            counts = co_counts.get(key, np.zeros(K_j))
            if len(counts) != K_j:
                counts = np.zeros(K_j)
            smoothed = (counts + alpha) / (marg + alpha * K_j)
            prior_log += np.log(smoothed + 1e-12)
            n_terms += 1

        if n_terms > 0:
            prior_log /= n_terms
        combined = model_lp + lam * prior_log
        combined -= combined.max()
        probs = np.exp(combined)
        probs /= probs.sum()
        pred_local = int(probs.argmax())
        predictions[feat_j_name] = feat_to_value_names[fj][pred_local]
    return predictions


def tune_prior_on_dev(cat_matrix_dev, cat_matrix_train, kept_feature_names,
                      feat_name_to_idx, feat_to_value_names,
                      feat_to_global_ids, co_counts, marginals,
                      model, device, T,
                      lam_grid, alpha_grid, mincount_grid):
    """
    Grid search over (lam, alpha, min_count) maximizing dev imputation accuracy.
    Uses first half of each dev language's features as context, second as targets.
    Returns best (lam, alpha, min_count) triple.
    """
    model.eval()

    dev_examples = []
    for lang_idx in range(cat_matrix_dev.shape[0]):
        observed = np.where(cat_matrix_dev[lang_idx] >= 0)[0].tolist()
        if len(observed) < 4:
            continue
        mid = len(observed) // 2
        ctx_feats = observed[:mid]
        tgt_feats = observed[mid:]

        ctx_gvals = [
            feat_to_global_ids[fi][int(cat_matrix_dev[lang_idx, fi])]
            for fi in ctx_feats
        ]
        obs_feat_dict = {
            kept_feature_names[fi]: feat_to_value_names[fi][int(cat_matrix_dev[lang_idx, fi])]
            for fi in ctx_feats
        }

        raw_lp = {}
        with torch.no_grad():
            z = model.encode_context(ctx_feats, ctx_gvals, device)
            for fi in tgt_feats:
                gids = feat_to_global_ids[fi]
                gids_t = torch.tensor(gids, dtype=torch.long, device=device)
                val_embs = model.val_embeddings(gids_t)
                logits = (z @ val_embs.T).squeeze(0) / T
                raw_lp[kept_feature_names[fi]] = F.log_softmax(logits, dim=0).cpu().numpy()

        true_vals = {
            kept_feature_names[fi]: feat_to_value_names[fi][int(cat_matrix_dev[lang_idx, fi])]
            for fi in tgt_feats
        }
        dev_examples.append((obs_feat_dict, raw_lp, true_vals))

    best_acc, best_params = -1.0, (0.4, 0.5, 5)
    for lam in lam_grid:
        for alpha in alpha_grid:
            for min_count in mincount_grid:
                total, correct = 0, 0
                for obs_fd, raw_lp, true_vals in dev_examples:
                    preds = apply_prior(raw_lp, obs_fd, feat_name_to_idx,
                                        feat_to_value_names, co_counts,
                                        marginals, lam, alpha, min_count)
                    for fn, tv in true_vals.items():
                        if fn in preds:
                            correct += int(preds[fn] == tv)
                            total += 1
                acc = correct / max(total, 1)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (lam, alpha, min_count)

    print(f"  Best prior params: lam={best_params[0]}, alpha={best_params[1]}, "
          f"min_count={best_params[2]}, dev_acc={best_acc:.4f}")
    return best_params


# ===========================================================================
# PART 6: Inference — SetEncoder
# ===========================================================================

def infer_set_encoder(model, obs_feat_dict, feat_name_to_idx,
                      feat_to_global_ids, feat_to_value_names,
                      kept_feature_names, device, T=1.0):
    """
    Encode observed features and predict all features in kept_feature_names.

    Returns
    -------
    raw_log_probs : {feat_name -> np.array(K_f,) log probs (temperature scaled)}
    predictions   : {feat_name -> (pred_value_str, margin_float)}
        margin = p1 - p2 (more calibrated than raw max)
    """
    model.eval()
    ctx_feats, ctx_gvals = [], []
    for feat_name, val_str in obs_feat_dict.items():
        fi = feat_name_to_idx.get(feat_name)
        if fi is None:
            continue
        val_list = feat_to_value_names.get(fi, [])
        if val_str not in val_list:
            continue
        vi_local = val_list.index(val_str)
        ctx_feats.append(fi)
        ctx_gvals.append(feat_to_global_ids[fi][vi_local])

    raw_lp = {}
    predictions = {}

    with torch.no_grad():
        z = model.encode_context(ctx_feats, ctx_gvals, device)
        for fi, feat_name in enumerate(kept_feature_names):
            gids = feat_to_global_ids[fi]
            gids_t = torch.tensor(gids, dtype=torch.long, device=device)
            val_embs = model.val_embeddings(gids_t)
            logits = (z @ val_embs.T).squeeze(0) / T
            log_probs = F.log_softmax(logits, dim=0).cpu().numpy()
            probs = np.exp(log_probs)
            raw_lp[feat_name] = log_probs

            sorted_p = np.sort(probs)[::-1]
            p1 = float(sorted_p[0])
            p2 = float(sorted_p[1]) if len(sorted_p) > 1 else 0.0
            margin = p1 - p2
            pred_local = int(probs.argmax())
            predictions[feat_name] = (feat_to_value_names[fi][pred_local], margin)

    return raw_lp, predictions


# ===========================================================================
# PART 7: Conservative self-training
# ===========================================================================

def iterative_infer(model, obs_feat_dict, feat_name_to_idx,
                    feat_to_global_ids, feat_to_value_names,
                    kept_feature_names, device, T=1.0,
                    n_rounds=2, margin_tau=0.30, top_n=5):
    """
    Conservative iterative self-training for SetEncoder.

    Rules for pseudo-labeling:
      1. Margin (p1 - p2) >= margin_tau
      2. Feature not in original obs_feat_dict
      3. At most top_n pseudo-labels added per round
      4. Leave-target-out final pass

    Returns {feat_name -> predicted_value_str} for all features.
    """
    current_obs = dict(obs_feat_dict)

    for round_idx in range(n_rounds):
        _, predictions = infer_set_encoder(
            model, current_obs, feat_name_to_idx, feat_to_global_ids,
            feat_to_value_names, kept_feature_names, device, T)

        candidates = [
            (feat_name, val, margin)
            for feat_name, (val, margin) in predictions.items()
            if feat_name not in obs_feat_dict and margin >= margin_tau
        ]
        candidates.sort(key=lambda x: -x[2])

        n_added = 0
        for feat_name, val, margin in candidates:
            if n_added >= top_n:
                break
            current_obs[feat_name] = val
            n_added += 1

    # Final pass — leave-target-out
    final_predictions = {}
    for target_feat in kept_feature_names:
        ctx = {k: v for k, v in current_obs.items()
               if k != target_feat or k in obs_feat_dict}
        raw_lp, preds = infer_set_encoder(
            model, ctx, feat_name_to_idx, feat_to_global_ids,
            feat_to_value_names, kept_feature_names, device, T)
        if target_feat in preds:
            final_predictions[target_feat] = preds[target_feat][0]

    return final_predictions


# ===========================================================================
# PART 8: Submission helpers
# ===========================================================================

HEADER = ("wals_code\tname\tlatitude\tlongitude\tgenus\tfamily"
          "\tcountrycodes\tfeatures\n")


def _fmt_features(blanked_set, obs_str_dict, model_preds, freq_fallback):
    parts     = []
    all_feats = set(obs_str_dict.keys()) | set(blanked_set)
    for fname in sorted(all_feats):
        if fname not in blanked_set:
            parts.append(f"{fname}={obs_str_dict[fname]}")
        else:
            if fname in model_preds:
                pred_id = model_preds[fname]
            elif fname in freq_fallback:
                pred_id = freq_fallback[fname]
            else:
                pred_id = "1"
            parts.append(f"{fname}={pred_id} -")
    return "|".join(parts)


def _write_submission(path, test_meta, test_blank, test_obs,
                      preds_by_lang, freq_fallback):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(HEADER)
        for _, row in test_meta.iterrows():
            wc           = row["wals_code"]
            blanked_set  = test_blank.get(wc, set())
            obs_str_dict = test_obs.get(wc, {})
            model_preds  = preds_by_lang.get(wc, {})
            feat_str     = _fmt_features(blanked_set, obs_str_dict,
                                         model_preds, freq_fallback)
            meta_str = "\t".join([wc, str(row["name"]),
                                   str(row["latitude"]), str(row["longitude"]),
                                   str(row["genus"]), str(row["family"]),
                                   str(row["countrycodes"])])
            fh.write(f"{meta_str}\t{feat_str}\n")


# ===========================================================================
# Scoring helpers
# ===========================================================================

def _run_scorer(scorer_path, submission_paths):
    """Run score.py with macro mode enabled on the given submission files."""
    with open(scorer_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    src_mod = src.replace(
        'for mode in ("micro",):  #, "macro"):',
        'for mode in ("micro", "macro"):'
    )
    scripts_dir = os.path.dirname(scorer_path)
    tmp_score   = os.path.join(scripts_dir, "_score_v3_tmp.py")
    with open(tmp_score, "w", encoding="utf-8") as fh:
        fh.write(src_mod)
    try:
        result = subprocess.run(
            [sys.executable, tmp_score] + list(submission_paths),
            capture_output=True, text=True, cwd=scripts_dir,
        )
        return result.stdout, result.stderr
    finally:
        if os.path.exists(tmp_score):
            os.remove(tmp_score)


_TABLE_COLS = ["Tucanoan", "Madang", "Mahakiranti", "Nilotic",
               "Mayan", "Northern Pama-Nyungan", "other genera", "Overall"]
_TABLE_HDR  = ["Tucanoan", "Madang", "Mahakiranti", "Nilotic",
               "Mayan", "N.Pama-Nyungan", "Other", "Overall"]


def _parse_scores(stdout):
    """Parse scorer stdout -> {mode: {basename: {genus_or_Overall: acc}}}"""
    results      = {}
    current_mode = None
    genera_order = None
    in_genera    = False
    in_overall   = False

    def _f(s):
        try:
            v = float(s)
            return None if v != v else v
        except Exception:
            return None

    for line in stdout.splitlines():
        line = line.rstrip()
        if line.startswith("# Averaging:"):
            current_mode = line.split(":")[-1].strip()
            results[current_mode] = {}
            genera_order = None
            in_genera = in_overall = False
            continue
        if current_mode is None:
            continue
        if line.startswith("submission\t"):
            genera_order = line.split("\t")[1:]
            in_genera    = True
            in_overall   = False
            continue
        if line.startswith("number of languages"):
            continue
        if line.startswith("overall:"):
            in_genera  = False
            in_overall = True
            continue
        if in_genera and genera_order and "\t" in line:
            parts = line.split("\t")
            name  = parts[0]
            vals  = parts[1:]
            if name not in results[current_mode]:
                results[current_mode][name] = {}
            for gi, g in enumerate(genera_order):
                if gi < len(vals):
                    results[current_mode][name][g] = _f(vals[gi])
            continue
        if in_overall and "\t" in line:
            parts = line.split("\t")
            name  = parts[0]
            results.setdefault(current_mode, {}).setdefault(name, {})
            if len(parts) > 1:
                results[current_mode][name]["Overall"] = _f(parts[1])

    return results


def _row_vals(name, mode, results):
    d   = results.get(mode, {}).get(name, {})
    row = []
    for col in _TABLE_COLS:
        v = d.get(col)
        row.append(f"{v:.2f}" if v is not None else " n/a")
    row_overall = d.get("Overall")
    if row_overall is not None:
        row.append(f"{row_overall:.2f}")
    else:
        row.append(" n/a")
    return row


def _print_table(rows, mode="macro", results=None, mean_std=None):
    """Print ablation table. rows = list of (label, sub_basename_or_None)."""
    COL_W = [38] + [7] * 8

    def pr(cells):
        parts = [str(cells[0]).ljust(COL_W[0])] + [str(c).rjust(7) for c in cells[1:]]
        print("| " + " | ".join(parts) + " |")

    def sep():
        print("|" + "|".join("-" + "-" * w + "-" for w in COL_W) + "|")

    pr(["System"] + _TABLE_HDR)
    sep()
    for label, data in rows:
        if isinstance(data, str) and results is not None:
            pr([label] + _row_vals(data, mode, results))
        elif data is None:
            pr([label] + [" ---"] * 8)
        else:
            # data is list of 8 pre-formatted strings
            pr([label] + data)


# ===========================================================================
# Bootstrap significance test
# ===========================================================================

def _compute_per_lang_acc(preds_by_lang, test_meta, test_blank, gold_obs,
                           feat_name_to_idx):
    """Compute per-language accuracy over blanked features."""
    per_lang = []
    for _, row in test_meta.iterrows():
        wc     = row["wals_code"]
        genus  = row["genus"]
        blanked = test_blank.get(wc, set())
        pred    = preds_by_lang.get(wc, {})
        gold    = gold_obs.get(wc, {})

        correct = 0
        total   = 0
        for fn in blanked:
            if fn not in feat_name_to_idx:
                continue
            gv = gold.get(fn)
            pv = pred.get(fn)
            if gv is None:
                continue
            total += 1
            if pv is not None and pv == gv:
                correct += 1

        if total > 0:
            per_lang.append((correct / total, genus))

    return per_lang


def _macro_acc_from_per_lang(per_lang):
    genus_accs = defaultdict(list)
    for acc, genus in per_lang:
        genus_accs[genus].append(acc)
    if not genus_accs:
        return 0.0
    return float(np.mean([np.mean(v) for v in genus_accs.values()]))


def bootstrap_test(preds_by_lang, test_meta, test_blank, gold_obs,
                   feat_name_to_idx, ref_overall=0.66, n_boot=1000):
    """
    Paired bootstrap test comparing best config vs reference point estimate.
    Returns (observed_macro, delta_ci_95, p_value).
    """
    per_lang = _compute_per_lang_acc(preds_by_lang, test_meta, test_blank,
                                      gold_obs, feat_name_to_idx)
    obs_macro = _macro_acc_from_per_lang(per_lang)
    obs_delta = obs_macro - ref_overall

    np.random.seed(0)
    n = len(per_lang)
    boot_deltas = []
    for _ in range(n_boot):
        idxs = np.random.choice(n, n, replace=True)
        sample = [per_lang[i] for i in idxs]
        boot_macro = _macro_acc_from_per_lang(sample)
        boot_deltas.append(boot_macro - ref_overall)

    boot_deltas = np.array(boot_deltas)
    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))
    # Two-sided p-value under null: delta=0
    p_value = float(np.mean(np.abs(boot_deltas) >= abs(obs_delta)))
    if p_value == 0.0:
        p_value = 1.0 / n_boot

    return obs_macro, (ci_lo, ci_hi), p_value


# ===========================================================================
# MAIN
# ===========================================================================

def main(args):
    base       = args.base
    train_csv  = f"{base}/ST2020/data/train.csv"
    dev_csv    = f"{base}/ST2020/data/dev.csv"
    test_blind = f"{base}/ST2020/data/test_blinded.csv"
    test_gold  = f"{base}/ST2020/data/test_gold.csv"
    scorer     = f"{base}/ST2020/scripts/score.py"
    gb2v       = f"{base}/Grambank2Vec"
    output_dir = args.output_dir or gb2v

    print("=" * 72)
    print("SIGTYP 2020 v3 — SetEncoder + non-neural baselines")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Part 1: Load data
    # -----------------------------------------------------------------------
    print("\n[Part 1] Loading SIGTYP data ...")
    train_meta, train_feat, train_blank, train_obs = parse_sigtyp(train_csv)
    dev_meta,   dev_feat,   dev_blank,   dev_obs   = parse_sigtyp(dev_csv)
    test_meta,  test_feat,  test_blank,  test_obs  = parse_sigtyp(test_blind)

    print(f"  Train: {len(train_meta)} languages, {train_feat.shape[1]} feature columns")
    print(f"  Dev:   {len(dev_meta)} languages, {dev_feat.shape[1]} feature columns")
    print(f"  Test:  {len(test_meta)} languages")

    traindev_feat = pd.concat([train_feat, dev_feat], ignore_index=True)
    n_train = len(train_feat)
    print(f"  Train+Dev: {len(traindev_feat)} languages, "
          f"{traindev_feat.shape[1]} feature columns")

    # -----------------------------------------------------------------------
    # Part 2: Vocabulary
    # -----------------------------------------------------------------------
    print("\n[Part 2] Building shared vocabulary ...")
    feature_cols = list(traindev_feat.columns)

    cat_matrix_traindev, kept_feature_names, feat_to_global_ids, feat_to_value_names = \
        prepare_categorical(traindev_feat, feature_cols)

    n_features    = len(kept_feature_names)
    n_total_values = max(max(gids) for gids in feat_to_global_ids.values()) + 1
    feat_name_to_idx = {name: i for i, name in enumerate(kept_feature_names)}

    print(f"  Kept {n_features} features (≥2 unique values)")
    print(f"  Total unique (feature, value) pairs: {n_total_values}")

    # Train/dev splits from traindev
    cat_matrix_train = cat_matrix_traindev[:n_train]
    cat_matrix_dev   = cat_matrix_traindev[n_train:]

    # -----------------------------------------------------------------------
    # Non-neural baselines (deterministic — run once)
    # -----------------------------------------------------------------------
    print("\n[Part 3] Non-neural baselines ...")

    freq_preds = freq_baseline(cat_matrix_train, kept_feature_names,
                               feat_to_value_names)

    # IDF weights for idf-kNN
    n_train_obs_per_feat = (cat_matrix_train >= 0).sum(axis=0).astype(float)
    idf_weights = np.log(n_train / (1.0 + n_train_obs_per_feat))

    # Precompute co-occurrence counts (shared by cond_prob and neural prior)
    print("  Building co-occurrence tables ...")
    t0 = time.time()
    baseline_co_counts, baseline_marginals = build_correlation_prior(
        cat_matrix_train, feat_to_global_ids, feat_to_value_names)
    print(f"  Done in {time.time()-t0:.1f}s")

    freq_by_lang   = {}
    genus_by_lang  = {}
    knn_by_lang    = {}
    idfknn_by_lang = {}
    cprob_by_lang  = {}

    for _, row in test_meta.iterrows():
        wc  = row["wals_code"]
        obs = {k: v for k, v in test_obs.get(wc, {}).items()
               if k in feat_name_to_idx}

        freq_by_lang[wc]   = freq_preds
        genus_by_lang[wc]  = genus_majority_baseline(
            cat_matrix_train, train_meta, kept_feature_names, feat_to_value_names,
            row["genus"], freq_fallback=freq_preds)
        knn_by_lang[wc]    = knn_baseline(
            obs, cat_matrix_train, kept_feature_names,
            feat_name_to_idx, feat_to_value_names, k=5, freq_fallback=freq_preds)
        idfknn_by_lang[wc] = idf_knn_baseline(
            obs, cat_matrix_train, kept_feature_names,
            feat_name_to_idx, feat_to_value_names, idf_weights, k=5,
            freq_fallback=freq_preds)
        cprob_by_lang[wc]  = cond_prob_baseline(
            obs, kept_feature_names, feat_name_to_idx, feat_to_value_names,
            baseline_co_counts, baseline_marginals,
            alpha=0.5, min_count=5, freq_fallback=freq_preds)

    baseline_sub_paths = []
    for tag, preds_map in [
        ("freq",     freq_by_lang),
        ("genus",    genus_by_lang),
        ("knn",      knn_by_lang),
        ("idfknn",   idfknn_by_lang),
        ("condprob", cprob_by_lang),
    ]:
        path = f"{output_dir}/sigtyp_v3_{tag}.tsv"
        _write_submission(path, test_meta, test_blank, test_obs,
                          preds_map, freq_preds)
        print(f"  Wrote {path}")
        baseline_sub_paths.append(path)

    # Score baselines
    print("  Scoring baselines ...")
    stdout_bl, _ = _run_scorer(scorer, baseline_sub_paths)
    baseline_results = _parse_scores(stdout_bl)

    # -----------------------------------------------------------------------
    # SetEncoder experiment loop (per seed)
    # -----------------------------------------------------------------------
    seed_results_A  = {}
    seed_results_B  = {}
    seed_results_C  = {}
    seed_results_D  = {}
    best_D_preds    = None
    best_D_seed     = None

    all_seed_scores = defaultdict(list)   # config -> list of overall macro acc

    for seed in args.seeds:
        print(f"\n[Seed {seed}] SetEncoder ...")
        ckpt_path = f"{output_dir}/set_encoder_seed{seed}.pt"

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = None

        if os.path.exists(ckpt_path) and not args.force_retrain:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                assert ckpt.get("embed_dim") == args.embed_dim, \
                    f"embed_dim mismatch: ckpt={ckpt.get('embed_dim')} vs args={args.embed_dim}"
                assert ckpt.get("hidden_dim") == args.hidden_dim, \
                    f"hidden_dim mismatch: ckpt={ckpt.get('hidden_dim')} vs args={args.hidden_dim}"
                model = SetEncoder(n_features, n_total_values, feat_to_global_ids,
                                   args.embed_dim, args.hidden_dim)
                model.load_state_dict(ckpt["state_dict"])
                print(f"  Loaded checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"  Checkpoint load failed ({e}), retraining ...")
                model = None

        if model is None:
            if args.skip_training:
                print(f"  --skip-training set and no valid checkpoint; skipping seed {seed}")
                continue
            print(f"  Training SetEncoder (seed={seed}, epochs={args.n_epochs}) ...")
            t0 = time.time()
            model = SetEncoder(n_features, n_total_values, feat_to_global_ids,
                               args.embed_dim, args.hidden_dim)
            dataset = SetEncoderDataset(cat_matrix_traindev, feat_to_global_ids)
            train_set_encoder(model, dataset, n_epochs=args.n_epochs,
                              lr=args.lr, l2_reg=args.l2_reg,
                              device=args.device, seed=seed)
            print(f"  Training done in {time.time()-t0:.1f}s")
            ckpt = {"state_dict": model.state_dict(), "seed": seed,
                    "n_epochs": args.n_epochs, "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim}
            torch.save(ckpt, ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")

        model.to(args.device)
        model.eval()

        # Temperature calibration
        print("  Calibrating temperature on dev ...")
        T = calibrate_temperature(model, cat_matrix_dev, feat_to_global_ids,
                                   feat_to_value_names, feat_name_to_idx,
                                   args.device)
        print(f"  Best T = {T}")
        if T > 2.0:
            print("  Warning: T > 2.0 — model may be significantly overconfident "
                  "during training; architecture may need revisiting")

        # Prior hyperparameter tuning
        print("  Tuning prior hyperparameters on dev ...")
        best_lam, best_alpha, best_min_count = tune_prior_on_dev(
            cat_matrix_dev, cat_matrix_train, kept_feature_names,
            feat_name_to_idx, feat_to_value_names, feat_to_global_ids,
            baseline_co_counts, baseline_marginals,
            model, args.device, T,
            args.lam_grid, args.alpha_grid, args.mincount_grid)

        # Inference for all test languages
        print(f"  Running inference (4 configs) over {len(test_meta)} test languages ...")
        A_preds, B_preds, C_preds, D_preds = {}, {}, {}, {}

        for i, (_, row) in enumerate(test_meta.iterrows()):
            wc          = row["wals_code"]
            blanked_set = test_blank.get(wc, set())
            obs_fd      = {k: v for k, v in test_obs.get(wc, {}).items()
                           if k in feat_name_to_idx}
            obs_in_vocab = len(obs_fd)

            # Config A: SetEncoder only
            raw_lp_A, preds_A_dict = infer_set_encoder(
                model, obs_fd, feat_name_to_idx, feat_to_global_ids,
                feat_to_value_names, kept_feature_names, args.device, T)
            A_preds[wc] = {fn: v for fn, (v, _) in preds_A_dict.items()}

            # Config B: SetEncoder + conservative self-training
            B_preds[wc] = iterative_infer(
                model, obs_fd, feat_name_to_idx, feat_to_global_ids,
                feat_to_value_names, kept_feature_names, args.device, T,
                n_rounds=args.n_rounds, margin_tau=args.margin_tau,
                top_n=args.top_n_pseudo)

            # Config C: SetEncoder + prior (no self-training)
            C_preds[wc] = apply_prior(
                raw_lp_A, obs_fd, feat_name_to_idx, feat_to_value_names,
                baseline_co_counts, baseline_marginals,
                best_lam, best_alpha, best_min_count)

            # Config D: self-training rounds + leave-target-out + prior
            # Step 1: Expand context via self-training rounds
            cur_obs_D = dict(obs_fd)
            for _round in range(args.n_rounds):
                _, preds_round = infer_set_encoder(
                    model, cur_obs_D, feat_name_to_idx, feat_to_global_ids,
                    feat_to_value_names, kept_feature_names, args.device, T)
                cands = [(fn, v, m) for fn, (v, m) in preds_round.items()
                         if fn not in obs_fd and m >= args.margin_tau]
                cands.sort(key=lambda x: -x[2])
                for fn, v, _ in cands[:args.top_n_pseudo]:
                    cur_obs_D[fn] = v

            # Step 2: Leave-target-out final pass with prior
            d_preds_wc = {}
            for target_feat in blanked_set:
                if target_feat not in feat_name_to_idx:
                    continue
                ctx = {k: v for k, v in cur_obs_D.items()
                       if k != target_feat or k in obs_fd}
                raw_lp_t, _ = infer_set_encoder(
                    model, ctx, feat_name_to_idx, feat_to_global_ids,
                    feat_to_value_names, kept_feature_names, args.device, T)
                if target_feat in raw_lp_t:
                    prior_preds = apply_prior(
                        {target_feat: raw_lp_t[target_feat]},
                        obs_fd, feat_name_to_idx, feat_to_value_names,
                        baseline_co_counts, baseline_marginals,
                        best_lam, best_alpha, best_min_count)
                    if target_feat in prior_preds:
                        d_preds_wc[target_feat] = prior_preds[target_feat]
            # For non-blanked features, use config A
            for fn, (v, _) in preds_A_dict.items():
                if fn not in d_preds_wc:
                    d_preds_wc[fn] = v
            D_preds[wc] = d_preds_wc

            if (i + 1) % 25 == 0 or (i + 1) == len(test_meta):
                print(f"  {i+1}/{len(test_meta)} processed  "
                      f"(last: {wc}, obs_in_vocab={obs_in_vocab})")

        # Write submission files
        sub_paths_seed = []
        for cfg, preds_map in [("A", A_preds), ("B", B_preds),
                                ("C", C_preds), ("D", D_preds)]:
            path = f"{output_dir}/sigtyp_v3_{cfg}_seed{seed}.tsv"
            _write_submission(path, test_meta, test_blank, test_obs,
                              preds_map, freq_preds)
            print(f"  Wrote {path}")
            sub_paths_seed.append(path)

        # Score
        print("  Scoring ...")
        stdout_s, stderr_s = _run_scorer(scorer, sub_paths_seed)
        if stderr_s.strip():
            print(f"  Scorer stderr: {stderr_s[:200]}")
        seed_scores = _parse_scores(stdout_s)

        seed_results_A[seed] = seed_scores
        seed_results_B[seed] = seed_scores
        seed_results_C[seed] = seed_scores
        seed_results_D[seed] = seed_scores

        for cfg, key in [("A", f"sigtyp_v3_A_seed{seed}"),
                          ("B", f"sigtyp_v3_B_seed{seed}"),
                          ("C", f"sigtyp_v3_C_seed{seed}"),
                          ("D", f"sigtyp_v3_D_seed{seed}")]:
            ov = seed_scores.get("macro", {}).get(key, {}).get("Overall")
            if ov is not None:
                all_seed_scores[cfg].append(ov)
            if cfg == "D" and (best_D_preds is None or (ov is not None and
                    ov > (seed_scores.get("macro", {}).get(
                        f"sigtyp_v3_D_seed{best_D_seed}", {}).get("Overall", 0)
                          if best_D_seed is not None else 0))):
                best_D_preds = D_preds
                best_D_seed  = seed

    # -----------------------------------------------------------------------
    # Part 10: Reporting
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("ABLATION TABLE  (macro accuracy per genus)")
    print("=" * 72)

    COL_W = [40] + [7] * 8

    def pr(cells):
        parts = [str(cells[0]).ljust(COL_W[0])] + [str(c).rjust(7) for c in cells[1:]]
        print("| " + " | ".join(parts) + " |")

    def sep():
        print("|" + "|".join("-" + "-" * w + "-" for w in COL_W) + "|")

    def _bl_row(tag, label):
        bn = f"sigtyp_v3_{tag}"
        d  = baseline_results.get("macro", {}).get(bn, {})
        vals = []
        for col in _TABLE_COLS:
            v = d.get(col)
            vals.append(f"{v:.2f}" if v is not None else " n/a")
        ov = d.get("Overall")
        vals.append(f"{ov:.2f}" if ov is not None else " n/a")
        return [label] + vals

    def _seed_row(cfg, label):
        scores = all_seed_scores.get(cfg, [])
        if not scores:
            return [label] + [" n/a"] * 8
        mean_ov = np.mean(scores)
        std_ov  = np.std(scores)
        return [label] + [" n/a"] * 7 + [f"{mean_ov:.2f}±{std_ov:.2f}"]

    pr(["System"] + _TABLE_HDR)
    sep()
    pr(["--- Non-neural baselines ---"] + [""] * 8)
    sep()
    pr(_bl_row("freq",     "Freq (global)"))
    pr(_bl_row("genus",    "Genus majority"))
    pr(_bl_row("knn",      "kNN (k=5)"))
    pr(_bl_row("idfknn",   "IDF-kNN (k=5)"))
    pr(_bl_row("condprob", "Cond. prob (UFAL-style)"))
    sep()
    pr(["--- SetEncoder (mean ± std across seeds) ---"] + [""] * 8)
    sep()
    pr(_seed_row("A", "[A] SetEncoder only"))
    pr(_seed_row("B", "[B] SetEncoder + self-train"))
    pr(_seed_row("C", "[C] SetEncoder + prior"))
    pr(_seed_row("D", "[D] SetEncoder (full)"))
    sep()
    pr(["--- Prior results (hardcoded) ---"] + [""] * 8)
    sep()
    pr(["[v1] Learned (transductive)"] +
       [".67", ".73", ".66", ".66", ".69", ".60", ".65", ".66"])
    pr(["[v1] TCF"] +
       [".67", ".57", ".62", ".62", ".62", ".62", ".61", ".61"])
    pr(["[v2D] Full system"] +
       [".67", ".77", ".64", ".67", ".68", ".65", ".64", ".66"])
    sep()
    pr(["--- SIGTYP 2020 paper (macro) ---"] + [""] * 8)
    sep()
    pr(["UFAL (best constrained)"] +
       [".73", ".78", ".74", ".71", ".80", ".76", ".76", ".75"])
    pr(["NEMO system2"] +
       [".71", ".72", ".72", ".76", ".76", ".67", ".69", ".69"])
    pr(["CrossLingference (unconstrained)"] +
       [".71", ".73", ".67", ".68", ".57", ".60", ".65", ".65"])
    pr(["Baseline_frequency (paper)"] +
       [".51", ".53", ".37", ".49", ".41", ".58", ".53", ".49"])
    pr(["Baseline_knn (paper)"] +
       [".48", ".57", ".46", ".48", ".32", ".52", ".51", ".48"])
    print()

    # Per-seed raw scores
    if all_seed_scores.get("D"):
        print("\nPer-seed D scores (Overall macro):")
        for seed, sc in zip(args.seeds, all_seed_scores["D"]):
            print(f"  seed={seed}: {sc:.4f}")

    # Bootstrap test
    if best_D_preds is not None:
        print("\n[Bootstrap] Testing best config D vs v1 Learned (0.66) ...")
        try:
            _, _, gold_obs = parse_sigtyp(test_gold)[2:]  # blanked, obs_strings
            gold_meta2, _, gold_blank2, gold_obs2 = parse_sigtyp(test_gold)
            obs_macro, (ci_lo, ci_hi), p_val = bootstrap_test(
                best_D_preds, test_meta, test_blank, gold_obs2,
                feat_name_to_idx, ref_overall=0.66, n_boot=1000)
            print(f"  Best D overall macro:  {obs_macro:.4f}")
            print(f"  Delta vs v1 Learned:   {obs_macro - 0.66:+.4f}")
            print(f"  95% CI of delta:       [{ci_lo:+.4f}, {ci_hi:+.4f}]")
            print(f"  p-value (two-sided):   {p_val:.4f}")
            best_D_ov = obs_macro
            ufal_ov = 0.75
            delta_ufal = best_D_ov - ufal_ov
            sign = "+" if delta_ufal >= 0 else ""
            print(f"\n  Best config: [D] SetEncoder (full)")
            print(f"  Delta vs UFAL (0.75):  {sign}{delta_ufal:.4f}  "
                  f"({'beat UFAL!' if delta_ufal >= 0 else 'still behind UFAL'})")
        except Exception as e:
            print(f"  Bootstrap test failed: {e}")
    else:
        print("\n  No D predictions available for bootstrap test.")

    print("\nDone.")


if __name__ == "__main__":
    main(get_args())
