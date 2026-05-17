#!/usr/bin/env python3
"""
SIGTYP 2020 v4: CrossAttentionSetEncoder + PMI/MI expert + embedding-aware
correlation smoothing.
Run: python3 sigtyp_eval_v4.py 2>&1 | tee sigtyp_eval_v4_log.txt
"""

import argparse
import os
import pickle
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
from model_set import SetEncoderDataset
from model_set_v4 import CrossAttentionSetEncoder, train_cross_attention

# ===========================================================================
# Verbatim helpers from sigtyp_eval_v3.py
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
    meta_df    : DataFrame [wals_code, name, latitude, longitude, genus, family, countrycodes]
    feat_df    : DataFrame rows=languages, cols=lowercase feature names; cells = value str or NaN
    blanked    : dict  wals_code -> set of lowercase feature names with value "?"
    obs_strings: dict  wals_code -> {lowercase feat_name: "id label" string}
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

        wals_code      = parts[0]
        name           = parts[1]
        try:
            latitude   = float(parts[2])
            longitude  = float(parts[3])
        except ValueError:
            latitude, longitude = float("nan"), float("nan")
        genus          = parts[4]
        family         = parts[5]
        countrycodes   = parts[6]
        features_field = "\t".join(parts[7:])

        meta_rows.append({"wals_code": wals_code, "name": name,
                          "latitude": latitude, "longitude": longitude,
                          "genus": genus, "family": family,
                          "countrycodes": countrycodes})

        feat_dict = {}
        blank_set = set()
        obs_str_d = {}

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
                feat_dict[fname_lc]  = tokens[0]
                obs_str_d[fname_lc]  = val_str

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


def _strip_to_id(full_val_str):
    """'5 SOV' -> '5'.  Already-stripped strings pass through."""
    return full_val_str.split()[0] if full_val_str.strip() else full_val_str


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


def _run_scorer(scorer_path, submission_paths):
    with open(scorer_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    src_mod = src.replace(
        'for mode in ("micro",):  #, "macro"):',
        'for mode in ("micro", "macro"):'
    )
    scripts_dir = os.path.dirname(scorer_path)
    tmp_score   = os.path.join(scripts_dir, "_score_v4_tmp.py")
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
_TABLE_HDR  = ["Tuc", "Mad", "Mah", "Nil", "May", "NPY", "Other", "Overall"]


def _parse_scores(stdout):
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


# ===========================================================================
# CLI
# ===========================================================================

def get_args():
    p = argparse.ArgumentParser(description="SIGTYP 2020 v4: CrossAttentionSetEncoder")
    p.add_argument("--base",          default="/home/user")
    p.add_argument("--seeds",         nargs="+", type=int,
                   default=[13, 21, 42, 87, 100])
    p.add_argument("--embed-dim",     type=int,   default=64)
    p.add_argument("--hidden-dim",    type=int,   default=128)
    p.add_argument("--n-epochs",      type=int,   default=200)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--l2-reg",        type=float, default=0.1)
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--output-dir",    default=None)
    p.add_argument("--device",        default=("cuda" if torch.cuda.is_available()
                                               else "cpu"))
    return p.parse_args()


# ===========================================================================
# PART 2: PMI statistics
# ===========================================================================

def build_pmi_stats(cat_matrix_train, feat_to_value_names, min_count=3):
    """
    Precompute from training data:
      marginals[fi]               : np.array(K_fi,) probabilities (Laplace smoothed)
      raw_cocounts[(fi,vi,fj)]    : np.array(K_fj,) co-occurrence counts
      marginal_counts[fi]         : np.array(K_fi,) raw counts
      mi_matrix                   : np.ndarray (n_feats, n_feats) mutual information
      idf_weights                 : np.ndarray (n_feats,)
    """
    n_langs, n_feats = cat_matrix_train.shape
    print(f"  Building PMI stats from {n_langs} langs × {n_feats} feats ...")

    # Marginal counts
    marginal_counts = {}
    for fi in range(n_feats):
        col   = cat_matrix_train[:, fi]
        K_fi  = len(feat_to_value_names[fi])
        counts = np.zeros(K_fi, dtype=float)
        for vi in range(K_fi):
            counts[vi] = float((col == vi).sum())
        marginal_counts[fi] = counts

    # Marginal probabilities (Laplace smoothed)
    marginals = {}
    for fi in range(n_feats):
        cnt = marginal_counts[fi] + 0.5
        marginals[fi] = cnt / cnt.sum()

    # Co-occurrence counts
    raw_cocounts = {}
    for lang in range(n_langs):
        obs = [(int(fi), int(cat_matrix_train[lang, fi]))
               for fi in range(n_feats)
               if cat_matrix_train[lang, fi] >= 0]
        for (fi, vi) in obs:
            for (fj, vj) in obs:
                if fi == fj:
                    continue
                key = (fi, vi, fj)
                K_fj = len(feat_to_value_names[fj])
                if key not in raw_cocounts:
                    raw_cocounts[key] = np.zeros(K_fj, dtype=float)
                raw_cocounts[key][vj] += 1.0

    # Mutual information matrix
    print("  Computing MI matrix ...")
    mi_matrix = np.zeros((n_feats, n_feats), dtype=float)
    for fi in range(n_feats):
        K_fi = len(feat_to_value_names[fi])
        for fj in range(fi + 1, n_feats):
            K_fj = len(feat_to_value_names[fj])
            mi   = 0.0
            for vi in range(K_fi):
                cnt_vi = marginal_counts[fi][vi]
                if cnt_vi < min_count:
                    continue
                key    = (fi, vi, fj)
                cocount = raw_cocounts.get(key, np.zeros(K_fj))
                for vj in range(K_fj):
                    c_joint = cocount[vj]
                    if c_joint < 1.0:
                        continue
                    p_joint = c_joint / n_langs
                    p_fi    = marginals[fi][vi]
                    p_fj    = marginals[fj][vj]
                    if p_joint > 0 and p_fi > 0 and p_fj > 0:
                        mi += p_joint * np.log(p_joint / (p_fi * p_fj))
            mi_matrix[fi, fj] = max(mi, 0.0)
            mi_matrix[fj, fi] = mi_matrix[fi, fj]

    # IDF weights
    obs_per_feat = (cat_matrix_train >= 0).sum(axis=0).astype(float)
    idf_weights  = np.log(n_langs / (1.0 + obs_per_feat))

    return marginals, raw_cocounts, marginal_counts, mi_matrix, idf_weights


# ===========================================================================
# PART 3: Expert distributions
# ===========================================================================

def p_freq(fj, marginals):
    """Frequency baseline: marginal distribution over fj values."""
    return marginals[fj].copy()


def p_idf_knn(obs_feat_dict, fj, cat_matrix_train, feat_name_to_idx,
              feat_to_value_names, idf_weights, marginals, k=5):
    """
    IDF-weighted kNN vote for feature fj.
    Smooth toward p_freq: beta = 1 / (1 + neighbor_support).
    """
    n_train = cat_matrix_train.shape[0]
    scores  = np.zeros(n_train, dtype=np.float64)

    for feat_name, val_str in obs_feat_dict.items():
        fi = feat_name_to_idx.get(feat_name)
        if fi is None:
            continue
        val_list = feat_to_value_names.get(fi, [])
        if val_str not in val_list:
            continue
        vi_local = val_list.index(val_str)
        match    = (cat_matrix_train[:, fi] == vi_local).astype(np.float64)
        scores  += match * idf_weights[fi]

    n_with = int((scores > 0).sum())
    p_f    = marginals[fj]

    if n_with == 0:
        return p_f.copy()

    k_use  = min(k, n_with)
    top_k  = np.argsort(scores)[-k_use:]

    K_fj             = len(feat_to_value_names[fj])
    vote             = np.zeros(K_fj, dtype=np.float64)
    neighbor_support = 0.0

    for li in top_k:
        vj_val = int(cat_matrix_train[li, fj])
        if vj_val >= 0:
            vote[vj_val]     += scores[li]
            neighbor_support += scores[li]

    beta = 1.0 / (1.0 + neighbor_support)
    if neighbor_support > 0:
        vote /= neighbor_support
        return (1.0 - beta) * vote + beta * p_f
    return p_f.copy()


def _compute_corr_scores(obs_feat_dict, fj, feat_name_to_idx,
                         feat_to_value_names, marginals,
                         raw_cocounts, marginal_counts,
                         mi_matrix, min_count, alpha):
    """
    Core PMI/MI computation. Returns (s_corr, weights_dict, vi_locals_dict).
    s_corr     : np.array(K_fj,) raw MI-weighted PMI scores (unnormalized)
    weights_dict: {fi_idx: mi_weight_normalized}
    vi_locals  : {fi_idx: vi_local}
    Returns (zeros, {}, {}) if no usable context.
    """
    K_fj = len(feat_to_value_names[fj])

    # Collect usable context observations
    usable = {}   # fi_idx -> vi_local
    mi_raw = {}   # fi_idx -> I(fi; fj)

    for feat_name, val_str in obs_feat_dict.items():
        fi = feat_name_to_idx.get(feat_name)
        if fi is None or fi == fj:
            continue
        val_list = feat_to_value_names.get(fi, [])
        if val_str not in val_list:
            continue
        vi_local  = val_list.index(val_str)
        cnt_fi_vi = marginal_counts[fi][vi_local]
        if cnt_fi_vi < min_count:
            continue
        usable[fi] = vi_local
        mi_raw[fi] = mi_matrix[fi, fj]

    if not usable:
        return np.zeros(K_fj), {}, {}

    # Normalize MI weights
    total_mi = sum(mi_raw.values())
    if total_mi < 1e-10:
        n = len(usable)
        weights = {fi: 1.0 / n for fi in usable}
    else:
        weights = {fi: mi_raw[fi] / total_mi for fi in usable}

    # Accumulate weighted PMI scores
    s_corr = np.zeros(K_fj)
    p_fj   = marginals[fj]

    for fi, vi_local in usable.items():
        w       = weights[fi]
        cnt     = marginal_counts[fi][vi_local]
        key     = (fi, vi_local, fj)
        cocount = raw_cocounts.get(key, np.zeros(K_fj))
        if len(cocount) != K_fj:
            cocount = np.zeros(K_fj)
        denom  = cnt + alpha
        p_cond = (cocount + alpha * p_fj) / denom
        pmi    = np.log(p_cond / (p_fj + 1e-10) + 1e-10)
        s_corr += w * pmi

    return s_corr, weights, usable


def _softmax_np(x):
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-300)


def p_corr(obs_feat_dict, fj, feat_name_to_idx, feat_to_value_names,
           marginals, raw_cocounts, marginal_counts, mi_matrix,
           min_count, alpha, T_corr):
    """
    MI-weighted PMI expert. Returns (probs, s_corr, weights_dict, vi_locals).
    Falls back to marginals[fj] if no usable context.
    """
    s_corr, weights, vi_locals = _compute_corr_scores(
        obs_feat_dict, fj, feat_name_to_idx, feat_to_value_names,
        marginals, raw_cocounts, marginal_counts, mi_matrix, min_count, alpha)

    if not weights:
        return marginals[fj].copy(), s_corr, {}, {}

    return _softmax_np(s_corr / T_corr), s_corr, weights, vi_locals


def p_embcorr(fj, feat_to_value_names, feat_to_global_ids,
              val_emb_weight, s_corr, weights_dict, vi_locals_dict,
              rho, T_corr):
    """
    Augments s_corr with embedding cosine similarity.
    Returns softmax((s_corr + rho * s_emb) / T_corr).
    """
    if not weights_dict or rho == 0.0:
        return _softmax_np(s_corr / T_corr)

    K_fj    = len(feat_to_value_names[fj])
    s_emb   = np.zeros(K_fj)
    gids_fj = feat_to_global_ids[fj]

    # Precompute fj value embeddings (K_fj, d)
    emb_fj = val_emb_weight[gids_fj]                  # (K_fj, d)
    norms_fj = np.linalg.norm(emb_fj, axis=1) + 1e-8  # (K_fj,)

    for fi, w in weights_dict.items():
        if w == 0.0:
            continue
        vi_local   = vi_locals_dict[fi]
        gid_vi     = feat_to_global_ids[fi][vi_local]
        emb_vi     = val_emb_weight[gid_vi]            # (d,)
        norm_vi    = np.linalg.norm(emb_vi) + 1e-8
        cos_sims   = (emb_fj @ emb_vi) / (norms_fj * norm_vi)  # (K_fj,)
        s_emb     += w * cos_sims

    return _softmax_np((s_corr + rho * s_emb) / T_corr)


# ===========================================================================
# PART 4: Temperature calibration
# ===========================================================================

def calibrate_temperature(model, cat_matrix_dev, feat_to_global_ids,
                           feat_to_value_names, feat_name_to_idx,
                           device, temp_grid=None):
    """Grid-search temperature minimising dev NLL. Uses raw logits / T."""
    if temp_grid is None:
        temp_grid = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    model.eval()
    best_T, best_nll = 1.0, float("inf")
    kept_feature_names_idx = {i: i for i in range(cat_matrix_dev.shape[1])}

    for T in temp_grid:
        total_nll, n_preds = 0.0, 0

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
                for fi in tgt_feats:
                    true_local = int(cat_matrix_dev[lang_idx, fi])
                    logits = model.logits_for_feature(
                        ctx_feats, ctx_gvals, fi, device)
                    log_probs  = F.log_softmax(logits / T, dim=0)
                    total_nll -= log_probs[true_local].item()
                    n_preds   += 1

        avg_nll = total_nll / max(n_preds, 1)
        if avg_nll < best_nll:
            best_nll = avg_nll
            best_T   = T

    return best_T


# ===========================================================================
# PART 5: Hyperparameter tuning on dev
# ===========================================================================

def _build_dev_examples(cat_matrix_dev, feat_to_global_ids,
                        feat_to_value_names, kept_feature_names):
    """Split each dev language in half; return list of (ctx_feats, ctx_gvals,
    obs_feat_dict, tgt_feats, true_vals)."""
    examples = []
    for lang_idx in range(cat_matrix_dev.shape[0]):
        observed = np.where(cat_matrix_dev[lang_idx] >= 0)[0].tolist()
        if len(observed) < 4:
            continue
        mid       = len(observed) // 2
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
        true_vals = {
            kept_feature_names[fi]: feat_to_value_names[fi][int(cat_matrix_dev[lang_idx, fi])]
            for fi in tgt_feats
        }
        examples.append((ctx_feats, ctx_gvals, obs_feat_dict, tgt_feats, true_vals))
    return examples


def tune_corr_params_nomodel(dev_examples, kept_feature_names, feat_name_to_idx,
                              feat_to_value_names, marginals, raw_cocounts,
                              marginal_counts, mi_matrix,
                              alpha_grid, mincount_grid, Tcorr_grid):
    """Stage 1 without model: tune (alpha, min_count, T_corr) for Config F."""
    best_acc, best_params = -1.0, (0.5, 3, 1.0)

    for alpha in alpha_grid:
        for min_count in mincount_grid:
            for T_corr in Tcorr_grid:
                total, correct = 0, 0
                for _, _, obs_fd, tgt_feats, true_vals in dev_examples:
                    for fi in tgt_feats:
                        fn = kept_feature_names[fi]
                        tv = true_vals[fn]
                        probs, _, _, _ = p_corr(
                            obs_fd, fi, feat_name_to_idx, feat_to_value_names,
                            marginals, raw_cocounts, marginal_counts, mi_matrix,
                            min_count, alpha, T_corr)
                        pred = feat_to_value_names[fi][int(probs.argmax())]
                        correct += int(pred == tv)
                        total   += 1
                acc = correct / max(total, 1)
                if acc > best_acc:
                    best_acc    = acc
                    best_params = (alpha, min_count, T_corr)

    print(f"  Config F corr params: alpha={best_params[0]}, "
          f"min_count={best_params[1]}, T_corr={best_params[2]}, "
          f"dev_acc={best_acc:.4f}")
    return best_params


def tune_hyperparams_on_dev(dev_examples, kept_feature_names, feat_name_to_idx,
                             feat_to_value_names, feat_to_global_ids,
                             cat_matrix_train,
                             marginals, raw_cocounts, marginal_counts,
                             mi_matrix, idf_weights, model, device, T_set,
                             alpha_grid, mincount_grid, Tcorr_grid,
                             rho_grid, k_grid,
                             lam_set_grid, lam_corr_grid,
                             lam_knn_grid, lam_freq_grid):
    """
    3-stage grid search on dev.
    Stage 1: tune (alpha, min_count, T_corr) maximising p_corr accuracy.
    Stage 2: tune (rho, k) with equal-weight ensemble.
    Stage 3: tune lambdas (lam_set, lam_corr, lam_knn, lam_freq).
    """
    # ------- Stage 1 -------
    print("  Stage 1: tuning corr params ...")
    best_acc1, best_p1 = -1.0, (0.5, 3, 1.0)
    for alpha in alpha_grid:
        for min_count in mincount_grid:
            for T_corr in Tcorr_grid:
                total, correct = 0, 0
                for _, _, obs_fd, tgt_feats, true_vals in dev_examples:
                    for fi in tgt_feats:
                        fn = kept_feature_names[fi]
                        tv = true_vals[fn]
                        probs, _, _, _ = p_corr(
                            obs_fd, fi, feat_name_to_idx, feat_to_value_names,
                            marginals, raw_cocounts, marginal_counts, mi_matrix,
                            min_count, alpha, T_corr)
                        pred = feat_to_value_names[fi][int(probs.argmax())]
                        correct += int(pred == tv)
                        total   += 1
                acc = correct / max(total, 1)
                if acc > best_acc1:
                    best_acc1 = acc
                    best_p1   = (alpha, min_count, T_corr)

    best_alpha, best_min_count, best_T_corr = best_p1
    print(f"  Stage 1 best: alpha={best_alpha}, min_count={best_min_count}, "
          f"T_corr={best_T_corr}, acc={best_acc1:.4f}")

    # ------- Stage 2 -------
    print("  Stage 2: tuning rho and k ...")
    model.eval()
    val_emb_w = model.val_embeddings.weight.detach().cpu().numpy()

    # Precompute model log-probs and corr scores for all dev examples
    dev_precomp = []
    for ctx_feats, ctx_gvals, obs_fd, tgt_feats, true_vals in dev_examples:
        per_fi = {}
        for fi in tgt_feats:
            fn = kept_feature_names[fi]
            with torch.no_grad():
                logits   = model.logits_for_feature(ctx_feats, ctx_gvals, fi, device)
                lp_set   = F.log_softmax(logits / T_set, dim=0).cpu().numpy()
            probs_c, s_c, w_c, vi_c = p_corr(
                obs_fd, fi, feat_name_to_idx, feat_to_value_names,
                marginals, raw_cocounts, marginal_counts, mi_matrix,
                best_min_count, best_alpha, best_T_corr)
            lp_freq = np.log(marginals[fi] + 1e-300)
            per_fi[fi] = {
                "fn": fn, "tv": true_vals[fn],
                "lp_set": lp_set,
                "lp_corr": np.log(probs_c + 1e-300),
                "s_corr": s_c, "w_c": w_c, "vi_c": vi_c,
                "lp_freq": lp_freq,
            }
        dev_precomp.append((obs_fd, per_fi))

    best_acc2, best_p2 = -1.0, (0.0, 5)
    for rho in rho_grid:
        for k in k_grid:
            total, correct = 0, 0
            for obs_fd, per_fi in dev_precomp:
                for fi, d in per_fi.items():
                    probs_e = p_embcorr(
                        fi, feat_to_value_names, feat_to_global_ids,
                        val_emb_w, d["s_corr"], d["w_c"], d["vi_c"],
                        rho, best_T_corr)
                    lp_emb = np.log(probs_e + 1e-300)
                    probs_knn = p_idf_knn(
                        obs_fd, fi, cat_matrix_train, feat_name_to_idx,
                        feat_to_value_names, idf_weights, marginals, k)
                    lp_knn = np.log(probs_knn + 1e-300)
                    combined = d["lp_set"] + lp_emb + lp_knn + d["lp_freq"]
                    pred = feat_to_value_names[fi][int(combined.argmax())]
                    correct += int(pred == d["tv"])
                    total   += 1
            acc = correct / max(total, 1)
            if acc > best_acc2:
                best_acc2 = acc
                best_p2   = (rho, k)

    best_rho, best_k = best_p2
    print(f"  Stage 2 best: rho={best_rho}, k={best_k}, acc={best_acc2:.4f}")

    # Rebuild dev_precomp with best_k for kNN
    dev_precomp2 = []
    for obs_fd, per_fi in dev_precomp:
        per_fi2 = {}
        for fi, d in per_fi.items():
            probs_knn = p_idf_knn(obs_fd, fi, cat_matrix_train, feat_name_to_idx,
                                   feat_to_value_names, idf_weights,
                                   marginals, best_k)
            probs_e   = p_embcorr(fi, feat_to_value_names, feat_to_global_ids,
                                   val_emb_w, d["s_corr"], d["w_c"], d["vi_c"],
                                   best_rho, best_T_corr)
            per_fi2[fi] = dict(d,
                                lp_corr_emb=np.log(probs_e + 1e-300),
                                lp_knn=np.log(probs_knn + 1e-300))
        dev_precomp2.append(per_fi2)

    # ------- Stage 3 -------
    print("  Stage 3: tuning ensemble lambdas ...")
    best_acc3, best_p3 = -1.0, (1.0, 1.0, 0.25, 0.1)
    for ls in lam_set_grid:
        for lc in lam_corr_grid:
            for lk in lam_knn_grid:
                for lf in lam_freq_grid:
                    total, correct = 0, 0
                    for per_fi2 in dev_precomp2:
                        for fi, d in per_fi2.items():
                            combined = (ls * d["lp_set"]
                                        + lc * d["lp_corr_emb"]
                                        + lk * d["lp_knn"]
                                        + lf * d["lp_freq"])
                            pred = feat_to_value_names[fi][int(combined.argmax())]
                            correct += int(pred == d["tv"])
                            total   += 1
                    acc = correct / max(total, 1)
                    if acc > best_acc3:
                        best_acc3 = acc
                        best_p3   = (ls, lc, lk, lf)

    best_ls, best_lc, best_lk, best_lf = best_p3
    print(f"  Stage 3 best: lam_set={best_ls}, lam_corr={best_lc}, "
          f"lam_knn={best_lk}, lam_freq={best_lf}, acc={best_acc3:.4f}")

    return {
        "alpha": best_alpha, "min_count": best_min_count,
        "T_corr": best_T_corr, "rho": best_rho, "k": best_k,
        "lam_set": best_ls, "lam_corr": best_lc,
        "lam_knn": best_lk, "lam_freq": best_lf,
    }


# ===========================================================================
# PART 6: Inference — all experts for one language
# ===========================================================================

def infer_all_experts(obs_feat_dict, blanked_feats, model,
                      kept_feature_names, feat_name_to_idx,
                      feat_to_value_names, feat_to_global_ids,
                      cat_matrix_train, marginals, raw_cocounts,
                      marginal_counts, mi_matrix, idf_weights,
                      device, T_set, params, val_emb_w):
    """
    Returns per-feature dicts for each blanked feature:
      lp_set, lp_corr, s_corr, weights_dict, vi_locals,
      lp_knn, lp_freq
    """
    # Build context
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

    alpha     = params["alpha"]
    min_count = params["min_count"]
    T_corr    = params["T_corr"]
    rho       = params["rho"]
    k         = params["k"]

    model.eval()
    results = {}
    with torch.no_grad():
        for feat_name in blanked_feats:
            fi = feat_name_to_idx.get(feat_name)
            if fi is None:
                continue
            K_fj = len(feat_to_value_names[fi])

            # SetEncoder
            logits = model.logits_for_feature(ctx_feats, ctx_gvals, fi, device)
            lp_set = F.log_softmax(logits / T_set, dim=0).cpu().numpy()

            # PMI/MI expert
            probs_c, s_c, w_c, vi_c = p_corr(
                obs_feat_dict, fi, feat_name_to_idx, feat_to_value_names,
                marginals, raw_cocounts, marginal_counts, mi_matrix,
                min_count, alpha, T_corr)
            lp_corr = np.log(probs_c + 1e-300)

            # IDF-kNN
            probs_knn = p_idf_knn(
                obs_feat_dict, fi, cat_matrix_train, feat_name_to_idx,
                feat_to_value_names, idf_weights, marginals, k)
            lp_knn = np.log(probs_knn + 1e-300)

            # Frequency
            lp_freq = np.log(marginals[fi] + 1e-300)

            results[feat_name] = {
                "fi": fi, "K": K_fj,
                "lp_set": lp_set, "lp_corr": lp_corr,
                "s_corr": s_c, "weights": w_c, "vi_locals": vi_c,
                "lp_knn": lp_knn, "lp_freq": lp_freq,
            }

    return results


# ===========================================================================
# PART 7: Per-config predictions
# ===========================================================================

def predict_configs(wc, blanked_feats, expert_results,
                    feat_to_value_names, feat_to_global_ids,
                    feat_name_to_idx, obs_feat_dict,
                    model, val_emb_w, params, freq_fallback):
    """
    Returns dict: cfg -> {feat_name: pred_str}
    for configs E, G, H, I.
    """
    alpha     = params["alpha"]
    min_count = params["min_count"]
    T_corr    = params["T_corr"]
    rho       = params["rho"]
    lam_set   = params["lam_set"]
    lam_corr  = params["lam_corr"]
    lam_knn   = params["lam_knn"]
    lam_freq  = params["lam_freq"]

    cfg_E, cfg_G, cfg_H, cfg_I = {}, {}, {}, {}

    for feat_name, d in expert_results.items():
        fi    = d["fi"]
        K     = d["K"]
        lp_s  = d["lp_set"]
        lp_c  = d["lp_corr"]
        s_c   = d["s_corr"]
        w_c   = d["weights"]
        vi_c  = d["vi_locals"]
        lp_k  = d["lp_knn"]
        lp_f  = d["lp_freq"]

        # Embedding-augmented corr
        probs_e = p_embcorr(fi, feat_to_value_names, feat_to_global_ids,
                             val_emb_w, s_c, w_c, vi_c, rho, T_corr)
        lp_ce = np.log(probs_e + 1e-300)

        vals = feat_to_value_names[fi]

        cfg_E[feat_name] = vals[int(lp_s.argmax())]
        cfg_G[feat_name] = vals[int((lam_set*lp_s + lam_corr*lp_c).argmax())]
        cfg_H[feat_name] = vals[int((lam_set*lp_s + lam_corr*lp_c
                                     + lam_knn*lp_k + lam_freq*lp_f).argmax())]
        cfg_I[feat_name] = vals[int((lam_set*lp_s + lam_corr*lp_ce
                                     + lam_knn*lp_k + lam_freq*lp_f).argmax())]

    return {"E": cfg_E, "G": cfg_G, "H": cfg_H, "I": cfg_I}


# ===========================================================================
# Bootstrap / scoring helpers
# ===========================================================================

def _compute_per_lang_acc(preds_by_lang, test_meta, test_blank,
                           gold_obs, feat_name_to_idx):
    per_lang = []
    for _, row in test_meta.iterrows():
        wc     = row["wals_code"]
        genus  = row["genus"]
        blanked = test_blank.get(wc, set())
        pred    = preds_by_lang.get(wc, {})
        gold    = gold_obs.get(wc, {})
        correct = total = 0
        for fn in blanked:
            if fn not in feat_name_to_idx:
                continue
            gv = gold.get(fn)
            pv = pred.get(fn)
            if gv is None:
                continue
            total   += 1
            if pv is not None and pv == gv:
                correct += 1
        if total > 0:
            per_lang.append((correct / total, genus))
    return per_lang


def _macro_acc(per_lang):
    ga = defaultdict(list)
    for acc, genus in per_lang:
        ga[genus].append(acc)
    if not ga:
        return 0.0
    return float(np.mean([np.mean(v) for v in ga.values()]))


def bootstrap_test(preds_by_lang, test_meta, test_blank, gold_obs,
                   feat_name_to_idx, ref=0.66, n_boot=1000):
    per_lang  = _compute_per_lang_acc(preds_by_lang, test_meta, test_blank,
                                       gold_obs, feat_name_to_idx)
    obs_macro = _macro_acc(per_lang)
    obs_delta = obs_macro - ref
    np.random.seed(0)
    n = len(per_lang)
    boot_deltas = []
    for _ in range(n_boot):
        idx   = np.random.choice(n, n, replace=True)
        samp  = [per_lang[i] for i in idx]
        boot_deltas.append(_macro_acc(samp) - ref)
    bd = np.array(boot_deltas)
    ci = (float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5)))
    p  = float(np.mean(np.abs(bd) >= abs(obs_delta)))
    if p == 0.0:
        p = 1.0 / n_boot
    return obs_macro, ci, p


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
    out_dir    = args.output_dir or gb2v
    pmi_cache  = f"{out_dir}/pmi_stats.pkl"

    print("=" * 72)
    print("SIGTYP 2020 v4 — CrossAttentionSetEncoder + PMI/MI expert")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------
    print("\n[Part 1] Loading SIGTYP data ...")
    train_meta, train_feat, train_blank, train_obs = parse_sigtyp(train_csv)
    dev_meta,   dev_feat,   dev_blank,   dev_obs   = parse_sigtyp(dev_csv)
    test_meta,  test_feat,  test_blank,  test_obs  = parse_sigtyp(test_blind)

    print(f"  Train: {len(train_meta)} languages, "
          f"Dev: {len(dev_meta)}, Test: {len(test_meta)}")

    traindev_feat = pd.concat([train_feat, dev_feat], ignore_index=True)
    n_train       = len(train_feat)

    # -----------------------------------------------------------------------
    # Vocabulary
    # -----------------------------------------------------------------------
    print("\n[Part 2] Building vocabulary ...")
    feature_cols = list(traindev_feat.columns)
    cat_matrix_traindev, kept_feature_names, feat_to_global_ids, feat_to_value_names = \
        prepare_categorical(traindev_feat, feature_cols)

    n_features     = len(kept_feature_names)
    n_total_values = max(max(gids) for gids in feat_to_global_ids.values()) + 1
    feat_name_to_idx = {name: i for i, name in enumerate(kept_feature_names)}

    print(f"  {n_features} features, {n_total_values} total value IDs")

    cat_matrix_train = cat_matrix_traindev[:n_train]
    cat_matrix_dev   = cat_matrix_traindev[n_train:]

    # Frequency fallback
    freq_fallback = {}
    for fi, name in enumerate(kept_feature_names):
        col = cat_matrix_train[:, fi]
        obs = col[col >= 0]
        if len(obs) == 0:
            freq_fallback[name] = feat_to_value_names[fi][0]
        else:
            cnt = Counter(obs.tolist())
            freq_fallback[name] = feat_to_value_names[fi][cnt.most_common(1)[0][0]]

    # -----------------------------------------------------------------------
    # PMI statistics (cached)
    # -----------------------------------------------------------------------
    print("\n[Part 3] PMI statistics ...")
    if os.path.exists(pmi_cache) and not args.force_retrain:
        print(f"  Loading from cache: {pmi_cache}")
        with open(pmi_cache, "rb") as fh:
            marginals, raw_cocounts, marginal_counts, mi_matrix, idf_weights = \
                pickle.load(fh)
    else:
        t0 = time.time()
        marginals, raw_cocounts, marginal_counts, mi_matrix, idf_weights = \
            build_pmi_stats(cat_matrix_train, feat_to_value_names, min_count=3)
        print(f"  Built in {time.time()-t0:.1f}s. Saving to {pmi_cache} ...")
        with open(pmi_cache, "wb") as fh:
            pickle.dump((marginals, raw_cocounts, marginal_counts,
                         mi_matrix, idf_weights), fh)

    # -----------------------------------------------------------------------
    # Config F: PMI expert only (deterministic, once)
    # -----------------------------------------------------------------------
    print("\n[Config F] Tuning corr params for PMI-only expert ...")
    dev_examples = _build_dev_examples(
        cat_matrix_dev, feat_to_global_ids, feat_to_value_names,
        kept_feature_names)

    alpha_grid    = [0.1, 0.5, 1.0, 2.0, 5.0]
    mincount_grid = [1, 3, 5, 10]
    Tcorr_grid    = [0.5, 1.0, 1.5, 2.0, 3.0]

    F_alpha, F_min_count, F_T_corr = tune_corr_params_nomodel(
        dev_examples, kept_feature_names, feat_name_to_idx,
        feat_to_value_names, marginals, raw_cocounts, marginal_counts,
        mi_matrix, alpha_grid, mincount_grid, Tcorr_grid)

    print("  Running Config F inference ...")
    F_preds = {}
    for _, row in test_meta.iterrows():
        wc  = row["wals_code"]
        obs = {k: _strip_to_id(v)
               for k, v in test_obs.get(wc, {}).items()
               if k in feat_name_to_idx}
        blanked = test_blank.get(wc, set())
        preds_f = {}
        for feat_name in blanked:
            fi = feat_name_to_idx.get(feat_name)
            if fi is None:
                preds_f[feat_name] = freq_fallback.get(feat_name, "1")
                continue
            probs, _, _, _ = p_corr(obs, fi, feat_name_to_idx,
                                     feat_to_value_names, marginals,
                                     raw_cocounts, marginal_counts, mi_matrix,
                                     F_min_count, F_alpha, F_T_corr)
            preds_f[feat_name] = feat_to_value_names[fi][int(probs.argmax())]
        F_preds[wc] = preds_f

    F_path = f"{out_dir}/sigtyp_v4_F.tsv"
    _write_submission(F_path, test_meta, test_blank, test_obs, F_preds, freq_fallback)
    print(f"  Wrote {F_path}")

    # -----------------------------------------------------------------------
    # Seed loop: CrossAttentionSetEncoder
    # -----------------------------------------------------------------------
    rho_grid      = [0.0, 0.1, 0.2, 0.4, 0.8]
    k_grid        = [3, 5, 10]
    lam_set_grid  = [0.5, 0.75, 1.0, 1.5]
    lam_corr_grid = [0.5, 0.75, 1.0, 1.5]
    lam_knn_grid  = [0.0, 0.25, 0.5, 1.0]
    lam_freq_grid = [0.0, 0.1, 0.25]

    seed_params   = {}
    seed_T        = {}
    all_E_preds   = {}
    all_I_preds   = {}   # seed -> {wc -> {fn -> pred}}
    sub_paths_all = [F_path]

    # For cross-seed analysis
    first_seed_I_preds = None

    for seed in args.seeds:
        print(f"\n[Seed {seed}] CrossAttentionSetEncoder ...")
        ckpt_path = f"{out_dir}/set_encoder_v4_seed{seed}.pt"

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = None

        if os.path.exists(ckpt_path) and not args.force_retrain:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                assert ckpt.get("embed_dim")  == args.embed_dim
                assert ckpt.get("hidden_dim") == args.hidden_dim
                assert ckpt.get("n_epochs")   == args.n_epochs
                model = CrossAttentionSetEncoder(
                    n_features, n_total_values, feat_to_global_ids,
                    args.embed_dim, args.hidden_dim)
                model.load_state_dict(ckpt["state_dict"])
                print(f"  Loaded checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"  Checkpoint load failed ({e}), retraining ...")
                model = None

        if model is None:
            if args.skip_training:
                print(f"  --skip-training set, no valid checkpoint; skipping seed {seed}")
                continue
            print(f"  Training ({args.n_epochs} epochs) ...")
            t0      = time.time()
            model   = CrossAttentionSetEncoder(
                n_features, n_total_values, feat_to_global_ids,
                args.embed_dim, args.hidden_dim)
            dataset = SetEncoderDataset(cat_matrix_traindev, feat_to_global_ids)
            train_cross_attention(model, dataset, n_epochs=args.n_epochs,
                                   lr=args.lr, l2_reg=args.l2_reg,
                                   device=args.device, seed=seed)
            print(f"  Training done in {time.time()-t0:.1f}s")
            ckpt = {"state_dict": model.state_dict(), "seed": seed,
                    "n_epochs": args.n_epochs, "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim}
            torch.save(ckpt, ckpt_path)
            print(f"  Saved to {ckpt_path}")

        model.to(args.device)
        model.eval()
        val_emb_w = model.val_embeddings.weight.detach().cpu().numpy()

        # Temperature calibration
        print("  Calibrating temperature ...")
        T_set = calibrate_temperature(
            model, cat_matrix_dev, feat_to_global_ids, feat_to_value_names,
            feat_name_to_idx, args.device)
        seed_T[seed] = T_set
        print(f"  T_set = {T_set}")
        if T_set > 2.0:
            print("  Warning: T > 2.0 — model may be overconfident")

        # Hyperparameter tuning
        print("  Tuning hyperparameters on dev ...")
        params = tune_hyperparams_on_dev(
            dev_examples, kept_feature_names, feat_name_to_idx,
            feat_to_value_names, feat_to_global_ids,
            cat_matrix_train,
            marginals, raw_cocounts, marginal_counts, mi_matrix, idf_weights,
            model, args.device, T_set,
            alpha_grid, mincount_grid, Tcorr_grid,
            rho_grid, k_grid,
            lam_set_grid, lam_corr_grid, lam_knn_grid, lam_freq_grid)
        seed_params[seed] = params

        # Inference
        print(f"  Inference over {len(test_meta)} test languages ...")
        E_preds, G_preds, H_preds, I_preds = {}, {}, {}, {}

        for i, (_, row) in enumerate(test_meta.iterrows()):
            wc      = row["wals_code"]
            blanked = test_blank.get(wc, set())
            obs_fd  = {k: _strip_to_id(v)
                       for k, v in test_obs.get(wc, {}).items()
                       if k in feat_name_to_idx}
            obs_in_vocab = len(obs_fd)

            expert_results = infer_all_experts(
                obs_fd, blanked, model, kept_feature_names, feat_name_to_idx,
                feat_to_value_names, feat_to_global_ids,
                cat_matrix_train, marginals, raw_cocounts, marginal_counts,
                mi_matrix, idf_weights, args.device, T_set, params, val_emb_w)

            cfgs = predict_configs(wc, blanked, expert_results,
                                    feat_to_value_names, feat_to_global_ids,
                                    feat_name_to_idx, obs_fd,
                                    model, val_emb_w, params, freq_fallback)

            E_preds[wc] = cfgs["E"]
            G_preds[wc] = cfgs["G"]
            H_preds[wc] = cfgs["H"]
            I_preds[wc] = cfgs["I"]

            if (i + 1) % 25 == 0 or (i + 1) == len(test_meta):
                print(f"  {i+1}/{len(test_meta)}  (last: {wc}, "
                      f"obs_in_vocab={obs_in_vocab})")

        all_E_preds[seed] = E_preds
        all_I_preds[seed] = I_preds
        if first_seed_I_preds is None:
            first_seed_I_preds = I_preds

        # Write submissions
        for cfg_name, preds in [("E", E_preds), ("G", G_preds),
                                  ("H", H_preds), ("I", I_preds)]:
            path = f"{out_dir}/sigtyp_v4_{cfg_name}_seed{seed}.tsv"
            _write_submission(path, test_meta, test_blank, test_obs,
                              preds, freq_fallback)
            print(f"  Wrote {path}")
            sub_paths_all.append(path)

    # -----------------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------------
    print("\n[Scoring] Running score.py ...")
    all_scores = {}
    if sub_paths_all:
        stdout, stderr = _run_scorer(scorer, sub_paths_all)
        if stderr.strip():
            print(f"  Scorer stderr: {stderr[:300]}")
        all_scores = _parse_scores(stdout)

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("ABLATION TABLE  (macro accuracy per genus)")
    print("=" * 72)

    COL_W = [42] + [6] * 8

    def pr(cells):
        parts = [str(cells[0]).ljust(COL_W[0])] + [str(c).rjust(6) for c in cells[1:]]
        print("| " + " | ".join(parts) + " |")

    def sep():
        print("|" + "|".join("-" + "-" * w + "-" for w in COL_W) + "|")

    def _get_row(bn, mode="macro"):
        d    = all_scores.get(mode, {}).get(bn, {})
        vals = [f"{d.get(c):.2f}" if d.get(c) is not None else " n/a"
                for c in _TABLE_COLS]
        return vals

    def _seed_row_str(cfg, label):
        ovs = []
        for seed in args.seeds:
            bn = f"sigtyp_v4_{cfg}_seed{seed}"
            ov = all_scores.get("macro", {}).get(bn, {}).get("Overall")
            if ov is not None:
                ovs.append(ov)
        if not ovs:
            return [label] + [" n/a"] * 7 + [" n/a"]
        return [label] + [" n/a"] * 7 + [f"{np.mean(ovs):.2f}±{np.std(ovs):.2f}"]

    pr(["System"] + _TABLE_HDR)
    sep()
    pr(["--- v3 non-neural baselines (reference) ---"] + [""] * 8)
    sep()
    for tag, label in [("freq", "Freq"), ("genus", "Genus majority"),
                        ("knn", "kNN k=5"), ("idfknn", "IDF-kNN k=5"),
                        ("condprob", "Cond. prob")]:
        bn = f"sigtyp_v3_{tag}"
        pr([label] + _get_row(bn))
    sep()
    pr(["--- v3 SetEncoder (reference) ---"] + [""] * 8)
    sep()
    pr(["[A] SetEncoder v3 (seed13 reported)"] +
       [".72", ".77", ".70", ".63", ".74", ".63", ".67", ".67"])
    sep()
    pr(["--- v4 CrossAttentionSetEncoder ---"] + [""] * 8)
    sep()
    pr(_seed_row_str("E", "[E] Cross-attn only      (mean±std)"))
    bn_F = "sigtyp_v4_F"
    pr(["[F] PMI/MI expert only   (single run)"] + _get_row(bn_F))
    pr(_seed_row_str("G", "[G] Cross-attn + PMI     (mean±std)"))
    pr(_seed_row_str("H", "[H] + IDF-kNN + freq     (mean±std)"))
    pr(_seed_row_str("I", "[I] + emb-aware corr     (mean±std)"))
    sep()
    pr(["--- v1/v2 (reference) ---"] + [""] * 8)
    sep()
    pr(["[v1] Learned (transductive)"] +
       [".67", ".73", ".66", ".66", ".69", ".60", ".65", ".66"])
    pr(["[v2D] Full system"] +
       [".67", ".77", ".64", ".67", ".68", ".65", ".64", ".66"])
    sep()
    pr(["--- SIGTYP 2020 paper (macro) ---"] + [""] * 8)
    sep()
    pr(["UFAL (best constrained)"] +
       [".73", ".78", ".74", ".71", ".80", ".76", ".76", ".75"])
    pr(["NEMO system2"] +
       [".71", ".72", ".72", ".76", ".76", ".67", ".69", ".69"])
    pr(["CrossLingference (unconstr.)"] +
       [".71", ".73", ".67", ".68", ".57", ".60", ".65", ".65"])
    pr(["Baseline_frequency (paper)"] +
       [".51", ".53", ".37", ".49", ".41", ".58", ".53", ".49"])
    print()

    # 1. Per-seed I scores
    print("Per-seed Config I overall macro:")
    best_I_preds = None
    best_I_seed  = None
    best_I_ov    = -1.0
    for seed in args.seeds:
        bn = f"sigtyp_v4_I_seed{seed}"
        ov = all_scores.get("macro", {}).get(bn, {}).get("Overall")
        flag = ""
        if ov is not None and ov > best_I_ov:
            best_I_ov    = ov
            best_I_seed  = seed
            best_I_preds = all_I_preds.get(seed)
        print(f"  seed={seed}: {f'{ov:.4f}' if ov is not None else 'n/a'}{flag}")

    # 2. Bootstrap test
    if best_I_preds is not None:
        print("\n[Bootstrap] Best config I vs ref=0.66 (n_boot=1000) ...")
        try:
            gold_meta2, _, gold_blank2, gold_obs2 = parse_sigtyp(test_gold)
            obs_macro, (ci_lo, ci_hi), p_val = bootstrap_test(
                best_I_preds, test_meta, test_blank, gold_obs2,
                feat_name_to_idx, ref=0.66)
            print(f"  Best I macro:           {obs_macro:.4f}")
            print(f"  Delta vs v1 Learned:    {obs_macro - 0.66:+.4f}  "
                  f"(p={p_val:.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}])")
            print(f"  Delta vs UFAL (0.75):   {obs_macro - 0.75:+.4f}  "
                  f"({'beat UFAL!' if obs_macro >= 0.75 else 'still behind UFAL'})")
        except Exception as e:
            print(f"  Bootstrap failed: {e}")

    # 3. Low-obs vs high-obs accuracy
    if best_I_preds is not None:
        print("\n[Low/high-obs analysis] ≤4 vs >4 observed in-vocab features:")
        try:
            gold_meta2, _, _, gold_obs2 = parse_sigtyp(test_gold)
            for cfg_label, preds_map in [
                    ("v3-A  (SetEncoder v3 approx)", all_E_preds.get(args.seeds[0], {})),
                    ("best I (v4)", best_I_preds)]:
                low_acc, high_acc = [], []
                for _, row in test_meta.iterrows():
                    wc  = row["wals_code"]
                    obs = {k: _strip_to_id(v)
                           for k, v in test_obs.get(wc, {}).items()
                           if k in feat_name_to_idx}
                    n_obs    = len(obs)
                    blanked  = test_blank.get(wc, set())
                    pred     = preds_map.get(wc, {})
                    gold     = gold_obs2.get(wc, {})
                    c, t = 0, 0
                    for fn in blanked:
                        if fn not in feat_name_to_idx:
                            continue
                        gv = gold.get(fn)
                        pv = pred.get(fn)
                        if gv is None:
                            continue
                        t += 1
                        if pv is not None and pv == gv:
                            c += 1
                    if t > 0:
                        acc = c / t
                        if n_obs <= 4:
                            low_acc.append(acc)
                        else:
                            high_acc.append(acc)
                la = f"{np.mean(low_acc):.3f}"  if low_acc  else "n/a"
                ha = f"{np.mean(high_acc):.3f}" if high_acc else "n/a"
                print(f"  {cfg_label}: low-obs={la}, high-obs={ha}")
        except Exception as e:
            print(f"  Low/high-obs analysis failed: {e}")

    # 4. Temperature per seed
    print("\nCalibrated temperatures:")
    for seed in args.seeds:
        T = seed_T.get(seed)
        if T is not None:
            warn = "  *** OVERCONFIDENT ***" if T > 2.0 else ""
            print(f"  seed={seed}: T_set={T}{warn}")

    # 5. Best hyperparameters
    if seed_params:
        print("\nBest hyperparameters (per seed, then mean):")
        keys = ["alpha", "min_count", "T_corr", "rho", "k",
                "lam_set", "lam_corr", "lam_knn", "lam_freq"]
        for k in keys:
            vals = [seed_params[s][k] for s in args.seeds if s in seed_params]
            if vals:
                print(f"  {k:12s}: {vals}  mean={np.mean(vals):.3f}")

    # 6. Examples: where PMI expert fixes SetEncoder
    print("\nExamples where PMI expert changed Config E prediction:")
    if best_I_preds is not None:
        try:
            gold_meta2, _, _, gold_obs2 = parse_sigtyp(test_gold)
            E_preds_best = all_E_preds.get(best_I_seed, {})
            examples_pmi_fixes = []
            examples_set_fixes = []
            for _, row in test_meta.iterrows():
                wc  = row["wals_code"]
                obs = {k: _strip_to_id(v)
                       for k, v in test_obs.get(wc, {}).items()
                       if k in feat_name_to_idx}
                for fn in test_blank.get(wc, set()):
                    if fn not in feat_name_to_idx:
                        continue
                    gv   = gold_obs2.get(wc, {}).get(fn)
                    pv_E = E_preds_best.get(wc, {}).get(fn)
                    pv_I = best_I_preds.get(wc, {}).get(fn)
                    if gv is None or pv_E is None or pv_I is None:
                        continue
                    if pv_E != gv and pv_I == gv:
                        examples_pmi_fixes.append((wc, row["name"], fn, pv_E, pv_I, gv, obs))
                    elif pv_E == gv and pv_I != gv:
                        examples_set_fixes.append((wc, row["name"], fn, pv_E, pv_I, gv, obs))

            for label, examples in [
                    ("PMI expert fixed SetEncoder (E wrong, I right):", examples_pmi_fixes[:3]),
                    ("SetEncoder fixed PMI hurt (E right, I wrong):", examples_set_fixes[:3])]:
                print(f"\n  {label}")
                for wc, name, fn, pe, pi, gv, obs in examples:
                    print(f"    Language: {name} ({wc}), Feature: {fn}")
                    print(f"    SetEncoder: {pe}, PMI→: {pi}, Gold: {gv}")
                    ctx_feats = list(obs.keys())[:5]
                    print(f"    Context: {ctx_feats}")
        except Exception as e:
            print(f"  Example analysis failed: {e}")

    # 8. Average lambda weights
    if seed_params:
        ls = np.mean([seed_params[s]["lam_set"]  for s in seed_params])
        lc = np.mean([seed_params[s]["lam_corr"] for s in seed_params])
        lk = np.mean([seed_params[s]["lam_knn"]  for s in seed_params])
        lf = np.mean([seed_params[s]["lam_freq"] for s in seed_params])
        print(f"\nMean ensemble weights: "
              f"λ_set={ls:.3f}  λ_corr={lc:.3f}  "
              f"λ_knn={lk:.3f}  λ_freq={lf:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main(get_args())
