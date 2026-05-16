#!/usr/bin/env python3
"""
SIGTYP 2020 v2: masked-reconstruction pretraining + kNN warm-start +
iterative self-training + feature correlation prior.
Run: python3 sigtyp_eval_v2.py 2>&1 | tee sigtyp_eval_v2_log.txt
"""

import os
import sys
import time
import subprocess
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/user/Grambank2Vec")
from model         import TypologicalMF,         WALSDataset,          train_model
from model_learned import (TypologicalMF_Learned, CategoricalTypDataset,
                            prepare_categorical,  train_model_learned,  evaluate_learned)
from data_preparation import binarise_features

# ===========================================================================
# PART 0: Constants
# ===========================================================================
_BASE            = "/home/user"
TRAIN_CSV        = f"{_BASE}/ST2020/data/train.csv"
DEV_CSV          = f"{_BASE}/ST2020/data/dev.csv"
TEST_BLIND       = f"{_BASE}/ST2020/data/test_blinded.csv"
TEST_GOLD        = f"{_BASE}/ST2020/data/test_gold.csv"
SCORER           = f"{_BASE}/ST2020/scripts/score.py"
GB2V             = f"{_BASE}/Grambank2Vec"
MODEL_LEARNED_PT = f"{GB2V}/sigtyp_v2_learned.pt"
MODEL_TCF_PT     = f"{GB2V}/sigtyp_v2_tcf.pt"

EMBED    = 64
EPOCHS   = 200
BATCH    = 64
LR       = 1e-3
L2_TRAIN = 0.1
L2_FT    = 0.05
SEED     = 42

# v1 hardcoded macro results (col order: Tuc Mad Mah Nil May NPY Other Overall)
V1_ROWS = [
    ("[v1] Learned (random init, 1 pass)", [0.67,0.73,0.66,0.66,0.69,0.60,0.65,0.66]),
    ("[v1] TCF    (random init, 1 pass)",  [0.67,0.57,0.62,0.62,0.62,0.62,0.61,0.61]),
]
PAPER_TABLE = [
    ("UFAL (best constrained)",          [.73,.78,.74,.71,.80,.76,.76,.75]),
    ("NEMO system2",                      [.71,.72,.72,.76,.76,.67,.69,.69]),
    ("CrossLingference (unconstrained)", [.71,.73,.67,.68,.57,.60,.65,.65]),
    ("Baseline_frequency (paper)",        [.51,.53,.37,.49,.41,.58,.53,.49]),
]
TABLE_COLS   = ["Tucanoan","Madang","Mahakiranti","Nilotic",
                "Mayan","Northern Pama-Nyungan","other genera","Overall"]
TABLE_HEADER = ["Tucanoan","Madang","Mahakiranti","Nilotic",
                "Mayan","N.Pama-Nyungan","Other","Overall"]

# ===========================================================================
# PART 1: Parse SIGTYP data  (identical to v1)
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
    meta_rows, feat_rows, blanked, obs_strings = [], {}, {}, {}
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
        wals_code = parts[0]
        name = parts[1]
        try:
            lat, lon = float(parts[2]), float(parts[3])
        except ValueError:
            lat, lon = float("nan"), float("nan")
        genus, family, cc = parts[4], parts[5], parts[6]
        features_field = "\t".join(parts[7:])
        meta_rows.append({"wals_code": wals_code, "name": name,
                          "latitude": lat, "longitude": lon,
                          "genus": genus, "family": family, "countrycodes": cc})
        feat_dict, blank_set, obs_str_d = {}, set(), {}
        cleaned = _apply_fixes(features_field)
        for piece in cleaned.split("|"):
            piece = piece.strip()
            if not piece or "=" not in piece:
                continue
            eq_idx = piece.index("=")
            fname_lc = piece[:eq_idx].lower()
            val_str  = piece[eq_idx+1:]
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
    rows_data = [{fn: feat_rows[wc].get(fn, np.nan) for fn in all_feat_names}
                 for wc in all_wals_codes]
    feat_df = pd.DataFrame(rows_data, columns=all_feat_names)
    return meta_df, feat_df, blanked, obs_strings

# ===========================================================================
# PART 4: kNN warm-start
# ===========================================================================
def knn_warmstart(obs_feat_dict, cat_matrix, traindev_feat_df,
                  feat_name_to_idx, feat_name_to_value_list,
                  model_lang_embeddings, k=5, device="cpu"):
    """Return mean embedding of top-k most-similar training languages."""
    n_langs = cat_matrix.shape[0]

    obs_pairs = []
    for feat_name, val_str in obs_feat_dict.items():
        fi = feat_name_to_idx.get(feat_name)
        if fi is None:
            continue
        val_list = feat_name_to_value_list.get(feat_name, [])
        if val_str not in val_list:
            continue
        obs_pairs.append((fi, val_list.index(val_str)))

    if not obs_pairs:
        with torch.no_grad():
            mean_emb = model_lang_embeddings.weight.mean(dim=0, keepdim=True)
        return mean_emb.detach().clone().to(device)

    scores = np.zeros(n_langs, dtype=np.float32)
    for fi, vi_local in obs_pairs:
        col = cat_matrix[:, fi]
        scores += (col == vi_local).astype(np.float32)

    if scores.max() == 0:
        with torch.no_grad():
            mean_emb = model_lang_embeddings.weight.mean(dim=0, keepdim=True)
        return mean_emb.detach().clone().to(device)

    n_with_overlap = int((scores > 0).sum())
    k_use = min(k, n_with_overlap)
    top_k_indices = np.argsort(scores)[-k_use:]

    with torch.no_grad():
        top_k_t  = torch.tensor(top_k_indices, dtype=torch.long, device=device)
        mean_emb = model_lang_embeddings(top_k_t).mean(dim=0, keepdim=True)
    return mean_emb.detach().clone()

# ===========================================================================
# PART 5: Fine-tuning helpers
# ===========================================================================
def _obs_pairs_learned(obs_feat_dict, feat_name_to_idx, feat_name_to_value_list):
    pairs = []
    for feat_name, val_str in obs_feat_dict.items():
        fi = feat_name_to_idx.get(feat_name)
        if fi is None:
            continue
        val_list = feat_name_to_value_list.get(feat_name, [])
        if val_str not in val_list:
            continue
        pairs.append((fi, val_list.index(val_str)))
    return pairs


def fine_tune_learned(model, init_emb, obs_feat_dict, feat_name_to_idx,
                      feat_to_global_ids, feat_name_to_value_list,
                      device, n_steps=300, lr=0.05, l2_reg=0.05):
    """Transductive fine-tuning for Learned model.

    Returns (new_emb_detached, predictions)
    predictions: {feat_name: (pred_value_str, confidence_float)}
    """
    obs_pairs = _obs_pairs_learned(obs_feat_dict, feat_name_to_idx,
                                   feat_name_to_value_list)
    new_emb = nn.Parameter(init_emb.clone().to(device))
    for p in model.parameters():
        p.requires_grad_(False)

    if obs_pairs:
        optimizer    = torch.optim.Adam([new_emb], lr=lr, weight_decay=0)
        loss_history = []
        model.eval()
        for step in range(n_steps):
            losses_i = []
            for fi, val_local in obs_pairs:
                gids_t   = torch.tensor(feat_to_global_ids[fi],
                                        dtype=torch.long, device=device)
                val_embs = model.value_embeddings(gids_t)
                scores   = new_emb @ val_embs.T
                losses_i.append(-F.log_softmax(scores, dim=-1)[0, val_local])
            nll        = torch.stack(losses_i).mean()
            total_loss = nll + (l2_reg / 2.0) * new_emb.pow(2).mean()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lv = total_loss.item()
            loss_history.append(lv)
            if lv < 0.005:
                break
            if step >= 30 and (loss_history[-31] - lv) < 0.001:
                break

    preds = {}
    model.eval()
    with torch.no_grad():
        e = new_emb.detach()
        for feat_name, val_list in feat_name_to_value_list.items():
            fi = feat_name_to_idx.get(feat_name)
            if fi is None or fi not in feat_to_global_ids:
                continue
            gids_t   = torch.tensor(feat_to_global_ids[fi],
                                    dtype=torch.long, device=device)
            val_embs = model.value_embeddings(gids_t)
            probs    = F.softmax((e @ val_embs.T)[0], dim=0)
            pred_local = int(probs.argmax().item())
            preds[feat_name] = (val_list[pred_local], float(probs[pred_local].item()))

    for p in model.parameters():
        p.requires_grad_(True)
    return new_emb.detach(), preds


def fine_tune_tcf(model, init_emb, obs_feat_dict, feat_name_to_idx,
                  feature_groups, feature_value_names,
                  binary_col_names, device, n_steps=300, lr=0.05, l2_reg=0.05):
    """Transductive fine-tuning for TCF model.

    Returns (new_emb_detached, predictions)
    predictions: {feat_name: (pred_value_str, confidence_float)}
    """
    obs_cols, obs_targets = [], []
    for feat_name, val_str in obs_feat_dict.items():
        if feat_name not in feature_groups:
            continue
        col_indices = feature_groups[feat_name]
        vals        = feature_value_names[feat_name]
        if val_str not in vals:
            continue
        if len(col_indices) == 1:
            obs_cols.append(col_indices[0])
            obs_targets.append(1.0 if val_str == vals[1] else 0.0)
        else:
            vi = vals.index(val_str)
            for ci, col_idx in enumerate(col_indices):
                obs_cols.append(col_idx)
                obs_targets.append(1.0 if ci == vi else 0.0)

    new_emb = nn.Parameter(init_emb.clone().to(device))
    for p in model.parameters():
        p.requires_grad_(False)

    if obs_cols:
        obs_cols_t    = torch.tensor(obs_cols,    dtype=torch.long,  device=device)
        obs_targets_t = torch.tensor(obs_targets, dtype=torch.float, device=device)
        optimizer     = torch.optim.Adam([new_emb], lr=lr, weight_decay=0)
        loss_history  = []
        model.eval()
        for step in range(n_steps):
            all_embs   = model.feat_embeddings.weight
            scores     = new_emb @ all_embs.T
            probs      = torch.sigmoid(scores)
            bce        = F.binary_cross_entropy(probs[0, obs_cols_t], obs_targets_t)
            total_loss = bce + (l2_reg / 2.0) * new_emb.pow(2).mean()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lv = total_loss.item()
            loss_history.append(lv)
            if lv < 0.005:
                break
            if step >= 30 and (loss_history[-31] - lv) < 0.001:
                break

    preds = {}
    model.eval()
    with torch.no_grad():
        e        = new_emb.detach()
        all_embs = model.feat_embeddings.weight.detach()
        scores   = (e @ all_embs.T)[0]
        for feat_name, col_indices in feature_groups.items():
            vals       = feature_value_names[feat_name]
            feat_scores = scores[col_indices]
            if len(col_indices) == 1:
                p_val      = torch.sigmoid(feat_scores[0]).item()
                pred_local = 1 if p_val > 0.5 else 0
                conf       = p_val if pred_local == 1 else 1 - p_val
            else:
                feat_probs = F.softmax(feat_scores, dim=0)
                pred_local = int(feat_probs.argmax().item())
                conf       = float(feat_probs[pred_local].item())
            preds[feat_name] = (vals[pred_local], conf)

    for p in model.parameters():
        p.requires_grad_(True)
    return new_emb.detach(), preds


def _raw_probs_learned(model, emb, feat_to_global_ids,
                       feat_name_to_idx, feat_name_to_value_list, device):
    raw = {}
    with torch.no_grad():
        e = emb.to(device)
        for feat_name, val_list in feat_name_to_value_list.items():
            fi = feat_name_to_idx.get(feat_name)
            if fi is None or fi not in feat_to_global_ids:
                continue
            gids_t   = torch.tensor(feat_to_global_ids[fi],
                                    dtype=torch.long, device=device)
            val_embs = model.value_embeddings(gids_t)
            raw[feat_name] = F.softmax((e @ val_embs.T).squeeze(0),
                                       dim=0).cpu().numpy()
    return raw


def _raw_probs_tcf(model, emb, feature_groups, feature_value_names, device):
    raw = {}
    with torch.no_grad():
        e        = emb.to(device)
        all_embs = model.feat_embeddings.weight.detach()
        scores   = (e @ all_embs.T).squeeze(0)
        for feat_name, col_indices in feature_groups.items():
            feat_scores = scores[col_indices]
            if len(col_indices) == 1:
                p_val = torch.sigmoid(feat_scores[0]).item()
                raw[feat_name] = np.array([1 - p_val, p_val])
            else:
                raw[feat_name] = F.softmax(feat_scores, dim=0).cpu().numpy()
    return raw


def infer_iterative(model, model_type, obs_feat_dict, cat_matrix, traindev_feat_df,
                    feat_name_to_idx, feat_to_global_ids, feat_name_to_value_list,
                    feature_groups, feature_value_names,
                    binary_col_names, embed_dim, device,
                    n_rounds=3, n_steps=300, lr=0.05, l2_reg=0.05,
                    confidence_thresholds=(0.95, 0.90, 0.85),
                    prior=None, k_nn=5):
    """Iterative self-training with optional correlation prior on final pass.

    Returns (predictions_dict, n_added_per_round_list, raw_probs_or_None)
    predictions_dict: {feat_name: value_str}
    """
    current_obs      = dict(obs_feat_dict)
    n_added_per_round = []

    for round_idx in range(n_rounds):
        threshold = confidence_thresholds[round_idx]
        init_emb  = knn_warmstart(current_obs, cat_matrix, traindev_feat_df,
                                  feat_name_to_idx, feat_name_to_value_list,
                                  model.lang_embeddings, k=k_nn, device=device)
        if model_type == "learned":
            _, predictions = fine_tune_learned(
                model, init_emb, current_obs, feat_name_to_idx,
                feat_to_global_ids, feat_name_to_value_list,
                device, n_steps, lr, l2_reg)
        else:
            _, predictions = fine_tune_tcf(
                model, init_emb, current_obs, feat_name_to_idx,
                feature_groups, feature_value_names,
                binary_col_names, device, n_steps, lr, l2_reg)

        n_added = 0
        for feat_name, (pred_val, conf) in predictions.items():
            if feat_name not in obs_feat_dict and conf >= threshold:
                current_obs[feat_name] = pred_val
                n_added += 1
        n_added_per_round.append(n_added)

    # Final pass with fully expanded current_obs
    init_emb = knn_warmstart(current_obs, cat_matrix, traindev_feat_df,
                             feat_name_to_idx, feat_name_to_value_list,
                             model.lang_embeddings, k=k_nn, device=device)
    if model_type == "learned":
        final_emb, final_preds = fine_tune_learned(
            model, init_emb, current_obs, feat_name_to_idx,
            feat_to_global_ids, feat_name_to_value_list,
            device, n_steps, lr, l2_reg)
        if prior is not None:
            raw = _raw_probs_learned(model, final_emb, feat_to_global_ids,
                                     feat_name_to_idx, feat_name_to_value_list,
                                     device)
            return (apply_correlation_prior(raw, obs_feat_dict, feat_name_to_idx,
                                            feat_name_to_value_list, prior),
                    n_added_per_round, raw)
    else:
        final_emb, final_preds = fine_tune_tcf(
            model, init_emb, current_obs, feat_name_to_idx,
            feature_groups, feature_value_names,
            binary_col_names, device, n_steps, lr, l2_reg)
        if prior is not None:
            raw = _raw_probs_tcf(model, final_emb, feature_groups,
                                 feature_value_names, device)
            return (apply_correlation_prior(raw, obs_feat_dict, feat_name_to_idx,
                                            feat_name_to_value_list, prior),
                    n_added_per_round, raw)

    return ({f: v for f, (v, _) in final_preds.items()}, n_added_per_round, None)

# ===========================================================================
# PART 6: Feature correlation prior
# ===========================================================================
def build_correlation_prior(cat_matrix, kept_feature_names,
                             feat_to_value_names, min_count=10):
    """Precompute P(fj=vj | fi=vi) for all (fi, vi, fj) with enough data."""
    n_langs, n_feats = cat_matrix.shape
    prior = defaultdict(dict)

    for fi in range(n_feats):
        n_vals_fi = len(feat_to_value_names[fi])
        col_fi    = cat_matrix[:, fi]

        for vi in range(n_vals_fi):
            mask_vi  = (col_fi == vi)
            count_vi = int(mask_vi.sum())
            if count_vi < min_count:
                continue

            for fj in range(n_feats):
                if fi == fj:
                    continue
                n_vals_fj  = len(feat_to_value_names[fj])
                col_fj_sub = cat_matrix[mask_vi, fj]
                obs_mask   = col_fj_sub >= 0
                if int(obs_mask.sum()) < min_count:
                    continue

                col_fj_obs = col_fj_sub[obs_mask].astype(int)
                vj_counts  = np.bincount(col_fj_obs,
                                         minlength=n_vals_fj).astype(np.float64)

                # If any observed value has < 5 co-occurrences, use uniform
                nonzero = vj_counts[vj_counts > 0]
                if len(nonzero) > 0 and np.any(nonzero < 5):
                    probs = np.ones(n_vals_fj) / n_vals_fj
                else:
                    total = vj_counts.sum()
                    probs = vj_counts / total if total > 0 else np.ones(n_vals_fj) / n_vals_fj
                prior[(fi, vi)][fj] = probs

    feat_majority = {}
    for fi in range(n_feats):
        col = cat_matrix[:, fi]
        obs = col[col >= 0]
        if len(obs) > 0:
            feat_majority[fi] = int(Counter(obs.tolist()).most_common(1)[0][0])
        else:
            feat_majority[fi] = 0

    return dict(prior), feat_majority


def apply_correlation_prior(raw_predictions, obs_feat_dict,
                             feat_name_to_idx, feat_name_to_value_list,
                             prior, lam=0.4):
    """Log-linear reranking using feature co-occurrence statistics.

    raw_predictions: {feat_name: np.array of softmax probs}
    obs_feat_dict:   {feat_name: value_str}  — GOLD observations only
    Returns: {feat_name: argmax_value_str after reranking}
    """
    result = {}
    for feat_j_name, raw_probs in raw_predictions.items():
        K_j    = len(raw_probs)
        fj     = feat_name_to_idx.get(feat_j_name)
        val_list_j = feat_name_to_value_list.get(feat_j_name, [])

        log_prior_j = np.zeros(K_j)
        n_terms     = 0

        if fj is not None:
            for feat_i_name, val_i_str in obs_feat_dict.items():
                fi = feat_name_to_idx.get(feat_i_name)
                if fi is None:
                    continue
                val_list_i = feat_name_to_value_list.get(feat_i_name, [])
                if val_i_str not in val_list_i:
                    continue
                vi_local = val_list_i.index(val_i_str)
                key = (fi, vi_local)
                if key not in prior or fj not in prior[key]:
                    continue
                prior_dist = prior[key][fj]
                if len(prior_dist) != K_j:
                    continue
                log_prior_j += np.log(prior_dist + 1e-10)
                n_terms += 1

        if n_terms > 0:
            log_prior_j /= n_terms

        log_model = np.log(raw_probs + 1e-10)
        combined  = log_model + lam * log_prior_j
        combined -= combined.max()
        reranked   = np.exp(combined)
        reranked  /= reranked.sum()

        pred_local = int(np.argmax(reranked))
        result[feat_j_name] = (val_list_j[pred_local]
                               if pred_local < len(val_list_j) else "1")
    return result

# ===========================================================================
# Submission helpers  (identical to v1)
# ===========================================================================
HEADER = "wals_code\tname\tlatitude\tlongitude\tgenus\tfamily\tcountrycodes\tfeatures\n"

def _fmt_features(blanked_set, obs_str_dict, model_preds, freq_preds):
    parts     = []
    all_feats = set(obs_str_dict.keys()) | set(blanked_set)
    for fname in sorted(all_feats):
        if fname not in blanked_set:
            parts.append(f"{fname}={obs_str_dict[fname]}")
        else:
            if fname in model_preds:
                pred_id = model_preds[fname]
            elif fname in freq_preds:
                pred_id = freq_preds[fname]
            else:
                pred_id = "1"
            parts.append(f"{fname}={pred_id} -")
    return "|".join(parts)


def _write_submission(path, test_meta, test_blank, test_obs,
                      preds_by_wals, freq_baseline):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(HEADER)
        for _, row in test_meta.iterrows():
            wals_code    = row["wals_code"]
            blanked_set  = test_blank.get(wals_code, set())
            obs_str_dict = test_obs.get(wals_code, {})
            model_preds  = preds_by_wals.get(wals_code, {})
            feat_str     = _fmt_features(blanked_set, obs_str_dict,
                                         model_preds, freq_baseline)
            meta_str = "\t".join([wals_code, row["name"],
                                   str(row["latitude"]), str(row["longitude"]),
                                   row["genus"], row["family"], row["countrycodes"]])
            fh.write(f"{meta_str}\t{feat_str}\n")


def compute_frequency_baseline(cat_matrix, kept_feature_names, feat_to_value_names):
    freq = {}
    for fi, name in enumerate(kept_feature_names):
        col = cat_matrix[:, fi]
        obs = col[col >= 0]
        if len(obs) == 0:
            freq[name] = feat_to_value_names[fi][0]
        else:
            cnt = Counter(obs.tolist())
            freq[name] = feat_to_value_names[fi][cnt.most_common(1)[0][0]]
    return freq

# ===========================================================================
# Scoring helpers
# ===========================================================================
def _run_scorer(submission_paths):
    """Run modified score.py (macro enabled) on the given files."""
    with open(SCORER, "r", encoding="utf-8") as fh:
        src = fh.read()
    src_mod = src.replace(
        'for mode in ("micro",):  #, "macro"):',
        'for mode in ("micro", "macro"):'
    )
    scripts_dir = os.path.dirname(SCORER)
    tmp_score   = os.path.join(scripts_dir, "_score_v2_tmp.py")
    with open(tmp_score, "w", encoding="utf-8") as fh:
        fh.write(src_mod)
    try:
        result = subprocess.run(
            [sys.executable, tmp_score] + submission_paths,
            capture_output=True, text=True, cwd=scripts_dir,
        )
        return result.stdout, result.stderr
    finally:
        if os.path.exists(tmp_score):
            os.remove(tmp_score)


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
            if name not in results.get(current_mode, {}):
                results.setdefault(current_mode, {})[name] = {}
            if len(parts) > 1:
                results[current_mode][name]["Overall"] = _f(parts[1])
    return results


def _row_vals(name, mode, results):
    d   = results.get(mode, {}).get(name, {})
    row = []
    for col in TABLE_COLS:
        v = d.get(col)
        row.append(f"{v:.2f}" if v is not None else " n/a")
    return row


def _print_ablation_table(rows, results):
    """Print ablation table with v1 reference rows and paper baselines."""
    COL_W = [38] + [7] * 8

    def pr(cells):
        parts = [str(c).ljust(COL_W[0])] + [str(c).rjust(7) for c in cells[1:]]
        print("| " + " | ".join(parts) + " |")

    def sep():
        print("|" + "|".join("-" + "-"*w + "-" for w in COL_W) + "|")

    pr(["System"] + TABLE_HEADER)
    sep()
    # v1 rows (hardcoded)
    for label, vals in V1_ROWS:
        pr([label] + [f"{v:.2f}" for v in vals])
    sep()
    # ablation rows (from scorer)
    for label, sub_basename in rows:
        row = _row_vals(sub_basename, "macro", results)
        pr([label] + row)
    sep()
    pr(["--- SIGTYP 2020 paper (MACRO, reference) ---"] + [""]*8)
    sep()
    for sys_name, vals in PAPER_TABLE:
        pr([sys_name] + [f"{v:.2f}" for v in vals])
    print()


def _print_delta_table(rows, results, ref_label):
    """Print delta over v1 Learned (0.66 overall)."""
    v1_learned_overall = 0.66
    print(f"\nDelta table (vs {ref_label}, overall macro):")
    for label, sub_basename in rows:
        d    = results.get("macro", {}).get(sub_basename, {})
        val  = d.get("Overall")
        if val is not None:
            delta = val - v1_learned_overall
            sign  = "+" if delta >= 0 else ""
            print(f"  {label:<40s}  {val:.2f}  ({sign}{delta:.2f})")

# ===========================================================================
# ========================= MAIN ============================================
# ===========================================================================
print("=" * 72)
print("SIGTYP 2020 v2 — masked pretraining + kNN + iterative + prior")
print("=" * 72)

# ---------------------------------------------------------------------------
# Part 1: Load data
# ---------------------------------------------------------------------------
print("\n[Part 1] Loading SIGTYP data ...")
train_meta, train_feat, train_blank, train_obs = parse_sigtyp(TRAIN_CSV)
dev_meta,   dev_feat,   dev_blank,   dev_obs   = parse_sigtyp(DEV_CSV)
test_meta,  test_feat,  test_blank,  test_obs  = parse_sigtyp(TEST_BLIND)

print(f"  Train: {len(train_meta)} languages, {train_feat.shape[1]} feature columns")
print(f"  Dev:   {len(dev_meta)} languages, {dev_feat.shape[1]} feature columns")
print(f"  Test:  {len(test_meta)} languages")

traindev_feat = pd.concat([train_feat, dev_feat], ignore_index=True)
traindev_meta = pd.concat([train_meta, dev_meta], ignore_index=True)
print(f"  Train+Dev: {len(traindev_feat)} languages, {traindev_feat.shape[1]} feature columns")

# ---------------------------------------------------------------------------
# Part 2: Build shared vocabulary  (identical to v1)
# ---------------------------------------------------------------------------
print("\n[Part 2] Building shared vocabulary ...")
feature_cols = list(traindev_feat.columns)

cat_matrix, kept_feature_names, feat_to_global_ids, feat_to_value_names = \
    prepare_categorical(traindev_feat, feature_cols)

n_features     = len(kept_feature_names)
n_total_values = max(max(gids) for gids in feat_to_global_ids.values()) + 1
print(f"  Kept {n_features} features after filtering (≥2 unique values)")
print(f"  Total unique (feature, value) pairs: {n_total_values}")

binary_matrix, binary_col_names, feature_groups, feature_value_names = \
    binarise_features(traindev_feat, kept_feature_names)
n_binary_cols = binary_matrix.shape[1]
print(f"  Binary columns: {n_binary_cols}")

feat_name_to_idx        = {name: idx for idx, name in enumerate(kept_feature_names)}
feat_name_to_value_list = {name: feat_to_value_names[idx]
                           for idx, name in enumerate(kept_feature_names)}

# ---------------------------------------------------------------------------
# Part 3: Masked-reconstruction pretraining (or load from .pt)
# ---------------------------------------------------------------------------
print("\n[Part 3] Masked-reconstruction pretraining ...")

# Build datasets (same as v1)
ls, fs, vs = [], [], []
for li in range(cat_matrix.shape[0]):
    for fi in range(cat_matrix.shape[1]):
        if cat_matrix[li, fi] >= 0:
            ls.append(li); fs.append(fi); vs.append(cat_matrix[li, fi])
train_ds_learned = CategoricalTypDataset(
    np.array(ls, dtype=np.int64),
    np.array(fs, dtype=np.int64),
    np.array(vs, dtype=np.int64),
)

lang_idxs, col_idxs = np.where(~np.isnan(binary_matrix))
values_bin = binary_matrix[lang_idxs, col_idxs]
train_ds_tcf = WALSDataset(
    lang_idxs.astype(np.int64),
    col_idxs.astype(np.int64),
    values_bin,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

N_LANGS = len(traindev_feat)

# --- Learned model ---
torch.manual_seed(SEED); np.random.seed(SEED)
model_learned = TypologicalMF_Learned(
    n_languages=N_LANGS, n_total_values=n_total_values,
    feat_to_global_ids=feat_to_global_ids, embed_dim=EMBED,
).to(device)

if os.path.exists(MODEL_LEARNED_PT):
    print(f"  Loading Learned model from {MODEL_LEARNED_PT}")
    model_learned.load_state_dict(
        torch.load(MODEL_LEARNED_PT, map_location=device))
    time_learned = 0.0
else:
    print(f"  Training Learned model ({EPOCHS} epochs, 50% mask) ...")
    loader_l = DataLoader(train_ds_learned, batch_size=BATCH,
                          shuffle=True, drop_last=False)
    opt_l    = torch.optim.Adam(model_learned.parameters(), lr=LR, weight_decay=0)
    t0       = time.time()
    for epoch in range(EPOCHS):
        model_learned.train()
        ep_loss, n_batches = 0.0, 0
        for lang_idx, feat_idx, val_idx in loader_l:
            lang_idx = lang_idx.to(device)
            feat_idx = feat_idx.to(device)
            val_idx  = val_idx.to(device)
            keep = torch.rand(len(lang_idx), device=device) > 0.5
            if keep.sum() == 0:
                continue
            log_probs, batch_l2 = model_learned(
                lang_idx[keep], feat_idx[keep], val_idx[keep])
            nll  = -log_probs.mean()
            loss = nll + (L2_TRAIN / 2.0) * batch_l2
            opt_l.zero_grad()
            loss.backward()
            opt_l.step()
            ep_loss  += nll.item()
            n_batches += 1
        avg = ep_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{EPOCHS}  NLL={avg:.4f}")
    time_learned = time.time() - t0
    print(f"  Learned training done in {time_learned:.1f}s")
    torch.save(model_learned.state_dict(), MODEL_LEARNED_PT)
    print(f"  Saved to {MODEL_LEARNED_PT}")

# --- TCF model ---
torch.manual_seed(SEED); np.random.seed(SEED)
model_tcf = TypologicalMF(
    n_languages=N_LANGS, n_features=n_binary_cols, embed_dim=EMBED,
).to(device)

if os.path.exists(MODEL_TCF_PT):
    print(f"  Loading TCF model from {MODEL_TCF_PT}")
    model_tcf.load_state_dict(
        torch.load(MODEL_TCF_PT, map_location=device))
    time_tcf = 0.0
else:
    print(f"  Training TCF model ({EPOCHS} epochs, 50% mask) ...")
    loader_t = DataLoader(train_ds_tcf, batch_size=BATCH,
                          shuffle=True, drop_last=False)
    opt_t    = torch.optim.Adam(model_tcf.parameters(), lr=LR, weight_decay=0)
    t0       = time.time()
    for epoch in range(EPOCHS):
        model_tcf.train()
        ep_loss, n_batches = 0.0, 0
        for lang_idx, feat_idx, vals in loader_t:
            lang_idx = lang_idx.to(device)
            feat_idx = feat_idx.to(device)
            vals     = vals.to(device)
            keep = torch.rand(len(lang_idx), device=device) > 0.5
            if keep.sum() == 0:
                continue
            preds, batch_l2 = model_tcf(lang_idx[keep], feat_idx[keep])
            bce  = F.binary_cross_entropy(preds, vals[keep])
            loss = bce + (L2_TRAIN / 2.0) * batch_l2
            opt_t.zero_grad()
            loss.backward()
            opt_t.step()
            ep_loss  += bce.item()
            n_batches += 1
        avg = ep_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={avg:.4f}")
    time_tcf = time.time() - t0
    print(f"  TCF training done in {time_tcf:.1f}s")
    torch.save(model_tcf.state_dict(), MODEL_TCF_PT)
    print(f"  Saved to {MODEL_TCF_PT}")

model_learned.eval()
model_tcf.eval()

# ---------------------------------------------------------------------------
# Frequency baseline
# ---------------------------------------------------------------------------
freq_baseline = compute_frequency_baseline(
    cat_matrix, kept_feature_names, feat_to_value_names)

# ---------------------------------------------------------------------------
# Part 6: Build correlation prior
# ---------------------------------------------------------------------------
print("\n[Part 6] Building feature correlation prior ...")
t0 = time.time()
prior, feat_majority = build_correlation_prior(
    cat_matrix, kept_feature_names, feat_to_value_names, min_count=10)
print(f"  Done in {time.time()-t0:.1f}s  "
      f"({len(prior)} conditioning (feat,val) pairs with entries)")

# Coverage diagnostic data (filled in during Part 7)
prior_coverage_num   = 0
prior_coverage_denom = 0

# ---------------------------------------------------------------------------
# Part 7: Build obs_feat_dicts once, run all 4 configs
# ---------------------------------------------------------------------------
print("\n[Part 7] Running inference configs A, B, C, D ...")

# Pre-build obs dicts for all test languages
obs_feat_dicts = {}
for row_idx, row in test_meta.iterrows():
    wals_code = row["wals_code"]
    ofd = {}
    if row_idx < len(test_feat):
        for col in test_feat.columns:
            val = test_feat.loc[row_idx, col]
            if pd.notna(val) and col not in test_blank.get(wals_code, set()):
                ofd[col] = str(val)
    obs_feat_dicts[wals_code] = ofd

# Containers: preds[config][model][wals_code] = {feat: val_str}
all_preds = {cfg: {"learned": {}, "tcf": {}}
             for cfg in ("A", "B", "C", "D")}

# Timing
times_per_cfg = {cfg: [] for cfg in ("A", "B", "C", "D")}

# Extra data for D
raw_probs_d = {"learned": {}, "tcf": {}}     # wals_code -> raw_probs dict
n_added_stats = {"C": [], "D": []}           # for diagnostics

N_TEST = len(test_meta)
for row_idx, row in test_meta.iterrows():
    wals_code   = row["wals_code"]
    obs_fd      = obs_feat_dicts[wals_code]

    # ---------- CONFIG A: masked train only, random init, single pass ----------
    t0 = time.time()
    rand_emb_l = torch.zeros(1, EMBED, device=device)
    nn.init.normal_(rand_emb_l, std=0.01)
    _, preds_l = fine_tune_learned(
        model_learned, rand_emb_l, obs_fd, feat_name_to_idx,
        feat_to_global_ids, feat_name_to_value_list,
        device, n_steps=300, lr=0.05, l2_reg=L2_FT)
    rand_emb_t = torch.zeros(1, EMBED, device=device)
    nn.init.normal_(rand_emb_t, std=0.01)
    _, preds_t = fine_tune_tcf(
        model_tcf, rand_emb_t, obs_fd, feat_name_to_idx,
        feature_groups, feature_value_names,
        binary_col_names, device, n_steps=300, lr=0.05, l2_reg=L2_FT)
    times_per_cfg["A"].append(time.time() - t0)
    all_preds["A"]["learned"][wals_code] = {f: v for f, (v, _) in preds_l.items()}
    all_preds["A"]["tcf"][wals_code]     = {f: v for f, (v, _) in preds_t.items()}

    # ---------- CONFIG B: +kNN, single pass ----------
    t0 = time.time()
    knn_emb_l = knn_warmstart(obs_fd, cat_matrix, traindev_feat,
                               feat_name_to_idx, feat_name_to_value_list,
                               model_learned.lang_embeddings, k=5, device=device)
    _, preds_l = fine_tune_learned(
        model_learned, knn_emb_l, obs_fd, feat_name_to_idx,
        feat_to_global_ids, feat_name_to_value_list,
        device, n_steps=300, lr=0.05, l2_reg=L2_FT)
    knn_emb_t = knn_warmstart(obs_fd, cat_matrix, traindev_feat,
                               feat_name_to_idx, feat_name_to_value_list,
                               model_tcf.lang_embeddings, k=5, device=device)
    _, preds_t = fine_tune_tcf(
        model_tcf, knn_emb_t, obs_fd, feat_name_to_idx,
        feature_groups, feature_value_names,
        binary_col_names, device, n_steps=300, lr=0.05, l2_reg=L2_FT)
    times_per_cfg["B"].append(time.time() - t0)
    all_preds["B"]["learned"][wals_code] = {f: v for f, (v, _) in preds_l.items()}
    all_preds["B"]["tcf"][wals_code]     = {f: v for f, (v, _) in preds_t.items()}

    # ---------- CONFIG C: +kNN +iterative, no prior ----------
    t0 = time.time()
    preds_l, n_add_l, _ = infer_iterative(
        model_learned, "learned", obs_fd, cat_matrix, traindev_feat,
        feat_name_to_idx, feat_to_global_ids, feat_name_to_value_list,
        feature_groups, feature_value_names, binary_col_names, EMBED, device,
        n_rounds=3, n_steps=300, lr=0.05, l2_reg=L2_FT,
        confidence_thresholds=(0.95, 0.90, 0.85), prior=None)
    preds_t, n_add_t, _ = infer_iterative(
        model_tcf, "tcf", obs_fd, cat_matrix, traindev_feat,
        feat_name_to_idx, feat_to_global_ids, feat_name_to_value_list,
        feature_groups, feature_value_names, binary_col_names, EMBED, device,
        n_rounds=3, n_steps=300, lr=0.05, l2_reg=L2_FT,
        confidence_thresholds=(0.95, 0.90, 0.85), prior=None)
    times_per_cfg["C"].append(time.time() - t0)
    all_preds["C"]["learned"][wals_code] = preds_l
    all_preds["C"]["tcf"][wals_code]     = preds_t
    n_added_stats["C"].append(n_add_l + n_add_t)   # combined for diagnostic

    # ---------- CONFIG D: +kNN +iterative +prior ----------
    t0 = time.time()
    preds_l, n_add_l, raw_l = infer_iterative(
        model_learned, "learned", obs_fd, cat_matrix, traindev_feat,
        feat_name_to_idx, feat_to_global_ids, feat_name_to_value_list,
        feature_groups, feature_value_names, binary_col_names, EMBED, device,
        n_rounds=3, n_steps=300, lr=0.05, l2_reg=L2_FT,
        confidence_thresholds=(0.95, 0.90, 0.85), prior=prior)
    preds_t, n_add_t, raw_t = infer_iterative(
        model_tcf, "tcf", obs_fd, cat_matrix, traindev_feat,
        feat_name_to_idx, feat_to_global_ids, feat_name_to_value_list,
        feature_groups, feature_value_names, binary_col_names, EMBED, device,
        n_rounds=3, n_steps=300, lr=0.05, l2_reg=L2_FT,
        confidence_thresholds=(0.95, 0.90, 0.85), prior=prior)
    times_per_cfg["D"].append(time.time() - t0)
    all_preds["D"]["learned"][wals_code] = preds_l
    all_preds["D"]["tcf"][wals_code]     = preds_t
    raw_probs_d["learned"][wals_code]    = raw_l
    raw_probs_d["tcf"][wals_code]        = raw_t
    n_added_stats["D"].append(n_add_l + n_add_t)

    # Prior coverage accumulation
    blanked_set = test_blank.get(wals_code, set())
    for feat_j in blanked_set:
        fj = feat_name_to_idx.get(feat_j)
        if fj is None:
            continue
        for feat_i, val_i_str in obs_fd.items():
            fi = feat_name_to_idx.get(feat_i)
            if fi is None:
                continue
            val_list_i = feat_name_to_value_list.get(feat_i, [])
            if val_i_str not in val_list_i:
                continue
            vi_local = val_list_i.index(val_i_str)
            prior_coverage_denom += 1
            key = (fi, vi_local)
            if key in prior and fj in prior[key]:
                prior_coverage_num += 1

    if (row_idx + 1) % 25 == 0 or (row_idx + 1) == N_TEST:
        n_in_vocab = sum(1 for fn in obs_fd if fn in feat_name_to_idx)
        print(f"  {row_idx+1}/{N_TEST} processed  "
              f"(last: {wals_code}, obs_in_vocab={n_in_vocab})")

# ---------------------------------------------------------------------------
# Write submission files
# ---------------------------------------------------------------------------
print("\n[Part 7b] Writing submission files ...")

sub_paths = {}  # key -> path
for cfg in ("A", "B", "C", "D"):
    for mdl in ("learned", "tcf"):
        key  = f"{cfg}_{mdl}"
        path = f"{GB2V}/sigtyp_v2_{key}.tsv"
        _write_submission(path, test_meta, test_blank, test_obs,
                          all_preds[cfg][mdl], freq_baseline)
        sub_paths[key] = path
        print(f"  Wrote {path}")

# Frequency baseline (one file, reused across configs)
freq_path = f"{GB2V}/sigtyp_v2_freq.tsv"
_write_submission(freq_path, test_meta, test_blank, test_obs, {}, freq_baseline)
sub_paths["freq"] = freq_path
print(f"  Wrote {freq_path}")

# ---------------------------------------------------------------------------
# Part 8: Score all submissions
# ---------------------------------------------------------------------------
print("\n[Part 8] Scoring submissions ...")
all_sub_files = [sub_paths[k] for k in sorted(sub_paths)]
stdout, stderr = _run_scorer(all_sub_files)
if stderr:
    print("SCORER STDERR:", stderr[:1000])

results = _parse_scores(stdout)

# Row definitions for ablation table
ABLATION_ROWS = [
    ("[A]  Learned (masked train only)",        os.path.basename(sub_paths["A_learned"]).replace(".tsv","")),
    ("[A]  TCF    (masked train only)",          os.path.basename(sub_paths["A_tcf"]).replace(".tsv","")),
    ("[B]  Learned (+kNN)",                      os.path.basename(sub_paths["B_learned"]).replace(".tsv","")),
    ("[B]  TCF    (+kNN)",                       os.path.basename(sub_paths["B_tcf"]).replace(".tsv","")),
    ("[C]  Learned (+kNN +iterative)",           os.path.basename(sub_paths["C_learned"]).replace(".tsv","")),
    ("[C]  TCF    (+kNN +iterative)",            os.path.basename(sub_paths["C_tcf"]).replace(".tsv","")),
    ("[D]  Learned (+kNN +iterative +prior lam=0.4)", os.path.basename(sub_paths["D_learned"]).replace(".tsv","")),
    ("[D]  TCF    (+kNN +iterative +prior lam=0.4)", os.path.basename(sub_paths["D_tcf"]).replace(".tsv","")),
    ("[Freq] Frequency baseline",               os.path.basename(sub_paths["freq"]).replace(".tsv","")),
]

print("\n[Part 8] Ablation table (MACRO accuracy per genus):\n")
_print_ablation_table(ABLATION_ROWS, results)

# lam tuning: if D < C for either model, try lam=0.2 and lam=0.6
def _get_overall(basename, results, mode="macro"):
    return results.get(mode, {}).get(basename, {}).get("Overall")

c_l_score = _get_overall(os.path.basename(sub_paths["C_learned"]).replace(".tsv",""), results)
c_t_score = _get_overall(os.path.basename(sub_paths["C_tcf"]).replace(".tsv",""),     results)
d_l_score = _get_overall(os.path.basename(sub_paths["D_learned"]).replace(".tsv",""), results)
d_t_score = _get_overall(os.path.basename(sub_paths["D_tcf"]).replace(".tsv",""),     results)

for mdl_key, c_score, d_score, raw_by_wals in [
    ("learned", c_l_score, d_l_score, raw_probs_d["learned"]),
    ("tcf",     c_t_score, d_t_score, raw_probs_d["tcf"]),
]:
    if c_score is not None and d_score is not None and d_score < c_score:
        print(f"  D_{mdl_key} ({d_score:.3f}) < C_{mdl_key} ({c_score:.3f}), "
              f"trying lam=0.2 and lam=0.6 ...")
        best_lam   = 0.4
        best_score = d_score
        for lam in (0.2, 0.6):
            preds_by_wals_lam = {}
            for wc, raw in raw_by_wals.items():
                if raw is None:
                    preds_by_wals_lam[wc] = all_preds["D"][mdl_key][wc]
                else:
                    preds_by_wals_lam[wc] = apply_correlation_prior(
                        raw, obs_feat_dicts[wc], feat_name_to_idx,
                        feat_name_to_value_list, prior, lam=lam)
            path_lam = f"{GB2V}/sigtyp_v2_D_{mdl_key}_lam{int(lam*10)}.tsv"
            _write_submission(path_lam, test_meta, test_blank, test_obs,
                              preds_by_wals_lam, freq_baseline)
            stdout_lam, _ = _run_scorer([path_lam] + [sub_paths[f"C_{mdl_key}"]])
            res_lam  = _parse_scores(stdout_lam)
            bname    = os.path.basename(path_lam).replace(".tsv","")
            new_score = _get_overall(bname, res_lam)
            print(f"    lam={lam}  overall macro = {new_score}")
            if new_score is not None and new_score > best_score:
                best_score = new_score
                best_lam   = lam
                # Update canonical submission
                import shutil
                shutil.copy(path_lam, sub_paths[f"D_{mdl_key}"])
                # Update label
                for i, (lbl, bn) in enumerate(ABLATION_ROWS):
                    if f"[D]" in lbl and mdl_key in lbl.lower():
                        ABLATION_ROWS[i] = (
                            lbl.replace("lam=0.4", f"lam={best_lam}"), bn)
        print(f"    Best lam for D_{mdl_key}: {best_lam} (score={best_score:.3f})")

_print_delta_table(ABLATION_ROWS, results, "[v1] Learned (0.66)")

# Best config report
print("\n[Best system]")
best_label  = ""
best_overall = -1.0
for label, bn in ABLATION_ROWS:
    v = _get_overall(bn, results)
    if v is not None and v > best_overall:
        best_overall = v
        best_label   = label
ufal = 0.75
delta_ufal = best_overall - ufal
sign = "+" if delta_ufal >= 0 else ""
print(f"  {best_label.strip()}")
print(f"  Overall macro: {best_overall:.2f}  "
      f"(vs UFAL 0.75: {sign}{delta_ufal:.2f})")

# ---------------------------------------------------------------------------
# Part 9: Timing and diagnostics
# ---------------------------------------------------------------------------
print("\n[Part 9] Diagnostics")
if time_learned > 0 or time_tcf > 0:
    print(f"  Masked training — Learned: {time_learned:.1f}s   TCF: {time_tcf:.1f}s")
else:
    print("  Models loaded from .pt (no training this run)")

for cfg in ("A", "B", "C", "D"):
    ts = times_per_cfg[cfg]
    print(f"  Config {cfg} inference: mean={np.mean(ts):.2f}s  "
          f"min={np.min(ts):.2f}s  max={np.max(ts):.2f}s  "
          f"total={sum(ts):.1f}s")

print("\n  Pseudo-observations added per round (iterative configs C, D):")
for cfg in ("C", "D"):
    added_lists = n_added_stats[cfg]
    per_lang = [a / 2 for a in added_lists]   # /2 because we summed learned+tcf
    print(f"    Config {cfg}: mean per lang = {np.mean(per_lang):.1f}  "
          f"max = {np.max(per_lang):.0f}  "
          f"(hint: if ≈0, thresholds are too strict; if large, too loose)")

if prior_coverage_denom > 0:
    cov = prior_coverage_num / prior_coverage_denom
    print(f"\n  Correlation prior coverage: {prior_coverage_num}/{prior_coverage_denom} "
          f"({cov:.1%}) of (blanked_feat, obs_feat) pairs had a prior entry")
else:
    print("\n  Correlation prior coverage: N/A (no blanked features)")

print(f"\n  Vocabulary: {n_features} features, {n_total_values} total values, "
      f"{n_binary_cols} binary columns")

print("\n[Done]")
