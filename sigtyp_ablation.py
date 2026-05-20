#!/usr/bin/env python3
"""
SIGTYP 2020 Ablation: value representation × inference method.
Run: python3 sigtyp_ablation.py 2>&1 | tee sigtyp_ablation_log.txt

Rows:
  [1]  UFALNeural   binary cosine+sigmoid + transductive fine-tuning
  [1b] UFALNeural   binary cosine+sigmoid + random embedding (UFAL original behaviour)
  [2]  CategoricalNeural   cosine+softmax + transductive fine-tuning
  [3]  CategoricalGeomNeural  shared value geometry + transductive fine-tuning
  [4]  CrossAttentionSetEncoder  (existing checkpoints, config E)
  [5]  UFAL probabilistic  (Python reimpl of Dan Zeman's Perl system)
"""

import argparse
import os
import sys
import time
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/user/Grambank2Vec")
from model_ufal import (UFALNeural, CategoricalNeural, CategoricalGeomNeural,
                         build_ufal_vocab, build_training_data_ufal,
                         build_training_data_wals_only,
                         train_ufal_model, train_categorical_model,
                         transductive_finetune_emb)
from model_learned import prepare_categorical
from model_set_v4 import CrossAttentionSetEncoder

# ===========================================================================
# Verbatim helpers from sigtyp_eval_v4.py
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
    import subprocess
    with open(scorer_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    src_mod = src.replace(
        'for mode in ("micro",):  #, "macro"):',
        'for mode in ("micro", "macro"):'
    )
    scripts_dir = os.path.dirname(scorer_path)
    tmp_score   = os.path.join(scripts_dir, "_score_abl_tmp.py")
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
    p = argparse.ArgumentParser(description="SIGTYP 2020 ablation")
    p.add_argument("--base",             default="/home/user")
    p.add_argument("--seeds",            nargs="+", type=int,
                   default=[13, 21, 42, 87, 100])
    p.add_argument("--embed-dim",        type=int,   default=64)
    p.add_argument("--n-epochs",         type=int,   default=200)
    p.add_argument("--dropout",          type=float, default=0.5)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--l2-reg",           type=float, default=0.1)
    p.add_argument("--kmeans-clusters",  type=int,   default=300)
    p.add_argument("--force-retrain",    action="store_true")
    p.add_argument("--skip-training",    action="store_true")
    p.add_argument("--output-dir",       default=None)
    p.add_argument("--device",           default=("cuda" if torch.cuda.is_available()
                                                  else "cpu"))
    return p.parse_args()


# ===========================================================================
# PART 3: Calibration helper
# ===========================================================================

def calibrate_temperature(model, model_type, cat_matrix_dev,
                           feat_to_fv_ids_or_gids, feat_to_value_names,
                           kept_feature_names, feat_name_to_idx,
                           embed_dim, device, temp_grid=None):
    """
    Half-split dev: first half as context, second half as targets.
    Optimize NLL over temperature grid [0.5, 0.75, 1.0, 1.5, 2.0, 3.0].
    Returns best T. For UFALNeural: return T=1.0 (no temperature, binary).
    """
    if model_type == 'ufal':
        return 1.0

    if temp_grid is None:
        temp_grid = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    model.eval()
    best_T, best_nll = 1.0, float("inf")

    for T in temp_grid:
        total_nll, n_preds = 0.0, 0

        for lang_idx in range(cat_matrix_dev.shape[0]):
            observed = np.where(cat_matrix_dev[lang_idx] >= 0)[0].tolist()
            if len(observed) < 4:
                continue
            mid = len(observed) // 2
            ctx_feats = observed[:mid]
            tgt_feats = observed[mid:]

            # Build context embedding from first half
            ctx_obs = {}
            for fi in ctx_feats:
                vi = int(cat_matrix_dev[lang_idx, fi])
                fname = kept_feature_names[fi]
                ctx_obs[fname] = feat_to_value_names[fi][vi]

            new_emb = transductive_finetune_emb(
                model, model_type, ctx_obs, feat_name_to_idx, feat_to_value_names,
                feat_to_fv_ids_or_gids, embed_dim, device,
                n_steps=50, lr=0.05, temperature=T)

            with torch.no_grad():
                for fi in tgt_feats:
                    true_local = int(cat_matrix_dev[lang_idx, fi])
                    ids = feat_to_fv_ids_or_gids[fi]
                    fi_t = torch.tensor(ids, dtype=torch.long, device=device)

                    if model_type == 'categorical':
                        fe = model.fv_embeddings(fi_t)
                        logits = (new_emb @ fe.T).squeeze(0)
                    else:  # geom
                        ve = model.val_embeddings(fi_t)
                        logits = (new_emb @ ve.T).squeeze(0)

                    lp = F.log_softmax(logits / T, dim=0)
                    total_nll -= lp[true_local].item()
                    n_preds   += 1

        avg_nll = total_nll / max(n_preds, 1)
        if avg_nll < best_nll:
            best_nll = avg_nll
            best_T   = T

    return best_T


# ===========================================================================
# PART 4: Inference for all features of a test language
# ===========================================================================

def predict_test_language(model, model_type, new_emb, kept_feature_names,
                           feat_name_to_idx, feat_to_value_names,
                           feat_to_fv_ids_or_gids, device, temperature=1.0):
    """
    Predict all features given the transductively-finetuned new_emb.
    Returns dict: feat_name -> predicted_value_str
    """
    results = {}
    model.eval()
    with torch.no_grad():
        le_normed = F.normalize(new_emb, dim=-1)
        for fi, fname in enumerate(kept_feature_names):
            ids  = feat_to_fv_ids_or_gids[fi]
            fi_t = torch.tensor(ids, dtype=torch.long, device=device)

            if model_type == 'ufal':
                fe    = F.normalize(model.fv_embeddings(fi_t), dim=-1)
                cos   = (le_normed.expand(len(ids), -1) * fe).sum(-1, keepdim=True)
                probs = torch.sigmoid(model.output_layer(cos)).squeeze(-1).cpu().numpy()
                pred_local = int(probs.argmax())
            elif model_type == 'categorical':
                fe     = model.fv_embeddings(fi_t)
                logits = (new_emb @ fe.T).squeeze(0)
                pred_local = F.log_softmax(logits / temperature, dim=0).argmax().item()
            else:  # geom
                ve     = model.val_embeddings(fi_t)
                logits = (new_emb @ ve.T).squeeze(0)
                pred_local = F.log_softmax(logits / temperature, dim=0).argmax().item()

            results[fname] = feat_to_value_names[fi][pred_local]
    return results


# ===========================================================================
# PART 5b: UFAL probabilistic reimplementation
# ===========================================================================

def _lat_zone(lat):
    # Perl @lat = (90, 60, 35, 10, -10, -90) → 5 zones north→south
    if   lat > 60:  return "lat_0"   # Arctic/Subarctic
    elif lat > 35:  return "lat_1"   # Northern temperate
    elif lat > 10:  return "lat_2"   # Northern subtropical
    elif lat > -10: return "lat_3"   # Equatorial
    else:           return "lat_4"   # Southern hemisphere


# Perl @lon = (160, -140, -115, -95, -60, -25, 35, 70, 95, 118, 130, 160)
# 11 zones; zone 0 wraps over the International Date Line (lon ≥ 160 or lon < -140).
_LON_UPPER = [-140, -115, -95, -60, -25, 35, 70, 95, 118, 130, 160]

def _lon_zone(lon):
    if lon >= 160 or lon < -140:
        return "lon_0"
    for i, upper in enumerate(_LON_UPPER):
        if lon < upper:
            return f"lon_{i + 1}"
    return "lon_0"  # fallback


def _latlon_zone(lat, lon):
    return f"{_lat_zone(lat)}_{_lon_zone(lon)}"


def _get_source_features(meta_row, obs_feat_dict, feat_name_to_idx):
    """Build source feature dict for a language: WALS obs + metadata."""
    sources = {}
    # Observed WALS features
    for fname, vid in obs_feat_dict.items():
        sources[("wals", fname)] = vid
    # Metadata
    genus  = str(meta_row.get("genus", ""))
    family = str(meta_row.get("family", ""))
    lat    = float(meta_row.get("latitude",  0.0) or 0.0)
    lon    = float(meta_row.get("longitude", 0.0) or 0.0)
    cc     = str(meta_row.get("countrycodes", ""))

    if genus:
        sources[("meta", "genus")] = genus
    if family:
        sources[("meta", "family")] = family
    # lat/lon zones always included; countrycodes not used as a source feature
    # (UFAL default config{countrycodes}='' means US codes are not excluded,
    # but we omit countrycodes entirely as a feature since the task specifies
    # treating US as unknown/unreliable)
    sources[("meta", "latzone")] = _lat_zone(lat)
    sources[("meta", "lonzone")] = _lon_zone(lon)
    sources[("meta", "latlon")]  = _latlon_zone(lat, lon)

    return sources


def _build_cooccurrence(all_lang_sources, all_lang_targets, feat_names_idx):
    """
    Build co-occurrence tables for probabilistic model.
    Returns:
      joint_counts[(src_key, src_val, tgt_feat, tgt_val)] = count
      src_counts[(src_key, src_val)] = count of LANGUAGES with that source value
      tgt_marginal[tgt_feat][tgt_val] = count
    """
    joint_counts  = defaultdict(int)
    src_counts    = defaultdict(int)
    tgt_marginal  = defaultdict(lambda: defaultdict(int))

    for lang_src, lang_tgt in zip(all_lang_sources, all_lang_targets):
        # Count source features once per language (not once per target feature).
        # P(tj=y|si=x) = c(si=x, tj=y) / c(si=x) where c(si=x) counts all
        # languages with si=x, regardless of whether tj is observed.
        for src_key, src_val in lang_src.items():
            src_counts[(src_key, src_val)] += 1

        # Joint counts and target marginals
        for tgt_feat, tgt_val in lang_tgt.items():
            tgt_marginal[tgt_feat][tgt_val] += 1
            for src_key, src_val in lang_src.items():
                joint_counts[(src_key, src_val, tgt_feat, tgt_val)] += 1

    return joint_counts, src_counts, tgt_marginal


def _entropy(counts_dict):
    """Shannon entropy of a distribution given as {value: count}."""
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts_dict.values():
        if c > 0:
            p = c / total
            h -= p * np.log(p)
    return h


def _compute_pairwise_information(joint_counts):
    """
    Compute information[(src_key, tgt_feat)] over all co-observed pairs.

    Perl formula: information{f}{g} = fgventropy{f}{g} - centropy{f}{g}
      = H(g | f present) - H(g | f value)
      = entropy of target g over languages where source f is present (any value)
        minus the conditional entropy of g given f's specific value.

    Both quantities are computed only over languages where BOTH f and g are
    observed (i.e., from the joint counts). This measures how much knowing f's
    specific value (vs. just knowing it is observed) reduces uncertainty about g.
    """
    # For each (src_key, tgt_feat): collect marginal tgt distribution (over all
    # src_val) and per-src_val distributions.
    present_tgt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    per_val     = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

    for (src_key, src_val, tgt_feat, tgt_val), count in joint_counts.items():
        present_tgt[src_key][tgt_feat][tgt_val]         += count
        per_val[src_key][src_val][tgt_feat][tgt_val]    += count

    information = {}
    for src_key in present_tgt:
        for tgt_feat in present_tgt[src_key]:
            h_present = _entropy(present_tgt[src_key][tgt_feat])

            total_present = sum(present_tgt[src_key][tgt_feat].values())
            h_cond = 0.0
            if total_present > 0:
                for src_val in per_val[src_key]:
                    if tgt_feat not in per_val[src_key][src_val]:
                        continue
                    tgt_c   = per_val[src_key][src_val][tgt_feat]
                    total_v = sum(tgt_c.values())
                    h_cond += (total_v / total_present) * _entropy(tgt_c)

            information[(src_key, tgt_feat)] = max(0.0, h_present - h_cond)

    return information


def _plogcinf_score(src_key, src_val, tgt_feat, tgt_val,
                    joint_counts, src_counts, information):
    """
    Perl formula: plogcinf = plogc × information(si, tj)
      plogc = P(tj=y | si=x) × log(c(si=x, tj=y))
      information(si, tj) = H(tj | si present) - H(tj | si value)

    P(tj=y|si=x) = c(si=x, tj=y) / c(si=x)
    c(si=x) = number of languages with si=x (regardless of whether tj is observed).
    """
    c_joint = joint_counts.get((src_key, src_val, tgt_feat, tgt_val), 0)
    if c_joint == 0:
        return 0.0
    c_src = src_counts.get((src_key, src_val), 0)
    if c_src == 0:
        return 0.0
    info = information.get((src_key, tgt_feat), 0.0)
    if info == 0.0:
        return 0.0
    p_cond = c_joint / c_src
    return p_cond * np.log(c_joint) * info


def run_ufal_probabilistic(train_meta, train_feat_df, dev_meta, dev_feat_df,
                            test_meta, test_blank, test_obs,
                            cat_matrix_traindev, kept_feature_names,
                            feat_to_value_names, feat_name_to_idx,
                            freq_fallback):
    """
    Python reimplementation of UFAL's train_and_predict_dz.pl.
    score = plogcinf, model = strongest (single best source feature).
    train_on_dev = blind, train_on_test = yes.
    """
    print("  Building UFAL probabilistic co-occurrence tables ...")

    # Combine traindev + visible test features for co-occurrence
    # (train_on_dev=blind, train_on_test=yes)
    all_sources = []
    all_targets = []

    def _lang_targets_from_row(meta_row_dict, feat_df_row, feat_name_to_idx, feat_to_value_names):
        tgt = {}
        for fname, fi in feat_name_to_idx.items():
            v = feat_df_row.get(fname)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            v_str = str(v)
            if v_str in feat_to_value_names[fi]:
                tgt[(("wals", fname))] = v_str
        return tgt

    # Train + dev languages
    all_meta = pd.concat([train_meta, dev_meta], ignore_index=True)
    for i, row in all_meta.iterrows():
        meta_dict = row.to_dict()
        fi_row = i
        # Build obs from cat_matrix (full observed features)
        obs = {}
        for fname, fidx in feat_name_to_idx.items():
            vi = cat_matrix_traindev[fi_row, fidx] if fi_row < cat_matrix_traindev.shape[0] else -1
            if vi >= 0:
                obs[fname] = feat_to_value_names[fidx][vi]
        src = _get_source_features(meta_dict, obs, feat_name_to_idx)
        tgt = {("wals", fn): obs[fn] for fn in obs}
        all_sources.append(src)
        all_targets.append(tgt)

    # Test languages: add visible (non-blanked) features
    for i, row in test_meta.iterrows():
        wc = row["wals_code"]
        meta_dict = row.to_dict()
        obs_raw = test_obs.get(wc, {})
        obs = {k: _strip_to_id(v) for k, v in obs_raw.items()
               if k in feat_name_to_idx}
        src = _get_source_features(meta_dict, obs, feat_name_to_idx)
        tgt = {("wals", fn): v for fn, v in obs.items()}
        all_sources.append(src)
        all_targets.append(tgt)

    joint_counts, src_counts, tgt_marginal = _build_cooccurrence(
        all_sources, all_targets, feat_name_to_idx)

    # Compute pairwise information scores: H(tgt|src_present) - H(tgt|src_value)
    print("  Computing pairwise information scores ...")
    information = _compute_pairwise_information(joint_counts)
    print(f"  {len(information)} (src_feat, tgt_feat) information pairs computed.")

    # Predict for each test language
    preds = {}
    for i, row in test_meta.iterrows():
        wc = row["wals_code"]
        meta_dict = row.to_dict()
        obs_raw = test_obs.get(wc, {})
        obs = {k: _strip_to_id(v) for k, v in obs_raw.items()
               if k in feat_name_to_idx}
        test_sources = _get_source_features(meta_dict, obs, feat_name_to_idx)
        blanked = test_blank.get(wc, set())
        lang_preds = {}

        for tgt_fname in blanked:
            fi = feat_name_to_idx.get(tgt_fname)
            if fi is None:
                continue
            tgt_values = feat_to_value_names[fi]
            tgt_feat   = ("wals", tgt_fname)

            # Find single BEST source feature (strongest model):
            # best source = the one whose best-scoring target value has the
            # highest plogcinf score.
            best_src_key    = None
            best_src_val    = None
            best_max_score  = -1.0

            for src_key, src_val in test_sources.items():
                max_score_for_src = max(
                    _plogcinf_score(src_key, src_val, tgt_feat, tv,
                                    joint_counts, src_counts, information)
                    for tv in tgt_values
                )
                if max_score_for_src > best_max_score:
                    best_max_score = max_score_for_src
                    best_src_key   = src_key
                    best_src_val   = src_val

            if best_src_key is not None and best_max_score > 0.0:
                scores = [
                    _plogcinf_score(best_src_key, best_src_val, tgt_feat, tv,
                                    joint_counts, src_counts, information)
                    for tv in tgt_values
                ]
                pred_val = tgt_values[int(np.argmax(scores))]
            else:
                # Fallback: marginal mode (Perl leaves '?', we must output something)
                tgt_cnt = tgt_marginal.get(tgt_feat, {})
                if tgt_cnt:
                    pred_val = max(tgt_cnt, key=tgt_cnt.get)
                else:
                    pred_val = freq_fallback.get(tgt_fname, tgt_values[0])

            lang_preds[tgt_fname] = pred_val

        preds[wc] = lang_preds

    return preds


# ===========================================================================
# Bootstrap test helper
# ===========================================================================

def _compute_per_lang_acc(preds_by_lang, test_meta, test_blank,
                           gold_obs, feat_name_to_idx):
    per_lang = []
    for _, row in test_meta.iterrows():
        wc      = row["wals_code"]
        genus   = row["genus"]
        blanked = test_blank.get(wc, set())
        pred    = preds_by_lang.get(wc, {})
        gold    = gold_obs.get(wc, {})
        correct = total = 0
        for fn in blanked:
            if fn not in feat_name_to_idx:
                continue
            gv_raw = gold.get(fn)
            if gv_raw is None:
                continue
            # gold_obs from parse_sigtyp contains full strings e.g. "5 SOV";
            # model predictions use stripped IDs e.g. "5" from feat_to_value_names.
            gv = _strip_to_id(gv_raw)
            pv = pred.get(fn)
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


def bootstrap_delta(preds_a, preds_b, test_meta, test_blank, gold_obs,
                    feat_name_to_idx, n_boot=1000, rng_seed=0):
    """Bootstrap test for difference in macro accuracy between two systems."""
    pl_a = _compute_per_lang_acc(preds_a, test_meta, test_blank, gold_obs, feat_name_to_idx)
    pl_b = _compute_per_lang_acc(preds_b, test_meta, test_blank, gold_obs, feat_name_to_idx)
    obs_delta = _macro_acc(pl_a) - _macro_acc(pl_b)
    n = min(len(pl_a), len(pl_b))
    if n == 0:
        return obs_delta, (0.0, 0.0), 1.0
    np.random.seed(rng_seed)
    boot_deltas = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        da  = _macro_acc([pl_a[i] for i in idx if i < len(pl_a)])
        db  = _macro_acc([pl_b[i] for i in idx if i < len(pl_b)])
        boot_deltas.append(da - db)
    bd = np.array(boot_deltas)
    ci = (float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5)))
    p  = float(np.mean(np.abs(bd) >= abs(obs_delta)))
    if p == 0.0:
        p = 1.0 / n_boot
    return obs_delta, ci, p


# ===========================================================================
# Value geometry probe
# ===========================================================================

def print_value_geometry_probe(model, model_type, kept_feature_names,
                                feat_to_value_names, feat_to_fv_ids_or_gids,
                                feat_name_to_idx, device, top_n=5):
    """
    For the given model, print top-5 nearest neighbors in value embedding space
    for anchor values. Cross-feature neighbors allowed (for geom and crossattn).
    """
    if model_type == 'geom':
        weight = model.val_embeddings.weight.detach().cpu().numpy()
        all_gids = list(range(weight.shape[0]))
    elif model_type == 'crossattn':
        weight = model.val_embeddings.weight.detach().cpu().numpy()
        all_gids = list(range(weight.shape[0]))
    else:
        return

    # Build reverse map: global_id -> (feat_name, val_name)
    gid_to_label = {}
    for fi, fname in enumerate(kept_feature_names):
        gids = feat_to_fv_ids_or_gids[fi]
        for local_i, gid in enumerate(gids):
            vname = feat_to_value_names[fi][local_i]
            gid_to_label[gid] = f"{fname}={vname}"

    def find_anchor(target_feat_substr, target_val_substr):
        for fi, fname in enumerate(kept_feature_names):
            if target_feat_substr.lower() in fname.lower():
                gids = feat_to_fv_ids_or_gids[fi]
                for local_i, gid in enumerate(gids):
                    vname = feat_to_value_names[fi][local_i]
                    if target_val_substr.lower() in str(vname).lower():
                        return gid, fname, vname
        return None, None, None

    anchors = [
        ("order_of_subject", "sov"),
        ("order_of_subject", "svo"),
        ("tone", "no_tone"),
    ]

    print(f"\n[Value geometry probe — model={model_type}]")
    for feat_sub, val_sub in anchors:
        gid, fname, vname = find_anchor(feat_sub, val_sub)
        if gid is None:
            print(f"  Anchor not found: feat~'{feat_sub}', val~'{val_sub}'")
            continue
        anchor_emb = weight[gid]
        anchor_norm = anchor_emb / (np.linalg.norm(anchor_emb) + 1e-8)

        cos_sims = []
        for g in all_gids:
            if g == gid:
                continue
            e = weight[g]
            n = np.linalg.norm(e) + 1e-8
            cos_sims.append((g, float(np.dot(anchor_norm, e / n))))
        cos_sims.sort(key=lambda x: -x[1])
        print(f"\n  Anchor: {fname}={vname} (gid={gid})")
        for g, sim in cos_sims[:top_n]:
            label = gid_to_label.get(g, f"gid={g}")
            print(f"    {sim:+.4f}  {label}")


# ===========================================================================
# Blanking ratio vs accuracy correlation
# ===========================================================================

def blanking_ratio_pearson(preds_by_lang, test_meta, test_blank,
                            gold_obs, feat_name_to_idx):
    """Compute Pearson R between blanking ratio and per-language accuracy."""
    blank_ratios = []
    accs         = []
    for _, row in test_meta.iterrows():
        wc      = row["wals_code"]
        blanked = test_blank.get(wc, set())
        obs_fn  = {k for k in test_blank.get(wc, set())}  # blanked
        # total features = blanked + observed
        n_obs   = sum(1 for fn in feat_name_to_idx if fn not in blanked)
        n_all   = len(feat_name_to_idx)
        blank_r = len(blanked) / max(n_all, 1)

        pred = preds_by_lang.get(wc, {})
        gold = gold_obs.get(wc, {})
        correct = total = 0
        for fn in blanked:
            if fn not in feat_name_to_idx:
                continue
            gv_raw = gold.get(fn)
            if gv_raw is None:
                continue
            gv = _strip_to_id(gv_raw)
            pv = pred.get(fn)
            total   += 1
            if pv is not None and pv == gv:
                correct += 1
        if total > 0:
            blank_ratios.append(blank_r)
            accs.append(correct / total)

    if len(blank_ratios) < 3:
        return float("nan")
    from scipy.stats import pearsonr
    r, _ = pearsonr(blank_ratios, accs)
    return float(r)


# ===========================================================================
# MAIN
# ===========================================================================

def main(args):
    base      = args.base
    train_csv = f"{base}/ST2020/data/train.csv"
    dev_csv   = f"{base}/ST2020/data/dev.csv"
    test_blind= f"{base}/ST2020/data/test_blinded.csv"
    test_gold = f"{base}/ST2020/data/test_gold.csv"
    scorer    = f"{base}/ST2020/scripts/score.py"
    gb2v      = f"{base}/Grambank2Vec"
    out_dir   = args.output_dir or gb2v

    print("=" * 72)
    print("SIGTYP 2020 — Ablation: value representation")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # PART 1: Data loading
    # -----------------------------------------------------------------------
    print("\n[Part 1] Loading SIGTYP data ...")
    train_meta, train_feat, train_blank, train_obs = parse_sigtyp(train_csv)
    dev_meta,   dev_feat,   dev_blank,   dev_obs   = parse_sigtyp(dev_csv)
    test_meta,  test_feat,  test_blank,  test_obs  = parse_sigtyp(test_blind)

    n_train = len(train_meta)
    n_dev   = len(dev_meta)
    print(f"  Train: {n_train}, Dev: {n_dev}, Test: {len(test_meta)}")

    traindev_feat = pd.concat([train_feat, dev_feat], ignore_index=True)
    traindev_meta = pd.concat([train_meta, dev_meta], ignore_index=True)

    # -----------------------------------------------------------------------
    # PART 2: Vocabulary
    # -----------------------------------------------------------------------
    print("\n[Part 2] Building vocabulary ...")
    feature_cols = list(traindev_feat.columns)
    (cat_matrix_traindev, kept_feature_names,
     feat_to_global_ids, feat_to_value_names) = prepare_categorical(
        traindev_feat, feature_cols)

    cat_matrix_train = cat_matrix_traindev[:n_train]
    cat_matrix_dev   = cat_matrix_traindev[n_train:]

    n_features     = len(kept_feature_names)
    n_total_values = max(max(gids) for gids in feat_to_global_ids.values()) + 1
    feat_name_to_idx = {name: i for i, name in enumerate(kept_feature_names)}

    print(f"  {n_features} features, {n_total_values} total value IDs")

    # UFAL vocab: global fv IDs including metadata
    print("  Building UFAL vocab ...")
    (feat_to_fv_ids, metadata_fv_maps,
     n_global_fv_ids, km) = build_ufal_vocab(
        cat_matrix_traindev, kept_feature_names, feat_to_value_names,
        traindev_meta, kmeans_clusters=args.kmeans_clusters)

    cluster_id_offset = metadata_fv_maps['cluster_id_offset']
    print(f"  UFAL global fv IDs: {n_global_fv_ids}")

    # Positive pairs for UFALNeural training
    print("  Building UFAL training positives ...")
    positives = build_training_data_ufal(
        cat_matrix_traindev, traindev_meta, feat_to_fv_ids,
        metadata_fv_maps, km, cluster_id_offset, feat_to_value_names)
    print(f"  {len(positives)} positive pairs")

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

    n_langs_traindev = len(traindev_meta)
    embed_dim        = args.embed_dim

    # Storage for all predictions (for scoring and ablation table)
    all_preds = {}   # key -> {wc -> {fn -> pred_val_str}}

    # -----------------------------------------------------------------------
    # PART 5: Seed loop
    # -----------------------------------------------------------------------
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"[Seed {seed}]")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # -------------------------------------------------------------------
        # ROW 1: UFALNeural
        # -------------------------------------------------------------------
        ckpt_ufal = f"{out_dir}/ufal_neural_seed{seed}.pt"
        model_ufal = None

        if os.path.exists(ckpt_ufal) and not args.force_retrain:
            try:
                ckpt = torch.load(ckpt_ufal, map_location="cpu")
                model_ufal = UFALNeural(
                    n_langs_traindev, n_global_fv_ids,
                    embed_dim=embed_dim, dropout_rate=args.dropout)
                model_ufal.load_state_dict(ckpt["state_dict"])
                print(f"  [1] Loaded UFALNeural from {ckpt_ufal}")
            except Exception as e:
                print(f"  [1] Checkpoint load failed ({e}), retraining ...")
                model_ufal = None

        if model_ufal is None:
            if args.skip_training:
                print(f"  [1] --skip-training set; skipping UFALNeural seed {seed}")
            else:
                print(f"  [1] Training UFALNeural ...")
                t0 = time.time()
                model_ufal = UFALNeural(
                    n_langs_traindev, n_global_fv_ids,
                    embed_dim=embed_dim, dropout_rate=args.dropout)
                train_ufal_model(
                    model_ufal, positives, n_global_fv_ids,
                    n_epochs=args.n_epochs, lr=args.lr, l2_reg=args.l2_reg,
                    device=args.device, seed=seed)
                torch.save({"state_dict": model_ufal.state_dict(),
                            "seed": seed, "n_epochs": args.n_epochs},
                           ckpt_ufal)
                print(f"  [1] Trained in {time.time()-t0:.1f}s, saved to {ckpt_ufal}")

        if model_ufal is not None:
            model_ufal.to(args.device)
            model_ufal.eval()

            # ROW 1: transductive fine-tuning
            preds_ufal_trans = {}
            # ROW 1b: random embedding (UFAL original behaviour)
            preds_ufal_random = {}

            print(f"  [1/1b] Inference over {len(test_meta)} test languages ...")
            for _, row in test_meta.iterrows():
                wc      = row["wals_code"]
                blanked = test_blank.get(wc, set())
                obs_raw = test_obs.get(wc, {})
                obs_fd  = {k: _strip_to_id(v) for k, v in obs_raw.items()
                           if k in feat_name_to_idx}

                # [1] Transductive
                new_emb = transductive_finetune_emb(
                    model_ufal, 'ufal', obs_fd, feat_name_to_idx,
                    feat_to_value_names, feat_to_fv_ids, embed_dim,
                    args.device, n_steps=200, lr=0.05, temperature=1.0)
                preds_ufal_trans[wc] = predict_test_language(
                    model_ufal, 'ufal', new_emb, kept_feature_names,
                    feat_name_to_idx, feat_to_value_names,
                    feat_to_fv_ids, args.device, temperature=1.0)

                # [1b] Random (zero init, no optimization)
                rand_emb = torch.zeros(1, embed_dim, device=args.device)
                nn.init.normal_(rand_emb, std=0.01)
                preds_ufal_random[wc] = predict_test_language(
                    model_ufal, 'ufal', rand_emb, kept_feature_names,
                    feat_name_to_idx, feat_to_value_names,
                    feat_to_fv_ids, args.device, temperature=1.0)

            path_ufal = f"{out_dir}/sigtyp_abl_ufal_seed{seed}.tsv"
            _write_submission(path_ufal, test_meta, test_blank, test_obs,
                              preds_ufal_trans, freq_fallback)
            print(f"  [1] Wrote {path_ufal}")

            path_ufal_rand = f"{out_dir}/sigtyp_abl_ufal_random_seed{seed}.tsv"
            _write_submission(path_ufal_rand, test_meta, test_blank, test_obs,
                              preds_ufal_random, freq_fallback)
            print(f"  [1b] Wrote {path_ufal_rand}")

            all_preds[f"ufal_seed{seed}"]        = preds_ufal_trans
            all_preds[f"ufal_random_seed{seed}"] = preds_ufal_random

        # -------------------------------------------------------------------
        # ROW 2: CategoricalNeural
        # -------------------------------------------------------------------
        ckpt_cat = f"{out_dir}/cat_neural_seed{seed}.pt"
        model_cat = None

        if os.path.exists(ckpt_cat) and not args.force_retrain:
            try:
                ckpt = torch.load(ckpt_cat, map_location="cpu")
                model_cat = CategoricalNeural(
                    n_langs_traindev, n_global_fv_ids, feat_to_fv_ids,
                    embed_dim=embed_dim, dropout_rate=args.dropout)
                model_cat.load_state_dict(ckpt["state_dict"])
                print(f"  [2] Loaded CategoricalNeural from {ckpt_cat}")
            except Exception as e:
                print(f"  [2] Checkpoint load failed ({e}), retraining ...")
                model_cat = None

        if model_cat is None:
            if args.skip_training:
                print(f"  [2] --skip-training set; skipping CategoricalNeural seed {seed}")
            else:
                print(f"  [2] Training CategoricalNeural ...")
                t0 = time.time()
                model_cat = CategoricalNeural(
                    n_langs_traindev, n_global_fv_ids, feat_to_fv_ids,
                    embed_dim=embed_dim, dropout_rate=args.dropout)
                train_categorical_model(
                    model_cat, cat_matrix_traindev,
                    n_epochs=args.n_epochs, lr=args.lr, l2_reg=args.l2_reg,
                    device=args.device, seed=seed)
                torch.save({"state_dict": model_cat.state_dict(),
                            "seed": seed, "n_epochs": args.n_epochs},
                           ckpt_cat)
                print(f"  [2] Trained in {time.time()-t0:.1f}s, saved to {ckpt_cat}")

        if model_cat is not None:
            model_cat.to(args.device)
            model_cat.eval()
            print("  [2] Calibrating temperature ...")
            T_cat = calibrate_temperature(
                model_cat, 'categorical', cat_matrix_dev,
                feat_to_fv_ids, feat_to_value_names,
                kept_feature_names, feat_name_to_idx,
                embed_dim, args.device)
            print(f"  [2] T_cat = {T_cat}")

            preds_cat = {}
            print(f"  [2] Inference over {len(test_meta)} test languages ...")
            for _, row in test_meta.iterrows():
                wc      = row["wals_code"]
                obs_raw = test_obs.get(wc, {})
                obs_fd  = {k: _strip_to_id(v) for k, v in obs_raw.items()
                           if k in feat_name_to_idx}
                new_emb = transductive_finetune_emb(
                    model_cat, 'categorical', obs_fd, feat_name_to_idx,
                    feat_to_value_names, feat_to_fv_ids, embed_dim,
                    args.device, n_steps=200, lr=0.05, temperature=T_cat)
                preds_cat[wc] = predict_test_language(
                    model_cat, 'categorical', new_emb, kept_feature_names,
                    feat_name_to_idx, feat_to_value_names,
                    feat_to_fv_ids, args.device, temperature=T_cat)

            path_cat = f"{out_dir}/sigtyp_abl_cat_seed{seed}.tsv"
            _write_submission(path_cat, test_meta, test_blank, test_obs,
                              preds_cat, freq_fallback)
            print(f"  [2] Wrote {path_cat}")
            all_preds[f"cat_seed{seed}"] = preds_cat

        # -------------------------------------------------------------------
        # ROW 3: CategoricalGeomNeural
        # -------------------------------------------------------------------
        ckpt_geom = f"{out_dir}/cat_geom_seed{seed}.pt"
        model_geom = None

        if os.path.exists(ckpt_geom) and not args.force_retrain:
            try:
                ckpt = torch.load(ckpt_geom, map_location="cpu")
                model_geom = CategoricalGeomNeural(
                    n_langs_traindev, n_total_values, feat_to_global_ids,
                    embed_dim=embed_dim, dropout_rate=args.dropout)
                model_geom.load_state_dict(ckpt["state_dict"])
                print(f"  [3] Loaded CategoricalGeomNeural from {ckpt_geom}")
            except Exception as e:
                print(f"  [3] Checkpoint load failed ({e}), retraining ...")
                model_geom = None

        if model_geom is None:
            if args.skip_training:
                print(f"  [3] --skip-training set; skipping CategoricalGeomNeural seed {seed}")
            else:
                print(f"  [3] Training CategoricalGeomNeural ...")
                t0 = time.time()
                model_geom = CategoricalGeomNeural(
                    n_langs_traindev, n_total_values, feat_to_global_ids,
                    embed_dim=embed_dim, dropout_rate=args.dropout)
                train_categorical_model(
                    model_geom, cat_matrix_traindev,
                    n_epochs=args.n_epochs, lr=args.lr, l2_reg=args.l2_reg,
                    device=args.device, seed=seed)
                torch.save({"state_dict": model_geom.state_dict(),
                            "seed": seed, "n_epochs": args.n_epochs},
                           ckpt_geom)
                print(f"  [3] Trained in {time.time()-t0:.1f}s, saved to {ckpt_geom}")

        if model_geom is not None:
            model_geom.to(args.device)
            model_geom.eval()
            print("  [3] Calibrating temperature ...")
            T_geom = calibrate_temperature(
                model_geom, 'geom', cat_matrix_dev,
                feat_to_global_ids, feat_to_value_names,
                kept_feature_names, feat_name_to_idx,
                embed_dim, args.device)
            print(f"  [3] T_geom = {T_geom}")

            preds_geom = {}
            print(f"  [3] Inference over {len(test_meta)} test languages ...")
            for _, row in test_meta.iterrows():
                wc      = row["wals_code"]
                obs_raw = test_obs.get(wc, {})
                obs_fd  = {k: _strip_to_id(v) for k, v in obs_raw.items()
                           if k in feat_name_to_idx}
                new_emb = transductive_finetune_emb(
                    model_geom, 'geom', obs_fd, feat_name_to_idx,
                    feat_to_value_names, feat_to_global_ids, embed_dim,
                    args.device, n_steps=200, lr=0.05, temperature=T_geom)
                preds_geom[wc] = predict_test_language(
                    model_geom, 'geom', new_emb, kept_feature_names,
                    feat_name_to_idx, feat_to_value_names,
                    feat_to_global_ids, args.device, temperature=T_geom)

            path_geom = f"{out_dir}/sigtyp_abl_geom_seed{seed}.tsv"
            _write_submission(path_geom, test_meta, test_blank, test_obs,
                              preds_geom, freq_fallback)
            print(f"  [3] Wrote {path_geom}")
            all_preds[f"geom_seed{seed}"] = preds_geom

        # -------------------------------------------------------------------
        # ROW 4: CrossAttentionSetEncoder (existing checkpoint, config E)
        # -------------------------------------------------------------------
        ckpt_crossattn = f"{gb2v}/set_encoder_v4_seed{seed}.pt"
        crossattn_path = f"{out_dir}/sigtyp_abl_crossattn_seed{seed}.tsv"

        if os.path.exists(crossattn_path) and not args.force_retrain:
            print(f"  [4] CrossAttn output already exists: {crossattn_path}")
        elif os.path.exists(ckpt_crossattn):
            try:
                ckpt = torch.load(ckpt_crossattn, map_location="cpu")
                hidden_dim = ckpt.get("hidden_dim", 128)
                model_ca = CrossAttentionSetEncoder(
                    n_features, n_total_values, feat_to_global_ids,
                    embed_dim, hidden_dim)
                model_ca.load_state_dict(ckpt["state_dict"])
                model_ca.to(args.device)
                model_ca.eval()
                print(f"  [4] Loaded CrossAttn from {ckpt_crossattn}")

                # Calibrate temperature on dev (same as sigtyp_eval_v4.py)
                from sigtyp_eval_v4 import calibrate_temperature as ca_calibrate
                T_ca = ca_calibrate(model_ca, cat_matrix_dev, feat_to_global_ids,
                                    feat_to_value_names, feat_name_to_idx, args.device)
                print(f"  [4] T_ca = {T_ca}")

                # Config E: argmax of SetEncoder logits only
                preds_ca = {}
                print(f"  [4] Inference over {len(test_meta)} test languages ...")
                for _, row in test_meta.iterrows():
                    wc      = row["wals_code"]
                    blanked = test_blank.get(wc, set())
                    obs_raw = test_obs.get(wc, {})
                    obs_fd  = {k: _strip_to_id(v) for k, v in obs_raw.items()
                               if k in feat_name_to_idx}

                    ctx_feats, ctx_gvals = [], []
                    for fname, vstr in obs_fd.items():
                        fi = feat_name_to_idx.get(fname)
                        if fi is None:
                            continue
                        vlist = feat_to_value_names.get(fi, [])
                        if vstr not in vlist:
                            continue
                        ctx_feats.append(fi)
                        ctx_gvals.append(feat_to_global_ids[fi][vlist.index(vstr)])

                    lang_preds = {}
                    with torch.no_grad():
                        for fname in blanked:
                            fi = feat_name_to_idx.get(fname)
                            if fi is None:
                                continue
                            logits = model_ca.logits_for_feature(
                                ctx_feats, ctx_gvals, fi, args.device)
                            pred_local = F.log_softmax(
                                logits / T_ca, dim=0).argmax().item()
                            lang_preds[fname] = feat_to_value_names[fi][pred_local]
                    preds_ca[wc] = lang_preds

                _write_submission(crossattn_path, test_meta, test_blank, test_obs,
                                  preds_ca, freq_fallback)
                print(f"  [4] Wrote {crossattn_path}")
                all_preds[f"crossattn_seed{seed}"] = preds_ca

            except Exception as e:
                print(f"  [4] CrossAttn failed for seed {seed}: {e}")
        else:
            print(f"  [4] No checkpoint at {ckpt_crossattn}, skipping seed {seed}")

    # -----------------------------------------------------------------------
    # ROW 5: UFAL probabilistic (deterministic, run once)
    # -----------------------------------------------------------------------
    path_prob = f"{out_dir}/sigtyp_abl_prob.tsv"
    if os.path.exists(path_prob) and not args.force_retrain:
        print(f"\n[5] UFAL probabilistic output exists: {path_prob}")
    else:
        print("\n[5] Running UFAL probabilistic reimplementation ...")
        t0 = time.time()
        preds_prob = run_ufal_probabilistic(
            train_meta, train_feat, dev_meta, dev_feat,
            test_meta, test_blank, test_obs,
            cat_matrix_traindev, kept_feature_names,
            feat_to_value_names, feat_name_to_idx,
            freq_fallback)
        _write_submission(path_prob, test_meta, test_blank, test_obs,
                          preds_prob, freq_fallback)
        print(f"  [5] Wrote {path_prob} in {time.time()-t0:.1f}s")
        all_preds["prob"] = preds_prob

    # Collect all submission paths for scoring
    sub_paths = []
    for seed in args.seeds:
        for tag in ["ufal", "ufal_random", "cat", "geom", "crossattn"]:
            p = f"{out_dir}/sigtyp_abl_{tag}_seed{seed}.tsv"
            if os.path.exists(p):
                sub_paths.append(p)
    if os.path.exists(path_prob):
        sub_paths.append(path_prob)

    # -----------------------------------------------------------------------
    # PART 6: Score all submissions
    # -----------------------------------------------------------------------
    all_scores = {}
    if sub_paths and os.path.exists(scorer):
        print(f"\n[Part 6] Scoring {len(sub_paths)} submissions ...")
        stdout, stderr = _run_scorer(scorer, sub_paths)
        if stderr.strip():
            print(f"  Scorer stderr: {stderr[:300]}")
        all_scores = _parse_scores(stdout)
    else:
        print("\n[Part 6] Scorer not available or no submissions to score.")

    # -----------------------------------------------------------------------
    # PART 7: Ablation table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ABLATION TABLE  (macro accuracy per genus)")
    print("=" * 80)

    COL_W = [42] + [6] * 8

    def pr(cells):
        parts = [str(cells[0]).ljust(COL_W[0])] + [str(c).rjust(6) for c in cells[1:]]
        print("| " + " | ".join(parts) + " |")

    def sep():
        print("|" + "|".join("-" + "-" * w + "-" for w in COL_W) + "|")

    def _get_val(submission_bn, col, mode="macro"):
        v = all_scores.get(mode, {}).get(submission_bn, {}).get(col)
        return f"{v:.2f}" if v is not None else " n/a"

    def _mean_std_row(label, tag, mode="macro"):
        ovs = []
        per_col = defaultdict(list)
        for seed in args.seeds:
            bn = f"sigtyp_abl_{tag}_seed{seed}"
            d  = all_scores.get(mode, {}).get(bn, {})
            ov = d.get("Overall")
            if ov is not None:
                ovs.append(ov)
            for col in _TABLE_COLS:
                v = d.get(col)
                if v is not None:
                    per_col[col].append(v)
        cells = [label]
        for col in _TABLE_COLS:
            vs = per_col[col]
            if vs:
                cells.append(f"{np.mean(vs):.2f}")
            else:
                cells.append(" n/a")
        # Replace Overall with mean±std
        if ovs:
            cells[-1] = f"{np.mean(ovs):.2f}±{np.std(ovs):.2f}"
        return cells

    def _single_row(label, submission_bn, mode="macro"):
        cells = [label]
        for col in _TABLE_COLS:
            cells.append(_get_val(submission_bn, col, mode))
        return cells

    pr(["System"] + _TABLE_HDR)
    sep()
    pr(["--- Ablation: value representation (all with transductive FT) ---"] + [""] * 8)
    sep()
    pr(_mean_std_row("[1]  UFAL-binary (cos+sigmoid)", "ufal"))
    pr(_mean_std_row("[2]  + Categorical (cos+softmax)", "cat"))
    pr(_mean_std_row("[3]  + Value geometry (shared)", "geom"))
    pr(_mean_std_row("[4]  + Set conditioning (CrossAttn)", "crossattn"))
    sep()
    pr(["--- UFAL original behaviour ---"] + [""] * 8)
    sep()
    pr(_mean_std_row("[1b] UFAL-binary (random emb, no adaptation)", "ufal_random"))
    sep()
    pr(["--- Reference systems ---"] + [""] * 8)
    sep()

    prob_bn = os.path.splitext(os.path.basename(path_prob))[0]
    pr(_single_row("[5]  UFAL-probabilistic (reimpl.)", prob_bn))
    pr(["[6]  UFAL combined (paper)"] +
       [".73", ".78", ".74", ".71", ".80", ".76", ".76", ".75"])
    sep()
    print()

    # -----------------------------------------------------------------------
    # Incremental delta table with bootstrap tests
    # -----------------------------------------------------------------------
    print("INCREMENTAL DELTA TABLE")
    print("=" * 60)

    gold_obs2 = None
    try:
        _, _, _, gold_obs2 = parse_sigtyp(test_gold)
    except Exception as e:
        print(f"  Could not load gold: {e}")

    if gold_obs2 is not None:
        # Aggregate predictions across seeds
        def _avg_preds(tag):
            acc_by_lang = defaultdict(lambda: defaultdict(list))
            for seed in args.seeds:
                p = all_preds.get(f"{tag}_seed{seed}", {})
                for wc, pred in p.items():
                    for fn, v in pred.items():
                        acc_by_lang[wc][fn].append(v)
            # majority vote per cell
            voted = {}
            for wc, feat_votes in acc_by_lang.items():
                voted[wc] = {}
                for fn, vs in feat_votes.items():
                    voted[wc][fn] = Counter(vs).most_common(1)[0][0]
            return voted

        preds_1   = _avg_preds("ufal")
        preds_2   = _avg_preds("cat")
        preds_3   = _avg_preds("geom")
        preds_4   = _avg_preds("crossattn")
        preds_1b  = _avg_preds("ufal_random")

        # Use per-seed preds for bootstrap
        def _seed_preds(tag, seed):
            return all_preds.get(f"{tag}_seed{seed}", {})

        pairs = [
            ("[1]→[2]  binary → categorical",      "ufal", "cat"),
            ("[2]→[3]  string → shared geometry",   "cat",  "geom"),
            ("[3]→[4]  fixed → set-conditioned",    "geom", "crossattn"),
            ("[1]→[1b] transduction vs random",     "ufal", "ufal_random"),
        ]

        for label, tag_a, tag_b in pairs:
            pa = _avg_preds(tag_a)
            pb = _avg_preds(tag_b)
            delta, ci, pval = bootstrap_delta(
                pa, pb, test_meta, test_blank, gold_obs2,
                feat_name_to_idx, n_boot=1000)
            print(f"  {label}:")
            print(f"    delta={delta:+.4f}  p={pval:.4f}  "
                  f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    # -----------------------------------------------------------------------
    # PART 8: Value geometry probe (seed=42 only)
    # -----------------------------------------------------------------------
    print("\n[Part 8] Value geometry probe (seed=42) ...")

    seed42 = 42
    ckpt_geom42 = f"{out_dir}/cat_geom_seed{seed42}.pt"
    if os.path.exists(ckpt_geom42):
        try:
            ckpt = torch.load(ckpt_geom42, map_location="cpu")
            m42 = CategoricalGeomNeural(
                n_langs_traindev, n_total_values, feat_to_global_ids,
                embed_dim=embed_dim, dropout_rate=args.dropout)
            m42.load_state_dict(ckpt["state_dict"])
            m42.to(args.device)
            m42.eval()
            print_value_geometry_probe(
                m42, 'geom', kept_feature_names, feat_to_value_names,
                feat_to_global_ids, feat_name_to_idx, args.device)
        except Exception as e:
            print(f"  Geom probe failed: {e}")

    ckpt_ca42 = f"{gb2v}/set_encoder_v4_seed{seed42}.pt"
    if os.path.exists(ckpt_ca42):
        try:
            ckpt = torch.load(ckpt_ca42, map_location="cpu")
            hidden_dim = ckpt.get("hidden_dim", 128)
            mca42 = CrossAttentionSetEncoder(
                n_features, n_total_values, feat_to_global_ids,
                embed_dim, hidden_dim)
            mca42.load_state_dict(ckpt["state_dict"])
            mca42.to(args.device)
            mca42.eval()
            print_value_geometry_probe(
                mca42, 'crossattn', kept_feature_names, feat_to_value_names,
                feat_to_global_ids, feat_name_to_idx, args.device)
        except Exception as e:
            print(f"  CrossAttn probe failed: {e}")

    # -----------------------------------------------------------------------
    # Blanking ratio vs accuracy (Pearson R)
    # -----------------------------------------------------------------------
    print("\n[Blanking ratio vs accuracy — Pearson R]")
    if gold_obs2 is not None:
        for label, tag in [
            ("[1]  UFAL-binary (transductive)", "ufal"),
            ("[1b] UFAL-binary (random)",       "ufal_random"),
            ("[2]  Categorical",                "cat"),
            ("[3]  Geom",                       "geom"),
            ("[4]  CrossAttn",                  "crossattn"),
        ]:
            p = _avg_preds(tag) if f"{tag}_seed{args.seeds[0]}" in all_preds else {}
            if not p:
                continue
            try:
                r = blanking_ratio_pearson(p, test_meta, test_blank, gold_obs2,
                                           feat_name_to_idx)
                print(f"  {label}: R = {r:+.4f}")
            except Exception as e:
                print(f"  {label}: R computation failed ({e})")

    # -----------------------------------------------------------------------
    # PART 9: Key finding printout
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("=== KEY FINDING: UFAL test inference ===")
    print("""UFAL's neural system (models.py, callbacks.py) pre-allocates language
embedding slots for test languages but NEVER updates them during training.
Test language embeddings remain randomly initialized throughout.
At test time: prediction = argmax sigmoid(cos(random_vector, fv_emb) * w + b)
""")

    def _get_overall(tag, mode="macro"):
        ovs = []
        for seed in args.seeds:
            bn = f"sigtyp_abl_{tag}_seed{seed}"
            v  = all_scores.get(mode, {}).get(bn, {}).get("Overall")
            if v is not None:
                ovs.append(v)
        if ovs:
            return float(np.mean(ovs)), float(np.std(ovs))
        return None, None

    mu_1b, sd_1b = _get_overall("ufal_random")
    mu_1,  sd_1  = _get_overall("ufal")

    def _fmt(mu, sd):
        if mu is None:
            return "n/a"
        return f"{mu:.4f} (±{sd:.4f})"

    print(f"[1b] (random test emb, no adaptation): {_fmt(mu_1b, sd_1b)} macro")
    print(f"[1]  (binary + transductive):           {_fmt(mu_1, sd_1)} macro")
    if mu_1 is not None and mu_1b is not None:
        delta = mu_1 - mu_1b
        print(f"     (+{delta:.4f} from adaptation)")
        print(f"""
This shows UFAL's neural component for test languages degenerates toward
a learned frequency prior. The {delta:.2f}-point gain from transductive fine-tuning
(row [1] vs [1b]) demonstrates that adapting to observed features matters
more than the language embedding architecture.""")

    print("\nDone.")


if __name__ == "__main__":
    main(get_args())
