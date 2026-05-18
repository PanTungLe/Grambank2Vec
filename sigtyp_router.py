#!/usr/bin/env python3
"""
sigtyp_router.py: Meta-learner router selecting among 10 base system predictions.

Goal: ≥0.75 macro accuracy on SIGTYP 2020 test set.
UFAL baseline = 0.75; oracle ceiling with these systems = ~0.88.

Run:
    python3 sigtyp_router.py \\
        --v4e-path /path/to/sigtyp_v4_E_seed42.tsv \\
        --v4i-path /path/to/sigtyp_v4_I_seed42.tsv \\
        --v4-ckpt  /path/to/set_encoder_v4_seed42.pt \\
        2>&1 | tee sigtyp_router_log.txt
"""

import argparse
import os
import sys
import time
import pickle
import subprocess
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
BASE_DIR = "/home/user/Grambank2Vec"
DATA_DIR = "/home/user/ST2020/data"
SCORER   = "/home/user/ST2020/scripts/score.py"

TRAIN_CSV  = f"{DATA_DIR}/train.csv"
DEV_CSV    = f"{DATA_DIR}/dev.csv"
TEST_BLIND = f"{DATA_DIR}/test_blinded.csv"
TEST_GOLD  = f"{DATA_DIR}/test_gold.csv"

SYSTEM_FILES = {
    "v1_learned":  f"{BASE_DIR}/sigtyp_submission_learned.tsv",
    "v1_tcf":      f"{BASE_DIR}/sigtyp_submission_tcf.tsv",
    "v1_freq":     f"{BASE_DIR}/sigtyp_submission_freq.tsv",
    "v3_freq":     f"{BASE_DIR}/sigtyp_v3_freq.tsv",
    "v3_condprob": f"{BASE_DIR}/sigtyp_v3_condprob.tsv",
    "v3_knn":      f"{BASE_DIR}/sigtyp_v3_knn.tsv",
    "v3_idfknn":   f"{BASE_DIR}/sigtyp_v3_idfknn.tsv",
    "v3_genus":    f"{BASE_DIR}/sigtyp_v3_genus.tsv",
    "v4E":         None,   # from --v4e-path (optional)
    "v4I":         None,   # from --v4i-path (optional)
}
SYS_NAMES = list(SYSTEM_FILES.keys())   # 10 systems

CONTROLLED_GENERA = {
    "Tucanoan", "Madang", "Mahakiranti", "Nilotic", "Mayan", "Northern Pama-Nyungan"
}

KNN_K          = 5
CONDPROB_ALPHA = 0.5
CONDPROB_MIN   = 5

HEADER = "wals_code\tname\tlatitude\tlongitude\tgenus\tfamily\tcountrycodes\tfeatures\n"

# Feature names for the 32-dimensional feature vector
FEATURE_NAMES = [
    # language-level (7)
    "genus_in_train",
    "family_in_train",
    "n_obs_in_vocab",
    "lat",
    "lon",
    "min_dist_km",
    "controlled_genus",
    # feature-level (4)
    "feat_freq",
    "n_values",
    "feat_entropy",
    "majority_val_int",
    # agreement signals (8)
    "n_systems_predict_mode",
    "all_agree",
    "top2_agree_count",
    "v4E_in_plurality",
    "v3genus_in_plurality",
    "v4E_v3genus_agree",
    "v4E_condprob_agree",
    "v4I_v4E_agree",
    # candidate-level (13)
    "vote_count",
    "vote_frac",
    "is_train_majority",
    "is_v1_freq",
    "is_v3_freq",
    "is_v3_condprob",
    "is_v3_knn",
    "is_v3_idfknn",
    "is_v3_genus",
    "is_v4E",
    "is_v4I",
    "is_v1_neural",
    "candidate_val_int",
]
assert len(FEATURE_NAMES) == 32, f"Expected 32 features, got {len(FEATURE_NAMES)}"

# ---------------------------------------------------------------------------
# SIGTYP fixes (same as sigtyp_eval_v3)
# ---------------------------------------------------------------------------
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


# ===========================================================================
# Part 0 — TSV parsing
# ===========================================================================

def parse_tsv(path):
    """Parse a SIGTYP-format TSV file.

    Returns
    -------
    meta_df    : DataFrame [wals_code, name, latitude, longitude, genus, family, countrycodes]
    feat_df    : DataFrame rows=languages, cols=lowercase feature names; value=id string or NaN
    blanked    : {wals_code: set of lowercase feature names with "?"}
    obs_strs   : {wals_code: {feat_name_lc: full_value_string like "1 SVO"}}
    """
    meta_rows, feat_rows, blanked, obs_strs = [], {}, {}, {}
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    skip = True
    for line in lines:
        if not line.strip():
            continue
        if skip:
            skip = False
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        wc = parts[0]
        try:
            lat = float(parts[2])
            lon = float(parts[3])
        except ValueError:
            lat = lon = float("nan")
        meta_rows.append({
            "wals_code": wc, "name": parts[1],
            "latitude": lat, "longitude": lon,
            "genus": parts[4], "family": parts[5],
            "countrycodes": parts[6],
        })
        fd, bs, osd = {}, set(), {}
        for piece in _apply_fixes("\t".join(parts[7:])).split("|"):
            piece = piece.strip()
            if not piece or "=" not in piece:
                continue
            eq = piece.index("=")
            fn = piece[:eq].lower()
            vs = piece[eq + 1:]
            if vs.strip() == "?":
                bs.add(fn)
                fd[fn] = float("nan")
            else:
                toks = vs.split()
                if not toks:
                    continue
                try:
                    int(toks[0])
                except ValueError:
                    continue
                fd[fn] = toks[0]
                osd[fn] = vs
        feat_rows[wc] = fd
        blanked[wc]   = bs
        obs_strs[wc]  = osd

    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)
    wcs     = [r["wals_code"] for r in meta_rows]
    all_fn  = sorted({fn for fd in feat_rows.values()
                      for fn, v in fd.items() if isinstance(v, str)})
    feat_df = pd.DataFrame(
        [{fn: feat_rows[wc].get(fn, float("nan")) for fn in all_fn} for wc in wcs],
        columns=all_fn,
    )
    return meta_df, feat_df, blanked, obs_strs


def build_vocab(feat_df):
    """Build feature vocabulary from a combined feat_df.

    Returns (kept_names, feat_to_vals, val_to_loc, cat_matrix).
      kept_names  : list of feature names with ≥2 unique values
      feat_to_vals: fi -> sorted list of value-id strings
      val_to_loc  : fi -> {val_str: local_idx}
      cat_matrix  : int32 array (n_langs × n_feats), -1 = missing
    """
    kept_names   = []
    feat_to_vals = {}
    val_to_loc   = {}
    columns      = []

    for col in feat_df.columns:
        unique = sorted(feat_df[col].dropna().unique().tolist())
        if len(unique) <= 1:
            continue
        fi = len(kept_names)
        kept_names.append(col)
        feat_to_vals[fi] = unique
        v2l = {v: i for i, v in enumerate(unique)}
        val_to_loc[fi] = v2l
        col_data = np.full(len(feat_df), -1, dtype=np.int32)
        for row_idx, v in enumerate(feat_df[col]):
            if isinstance(v, str) and v in v2l:
                col_data[row_idx] = v2l[v]
        columns.append(col_data)

    if columns:
        cat_matrix = np.column_stack(columns).astype(np.int32)
    else:
        cat_matrix = np.empty((len(feat_df), 0), dtype=np.int32)
    return kept_names, feat_to_vals, val_to_loc, cat_matrix


def compute_idf(cat_matrix):
    """IDF weight per feature: log(n / (1 + n_obs))."""
    n = cat_matrix.shape[0]
    n_obs = (cat_matrix >= 0).sum(axis=0).astype(float)
    return np.log(n / (1.0 + n_obs))


def build_marginals(cat_matrix, feat_to_vals):
    """Laplace-smoothed marginal distributions.

    Returns dict fi -> np.array of shape (K_fi,), summing to 1.
    """
    n_feats = cat_matrix.shape[1]
    marginals = {}
    for fi in range(n_feats):
        K = len(feat_to_vals[fi])
        counts = np.zeros(K, dtype=np.float64)
        col = cat_matrix[:, fi]
        for vi in range(K):
            counts[vi] = float((col == vi).sum())
        counts += 0.5
        marginals[fi] = counts / counts.sum()
    return marginals


def build_co_counts(cat_matrix, feat_to_vals):
    """Build co-occurrence count tables from cat_matrix.

    Returns
    -------
    co_counts  : {(fi, vi, fj): np.array(K_fj)} of raw counts
    marg_counts: {(fi, vi): int count}
    """
    n_langs, n_feats = cat_matrix.shape
    co_counts   = {}
    marg_counts = {}
    for l in range(n_langs):
        obs = [(int(fi), int(cat_matrix[l, fi]))
               for fi in range(n_feats)
               if cat_matrix[l, fi] >= 0]
        for fi, vi in obs:
            marg_counts[(fi, vi)] = marg_counts.get((fi, vi), 0) + 1
            for fj, vj in obs:
                if fi == fj:
                    continue
                key = (fi, vi, fj)
                K_fj = len(feat_to_vals[fj])
                if key not in co_counts:
                    co_counts[key] = np.zeros(K_fj, dtype=np.float32)
                co_counts[key][vj] += 1
    return co_counts, marg_counts


def load_system_preds(path, test_blank, feat_name_to_idx, val_to_loc, feat_to_vals):
    """Load a single system's predictions as value strings for blanked test cells.

    Returns {wals_code: {feat_name: pred_val_str or None}}.
    """
    result = {}
    if path is None or not os.path.exists(path):
        if path is not None:
            print(f"  [WARN] system TSV not found: {path}")
        return result
    _, _, _, obs_s = parse_tsv(path)
    for wc, blank_set in test_blank.items():
        preds = {}
        lang_obs = obs_s.get(wc, {})
        for fn in blank_set:
            fi = feat_name_to_idx.get(fn)
            if fi is None:
                preds[fn] = None
                continue
            raw = lang_obs.get(fn)
            if raw is None:
                preds[fn] = None
                continue
            vid = raw.split()[0]
            if vid in val_to_loc[fi]:
                preds[fn] = vid
            else:
                preds[fn] = None
        result[wc] = preds
    return result


# ===========================================================================
# Utility functions
# ===========================================================================

def haversine_km_vec(lat, lon, lats_arr, lons_arr):
    """Vectorized haversine; returns minimum distance to any element."""
    R = 6371.0
    lat_r  = np.radians(lat)
    lon_r  = np.radians(lon)
    lats_r = np.radians(lats_arr)
    lons_r = np.radians(lons_arr)
    dlat = lats_r - lat_r
    dlon = lons_r - lon_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(lats_r) * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1))).min())


def feat_entropy(marginals_fi):
    p = marginals_fi
    return float(-np.sum(p * np.log(p + 1e-300)))


# ===========================================================================
# LOO-simulated system predictions
# ===========================================================================

def _knn_predict_loo(h, fi, remaining_idx, M_full, cat_matrix_train,
                     feat_to_vals, freq_fallback_local):
    """kNN LOO prediction for training lang h, feature fi."""
    col_fi = cat_matrix_train[:, fi]
    vi_h   = int(cat_matrix_train[h, fi])
    has_fi = col_fi >= 0
    agree_fi = (col_fi == vi_h) & has_fi & (vi_h >= 0)

    loo_scores = M_full[h, remaining_idx].copy()
    rem_has_fi  = has_fi[remaining_idx]
    rem_agree   = agree_fi[remaining_idx]
    loo_scores[rem_has_fi] -= rem_agree[rem_has_fi].astype(float)

    k_use = min(KNN_K, int((loo_scores > 0).sum()))
    if k_use == 0:
        return freq_fallback_local

    top_k      = np.argsort(loo_scores)[-k_use:]
    actual_top = remaining_idx[top_k]
    votes = [int(cat_matrix_train[n, fi])
             for n in actual_top if cat_matrix_train[n, fi] >= 0]
    if not votes:
        return freq_fallback_local
    best_local = Counter(votes).most_common(1)[0][0]
    return feat_to_vals[fi][best_local]


def _idfknn_predict_loo(h, fi, remaining_idx, M_idf_full, cat_matrix_train,
                        feat_to_vals, idf_rem, freq_fallback_local):
    """IDF-kNN LOO prediction for training lang h, feature fi."""
    col_fi = cat_matrix_train[:, fi]
    vi_h   = int(cat_matrix_train[h, fi])
    has_fi = col_fi >= 0
    agree_fi = (col_fi == vi_h) & has_fi & (vi_h >= 0)

    loo_scores = M_idf_full[h, remaining_idx].copy()
    rem_has_fi  = has_fi[remaining_idx]
    rem_agree   = agree_fi[remaining_idx]
    loo_scores[rem_has_fi] -= rem_agree[rem_has_fi].astype(float) * float(idf_rem[fi])

    k_use = min(KNN_K, int((loo_scores > 0).sum()))
    if k_use == 0:
        return freq_fallback_local

    top_k      = np.argsort(loo_scores)[-k_use:]
    actual_top = remaining_idx[top_k]
    votes = [int(cat_matrix_train[n, fi])
             for n in actual_top if cat_matrix_train[n, fi] >= 0]
    if not votes:
        return freq_fallback_local
    best_local = Counter(votes).most_common(1)[0][0]
    return feat_to_vals[fi][best_local]


def _condprob_predict_one(fi, obs_dict, feat_name_to_idx, feat_to_vals,
                          co_counts_rem, marg_rem, freq_fallback_local):
    """Condprob prediction for a single (fi, obs_dict) pair."""
    K_j     = len(feat_to_vals[fi])
    log_sc  = np.zeros(K_j, dtype=np.float64)
    n_terms = 0
    for feat_i_name, val_i_str in obs_dict.items():
        fk = feat_name_to_idx.get(feat_i_name)
        if fk is None or fk == fi:
            continue
        val_list_k = feat_to_vals.get(fk, [])
        if val_i_str not in val_list_k:
            continue
        vk_local = val_list_k.index(val_i_str)
        mc = marg_rem.get((fk, vk_local), 0)
        if mc < CONDPROB_MIN:
            continue
        counts = co_counts_rem.get((fk, vk_local, fi), np.zeros(K_j, dtype=np.float32))
        if len(counts) != K_j:
            counts = np.zeros(K_j, dtype=np.float32)
        smoothed = (counts + CONDPROB_ALPHA) / (mc + CONDPROB_ALPHA * K_j)
        log_sc  += np.log(smoothed + 1e-12)
        n_terms += 1
    if n_terms > 0:
        return feat_to_vals[fi][int(np.argmax(log_sc))]
    return freq_fallback_local


def _genus_predict_one(fi, genus, genus_tables_rem, freq_fallback_str):
    """Genus majority prediction for a single fi, using remaining data tables."""
    table = genus_tables_rem.get(genus, {})
    local_vi = table.get(fi)
    if local_vi is not None:
        return local_vi  # already a value string in our loo genus tables
    return freq_fallback_str


# ===========================================================================
# Feature row computation
# ===========================================================================

def make_feature_row(fi, wc, fn, sys_preds_cell, candidate_str,
                     lang_row, train_genera, train_families,
                     feat_freq, n_values, entropy_val, majority_val_int,
                     train_latlons, feat_to_vals, marginals_fi):
    """Compute the 32-feature vector for one (cell, candidate) pair.

    Parameters
    ----------
    fi                : feature index
    wc                : wals_code of the language
    fn                : feature name (lowercase)
    sys_preds_cell    : {sys_name: pred_val_str or None}
    candidate_str     : candidate value string being scored
    lang_row          : pandas Series with wals_code, genus, family, latitude, longitude
    train_genera      : set of genera in train
    train_families    : set of families in train
    feat_freq         : number of training langs with this feature
    n_values          : int, cardinality
    entropy_val       : float
    majority_val_int  : int
    train_latlons     : np.array shape (n_train, 2) [lat, lon]
    feat_to_vals      : fi -> list of value strings
    marginals_fi      : np.array of Laplace-smoothed marginals for fi
    """
    genus  = lang_row["genus"]
    family = lang_row["family"]
    lat    = float(lang_row["latitude"])
    lon    = float(lang_row["longitude"])

    # Language-level features
    genus_in_train    = int(genus in train_genera)
    family_in_train   = int(family in train_families)
    controlled_genus  = int(genus in CONTROLLED_GENERA)

    # n_obs_in_vocab: we pass it in as a precomputed int
    # We embed it as 0 here (to be patched in caller)
    n_obs_in_vocab = 0  # placeholder — overridden in caller

    if (not np.isnan(lat)) and train_latlons.shape[0] > 0:
        min_dist = haversine_km_vec(lat, lon, train_latlons[:, 0], train_latlons[:, 1])
    else:
        min_dist = 0.0

    # Agreement signals
    avail_preds = {s: v for s, v in sys_preds_cell.items() if v is not None}
    n_avail     = len(avail_preds)

    if n_avail == 0:
        mode_val   = None
        n_mode     = 0
        all_agree  = 0
        top2_agree = 0
    else:
        ctr        = Counter(avail_preds.values())
        mode_val, n_mode = ctr.most_common(1)[0]
        all_agree  = int(n_mode == n_avail and n_avail > 0)
        top2       = ctr.most_common(2)
        top2_agree = top2[1][1] if len(top2) > 1 else 0

    v4E  = avail_preds.get("v4E")
    v4I  = avail_preds.get("v4I")
    v3g  = avail_preds.get("v3_genus")
    v3cp = avail_preds.get("v3_condprob")

    v4E_in_plurality   = int(v4E is not None and v4E == mode_val)
    v3genus_in_plural  = int(v3g is not None and v3g == mode_val)
    v4E_v3genus_agree  = int(v4E is not None and v3g is not None and v4E == v3g)
    v4E_condprob_agree = int(v4E is not None and v3cp is not None and v4E == v3cp)
    v4I_v4E_agree      = int(v4I is not None and v4E is not None and v4I == v4E)

    n_systems_predict_mode = n_mode

    # Candidate-level
    vote_count_ = sum(1 for v in avail_preds.values() if v == candidate_str)
    vote_frac   = vote_count_ / max(n_avail, 1)
    is_train_majority = int(candidate_str == feat_to_vals[fi][int(np.argmax(marginals_fi))])

    # Per-system flags
    is_v1_freq     = int(sys_preds_cell.get("v1_freq")     == candidate_str)
    is_v3_freq     = int(sys_preds_cell.get("v3_freq")     == candidate_str)
    is_v3_condprob = int(sys_preds_cell.get("v3_condprob") == candidate_str)
    is_v3_knn      = int(sys_preds_cell.get("v3_knn")      == candidate_str)
    is_v3_idfknn   = int(sys_preds_cell.get("v3_idfknn")   == candidate_str)
    is_v3_genus    = int(sys_preds_cell.get("v3_genus")    == candidate_str)
    is_v4E         = int(sys_preds_cell.get("v4E")         == candidate_str)
    is_v4I         = int(sys_preds_cell.get("v4I")         == candidate_str)
    is_v1_learned  = int(sys_preds_cell.get("v1_learned")  == candidate_str)
    is_v1_tcf      = int(sys_preds_cell.get("v1_tcf")      == candidate_str)

    try:
        cand_val_int = int(candidate_str)
    except (ValueError, TypeError):
        cand_val_int = 0

    row = [
        genus_in_train, family_in_train, n_obs_in_vocab,
        lat, lon, min_dist, controlled_genus,
        feat_freq, n_values, entropy_val, majority_val_int,
        n_systems_predict_mode, all_agree, top2_agree,
        v4E_in_plurality, v3genus_in_plural, v4E_v3genus_agree,
        v4E_condprob_agree, v4I_v4E_agree,
        vote_count_, vote_frac, is_train_majority,
        is_v1_freq, is_v3_freq, is_v3_condprob, is_v3_knn, is_v3_idfknn,
        is_v3_genus, is_v4E, is_v4I, 0,  # is_v1_neural placeholder
        cand_val_int,
    ]
    return row


# ===========================================================================
# STEP 2 — LOO pseudo-test training data
# ===========================================================================

def build_loo_train_data(train_meta_df, cat_matrix_train, kept_names,
                         feat_name_to_idx, feat_to_vals, val_to_loc,
                         idf_weights, rng=None):
    """Build LOO pseudo-test training data using GroupKFold on training genera.

    Returns (X, y) arrays.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n_train, n_feats = cat_matrix_train.shape
    genera_train     = train_meta_df["genus"].tolist()
    families_train   = train_meta_df["family"].tolist()

    train_latlons = np.column_stack([
        train_meta_df["latitude"].values.astype(float),
        train_meta_df["longitude"].values.astype(float),
    ])
    all_genera_set   = set(genera_train)
    all_families_set = set(families_train)

    # Compute full marginals from entire training set
    marginals_full = build_marginals(cat_matrix_train, feat_to_vals)

    # Precompute match matrices ONCE
    print("  Precomputing match matrix (this may take a moment) ...")
    t0 = time.time()
    M_full     = np.zeros((n_train, n_train), dtype=np.float32)
    M_idf_full = np.zeros((n_train, n_train), dtype=np.float32)
    for fi in range(n_feats):
        col  = cat_matrix_train[:, fi]
        has  = col >= 0
        agree = (col[:, None] == col[None, :]) & has[:, None] & has[None, :]
        M_full     += agree.astype(np.float32)
        M_idf_full += agree.astype(np.float32) * float(idf_weights[fi])
    print(f"  Match matrix done in {time.time()-t0:.1f}s")

    gkf    = GroupKFold(n_splits=10)
    groups = np.array(genera_train)

    X_rows, y_rows = [], []

    n_obs_per_lang = (cat_matrix_train >= 0).sum(axis=1)

    for fold_idx, (tr_idx, held_idx) in enumerate(gkf.split(
            np.arange(n_train), np.arange(n_train), groups=groups)):
        print(f"  Fold {fold_idx+1}/10: {len(held_idx)} held-out langs ...")
        tr_idx_arr   = np.array(tr_idx)
        held_idx_arr = np.array(held_idx)

        # Build remaining-only stats
        cat_rem = cat_matrix_train[tr_idx_arr]
        n_rem   = cat_rem.shape[0]
        if n_rem == 0:
            continue

        idf_rem      = compute_idf(cat_rem)
        marginals_rem = build_marginals(cat_rem, feat_to_vals)

        # Genus tables from remaining
        genus_tables_rem = {}  # genus -> {fi: val_str}
        freq_rem = {}  # fi -> val_str
        for fi in range(n_feats):
            local_idx = int(np.argmax(marginals_rem[fi]))
            freq_rem[fi] = feat_to_vals[fi][local_idx]

        for g in set(groups[tr_idx_arr]):
            g_mask = groups[tr_idx_arr] == g
            g_rows = cat_rem[g_mask]
            gp = {}
            for fi in range(n_feats):
                col = g_rows[:, fi]
                obs = col[col >= 0]
                if len(obs) > 0:
                    gp[fi] = feat_to_vals[fi][Counter(obs.tolist()).most_common(1)[0][0]]
                else:
                    gp[fi] = freq_rem[fi]
            genus_tables_rem[g] = gp

        # Co-occurrence from remaining
        co_counts_rem, marg_rem = build_co_counts(cat_rem, feat_to_vals)

        for h in held_idx_arr:
            genus_h  = genera_train[h]
            family_h = families_train[h]
            lat_h    = float(train_meta_df.iloc[h]["latitude"])
            lon_h    = float(train_meta_df.iloc[h]["longitude"])

            observed_feats = [fi for fi in range(n_feats) if cat_matrix_train[h, fi] >= 0]
            if len(observed_feats) == 0:
                continue

            n_obs_vocab = len(observed_feats)

            # lang-level features (constant per h)
            genus_in_train_  = int(genus_h in all_genera_set)
            family_in_train_ = int(family_h in all_families_set)
            controlled_genus_= int(genus_h in CONTROLLED_GENERA)
            if not np.isnan(lat_h) and train_latlons.shape[0] > 0:
                min_dist_ = haversine_km_vec(lat_h, lon_h,
                                             train_latlons[:, 0],
                                             train_latlons[:, 1])
            else:
                min_dist_ = 0.0

            # Subsample blanked features (up to 5 random blankings)
            n_samples = 5
            for _s in range(n_samples):
                p          = rng.uniform(0.3, 0.7)
                n_blank    = max(1, int(len(observed_feats) * p))
                n_blank    = min(n_blank, len(observed_feats))
                blank_idxs = rng.choice(len(observed_feats), n_blank, replace=False)
                blanked_feats  = [observed_feats[i] for i in blank_idxs]
                context_feats  = [fi for fi in observed_feats if fi not in set(blanked_feats)]
                obs_dict       = {kept_names[fi]: feat_to_vals[fi][int(cat_matrix_train[h, fi])]
                                  for fi in context_feats}

                for fi in blanked_feats:
                    true_local = int(cat_matrix_train[h, fi])
                    true_val   = feat_to_vals[fi][true_local]

                    # Simulate system predictions
                    fp_freq  = freq_rem[fi]
                    fp_genus = genus_tables_rem.get(genus_h, {}).get(fi, fp_freq)
                    fp_knn   = _knn_predict_loo(
                        h, fi, tr_idx_arr, M_full, cat_matrix_train,
                        feat_to_vals, fp_freq)
                    fp_idfknn = _idfknn_predict_loo(
                        h, fi, tr_idx_arr, M_idf_full, cat_matrix_train,
                        feat_to_vals, idf_rem, fp_freq)
                    fp_condprob = _condprob_predict_one(
                        fi, obs_dict, feat_name_to_idx, feat_to_vals,
                        co_counts_rem, marg_rem, fp_freq)

                    sys_preds_cell = {
                        "v1_freq":    fp_freq,
                        "v3_freq":    fp_freq,
                        "v3_genus":   fp_genus,
                        "v3_knn":     fp_knn,
                        "v3_idfknn":  fp_idfknn,
                        "v3_condprob": fp_condprob,
                        "v1_learned": None,
                        "v1_tcf":     None,
                        "v4E":        None,
                        "v4I":        None,
                    }

                    candidates = list({v for v in sys_preds_cell.values() if v is not None})
                    if not candidates:
                        candidates = [fp_freq]
                    train_maj = fp_freq
                    if train_maj not in candidates:
                        candidates.append(train_maj)

                    marginals_fi = marginals_rem[fi]
                    feat_freq_   = int((cat_rem[:, fi] >= 0).sum())
                    n_values_    = len(feat_to_vals[fi])
                    entropy_val_ = feat_entropy(marginals_fi)
                    majority_val_int_ = int(feat_to_vals[fi][int(np.argmax(marginals_fi))])

                    avail_preds = {s: v for s, v in sys_preds_cell.items() if v is not None}
                    n_avail     = len(avail_preds)
                    if n_avail > 0:
                        ctr     = Counter(avail_preds.values())
                        mode_val, n_mode = ctr.most_common(1)[0]
                        all_agree_  = int(n_mode == n_avail)
                        top2        = ctr.most_common(2)
                        top2_agree_ = top2[1][1] if len(top2) > 1 else 0
                    else:
                        mode_val    = None
                        n_mode      = 0
                        all_agree_  = 0
                        top2_agree_ = 0

                    v4E_  = None
                    v4I_  = None
                    v3g_  = sys_preds_cell.get("v3_genus")
                    v3cp_ = sys_preds_cell.get("v3_condprob")

                    v4E_in_plurality_   = 0
                    v3genus_in_plural_  = int(v3g_ is not None and v3g_ == mode_val)
                    v4E_v3genus_agree_  = 0
                    v4E_condprob_agree_ = 0
                    v4I_v4E_agree_      = 0

                    for cand in candidates:
                        vote_count_ = sum(1 for v in avail_preds.values() if v == cand)
                        vote_frac_  = vote_count_ / max(n_avail, 1)
                        is_train_majority_ = int(cand == fp_freq)
                        is_v1_freq_     = int(sys_preds_cell.get("v1_freq")     == cand)
                        is_v3_freq_     = int(sys_preds_cell.get("v3_freq")     == cand)
                        is_v3_condprob_ = int(sys_preds_cell.get("v3_condprob") == cand)
                        is_v3_knn_      = int(sys_preds_cell.get("v3_knn")      == cand)
                        is_v3_idfknn_   = int(sys_preds_cell.get("v3_idfknn")   == cand)
                        is_v3_genus_    = int(sys_preds_cell.get("v3_genus")    == cand)
                        is_v4E_         = 0
                        is_v4I_         = 0
                        is_v1_neural_   = 0  # no neural in LOO sim
                        try:
                            cand_val_int_ = int(cand)
                        except (ValueError, TypeError):
                            cand_val_int_ = 0

                        row = [
                            genus_in_train_, family_in_train_, n_obs_vocab,
                            lat_h, lon_h, min_dist_, controlled_genus_,
                            feat_freq_, n_values_, entropy_val_, majority_val_int_,
                            n_mode, all_agree_, top2_agree_,
                            v4E_in_plurality_, v3genus_in_plural_,
                            v4E_v3genus_agree_, v4E_condprob_agree_, v4I_v4E_agree_,
                            vote_count_, vote_frac_, is_train_majority_,
                            is_v1_freq_, is_v3_freq_, is_v3_condprob_,
                            is_v3_knn_, is_v3_idfknn_, is_v3_genus_,
                            is_v4E_, is_v4I_, is_v1_neural_,
                            cand_val_int_,
                        ]
                        label = 1 if cand == true_val else 0
                        X_rows.append(row)
                        y_rows.append(label)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    print(f"  LOO training data: {len(X)} rows")
    return X, y


# ===========================================================================
# STEP 3 — Train meta-learners
# ===========================================================================

def train_meta_learner(X_train, y_train):
    """Train meta-learner A (LightGBM or GradientBoosting fallback)."""
    if HAS_LGB:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "min_child_samples": 10,
            "random_state": 42,
            "verbose": -1,
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        print("  Trained LightGBM meta-learner.")
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        clf.fit(X_train, y_train)
        print("  Trained GradientBoosting meta-learner (LightGBM not available).")
    return clf


def cv_estimate(X, y, n_splits=5):
    """5-fold CV estimate of meta-learner accuracy on pseudo-test data."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for tr, val in skf.split(X, y):
        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y[tr], y[val]
        clf_tmp = train_meta_learner(X_tr, y_tr)
        y_hat = clf_tmp.predict(X_val)
        accs.append(accuracy_score(y_val, y_hat))
    return float(np.mean(accs)), float(np.std(accs))


# ===========================================================================
# STEP 2B — Agreement heuristic (Router B)
# ===========================================================================

def agreement_heuristic(sys_preds_cell, controlled_genus):
    """Agreement-based heuristic selection among system predictions."""
    avail = {s: v for s, v in sys_preds_cell.items() if v is not None}
    if not avail:
        return None
    ctr = Counter(avail.values())
    plurality, n_mode = ctr.most_common(1)[0]

    if n_mode >= 6:
        return plurality

    v4E = avail.get("v4E")
    v4I = avail.get("v4I")
    v3g = avail.get("v3_genus")

    if controlled_genus and v4E and v3g:
        if v4E == v3g:
            return v4E
        return v4E   # v4E preferred for controlled genus

    if v4I is not None:
        return v4I
    if v4E is not None:
        return v4E

    return plurality


# ===========================================================================
# Submission writer
# ===========================================================================

def _write_submission(path, preds_by_lang, test_meta_df, test_blank,
                      test_obs_strs, freq_fallback_str):
    """Write a SIGTYP TSV submission file."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(HEADER)
        for _, row in test_meta_df.iterrows():
            wc  = row["wals_code"]
            blk = test_blank.get(wc, set())
            obs = test_obs_strs.get(wc, {})
            mp  = preds_by_lang.get(wc, {})
            parts = []
            for fn in sorted(set(obs.keys()) | blk):
                if fn not in blk:
                    parts.append(f"{fn}={obs[fn]}")
                else:
                    pid = mp.get(fn) or freq_fallback_str.get(fn) or "1"
                    parts.append(f"{fn}={pid} -")
            meta = "\t".join([wc, str(row["name"]),
                               str(row["latitude"]), str(row["longitude"]),
                               str(row["genus"]), str(row["family"]),
                               str(row["countrycodes"])])
            fh.write(f"{meta}\t{'|'.join(parts)}\n")


# ===========================================================================
# Scoring helpers
# ===========================================================================

def _macro_acc(per_genus):
    if not per_genus:
        return 0.0
    return float(np.mean([float(np.mean(v)) for v in per_genus.values() if v]))


def compute_macro(preds_by_lang, gold_by_lang, test_meta_df, test_blank):
    """Compute macro accuracy (per-genus mean of per-language accuracy)."""
    per_genus = defaultdict(list)
    for _, row in test_meta_df.iterrows():
        wc    = row["wals_code"]
        genus = row["genus"]
        blk   = test_blank.get(wc, set())
        gold  = gold_by_lang.get(wc, {})
        pred  = preds_by_lang.get(wc, {})
        c, t  = 0, 0
        for fn in blk:
            gv = gold.get(fn)
            pv = pred.get(fn)
            if gv is None:
                continue
            t += 1
            c += int(pv is not None and pv == gv)
        if t > 0:
            per_genus[genus].append(c / t)
    return _macro_acc(per_genus)


def _run_scorer(paths):
    """Run official scorer with macro mode enabled."""
    with open(SCORER, "r", encoding="utf-8") as fh:
        src = fh.read()
    src_mod = src.replace(
        'for mode in ("micro",):  #, "macro"):',
        'for mode in ("micro", "macro"):'
    )
    tmp = os.path.join(os.path.dirname(SCORER), "_router_score_tmp.py")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(src_mod)
        result = subprocess.run(
            [sys.executable, tmp] + list(paths),
            capture_output=True, text=True,
            cwd=os.path.dirname(SCORER),
        )
        return result.stdout, result.stderr
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def oracle_ceiling(gold_by_lang, test_blank, test_meta_df, all_sys_preds,
                   feat_name_to_idx, feat_to_vals):
    """Oracle ceiling: fraction of blanked cells where ≥1 base system was correct."""
    per_genus = defaultdict(list)
    for _, row in test_meta_df.iterrows():
        wc    = row["wals_code"]
        genus = row["genus"]
        blk   = test_blank.get(wc, set())
        gold  = gold_by_lang.get(wc, {})
        c, t  = 0, 0
        for fn in blk:
            gv = gold.get(fn)
            fi = feat_name_to_idx.get(fn)
            if gv is None or fi is None:
                continue
            t += 1
            any_correct = False
            for sn in SYS_NAMES:
                pv = all_sys_preds.get(sn, {}).get(wc, {}).get(fn)
                if pv is not None and pv == gv:
                    any_correct = True
                    break
            c += int(any_correct)
        if t > 0:
            per_genus[genus].append(c / t)
    return _macro_acc(per_genus)


# ===========================================================================
# Argparse
# ===========================================================================

def get_args():
    p = argparse.ArgumentParser(description="SIGTYP 2020 router meta-learner")
    p.add_argument("--v4e-path",           default=None,
                   help="Path to v4E system TSV file")
    p.add_argument("--v4i-path",           default=None,
                   help="Path to v4I system TSV file")
    p.add_argument("--v4-ckpt",            default=None,
                   help="Path to set_encoder_v4_seed42.pt checkpoint")
    p.add_argument("--force-rebuild-train", action="store_true",
                   help="Force rebuild of LOO training data even if cache exists")
    p.add_argument("--output-dir",         default=BASE_DIR,
                   help="Directory to write output TSV files and cache")
    return p.parse_args()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    args = get_args()
    t0   = time.time()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("SIGTYP 2020 Router — meta-learner selecting among 10 base systems")
    print("=" * 72)

    # Update paths from args
    if args.v4e_path:
        SYSTEM_FILES["v4E"] = args.v4e_path
    if args.v4i_path:
        SYSTEM_FILES["v4I"] = args.v4i_path

    # -----------------------------------------------------------------------
    # STEP 0 — Load everything
    # -----------------------------------------------------------------------
    print("\n[Step 0] Loading SIGTYP data ...")
    train_meta, train_feat, train_blank, train_obs = parse_tsv(TRAIN_CSV)
    dev_meta,   dev_feat,   dev_blank,   dev_obs   = parse_tsv(DEV_CSV)
    test_meta,  test_feat,  test_blank,  test_obs  = parse_tsv(TEST_BLIND)
    _,          _,          _,           gold_obs  = parse_tsv(TEST_GOLD)

    print(f"  Train: {len(train_meta)} langs  Dev: {len(dev_meta)}  Test: {len(test_meta)}")

    # Build gold_by_lang (value id strings)
    gold_by_lang = {}
    for wc, obs_d in gold_obs.items():
        gold_by_lang[wc] = {fn: vs.split()[0] for fn, vs in obs_d.items()}

    # Build vocab from train+dev
    print("\n[Step 0] Building vocabulary from train+dev ...")
    traindev_feat = pd.concat([train_feat, dev_feat], ignore_index=True)
    kept_names, feat_to_vals, val_to_loc, cat_matrix_td = build_vocab(traindev_feat)
    n_train_split = len(train_feat)
    cat_matrix_train = cat_matrix_td[:n_train_split]
    n_feats = len(kept_names)
    feat_name_to_idx = {name: i for i, name in enumerate(kept_names)}
    print(f"  {n_feats} features retained")

    # IDF weights from train
    idf_weights = compute_idf(cat_matrix_train)

    # Full train marginals
    marginals = build_marginals(cat_matrix_train, feat_to_vals)

    # Frequency fallback strings
    freq_fallback_str = {}
    for fi, name in enumerate(kept_names):
        local_vi = int(np.argmax(marginals[fi]))
        freq_fallback_str[name] = feat_to_vals[fi][local_vi]

    # Train-side sets for lang-level features
    train_genera   = set(train_meta["genus"].tolist())
    train_families = set(train_meta["family"].tolist())
    train_latlons  = np.column_stack([
        train_meta["latitude"].values.astype(float),
        train_meta["longitude"].values.astype(float),
    ])

    # Feature-level stats (from full train)
    feat_stats = {}
    for fi in range(n_feats):
        col = cat_matrix_train[:, fi]
        feat_freq_  = int((col >= 0).sum())
        n_values_   = len(feat_to_vals[fi])
        entropy_val_= feat_entropy(marginals[fi])
        try:
            majority_val_int_ = int(feat_to_vals[fi][int(np.argmax(marginals[fi]))])
        except (ValueError, TypeError):
            majority_val_int_ = 0
        feat_stats[fi] = (feat_freq_, n_values_, entropy_val_, majority_val_int_)

    # -----------------------------------------------------------------------
    # Load all system predictions for blanked test cells
    # -----------------------------------------------------------------------
    print("\n[Step 0] Loading base-system predictions ...")
    all_sys_preds = {}
    for sn in SYS_NAMES:
        path = SYSTEM_FILES.get(sn)
        print(f"  Loading {sn} ...")
        all_sys_preds[sn] = load_system_preds(
            path, test_blank, feat_name_to_idx, val_to_loc, feat_to_vals)

    # Per-system accuracy
    print("\n  Per-system test accuracy:")
    sys_macros = {}
    for sn in SYS_NAMES:
        sp = all_sys_preds.get(sn, {})
        pg = defaultdict(list)
        for _, row in test_meta.iterrows():
            wc  = row["wals_code"]
            g   = row["genus"]
            blk = test_blank.get(wc, set())
            gold = gold_by_lang.get(wc, {})
            c, t = 0, 0
            for fn in blk:
                gv = gold.get(fn)
                if gv is None:
                    continue
                pv = sp.get(wc, {}).get(fn)
                t += 1
                c += int(pv == gv)
            if t > 0:
                pg[g].append(c / t)
        mac = _macro_acc(pg)
        sys_macros[sn] = mac
        print(f"    {sn:15s}: {mac:.4f}")

    # Oracle ceiling
    oracle = oracle_ceiling(gold_by_lang, test_blank, test_meta, all_sys_preds,
                            feat_name_to_idx, feat_to_vals)
    print(f"\n  Oracle ceiling (any-system correct): {oracle:.4f}")

    # -----------------------------------------------------------------------
    # STEP 1/2 — Build LOO pseudo-test training data (with cache)
    # -----------------------------------------------------------------------
    cache_path = os.path.join(output_dir, "router_train_data.pkl")
    if os.path.exists(cache_path) and not args.force_rebuild_train:
        print(f"\n[Step 2] Loading cached LOO training data from {cache_path} ...")
        with open(cache_path, "rb") as fh:
            X_train, y_train = pickle.load(fh)
        print(f"  Loaded {len(X_train)} rows.")
    else:
        print("\n[Step 2] Building LOO pseudo-test training data ...")
        X_train, y_train = build_loo_train_data(
            train_meta, cat_matrix_train, kept_names,
            feat_name_to_idx, feat_to_vals, val_to_loc, idf_weights)
        with open(cache_path, "wb") as fh:
            pickle.dump((X_train, y_train), fh)
        print(f"  Saved cache to {cache_path}")

    print(f"  Training data shape: {X_train.shape}  positive rate: {y_train.mean():.3f}")

    # -----------------------------------------------------------------------
    # STEP 3 — Train meta-learner
    # -----------------------------------------------------------------------
    print("\n[Step 3] Training meta-learner A ...")
    clf = train_meta_learner(X_train, y_train)

    # 5-fold CV estimate (quick, on subsample if large)
    n_cv = min(len(X_train), 50000)
    if n_cv < len(X_train):
        rng_cv = np.random.RandomState(0)
        idx_cv = rng_cv.choice(len(X_train), n_cv, replace=False)
        X_cv, y_cv = X_train[idx_cv], y_train[idx_cv]
    else:
        X_cv, y_cv = X_train, y_train
    print("  Running 5-fold CV estimate ...")
    cv_mean, cv_std = cv_estimate(X_cv, y_cv)
    print(f"  CV binary accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

    # Feature importances
    if HAS_LGB:
        importances = clf.feature_importances_
        top15 = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])[:15]
        print("\n  LightGBM top-15 feature importances:")
        for name, imp in top15:
            print(f"    {name:30s}: {imp:.0f}")

    # -----------------------------------------------------------------------
    # STEP 4 — Test inference
    # -----------------------------------------------------------------------
    print("\n[Step 4] Test inference ...")

    preds_A = {}
    preds_B = {}

    # Pre-compute per-language n_obs_in_vocab for test
    for _, row in test_meta.iterrows():
        wc  = row["wals_code"]
        blk = test_blank.get(wc, set())
        obs = test_obs.get(wc, {})

        n_obs_in_vocab = sum(1 for fn in obs if fn in feat_name_to_idx)

        lang_preds_A = {}
        lang_preds_B = {}

        for fn in blk:
            fi = feat_name_to_idx.get(fn)
            if fi is None:
                fallback = freq_fallback_str.get(fn, "1")
                lang_preds_A[fn] = fallback
                lang_preds_B[fn] = fallback
                continue

            # Build sys predictions for this cell
            sp = {sn: all_sys_preds.get(sn, {}).get(wc, {}).get(fn)
                  for sn in SYS_NAMES}

            # Router B — agreement heuristic
            b_pred = agreement_heuristic(sp, row["genus"] in CONTROLLED_GENERA)
            if b_pred is None:
                b_pred = freq_fallback_str.get(fn, feat_to_vals[fi][0])
            lang_preds_B[fn] = b_pred

            # Router A — ranking meta-learner
            candidates = list({v for v in sp.values() if v is not None})
            train_maj  = feat_to_vals[fi][int(np.argmax(marginals[fi]))]
            if train_maj not in candidates:
                candidates.append(train_maj)
            if not candidates:
                lang_preds_A[fn] = train_maj
                continue

            feat_freq_, n_values_, entropy_val_, majority_val_int_ = feat_stats[fi]
            lat_ = float(row["latitude"])
            lon_ = float(row["longitude"])
            if not np.isnan(lat_) and train_latlons.shape[0] > 0:
                min_dist_ = haversine_km_vec(lat_, lon_,
                                             train_latlons[:, 0],
                                             train_latlons[:, 1])
            else:
                min_dist_ = 0.0

            genus_in_train_   = int(row["genus"] in train_genera)
            family_in_train_  = int(row["family"] in train_families)
            controlled_genus_ = int(row["genus"] in CONTROLLED_GENERA)

            avail_preds = {s: v for s, v in sp.items() if v is not None}
            n_avail     = len(avail_preds)
            if n_avail > 0:
                ctr      = Counter(avail_preds.values())
                mode_val, n_mode = ctr.most_common(1)[0]
                all_agree_  = int(n_mode == n_avail)
                top2        = ctr.most_common(2)
                top2_agree_ = top2[1][1] if len(top2) > 1 else 0
            else:
                mode_val    = None
                n_mode      = 0
                all_agree_  = 0
                top2_agree_ = 0

            v4E_  = avail_preds.get("v4E")
            v4I_  = avail_preds.get("v4I")
            v3g_  = avail_preds.get("v3_genus")
            v3cp_ = avail_preds.get("v3_condprob")

            v4E_in_plurality_   = int(v4E_ is not None and v4E_ == mode_val)
            v3genus_in_plural_  = int(v3g_ is not None and v3g_ == mode_val)
            v4E_v3genus_agree_  = int(v4E_ is not None and v3g_ is not None and v4E_ == v3g_)
            v4E_condprob_agree_ = int(v4E_ is not None and v3cp_ is not None and v4E_ == v3cp_)
            v4I_v4E_agree_      = int(v4I_ is not None and v4E_ is not None and v4I_ == v4E_)

            X_cands = []
            for cand in candidates:
                vote_count_    = sum(1 for v in avail_preds.values() if v == cand)
                vote_frac_     = vote_count_ / max(n_avail, 1)
                is_train_maj_  = int(cand == train_maj)
                is_v1_freq_    = int(sp.get("v1_freq")     == cand)
                is_v3_freq_    = int(sp.get("v3_freq")     == cand)
                is_v3_condprob_= int(sp.get("v3_condprob") == cand)
                is_v3_knn_     = int(sp.get("v3_knn")      == cand)
                is_v3_idfknn_  = int(sp.get("v3_idfknn")   == cand)
                is_v3_genus_   = int(sp.get("v3_genus")    == cand)
                is_v4E_        = int(sp.get("v4E")         == cand)
                is_v4I_        = int(sp.get("v4I")         == cand)
                is_v1_neural_  = int(
                    sp.get("v1_learned") == cand or sp.get("v1_tcf") == cand)
                try:
                    cand_val_int_ = int(cand)
                except (ValueError, TypeError):
                    cand_val_int_ = 0

                row_feat = [
                    genus_in_train_, family_in_train_, n_obs_in_vocab,
                    lat_, lon_, min_dist_, controlled_genus_,
                    feat_freq_, n_values_, entropy_val_, majority_val_int_,
                    n_mode, all_agree_, top2_agree_,
                    v4E_in_plurality_, v3genus_in_plural_,
                    v4E_v3genus_agree_, v4E_condprob_agree_, v4I_v4E_agree_,
                    vote_count_, vote_frac_, is_train_maj_,
                    is_v1_freq_, is_v3_freq_, is_v3_condprob_,
                    is_v3_knn_, is_v3_idfknn_, is_v3_genus_,
                    is_v4E_, is_v4I_, is_v1_neural_,
                    cand_val_int_,
                ]
                X_cands.append(row_feat)

            X_arr  = np.array(X_cands, dtype=np.float32)
            scores = clf.predict_proba(X_arr)[:, 1]
            best   = candidates[int(np.argmax(scores))]

            # Post-hoc neural boost: if v4E or v4I available and ≥3 neural agree
            neural_votes = [sp.get(s) for s in ["v4E", "v4I", "v1_learned", "v1_tcf"]
                            if sp.get(s) is not None]
            if len(neural_votes) >= 2:
                n_ctr = Counter(neural_votes)
                n_val, n_cnt = n_ctr.most_common(1)[0]
                if n_cnt >= 3 and n_val != best:
                    best = n_val

            lang_preds_A[fn] = best

        preds_A[wc] = lang_preds_A
        preds_B[wc] = lang_preds_B

    # -----------------------------------------------------------------------
    # Write submissions
    # -----------------------------------------------------------------------
    path_A = os.path.join(output_dir, "sigtyp_router_A.tsv")
    path_B = os.path.join(output_dir, "sigtyp_router_B.tsv")
    _write_submission(path_A, preds_A, test_meta, test_blank, test_obs, freq_fallback_str)
    _write_submission(path_B, preds_B, test_meta, test_blank, test_obs, freq_fallback_str)
    print(f"\n  Wrote Router A: {path_A}")
    print(f"  Wrote Router B: {path_B}")

    # -----------------------------------------------------------------------
    # STEP 5 — Score and report
    # -----------------------------------------------------------------------
    print("\n[Step 5] Scoring ...")

    macro_A = compute_macro(preds_A, gold_by_lang, test_meta, test_blank)
    macro_B = compute_macro(preds_B, gold_by_lang, test_meta, test_blank)
    print(f"\n  Router A macro: {macro_A:.4f}")
    print(f"  Router B macro: {macro_B:.4f}")
    print(f"  Oracle ceiling: {oracle:.4f}")
    print(f"  UFAL baseline:  0.7500")

    # Official scorer
    print("\n  Running official scorer ...")
    try:
        stdout, stderr = _run_scorer([TEST_GOLD, path_A, path_B])
        if stdout:
            print(stdout)
        if stderr.strip():
            print("SCORER STDERR:", stderr[:500])
    except Exception as e:
        print(f"  Official scorer failed: {e}")

    # Per-system accuracy table
    print("\n" + "=" * 60)
    print("Per-system accuracy comparison")
    print("=" * 60)
    print(f"{'System':<20} {'Macro':>8}")
    print("-" * 30)
    for sn in SYS_NAMES:
        m = sys_macros.get(sn, float("nan"))
        print(f"{sn:<20} {m:8.4f}")
    print("-" * 30)
    print(f"{'Router A':<20} {macro_A:8.4f}")
    print(f"{'Router B':<20} {macro_B:8.4f}")
    print(f"{'Oracle ceiling':<20} {oracle:8.4f}")
    print(f"{'UFAL target':<20} {'0.7500':>8}")
    for label, mac in [("Router A", macro_A), ("Router B", macro_B)]:
        gap = mac - 0.75
        sign = "+" if gap >= 0 else ""
        status = "BEATS UFAL!" if gap >= 0 else f"gap = {sign}{gap:.4f}"
        print(f"  {label}: {status}")
    print("=" * 60)

    # Per-genus routing decisions for Router A
    print("\n  Per-genus routing decisions (Router A):")
    genus_routing = defaultdict(lambda: defaultdict(int))
    for _, row in test_meta.iterrows():
        wc     = row["wals_code"]
        genus  = row["genus"]
        blk    = test_blank.get(wc, set())
        pred_a = preds_A.get(wc, {})
        for fn in blk:
            a_val = pred_a.get(fn)
            if a_val is None:
                continue
            matched = []
            for sn in SYS_NAMES:
                sys_val = all_sys_preds.get(sn, {}).get(wc, {}).get(fn)
                if sys_val == a_val:
                    matched.append(sn)
            if matched:
                genus_routing[genus][matched[0]] += 1
            else:
                genus_routing[genus]["freq_fallback"] += 1

    for genus in sorted(genus_routing.keys()):
        top_choices = Counter(genus_routing[genus]).most_common(3)
        total = sum(genus_routing[genus].values())
        choices_str = ", ".join(f"{s}={c}" for s, c in top_choices)
        print(f"    {genus:<30s}: total={total}  top={choices_str}")

    # Remaining gap analysis
    print("\n  Remaining gap: cells where router A is wrong but oracle is right:")
    n_gap = 0
    n_total_blanked = 0
    for _, row in test_meta.iterrows():
        wc     = row["wals_code"]
        blk    = test_blank.get(wc, set())
        gold   = gold_by_lang.get(wc, {})
        pred_a = preds_A.get(wc, {})
        for fn in blk:
            gv = gold.get(fn)
            if gv is None:
                continue
            n_total_blanked += 1
            a_val = pred_a.get(fn)
            if a_val != gv:
                # Check if any system was right
                any_right = any(
                    all_sys_preds.get(sn, {}).get(wc, {}).get(fn) == gv
                    for sn in SYS_NAMES
                )
                if any_right:
                    n_gap += 1
    print(f"    {n_gap} / {n_total_blanked} cells "
          f"({100*n_gap/max(n_total_blanked,1):.1f}%) could be improved by better routing")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
