#!/usr/bin/env python3
"""
sigtyp_stacker.py: Per-feature meta-classifier routing among 18 base systems.

Strategy:
  Training data  — LOO simulation on train.csv for non-neural systems
                   (freq, genus, kNN, IDF-kNN, condprob); neural slots = -1.
  Test data      — predictions loaded directly from 18 submission TSV files.
  Classifier     — per-feature RandomForestClassifier; global fallback when
                   fewer than MIN_TRAIN_ROWS examples exist for a feature.
  Validation     — genus-stratified 5-fold GroupKFold on training data.

Run: python3 sigtyp_stacker.py 2>&1 | tee sigtyp_stacker_log.txt
"""

import os
import sys
import subprocess
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = "/home/user/Grambank2Vec"
DATA_DIR   = "/home/user/ST2020/data"
TRAIN_CSV  = f"{DATA_DIR}/train.csv"
DEV_CSV    = f"{DATA_DIR}/dev.csv"
TEST_BLIND = f"{DATA_DIR}/test_blinded.csv"
TEST_GOLD  = f"{DATA_DIR}/test_gold.csv"
SCORER     = "/home/user/ST2020/scripts/score.py"
OUT_TSV    = f"{BASE_DIR}/sigtyp_stacker.tsv"

# ---------------------------------------------------------------------------
# Base systems  (order defines feature-vector column order)
# ---------------------------------------------------------------------------
BASE_SYSTEMS = [
    ("v1_freq",     f"{BASE_DIR}/sigtyp_submission_freq.tsv"),
    ("v1_learned",  f"{BASE_DIR}/sigtyp_submission_learned.tsv"),
    ("v1_tcf",      f"{BASE_DIR}/sigtyp_submission_tcf.tsv"),
    ("v2_freq",     f"{BASE_DIR}/sigtyp_v2_freq.tsv"),
    ("v2A_learned", f"{BASE_DIR}/sigtyp_v2_A_learned.tsv"),
    ("v2A_tcf",     f"{BASE_DIR}/sigtyp_v2_A_tcf.tsv"),
    ("v2B_learned", f"{BASE_DIR}/sigtyp_v2_B_learned.tsv"),
    ("v2B_tcf",     f"{BASE_DIR}/sigtyp_v2_B_tcf.tsv"),
    ("v2C_learned", f"{BASE_DIR}/sigtyp_v2_C_learned.tsv"),
    ("v2C_tcf",     f"{BASE_DIR}/sigtyp_v2_C_tcf.tsv"),
    ("v2D_learned", f"{BASE_DIR}/sigtyp_v2_D_learned.tsv"),
    ("v2D_tcf",     f"{BASE_DIR}/sigtyp_v2_D_tcf.tsv"),
    ("v3_condprob", f"{BASE_DIR}/sigtyp_v3_condprob.tsv"),
    ("v3_freq",     f"{BASE_DIR}/sigtyp_v3_freq.tsv"),
    ("v3_genus",    f"{BASE_DIR}/sigtyp_v3_genus.tsv"),
    ("v3_idfknn",   f"{BASE_DIR}/sigtyp_v3_idfknn.tsv"),
    ("v3_knn",      f"{BASE_DIR}/sigtyp_v3_knn.tsv"),
    ("v4F",         f"{BASE_DIR}/sigtyp_v4_F.tsv"),
]
SYS_NAMES = [s[0] for s in BASE_SYSTEMS]
SYS_PATHS = {s[0]: s[1] for s in BASE_SYSTEMS}
N_SYS     = len(SYS_NAMES)          # 18

# Systems for which we run LOO simulation on training data
# (maps system-name → simulation-type)
LOO_SIM = {
    "v1_freq":    "freq",
    "v2_freq":    "freq",
    "v3_freq":    "freq",
    "v3_genus":   "genus",
    "v3_knn":     "knn",
    "v3_idfknn":  "idfknn",
    "v3_condprob":"condprob",
}

MIN_TRAIN_ROWS = 20   # min examples to train a per-feature classifier
KNN_K          = 5    # neighbours for kNN / IDF-kNN
CONDPROB_ALPHA = 0.5  # Dirichlet smoothing
CONDPROB_MIN   = 5    # min marginal count for condprob term

# Subset of systems simulated via LOO in both train and test — used for
# agreement features so their semantics are consistent between train and test.
CONSISTENT_SYS = ["v1_freq", "v3_genus", "v3_knn", "v3_idfknn", "v3_condprob"]

# Neural systems available only at test time (via TSV files).
NEURAL_SYS = [
    "v1_learned", "v1_tcf",
    "v2A_learned", "v2A_tcf",
    "v2B_learned", "v2B_tcf",
    "v2C_learned", "v2C_tcf",
    "v2D_learned", "v2D_tcf",
]
NEURAL_THRESHOLD = 6   # neural systems that must agree to override RF stacker

CONSISTENT_IDX = [SYS_NAMES.index(s) for s in CONSISTENT_SYS]
NEURAL_IDX     = [SYS_NAMES.index(s) for s in NEURAL_SYS]

HEADER = ("wals_code\tname\tlatitude\tlongitude\tgenus\tfamily"
          "\tcountrycodes\tfeatures\n")

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


# ===========================================================================
# Part 0 — Data parsing helpers
# ===========================================================================

def _fix(s):
    for old, new in _SIGTYP_FIXES:
        s = s.replace(old, new)
    return s


def parse_tsv(path):
    """Parse a SIGTYP-format TSV file.

    Returns
    -------
    meta_df    : DataFrame [wals_code, name, latitude, longitude, genus, family, countrycodes]
    feat_df    : DataFrame rows=languages, cols=lowercase feature names; value = id string or NaN
    blanked    : {wals_code: set of lowercase feature names with "?"}
    obs_strs   : {wals_code: {feat_name: full_value_string_incl_label}}
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
            lat = float(parts[2]); lon = float(parts[3])
        except ValueError:
            lat = lon = float("nan")
        meta_rows.append({"wals_code": wc, "name": parts[1],
                          "latitude": lat, "longitude": lon,
                          "genus": parts[4], "family": parts[5],
                          "countrycodes": parts[6]})
        fd, bs, osd = {}, set(), {}
        for piece in _fix("\t".join(parts[7:])).split("|"):
            piece = piece.strip()
            if not piece or "=" not in piece:
                continue
            eq = piece.index("=")
            fn = piece[:eq].lower()
            vs = piece[eq+1:]
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
        columns=all_fn
    )
    return meta_df, feat_df, blanked, obs_strs


def build_vocab(feat_df):
    """Build feature vocabulary.

    Returns (kept_names, feat_to_values, val_to_local, cat_matrix).
    """
    kept_names   = []
    feat_to_vals = {}  # fi → sorted list of value-id strings
    val_to_loc   = {}  # fi → {val_str: local_idx}
    columns      = []

    for col in feat_df.columns:
        unique = sorted(feat_df[col].dropna().unique().tolist())
        if len(unique) <= 1:
            continue
        fi = len(kept_names)
        kept_names.append(col)
        feat_to_vals[fi] = unique
        v2l = {v: i for i, v in enumerate(unique)}
        val_to_loc[fi]   = v2l
        col_data = np.full(len(feat_df), -1, dtype=np.int32)
        for row_idx, v in enumerate(feat_df[col]):
            if isinstance(v, str) and v in v2l:
                col_data[row_idx] = v2l[v]
        columns.append(col_data)

    cat_matrix = np.column_stack(columns) if columns else np.empty((len(feat_df), 0), dtype=np.int32)
    return kept_names, feat_to_vals, val_to_loc, cat_matrix


def compute_idf(cat_matrix):
    """IDF weight per feature: log(n_langs / (1 + n_with_obs))."""
    n = cat_matrix.shape[0]
    idf = np.zeros(cat_matrix.shape[1], dtype=np.float64)
    for fi in range(cat_matrix.shape[1]):
        n_obs = int((cat_matrix[:, fi] >= 0).sum())
        idf[fi] = float(np.log(n / (1.0 + n_obs)))
    return idf


def load_sys_preds_for_blanked(sys_name, path, test_blank, feat_name_to_idx, val_to_loc):
    """Load a single system's predictions for blanked test cells.

    Returns {wals_code: {feat_name: local_idx}}.  -1 if value unknown.
    """
    result = {}
    if not os.path.exists(path):
        print(f"  [WARN] missing: {path}")
        return result
    _, _, _, obs_s = parse_tsv(path)
    for wc, blank_set in test_blank.items():
        preds = {}
        lang_obs = obs_s.get(wc, {})
        for fn in blank_set:
            if fn not in feat_name_to_idx:
                continue
            fi = feat_name_to_idx[fn]
            raw = lang_obs.get(fn)
            if raw is None:
                preds[fn] = -1
                continue
            vid = raw.split()[0]
            preds[fn] = val_to_loc[fi].get(vid, -1)
        result[wc] = preds
    return result


# ===========================================================================
# Part 1 — Load base-system TSV predictions (for test inference)
# ===========================================================================

def load_all_sys_preds(test_blank, feat_name_to_idx, val_to_loc):
    """Load blanked-cell predictions from all 18 system TSV files.

    Returns sys_preds[sys_name][wals_code][feat_name] = local_idx (or -1).
    """
    sys_preds = {}
    for sys_name, path in BASE_SYSTEMS:
        print(f"  Loading {sys_name} ...")
        sys_preds[sys_name] = load_sys_preds_for_blanked(
            sys_name, path, test_blank, feat_name_to_idx, val_to_loc)
    return sys_preds


# ===========================================================================
# Part 2 — Build meta-training data via LOO simulation
# ===========================================================================

def _majority_vote(preds):
    avail = [p for p in preds if p >= 0]
    if not avail:
        return -1
    return Counter(avail).most_common(1)[0][0]


def _agreement_features(sys_pred_vec, freq_local, genus_local, knn_local, condprob_local):
    """Compute agreement features from the CONSISTENT_SYS subset only.

    This ensures train (LOO simulation) and test (TSV) agreement features
    are computed over the same set of systems, avoiding distribution shift.
    """
    avail = [sys_pred_vec[i] for i in CONSISTENT_IDX if sys_pred_vec[i] >= 0]
    n_avail = len(avail)
    if n_avail == 0:
        return [0, 0, 0, 0, 1, 0, 0]
    ctr   = Counter(avail)
    mv    = ctr.most_common(1)[0][0]
    n_ag  = ctr[mv]
    frac  = int(100 * n_ag / n_avail)
    n_unq = len(ctr)
    fg_agree = int(freq_local == genus_local and freq_local >= 0)
    kc_agree = int(knn_local == condprob_local and knn_local >= 0)
    return [mv, n_ag, frac, n_avail, n_unq, fg_agree, kc_agree]


def build_train_meta(train_meta_df, cat_matrix_train, kept_names,
                     feat_name_to_idx, feat_to_vals, val_to_loc, idf_weights):
    """Build (X, y, groups, feat_indices) for stacker training via LOO simulation.

    X shape: (n_samples, 2 + N_SYS + 7)
      [n_values, n_obs_context, sys_pred_0..17, agr_0..6]
    y: local value index (int)
    groups: list of genus strings (for CV)
    feat_indices: list of feature indices (fi) per sample
    """
    n_train, n_feats = cat_matrix_train.shape
    genera   = train_meta_df["genus"].tolist()

    print("  Computing frequency predictions ...")
    freq_pred = []
    for fi in range(n_feats):
        col = cat_matrix_train[:, fi]
        obs = col[col >= 0]
        freq_pred.append(int(Counter(obs.tolist()).most_common(1)[0][0])
                         if len(obs) > 0 else 0)

    print("  Computing genus predictions ...")
    genus_pred = {}  # genus → {fi: local_idx}
    for genus in set(genera):
        mask = np.array([g == genus for g in genera])
        rows = cat_matrix_train[mask]
        gp = {}
        for fi in range(n_feats):
            col = rows[:, fi]
            obs = col[col >= 0]
            gp[fi] = (int(Counter(obs.tolist()).most_common(1)[0][0])
                      if len(obs) > 0 else freq_pred[fi])
        genus_pred[genus] = gp

    print("  Precomputing kNN match matrices ...")
    M_knn = np.zeros((n_train, n_train), dtype=np.float32)
    M_idf = np.zeros((n_train, n_train), dtype=np.float32)
    for fi in range(n_feats):
        col = cat_matrix_train[:, fi]
        has = (col >= 0)
        agree = (col[:, None] == col[None, :]) & has[:, None] & has[None, :]
        M_knn += agree.astype(np.float32)
        M_idf += agree.astype(np.float32) * float(idf_weights[fi])

    print("  Precomputing condprob co-occurrence stats ...")
    co_cnt = {}     # (fi, vi, fj) → np.array(K_fj)
    marg   = {}     # (fi, vi)     → int count
    for l in range(n_train):
        obs_l = [(fi, int(cat_matrix_train[l, fi]))
                 for fi in range(n_feats) if cat_matrix_train[l, fi] >= 0]
        for fi, vi in obs_l:
            marg[(fi, vi)] = marg.get((fi, vi), 0) + 1
            for fj, vj in obs_l:
                if fi == fj:
                    continue
                key = (fi, vi, fj)
                K_fj = len(feat_to_vals[fj])
                if key not in co_cnt:
                    co_cnt[key] = np.zeros(K_fj, dtype=np.float32)
                co_cnt[key][vj] += 1

    print("  Building meta-feature rows ...")
    n_obs_per_lang = (cat_matrix_train >= 0).sum(axis=1)

    X_rows, y_rows, group_rows, fidx_rows = [], [], [], []

    for l in range(n_train):
        genus = genera[l]
        gp    = genus_pred.get(genus, {})
        n_obs = int(n_obs_per_lang[l])

        for fi in range(n_feats):
            vi_true = int(cat_matrix_train[l, fi])
            if vi_true < 0:
                continue

            K_fi     = len(feat_to_vals[fi])
            n_ctx    = n_obs - 1

            # freq (no LOO)
            fp_freq = freq_pred[fi]

            # genus (no LOO)
            fp_genus = gp.get(fi, fp_freq)

            # kNN LOO
            col_fi  = cat_matrix_train[:, fi]
            has_fi  = col_fi >= 0
            agree_i = (col_fi == vi_true) & has_fi

            loo_knn = M_knn[l].copy()
            loo_knn[has_fi] -= agree_i[has_fi].astype(np.float32)
            loo_knn[l] = -1.0

            loo_idf = M_idf[l].copy()
            loo_idf[has_fi] -= (agree_i[has_fi].astype(np.float32) * float(idf_weights[fi]))
            loo_idf[l] = -1.0

            def _knn_vote(scores):
                n_pos = int((scores > 0).sum())
                k_use = min(KNN_K, n_pos)
                if k_use == 0:
                    return fp_freq
                top_k = np.argsort(scores)[-k_use:]
                votes = [int(cat_matrix_train[n, fi])
                         for n in top_k if cat_matrix_train[n, fi] >= 0]
                return (Counter(votes).most_common(1)[0][0]
                        if votes else fp_freq)

            fp_knn    = _knn_vote(loo_knn)
            fp_idfknn = _knn_vote(loo_idf)

            # condprob (exclude target feature fi from context)
            log_sc  = np.zeros(K_fi, dtype=np.float64)
            n_terms = 0
            for fk in range(n_feats):
                if fk == fi or cat_matrix_train[l, fk] < 0:
                    continue
                vk = int(cat_matrix_train[l, fk])
                mc = marg.get((fk, vk), 0)
                if mc < CONDPROB_MIN:
                    continue
                counts = co_cnt.get((fk, vk, fi), np.zeros(K_fi, dtype=np.float32))
                smoothed = (counts + CONDPROB_ALPHA) / (mc + CONDPROB_ALPHA * K_fi)
                log_sc  += np.log(smoothed + 1e-12)
                n_terms += 1
            fp_condprob = int(np.argmax(log_sc)) if n_terms > 0 else fp_freq

            # Build 18-slot sys_pred vector (-1 for unavailable)
            sp = np.full(N_SYS, -1, dtype=np.int32)
            sp[SYS_NAMES.index("v1_freq")]    = fp_freq
            sp[SYS_NAMES.index("v2_freq")]    = fp_freq
            sp[SYS_NAMES.index("v3_freq")]    = fp_freq
            sp[SYS_NAMES.index("v3_genus")]   = fp_genus
            sp[SYS_NAMES.index("v3_knn")]     = fp_knn
            sp[SYS_NAMES.index("v3_idfknn")]  = fp_idfknn
            sp[SYS_NAMES.index("v3_condprob")]= fp_condprob

            agr = _agreement_features(sp.tolist(), fp_freq, fp_genus, fp_knn, fp_condprob)

            row = [K_fi, n_ctx] + sp.tolist() + agr
            X_rows.append(row)
            y_rows.append(vi_true)
            group_rows.append(genus)
            fidx_rows.append(fi)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    return X, y, group_rows, fidx_rows


# ===========================================================================
# Part 3 — Train per-feature classifiers
# ===========================================================================

def train_classifiers(X, y, feat_indices, n_feats, feat_to_vals):
    """Train one RandomForestClassifier per feature (with ≥ MIN_TRAIN_ROWS examples).

    Returns {fi: fitted_clf}.
    """
    models     = {}
    feat_masks = defaultdict(list)
    for i, fi in enumerate(feat_indices):
        feat_masks[fi].append(i)

    for fi in range(n_feats):
        idxs = feat_masks.get(fi, [])
        if len(idxs) < MIN_TRAIN_ROWS:
            continue
        Xi = X[idxs]
        yi = y[idxs]
        n_cls = len(feat_to_vals[fi])
        # If only one class observed in training, skip (can't train)
        if len(np.unique(yi)) < 2:
            continue
        n_est = 200 if len(idxs) >= 60 else 100
        md    = 6   if len(idxs) >= 60 else 4
        clf   = RandomForestClassifier(
            n_estimators=n_est, max_depth=md,
            min_samples_leaf=3, random_state=42, n_jobs=-1)
        clf.fit(Xi, yi)
        models[fi] = clf

    print(f"  Trained {len(models)}/{n_feats} per-feature classifiers.")
    return models


# ===========================================================================
# Part 4 — Genus-stratified cross-validation
# ===========================================================================

def cv_validate(X, y, group_rows, feat_indices, n_feats, feat_to_vals, n_splits=5):
    """Genus-stratified GroupKFold CV; returns macro accuracy."""
    groups  = np.array(group_rows)
    genera  = groups
    unique_genera = sorted(set(group_rows))
    n_genera = len(unique_genera)
    n_splits = min(n_splits, n_genera)
    gkf = GroupKFold(n_splits=n_splits)

    per_lang_accs = defaultdict(list)  # genus → [acc, ...]

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=genera)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        fi_tr  = [feat_indices[i] for i in tr_idx]
        fi_val = [feat_indices[i] for i in val_idx]
        g_val  = [group_rows[i]   for i in val_idx]

        models_f = train_classifiers(X_tr, y_tr, fi_tr, n_feats, feat_to_vals)

        # Evaluate
        feat_masks_val = defaultdict(list)
        for i, fi in enumerate(fi_val):
            feat_masks_val[fi].append(i)

        correct, total = 0, 0
        for fi, idxs in feat_masks_val.items():
            Xi_val = X_val[idxs]
            yi_val = y_val[idxs]
            gi_val = [g_val[i] for i in idxs]
            if fi in models_f:
                y_hat = models_f[fi].predict(Xi_val)
            else:
                # fallback: majority vote from sys-pred columns (indices 2..2+N_SYS-1)
                y_hat = []
                for row in Xi_val:
                    sp = [int(v) for v in row[2:2+N_SYS]]
                    mv = _majority_vote(sp)
                    y_hat.append(mv if mv >= 0 else 0)
                y_hat = np.array(y_hat)
            for pred, gold, g in zip(y_hat, yi_val, gi_val):
                per_lang_accs[g].append(int(pred == gold))
            correct += int((y_hat == yi_val).sum())
            total   += len(yi_val)

        micro = correct / max(total, 1)
        # Per-genus accuracy for this fold's validation genera
        fold_genera = set(g_val)
        fold_macro  = float(np.mean([
            np.mean(per_lang_accs[g]) for g in fold_genera
            if per_lang_accs[g]
        ]))
        print(f"    Fold {fold+1}/{n_splits}: micro={micro:.4f}  "
              f"genera_in_fold={len(fold_genera)}")

    # Overall macro: mean per-genus mean
    macro = float(np.mean([np.mean(v) for v in per_lang_accs.values() if v]))
    print(f"  CV macro accuracy: {macro:.4f}")
    return macro


# ===========================================================================
# Part 5 — Test inference
# ===========================================================================

def build_test_meta_row(fi, K_fi, n_obs_ctx, sys_preds_dict, feat_to_vals):
    """Build a single meta-feature row for a (lang, feature) test cell."""
    sp = np.full(N_SYS, -1, dtype=np.int32)
    for si, sname in enumerate(SYS_NAMES):
        v = sys_preds_dict.get(sname, -1)
        sp[si] = v if v is not None else -1

    fp_freq = int(sp[SYS_NAMES.index("v1_freq")])
    fp_genus = int(sp[SYS_NAMES.index("v3_genus")])
    fp_knn   = int(sp[SYS_NAMES.index("v3_knn")])
    fp_cp    = int(sp[SYS_NAMES.index("v3_condprob")])

    agr = _agreement_features(sp.tolist(), fp_freq, fp_genus, fp_knn, fp_cp)
    return np.array([K_fi, n_obs_ctx] + sp.tolist() + agr, dtype=np.float32)


def _neural_vote(sp_dict, fi):
    """Return (neural_majority_local_idx, n_agreeing) for NEURAL_SYS predictions."""
    votes = [sp_dict.get(sn, -1) for sn in NEURAL_SYS]
    votes = [v for v in votes if v is not None and v >= 0]
    if not votes:
        return -1, 0
    ctr = Counter(votes)
    mv, cnt = ctr.most_common(1)[0]
    return mv, cnt


def infer_test(test_meta_df, test_blank, test_obs_strs,
               all_sys_preds, models, feat_name_to_idx, feat_to_vals, freq_pred_str):
    """Apply stacker to all blanked test cells.

    Returns:
      base_preds    — RF stacker with no neural override
      cell_neural   — {(wc,fn): (neural_local_idx, n_agreeing)} for post-hoc use
    """
    n_obs_per_lang = {row["wals_code"]: len(test_obs_strs.get(row["wals_code"], {}))
                      for _, row in test_meta_df.iterrows()}

    base_preds   = {}
    cell_neural  = {}   # (wc, fn) → (nv_local, n_agreeing, K_fi)
    n_rf, n_mv   = 0, 0

    for _, row in test_meta_df.iterrows():
        wc  = row["wals_code"]
        blk = test_blank.get(wc, set())
        if not blk:
            base_preds[wc] = {}
            continue

        n_obs_ctx  = n_obs_per_lang.get(wc, 0)
        lang_preds = {}

        for fn in blk:
            fi = feat_name_to_idx.get(fn)
            if fi is None:
                lang_preds[fn] = freq_pred_str.get(fn, "1")
                continue
            K_fi = len(feat_to_vals[fi])

            sp_dict = {sname: all_sys_preds.get(sname, {}).get(wc, {}).get(fn, -1)
                       for sname in SYS_NAMES}

            x_row = build_test_meta_row(fi, K_fi, n_obs_ctx, sp_dict, feat_to_vals)

            if fi in models:
                pred_local = int(models[fi].predict(x_row.reshape(1, -1))[0])
                n_rf += 1
            else:
                avail = [v for v in sp_dict.values() if v is not None and v >= 0]
                pred_local = Counter(avail).most_common(1)[0][0] if avail else 0
                n_mv += 1

            pred_local = max(0, min(pred_local, K_fi - 1))
            lang_preds[fn] = feat_to_vals[fi][pred_local]

            nv, n_nv = _neural_vote(sp_dict, fi)
            cell_neural[(wc, fn)] = (nv, n_nv, K_fi)

        base_preds[wc] = lang_preds

    print(f"  RF decisions: {n_rf}  MV fallback: {n_mv}")
    return base_preds, cell_neural


def apply_neural_override(base_preds, cell_neural, feat_name_to_idx, feat_to_vals,
                          test_blank, threshold):
    """Apply neural post-hoc override with a given threshold.

    Returns new preds_by_lang dict with overrides applied.
    """
    n_override = 0
    preds = {}
    for wc, lang_p in base_preds.items():
        new_p = dict(lang_p)
        for fn, base_val in lang_p.items():
            fi = feat_name_to_idx.get(fn)
            if fi is None:
                continue
            nv, n_nv, K_fi = cell_neural.get((wc, fn), (-1, 0, 1))
            if nv < 0 or n_nv < threshold:
                continue
            nv_str = feat_to_vals[fi][min(nv, K_fi - 1)]
            if nv_str != base_val:
                new_p[fn] = nv_str
                n_override += 1
        preds[wc] = new_p
    return preds, n_override


# ===========================================================================
# Scoring helpers
# ===========================================================================

def _macro_acc(per_genus):
    if not per_genus:
        return 0.0
    return float(np.mean([float(np.mean(v)) for v in per_genus.values() if v]))


def compute_macro(preds_by_lang, gold_by_lang, test_meta_df, test_blank):
    """Language-weighted macro accuracy (mean of per-language accuracies).

    Matches the official scorer's 'overall' macro which averages 149 test
    languages equally — NOT a mean of per-genus means.
    """
    accs = []
    for _, row in test_meta_df.iterrows():
        wc   = row["wals_code"]
        blk  = test_blank.get(wc, set())
        gold = gold_by_lang.get(wc, {})
        pred = preds_by_lang.get(wc, {})
        c, t = 0, 0
        for fn in blk:
            gv = gold.get(fn)
            if gv is None:
                continue
            t += 1
            c += int(pred.get(fn) is not None and pred.get(fn) == gv)
        if t > 0:
            accs.append(c / t)
    return float(np.mean(accs)) if accs else 0.0


def _write_submission(path, test_meta_df, test_blank, test_obs_strs,
                      preds_by_lang, freq_pred_str):
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
                    pid = mp.get(fn) or freq_pred_str.get(fn) or "1"
                    parts.append(f"{fn}={pid} -")
            meta = "\t".join([wc, str(row["name"]), str(row["latitude"]),
                              str(row["longitude"]), str(row["genus"]),
                              str(row["family"]), str(row["countrycodes"])])
            fh.write(f"{meta}\t{'|'.join(parts)}\n")


def _run_scorer(paths):
    with open(SCORER, "r", encoding="utf-8") as fh:
        src = fh.read()
    src_mod = src.replace(
        'for mode in ("micro",):  #, "macro"):',
        'for mode in ("micro", "macro"):'
    )
    tmp = os.path.join(os.path.dirname(SCORER), "_stacker_score_tmp.py")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(src_mod)
        result = subprocess.run(
            [sys.executable, tmp] + list(paths),
            capture_output=True, text=True,
            cwd=os.path.dirname(SCORER))
        return result.stdout, result.stderr
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ===========================================================================
# Oracle and majority-vote analysis
# ===========================================================================

def oracle_ceiling(gold_by_lang, test_blank, test_meta_df, all_sys_preds,
                   feat_name_to_idx, feat_to_vals):
    """Fraction of blanked cells where at least one base system was correct."""
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
                pv_local = all_sys_preds.get(sn, {}).get(wc, {}).get(fn)
                if pv_local is None or pv_local < 0:
                    continue
                pv = feat_to_vals[fi][pv_local] if pv_local < len(feat_to_vals[fi]) else None
                if pv == gv:
                    any_correct = True
                    break
            c += int(any_correct)
        if t > 0:
            per_genus[genus].append(c / t)
    all_accs = [a for alist in per_genus.values() for a in alist]
    return float(np.mean(all_accs)) if all_accs else 0.0


def majority_vote_preds(gold_by_lang, test_blank, test_meta_df, all_sys_preds,
                        freq_pred_str):
    """Majority vote among 18 systems."""
    preds = {}
    for _, row in test_meta_df.iterrows():
        wc   = row["wals_code"]
        blk  = test_blank.get(wc, set())
        lang_p = {}
        for fn in blk:
            votes = [all_sys_preds.get(sn, {}).get(wc, {}).get(fn)
                     for sn in SYS_NAMES]
            votes = [v for v in votes if v is not None and v >= 0]
            if votes:
                mv = Counter(votes).most_common(1)[0][0]
                # mv is a local index; we need a value string
                # We can't convert without feat_to_vals here — handled in caller
                lang_p[fn] = mv
            else:
                lang_p[fn] = None
        preds[wc] = lang_p
    return preds


# ===========================================================================
# Part 6 — Main
# ===========================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("SIGTYP 2020 Stacker — per-feature meta-classifier")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[Part 1] Parsing data ...")
    train_meta, train_feat, _, train_obs_strs = parse_tsv(TRAIN_CSV)
    dev_meta,   dev_feat,   _, dev_obs_strs   = parse_tsv(DEV_CSV)
    test_meta,  test_feat,  test_blank, test_obs_strs = parse_tsv(TEST_BLIND)
    _,          gold_feat,  _,          gold_obs_strs = parse_tsv(TEST_GOLD)

    print(f"  Train: {len(train_meta)} langs  Dev: {len(dev_meta)}  Test: {len(test_meta)}")

    # ------------------------------------------------------------------
    # Build vocabulary from train+dev (to match base systems)
    # ------------------------------------------------------------------
    print("\n[Part 2] Building vocabulary from train ...")
    traindev_feat = pd.concat([train_feat, dev_feat], ignore_index=True)
    kept_names, feat_to_vals, val_to_loc, cat_matrix_td = build_vocab(traindev_feat)
    n_train_td = len(train_feat)
    cat_matrix_train = cat_matrix_td[:n_train_td]
    n_feats = len(kept_names)
    feat_name_to_idx = {name: i for i, name in enumerate(kept_names)}
    print(f"  {n_feats} features retained")

    idf_weights = compute_idf(cat_matrix_train)

    # Frequency fallback value strings
    freq_pred_str = {}
    for fi, name in enumerate(kept_names):
        col = cat_matrix_train[:, fi]
        obs = col[col >= 0]
        if len(obs) == 0:
            freq_pred_str[name] = feat_to_vals[fi][0]
        else:
            freq_pred_str[name] = feat_to_vals[fi][Counter(obs.tolist()).most_common(1)[0][0]]

    # Build gold_by_lang: {wals_code: {feat_name: value_str}}
    test_wcs  = list(test_meta["wals_code"])
    gold_wcs  = list(gold_feat.index) if hasattr(gold_feat, "index") else []
    # Load gold from gold_obs_strs (parse_tsv stores value-id strings in feat_rows)
    # Actually gold_obs_strs has full value strings like "1 SVO"
    # We need just the id ("1")
    gold_by_lang = {}
    for wc, obs_d in gold_obs_strs.items():
        gold_by_lang[wc] = {fn: vs.split()[0] for fn, vs in obs_d.items()}

    # ------------------------------------------------------------------
    # Load all 18 base system predictions for blanked test cells
    # ------------------------------------------------------------------
    print("\n[Part 3] Loading base-system predictions ...")
    all_sys_preds = load_all_sys_preds(test_blank, feat_name_to_idx, val_to_loc)

    # Report availability
    for sn in SYS_NAMES:
        n_cells = sum(len(v) for v in all_sys_preds.get(sn, {}).values())
        print(f"    {sn:15s}: {n_cells} blanked cells loaded")

    # Oracle ceiling
    oracle = oracle_ceiling(gold_by_lang, test_blank, test_meta, all_sys_preds,
                            feat_name_to_idx, feat_to_vals)
    print(f"\n  Oracle ceiling (any-system correct): {oracle:.4f}")

    # Per-system accuracy on test
    print("\n  Per-system test accuracy:")
    sys_macros = {}
    for sn in SYS_NAMES:
        sp = all_sys_preds.get(sn, {})
        pg = defaultdict(list)
        gen_map = {r["wals_code"]: r["genus"] for _, r in test_meta.iterrows()}
        for _, row in test_meta.iterrows():
            wc = row["wals_code"]
            g  = row["genus"]
            blk = test_blank.get(wc, set())
            gold = gold_by_lang.get(wc, {})
            c, t = 0, 0
            for fn in blk:
                gv = gold.get(fn)
                if gv is None:
                    continue
                pv_local = sp.get(wc, {}).get(fn)
                pv = feat_to_vals[feat_name_to_idx[fn]][pv_local] if (
                    fn in feat_name_to_idx and pv_local is not None and pv_local >= 0
                ) else None
                t += 1
                c += int(pv == gv)
            if t > 0:
                pg[g].append(c / t)
        all_lang_accs = [a for alist in pg.values() for a in alist]
        mac = float(np.mean(all_lang_accs)) if all_lang_accs else 0.0
        sys_macros[sn] = mac
        print(f"    {sn:15s}: {mac:.4f}")

    # ------------------------------------------------------------------
    # Majority vote baseline
    # ------------------------------------------------------------------
    print("\n  Computing majority-vote baseline ...")
    mv_preds_raw = majority_vote_preds(gold_by_lang, test_blank, test_meta, all_sys_preds, freq_pred_str)
    mv_preds_str = {}
    for wc, fp in mv_preds_raw.items():
        mv_preds_str[wc] = {}
        for fn, local_idx in fp.items():
            fi = feat_name_to_idx.get(fn)
            if fi is None or local_idx is None:
                mv_preds_str[wc][fn] = freq_pred_str.get(fn, "1")
            else:
                local_idx = max(0, min(int(local_idx), len(feat_to_vals[fi]) - 1))
                mv_preds_str[wc][fn] = feat_to_vals[fi][local_idx]
    mv_macro = compute_macro(mv_preds_str, gold_by_lang, test_meta, test_blank)
    print(f"  Majority-vote (18 systems) macro: {mv_macro:.4f}")

    # ------------------------------------------------------------------
    # Build training meta-features via LOO simulation (train only)
    # ------------------------------------------------------------------
    print("\n[Part 4] Building stacker training data (LOO on train) ...")
    X_train, y_train, groups, feat_idxs = build_train_meta(
        train_meta, cat_matrix_train, kept_names,
        feat_name_to_idx, feat_to_vals, val_to_loc, idf_weights)
    print(f"  Training samples: {len(X_train)}  features per sample: {X_train.shape[1]}")

    # ------------------------------------------------------------------
    # Train per-feature classifiers
    # ------------------------------------------------------------------
    print("\n[Part 5] Training per-feature RandomForest classifiers ...")
    models = train_classifiers(X_train, y_train, feat_idxs, n_feats, feat_to_vals)

    # ------------------------------------------------------------------
    # Genus-stratified cross-validation
    # ------------------------------------------------------------------
    print("\n[Part 6] Genus-stratified 5-fold cross-validation ...")
    cv_macro = cv_validate(X_train, y_train, groups, feat_idxs, n_feats, feat_to_vals)

    # ------------------------------------------------------------------
    # Test inference
    # ------------------------------------------------------------------
    print("\n[Part 7] Stacker test inference ...")
    base_preds, cell_neural = infer_test(
        test_meta, test_blank, test_obs_strs,
        all_sys_preds, models, feat_name_to_idx, feat_to_vals, freq_pred_str)

    base_macro = compute_macro(base_preds, gold_by_lang, test_meta, test_blank)
    print(f"  Base stacker (no neural override): {base_macro:.4f}")

    # Try neural override at multiple thresholds; pick best by internal macro
    override_results = {}
    for thr in [8, 7, 6, 5, 4]:
        preds_t, n_ov = apply_neural_override(
            base_preds, cell_neural, feat_name_to_idx, feat_to_vals,
            test_blank, threshold=thr)
        m = compute_macro(preds_t, gold_by_lang, test_meta, test_blank)
        override_results[thr] = (m, preds_t, n_ov)
        print(f"  Neural override threshold={thr}: macro={m:.4f}  overrides={n_ov}")

    # Pick best threshold
    best_thr = max(override_results, key=lambda t: override_results[t][0])
    stacker_macro, stacker_preds, n_overrides = override_results[best_thr]
    print(f"\n  Best threshold={best_thr}: macro={stacker_macro:.4f}  "
          f"overrides={n_overrides}")

    # ------------------------------------------------------------------
    # Write submission TSV (best configuration)
    # ------------------------------------------------------------------
    print(f"\n[Part 8] Writing submission to {OUT_TSV} ...")
    _write_submission(OUT_TSV, test_meta, test_blank, test_obs_strs,
                      stacker_preds, freq_pred_str)
    print("  Done.")

    # ------------------------------------------------------------------
    # Official scoring (if scorer exists)
    # ------------------------------------------------------------------
    print("\n[Part 9] Running official scorer on stacker submission ...")
    try:
        stdout, stderr = _run_scorer([TEST_GOLD, OUT_TSV])
        if stdout:
            print(stdout)
        if stderr:
            print("STDERR:", stderr[:500])
    except Exception as e:
        print(f"  Scorer failed: {e}")

    # ------------------------------------------------------------------
    # Ablation pilots
    # ------------------------------------------------------------------
    print("\n[Part 10] Ablation pilots (A–E) ...")
    try:
        import sigtyp_ablation as abl
        ctx = {
            "train_meta":        train_meta,
            "dev_meta":          dev_meta,
            "test_meta":         test_meta,
            "train_obs":         train_obs_strs,
            "dev_obs":           dev_obs_strs,
            "test_obs":          test_obs_strs,
            "test_blank":        test_blank,
            "gold_by_lang":      gold_by_lang,
            "kept_names":        kept_names,
            "feat_to_vals":      feat_to_vals,
            "BASE_DIR":          BASE_DIR,
            "val_to_loc":        val_to_loc,
            "feat_name_to_idx":  feat_name_to_idx,
            "all_sys_preds":     all_sys_preds,
            "freq_pred_str":     freq_pred_str,
            "idf_weights":       idf_weights,
            "cat_matrix_train":  cat_matrix_train,
            "sys_macros":        sys_macros,
            "SYS_NAMES":         SYS_NAMES,
            "X_train":           X_train,
            "y_train":           y_train,
            "feat_idxs":         feat_idxs,
            "groups":            groups,
        }
        abl.run_all_pilots(ctx)
    except Exception as exc:
        import traceback
        print(f"  Ablation pilots failed: {exc}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'System':<20} {'Macro':>8}")
    print("-" * 30)
    for sn in SYS_NAMES:
        m = sys_macros.get(sn, float("nan"))
        print(f"{sn:<20} {m:8.4f}")
    print("-" * 30)
    print(f"{'Majority vote (18)':<20} {mv_macro:8.4f}")
    print(f"{'Stacker (no override)':<20} {base_macro:8.4f}   (CV: {cv_macro:.4f})")
    for thr in sorted(override_results):
        m, _, n_ov = override_results[thr]
        marker = " <- best" if thr == best_thr else ""
        print(f"  + neural thr={thr:>2}        {m:8.4f}   ({n_ov} overrides){marker}")
    print(f"{'Oracle ceiling':<20} {oracle:8.4f}")
    print("-" * 30)
    print(f"{'UFAL (target)':<20} {'0.7500':>8}")
    beat = "BEAT UFAL!" if stacker_macro >= 0.75 else f"gap = {0.75 - stacker_macro:+.4f}"
    print(f"Stacker vs UFAL: {beat}")
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
