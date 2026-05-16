#!/usr/bin/env python3
"""
SIGTYP 2020 shared task evaluation using both Grambank2Vec models.
Run: python3 sigtyp_eval.py 2>&1 | tee sigtyp_eval_log.txt
"""

import os
import sys
import time
import tempfile
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter

# Grambank2Vec on path
sys.path.insert(0, "/home/user/Grambank2Vec")

from model import TypologicalMF, WALSDataset, train_model
from model_learned import (
    TypologicalMF_Learned, CategoricalTypDataset,
    prepare_categorical, train_model_learned, evaluate_learned,
)
from data_preparation import binarise_features

# ============================================================================
# PART 0: Constants
# ============================================================================

_BASE = "/home/user"
TRAIN_CSV  = f"{_BASE}/ST2020/data/train.csv"
DEV_CSV    = f"{_BASE}/ST2020/data/dev.csv"
TEST_BLIND = f"{_BASE}/ST2020/data/test_blinded.csv"
TEST_GOLD  = f"{_BASE}/ST2020/data/test_gold.csv"
SCORER     = f"{_BASE}/ST2020/scripts/score.py"

SUBMISSION_LEARNED = f"{_BASE}/Grambank2Vec/sigtyp_submission_learned.tsv"
SUBMISSION_TCF     = f"{_BASE}/Grambank2Vec/sigtyp_submission_tcf.tsv"
SUBMISSION_FREQ    = f"{_BASE}/Grambank2Vec/sigtyp_submission_freq.tsv"

# Paper macro-averaged results (for reference)
PAPER_TABLE = [
    ("UFAL (best constrained)",              [.73, .78, .74, .71, .80, .76, .76, .75]),
    ("NEMO system2",                          [.71, .72, .72, .76, .76, .67, .69, .69]),
    ("CrossLingference (unconstrained)",      [.71, .73, .67, .68, .57, .60, .65, .65]),
    ("Baseline_frequency",                    [.51, .53, .37, .49, .41, .58, .53, .49]),
    ("Baseline_knn-imputation",               [.48, .57, .46, .48, .32, .52, .51, .48]),
    ("NUIG (lowest)",                         [.51, .56, .35, .45, .32, .45, .48, .45]),
]

# ============================================================================
# PART 1: Parse SIGTYP data
# ============================================================================

# Same string fixes as score.py Sample class
_SIGTYP_FIXES = [
    ("double negationPosition_of_negative",  "double negation|Position_of_negative"),
    ("double negationSVONeg_Order",           "double negation|SVONeg_Order"),
    ("double negationSNegVO_Order",           "double negation|SNegVO_Order"),
    ("double negationPreverbal_Negative",     "double negation|Preverbal_Negative"),
    ("1 Separate word, no double negation|Word&NoDoubleNeg",
     "1 Separate word, no double negation"),
    ("2 Prefix, no double negation|Prefix&NoDoubleNeg",
     "2 Prefix, no double negation"),
    (" (= ",                                 " (EQUALS "),
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
    feat_rows   = {}   # wals_code -> {feat_name_lc: str_val_or_nan}
    blanked     = {}   # wals_code -> set of lc names
    obs_strings = {}   # wals_code -> {lc name: "id label" str}

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
        # features may span additional tabs (shouldn't, but be robust)
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
                    int(tokens[0])          # must be integer
                except ValueError:
                    continue
                feat_dict[fname_lc] = tokens[0]   # store as string
                obs_str_d[fname_lc] = val_str

        feat_rows[wals_code]   = feat_dict
        blanked[wals_code]     = blank_set
        obs_strings[wals_code] = obs_str_d

    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)

    # Build feat_df: union of all feature names, rows in same order as meta_df
    all_wals_codes = [r["wals_code"] for r in meta_rows]
    all_feat_names = sorted(set(fn for fd in feat_rows.values() for fn in fd))

    rows_data = []
    for wc in all_wals_codes:
        fd = feat_rows[wc]
        rows_data.append({fn: fd.get(fn, np.nan) for fn in all_feat_names})

    feat_df = pd.DataFrame(rows_data, columns=all_feat_names)
    return meta_df, feat_df, blanked, obs_strings


# ============================================================================
# Main
# ============================================================================
print("=" * 70)
print("SIGTYP 2020 Evaluation — Grambank2Vec (Learned + TCF) + Frequency")
print("=" * 70)

# Load all four files
print("\n[Part 1] Loading SIGTYP data ...")
train_meta, train_feat, train_blank, train_obs = parse_sigtyp(TRAIN_CSV)
dev_meta,   dev_feat,   dev_blank,   dev_obs   = parse_sigtyp(DEV_CSV)
test_meta,  test_feat,  test_blank,  test_obs  = parse_sigtyp(TEST_BLIND)
gold_meta,  gold_feat,  gold_blank,  gold_obs  = parse_sigtyp(TEST_GOLD)

print(f"  Train: {len(train_meta)} languages, {train_feat.shape[1]} feature columns")
print(f"  Dev:   {len(dev_meta)} languages, {dev_feat.shape[1]} feature columns")
print(f"  Test:  {len(test_meta)} languages")

# Concatenate train + dev for vocabulary and model training
traindev_feat = pd.concat([train_feat, dev_feat], ignore_index=True)
traindev_meta = pd.concat([train_meta, dev_meta], ignore_index=True)
print(f"  Train+Dev: {len(traindev_feat)} languages, {traindev_feat.shape[1]} feature columns")

# ============================================================================
# PART 2: Build shared vocabulary
# ============================================================================
print("\n[Part 2] Building shared vocabulary ...")

feature_cols = list(traindev_feat.columns)

# prepare_categorical expects string or NaN values; our feat_df already stores strings
cat_matrix, kept_feature_names, feat_to_global_ids, feat_to_value_names = \
    prepare_categorical(traindev_feat, feature_cols)

print(f"  Kept {len(kept_feature_names)} features after filtering (≥2 unique values)")

n_features    = len(kept_feature_names)
n_total_values = max(max(gids) for gids in feat_to_global_ids.values()) + 1
print(f"  Total unique (feature, value) pairs: {n_total_values}")

# Binarise for TCF model (pass the same kept_feature_names)
binary_matrix, binary_col_names, feature_groups, feature_value_names = \
    binarise_features(traindev_feat, kept_feature_names)

n_binary_cols = binary_matrix.shape[1]
print(f"  Binary columns: {n_binary_cols}")

# Lookup structures
feat_name_to_idx        = {name: idx for idx, name in enumerate(kept_feature_names)}
feat_name_to_value_list = {name: feat_to_value_names[idx]
                           for idx, name in enumerate(kept_feature_names)}

# ============================================================================
# PART 3: Build training datasets
# ============================================================================
print("\n[Part 3] Building training datasets ...")

# Learned model dataset (categorical triples from cat_matrix)
ls, fs, vs = [], [], []
for li in range(cat_matrix.shape[0]):
    for fi in range(cat_matrix.shape[1]):
        if cat_matrix[li, fi] >= 0:
            ls.append(li)
            fs.append(fi)
            vs.append(cat_matrix[li, fi])
train_ds_learned = CategoricalTypDataset(
    np.array(ls, dtype=np.int64),
    np.array(fs, dtype=np.int64),
    np.array(vs, dtype=np.int64),
)
print(f"  Learned dataset: {len(train_ds_learned)} observed (lang, feat, val) triples")

# TCF model dataset (binary triples from binary_matrix)
lang_idxs, col_idxs = np.where(~np.isnan(binary_matrix))
values_bin = binary_matrix[lang_idxs, col_idxs]
train_ds_tcf = WALSDataset(
    lang_idxs.astype(np.int64),
    col_idxs.astype(np.int64),
    values_bin,
)
print(f"  TCF dataset: {len(train_ds_tcf)} observed (lang, col, 0/1) triples")

# ============================================================================
# PART 4: Train both models
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[Part 4] Training models on device: {device}")

N_LANGS  = len(traindev_feat)
EMBED    = 64
EPOCHS   = 150
BATCH    = 64
LR       = 1e-3
L2       = 0.1

# ---- Learned model ----
torch.manual_seed(42)
np.random.seed(42)
model_learned = TypologicalMF_Learned(
    n_languages=N_LANGS,
    n_total_values=n_total_values,
    feat_to_global_ids=feat_to_global_ids,
    embed_dim=EMBED,
)
print(f"\n  Training Learned model ({EPOCHS} epochs) ...")
t0 = time.time()
losses_learned = train_model_learned(
    model_learned, train_ds_learned,
    n_epochs=EPOCHS, batch_size=BATCH, lr=LR, l2_reg=L2, device=device,
)
time_learned = time.time() - t0
print(f"  Learned training done in {time_learned:.1f}s  "
      f"(final loss={losses_learned[-1]:.4f})")

# ---- TCF model ----
torch.manual_seed(42)
np.random.seed(42)
model_tcf = TypologicalMF(
    n_languages=N_LANGS,
    n_features=n_binary_cols,
    embed_dim=EMBED,
)
print(f"\n  Training TCF model ({EPOCHS} epochs) ...")
t0 = time.time()
losses_tcf = train_model(
    model_tcf, train_ds_tcf,
    n_epochs=EPOCHS, batch_size=BATCH, lr=LR, l2_reg=L2, device=device,
)
time_tcf = time.time() - t0
print(f"  TCF training done in {time_tcf:.1f}s  "
      f"(final loss={losses_tcf[-1]:.4f})")

# ============================================================================
# PART 5: Transductive fine-tuning for test languages
# ============================================================================
print("\n[Part 5] Transductive fine-tuning for test languages ...")

# Frequency baseline (train+dev only)
def compute_frequency_baseline(cat_matrix, kept_feature_names, feat_to_value_names):
    freq_pred = {}
    for fi, name in enumerate(kept_feature_names):
        col = cat_matrix[:, fi]
        observed = col[col >= 0]
        if len(observed) == 0:
            freq_pred[name] = feat_to_value_names[fi][0]
        else:
            cnt = Counter(observed.tolist())
            most_common_local = cnt.most_common(1)[0][0]
            freq_pred[name] = feat_to_value_names[fi][most_common_local]
    return freq_pred

freq_baseline = compute_frequency_baseline(
    cat_matrix, kept_feature_names, feat_to_value_names)

# Majority-class fallback (single most common integer code across all training)
_all_vals = [v for sub in feat_to_value_names.values() for v in sub]
_default_pred = Counter(_all_vals).most_common(1)[0][0] if _all_vals else "1"


def infer_test_language_learned(
    model, obs_feat_dict, feat_name_to_idx, feat_to_global_ids,
    feat_name_to_value_list, embed_dim, device, n_steps=300, lr=0.05
):
    """Transductive embedding optimisation for a single test language (Learned model).

    Parameters
    ----------
    obs_feat_dict : {lowercase_feat_name: value_str}  — observed non-? features
    Returns : {feat_name: predicted_value_str}  for all vocabulary features
    """
    # Build (feat_idx, value_local_idx) pairs from observed features
    obs_pairs = []
    for feat_name, val_str in obs_feat_dict.items():
        if feat_name not in feat_name_to_idx:
            continue
        fi = feat_name_to_idx[feat_name]
        val_list = feat_name_to_value_list[feat_name]
        if val_str not in val_list:
            continue
        val_local = val_list.index(val_str)
        obs_pairs.append((fi, val_local))

    # New language embedding
    new_emb = nn.Parameter(torch.zeros(1, embed_dim))
    nn.init.normal_(new_emb, std=0.01)
    new_emb = new_emb.to(device)

    # Freeze model
    for p in model.parameters():
        p.requires_grad_(False)

    if len(obs_pairs) == 0:
        # Restore and return freq baseline
        for p in model.parameters():
            p.requires_grad_(True)
        return dict(freq_baseline)  # copy

    optimizer = torch.optim.Adam([new_emb], lr=lr, weight_decay=0)

    model.eval()
    for _step in range(n_steps):
        losses_i = []
        for fi, val_local in obs_pairs:
            gids   = feat_to_global_ids[fi]
            gids_t = torch.tensor(gids, dtype=torch.long, device=device)
            val_embs   = model.value_embeddings(gids_t)         # (K, d)
            scores     = new_emb @ val_embs.T                   # (1, K)
            log_probs  = F.log_softmax(scores, dim=-1)
            loss_i     = -log_probs[0, val_local]
            losses_i.append(loss_i)

        total_loss = torch.stack(losses_i).mean() + L2 * new_emb.pow(2).mean() / 2
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if total_loss.item() < 0.005:
            break

    # Decode predictions for all vocabulary features
    preds = {}
    model.eval()
    with torch.no_grad():
        new_emb_d = new_emb.detach()
        for fi, feat_name in enumerate(kept_feature_names):
            gids     = feat_to_global_ids[fi]
            gids_t   = torch.tensor(gids, dtype=torch.long, device=device)
            val_embs = model.value_embeddings(gids_t)            # (K, d)
            scores   = new_emb_d @ val_embs.T                   # (1, K)
            pred_local = int(torch.argmax(scores[0]).item())
            preds[feat_name] = feat_name_to_value_list[feat_name][pred_local]

    # Restore gradients
    for p in model.parameters():
        p.requires_grad_(True)

    return preds


def infer_test_language_tcf(
    model, obs_feat_dict, feat_name_to_idx, feature_groups,
    feature_value_names, binary_col_names, embed_dim, device,
    n_steps=300, lr=0.05
):
    """Transductive embedding optimisation for a single test language (TCF model).

    Parameters
    ----------
    obs_feat_dict : {lowercase_feat_name: value_str}
    Returns : {feat_name: predicted_value_str}  for all feature_groups keys
    """
    # Build binary observed signal
    obs_cols    = []
    obs_targets = []
    for feat_name, val_str in obs_feat_dict.items():
        if feat_name not in feature_groups:
            continue
        col_indices = feature_groups[feat_name]
        vals        = feature_value_names[feat_name]
        if val_str not in vals:
            continue

        if len(col_indices) == 1:       # 2-valued feature
            target = 1.0 if val_str == vals[1] else 0.0
            obs_cols.append(col_indices[0])
            obs_targets.append(target)
        else:                            # one-hot
            val_idx = vals.index(val_str)
            for ci, col_idx in enumerate(col_indices):
                obs_cols.append(col_idx)
                obs_targets.append(1.0 if ci == val_idx else 0.0)

    new_emb = nn.Parameter(torch.zeros(1, embed_dim))
    nn.init.normal_(new_emb, std=0.01)
    new_emb = new_emb.to(device)

    for p in model.parameters():
        p.requires_grad_(False)

    if len(obs_cols) == 0:
        for p in model.parameters():
            p.requires_grad_(True)
        # Return freq predictions for tcf-known features
        preds = {}
        for feat_name in feature_groups:
            preds[feat_name] = freq_baseline.get(feat_name, _default_pred)
        return preds

    obs_cols_t    = torch.tensor(obs_cols,    dtype=torch.long,  device=device)
    obs_targets_t = torch.tensor(obs_targets, dtype=torch.float, device=device)

    optimizer = torch.optim.Adam([new_emb], lr=lr, weight_decay=0)
    model.eval()

    for _step in range(n_steps):
        col_embs = model.feat_embeddings.weight            # (n_binary_cols, d)
        scores   = new_emb @ col_embs.T                   # (1, n_binary_cols)
        probs    = torch.sigmoid(scores)
        bce      = F.binary_cross_entropy(
            probs[0, obs_cols_t], obs_targets_t)
        loss = bce + L2 * new_emb.pow(2).mean() / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item() < 0.005:
            break

    # Decode
    preds = {}
    model.eval()
    with torch.no_grad():
        new_emb_d = new_emb.detach()
        col_embs  = model.feat_embeddings.weight.detach()  # (n_binary_cols, d)
        scores    = (new_emb_d @ col_embs.T)[0]            # (n_binary_cols,)

        for feat_name, col_indices in feature_groups.items():
            vals          = feature_value_names[feat_name]
            scores_feat   = scores[col_indices]

            if len(col_indices) == 1:   # 2-valued
                pred_val = vals[1] if torch.sigmoid(scores_feat[0]).item() > 0.5 \
                           else vals[0]
            else:                        # one-hot
                pred_local = int(torch.argmax(scores_feat).item())
                pred_val   = vals[pred_local]

            preds[feat_name] = pred_val

    for p in model.parameters():
        p.requires_grad_(True)

    return preds


# ---- Run inference for all test languages ----
print(f"  Running inference for {len(test_meta)} test languages ...")

preds_learned = {}   # wals_code -> {feat_name: value_str}
preds_tcf     = {}
infer_times   = []
n_obs_counts  = {}   # wals_code -> n_observed_features_in_vocab

model_learned.eval()
model_tcf.eval()

for row_idx, row in test_meta.iterrows():
    wals_code = row["wals_code"]

    # Build obs_feat_dict from non-blanked, non-NaN cells in test_feat
    obs_feat_dict = {}
    if row_idx < len(test_feat):
        for col in test_feat.columns:
            val = test_feat.loc[row_idx, col]
            if pd.notna(val) and col not in test_blank.get(wals_code, set()):
                obs_feat_dict[col] = str(val)

    # Count observed features that are in the vocabulary
    n_in_vocab = sum(1 for fn in obs_feat_dict if fn in feat_name_to_idx)
    n_obs_counts[wals_code] = n_in_vocab

    t_start = time.time()
    pl = infer_test_language_learned(
        model_learned, obs_feat_dict,
        feat_name_to_idx, feat_to_global_ids,
        feat_name_to_value_list, EMBED, device,
    )
    pt = infer_test_language_tcf(
        model_tcf, obs_feat_dict,
        feat_name_to_idx, feature_groups,
        feature_value_names, binary_col_names, EMBED, device,
    )
    elapsed = time.time() - t_start

    preds_learned[wals_code] = pl
    preds_tcf[wals_code]     = pt
    infer_times.append(elapsed)

    if (row_idx + 1) % 25 == 0 or (row_idx + 1) == len(test_meta):
        print(f"    {row_idx+1}/{len(test_meta)} languages processed  "
              f"(last: {wals_code}, obs_in_vocab={n_in_vocab})")

# ============================================================================
# PART 6: Generate submission files
# ============================================================================
print("\n[Part 6] Writing submission files ...")

HEADER = "wals_code\tname\tlatitude\tlongitude\tgenus\tfamily\tcountrycodes\tfeatures\n"

def _fmt_features(wals_code, blanked_set, obs_str_dict,
                  model_preds, freq_preds, vocab_set,
                  oov_fallback="_default_pred"):
    """Build the pipe-delimited features string for a test language submission."""
    parts = []
    # Collect all feature names: observed + blanked
    all_feats = set(obs_str_dict.keys()) | set(blanked_set)

    for fname in sorted(all_feats):
        if fname not in blanked_set:
            # Observed — copy original "id label" string verbatim
            val_str = obs_str_dict[fname]
            parts.append(f"{fname}={val_str}")
        else:
            # To predict
            if fname in model_preds:
                pred_id = model_preds[fname]
            elif fname in freq_preds:
                pred_id = freq_preds[fname]
            else:
                pred_id = "1"   # hard fallback for complete OOV
            parts.append(f"{fname}={pred_id} -")

    return "|".join(parts)


# Track vocab-miss statistics
n_vocab_miss_total = defaultdict(int)

for model_name, submission_path, preds_dict in [
    ("learned", SUBMISSION_LEARNED, preds_learned),
    ("tcf",     SUBMISSION_TCF,     preds_tcf),
    ("freq",    SUBMISSION_FREQ,    {}),
]:
    with open(submission_path, "w", encoding="utf-8") as fh:
        fh.write(HEADER)
        for row_idx, row in test_meta.iterrows():
            wals_code    = row["wals_code"]
            blanked_set  = test_blank.get(wals_code, set())
            obs_str_dict = test_obs.get(wals_code, {})

            if model_name == "freq":
                model_preds = {}
            else:
                model_preds = preds_dict.get(wals_code, {})

            # Count vocabulary misses for blanked features
            for fname in blanked_set:
                if fname not in feat_name_to_idx:
                    n_vocab_miss_total[model_name] += 1

            feat_str = _fmt_features(
                wals_code, blanked_set, obs_str_dict,
                model_preds, freq_baseline,
                set(feat_name_to_idx.keys()),
            )

            meta_str = "\t".join([
                row["wals_code"],
                row["name"],
                str(row["latitude"]),
                str(row["longitude"]),
                row["genus"],
                row["family"],
                row["countrycodes"],
            ])
            fh.write(f"{meta_str}\t{feat_str}\n")

    print(f"  Wrote {submission_path}")

# ============================================================================
# PART 7: Score with official scorer (modified for micro + macro)
# ============================================================================
print("\n[Part 7] Running official scorer ...")

# Create a temp copy of score.py with macro averaging enabled
with open(SCORER, "r", encoding="utf-8") as fh:
    score_src = fh.read()

score_src_mod = score_src.replace(
    'for mode in ("micro",):  #, "macro"):',
    'for mode in ("micro", "macro"):'
)

# Write temp copy in the SAME scripts/ dir so DATA_PATH resolves correctly
# (score.py computes DATA_PATH relative to its own __file__)
scripts_dir = os.path.dirname(SCORER)
tmp_score   = os.path.join(scripts_dir, "_score_macro_tmp.py")
with open(tmp_score, "w", encoding="utf-8") as fh:
    fh.write(score_src_mod)

try:
    result = subprocess.run(
        [sys.executable, tmp_score,
         SUBMISSION_LEARNED, SUBMISSION_TCF, SUBMISSION_FREQ],
        capture_output=True, text=True,
        cwd=scripts_dir,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:2000])
finally:
    if os.path.exists(tmp_score):
        os.remove(tmp_score)

# ============================================================================
# PART 8: Parse scorer output and print comparison table
# ============================================================================
print("\n[Part 8] Comparison table ...")

# Parse scorer output
# Format:
#   # Averaging: <mode>
#   submission  <g1>  <g2> ...
#   number of languages  N1  N2 ...
#   <name>  acc1  acc2 ...
#   overall:
#   <name>  overall_acc

# controlled_genera sorted: Madang, Mahakiranti, Mayan, Nilotic,
#                           Northern Pama-Nyungan, Tucanoan, other genera
GENERA_SORTED = sorted(
    ["Mayan", "Tucanoan", "Madang", "Mahakiranti",
     "Northern Pama-Nyungan", "Nilotic", "other genera"]
)

# Table column order (spec):
# Tucanoan, Madang, Mahakiranti, Nilotic, Mayan, N.Pama-Nyungan, Other, Overall
TABLE_COLS   = ["Tucanoan", "Madang", "Mahakiranti", "Nilotic",
                "Mayan", "Northern Pama-Nyungan", "other genera", "Overall"]
TABLE_HEADER = ["Tucanoan", "Madang", "Mahakiranti", "Nilotic",
                "Mayan", "N.Pama-Nyungan", "Other", "Overall"]

def _safe_float(s):
    try:
        v = float(s)
        return v if not (v != v) else None   # NaN -> None
    except Exception:
        return None

# Parse scorer stdout
scorer_results = {}   # mode -> {submission_name -> {genus: acc, Overall: acc}}

current_mode    = None
in_genera_block = False
in_overall      = False
genera_order    = None

for line in result.stdout.splitlines():
    line = line.rstrip()
    if line.startswith("# Averaging:"):
        current_mode    = line.split(":")[-1].strip()
        scorer_results[current_mode] = {}
        in_genera_block = False
        in_overall      = False
        genera_order    = None
        continue

    if current_mode is None:
        continue

    if line.startswith("submission\t"):
        genera_order    = line.split("\t")[1:]
        in_genera_block = True
        in_overall      = False
        continue

    if line.startswith("number of languages"):
        continue

    if line.startswith("overall:"):
        in_genera_block = False
        in_overall      = True
        continue

    if in_genera_block and genera_order and "\t" in line:
        parts = line.split("\t")
        name  = parts[0]
        vals  = parts[1:]
        accs  = {}
        for gi, g in enumerate(genera_order):
            if gi < len(vals):
                accs[g] = _safe_float(vals[gi])
        if name not in scorer_results[current_mode]:
            scorer_results[current_mode][name] = {}
        scorer_results[current_mode][name].update(accs)
        continue

    if in_overall and "\t" in line:
        parts = line.split("\t")
        name  = parts[0]
        if current_mode not in scorer_results:
            scorer_results[current_mode] = {}
        if name not in scorer_results[current_mode]:
            scorer_results[current_mode][name] = {}
        scorer_results[current_mode][name]["Overall"] = _safe_float(parts[1]) \
            if len(parts) > 1 else None


def _row_vals(name, mode):
    """Extract a row of values for the comparison table in TABLE_COLS order."""
    d = scorer_results.get(mode, {}).get(name, {})
    row = []
    for col in TABLE_COLS:
        v = d.get(col)
        row.append(f"{v:.2f}" if v is not None else "  n/a")
    return row

# Submission base names as written by score.py
SUB_NAMES = {
    "learned": os.path.basename(SUBMISSION_LEARNED).replace(".tsv", ""),
    "tcf":     os.path.basename(SUBMISSION_TCF).replace(".tsv", ""),
    "freq":    os.path.basename(SUBMISSION_FREQ).replace(".tsv", ""),
}

# Print table
COL_W = [32] + [7] * 8

def _print_row(cells):
    parts = [str(c).ljust(w) if i == 0 else str(c).rjust(w)
             for i, (c, w) in enumerate(zip(cells, COL_W))]
    print("| " + " | ".join(parts) + " |")

def _print_sep():
    parts = ["-" * w for w in COL_W]
    print("|" + "|".join("-" + p + "-" for p in parts) + "|")

print()
_print_row(["System"] + TABLE_HEADER)
_print_sep()

for mode in ["micro", "macro"]:
    if mode not in scorer_results:
        continue
    for label, key in [
        (f"Grambank2Vec Learned ({mode})", SUB_NAMES["learned"]),
        (f"Grambank2Vec TCF ({mode})",     SUB_NAMES["tcf"]),
        (f"Grambank2Vec Freq ({mode})",    SUB_NAMES["freq"]),
    ]:
        row = _row_vals(key, mode)
        _print_row([label] + row)

_print_sep()
_print_row(["--- SIGTYP 2020 paper (MACRO, reference) ---"] + [""] * 8)
_print_sep()

for sys_name, vals in PAPER_TABLE:
    _print_row([sys_name] + [f"{v:.2f}" for v in vals])

print()
print("Notes:")
print("  - Paper results are MACRO-averaged over genera; our rows show both MICRO and MACRO.")
print("  - Constrained setting: SIGTYP train+dev only (no external data), same as paper.")
print("  - Transductive fine-tuning: up to 300 gradient steps on each test language's")
print("    observed features (analogous to kNN's nearest-neighbour lookup at inference time).")
print("  - Genus/geographic information NOT used during fine-tuning (unlike NEMO/CrossLingference).")
print("  - Features in test_blinded absent from train+dev vocabulary → frequency fallback.")

# ============================================================================
# PART 9: Timing and diagnostics
# ============================================================================
print("\n[Part 9] Diagnostics")
print(f"  Training time — Learned: {time_learned:.1f}s   TCF: {time_tcf:.1f}s")
print(f"  Transductive inference: "
      f"mean={np.mean(infer_times):.2f}s  "
      f"min={np.min(infer_times):.2f}s  "
      f"max={np.max(infer_times):.2f}s  "
      f"total={sum(infer_times):.1f}s")
print(f"  Vocabulary: {n_features} features, {n_total_values} total values, "
      f"{n_binary_cols} binary columns")

# Genus-level observed-feature counts
CONTROLLED = ["Tucanoan", "Madang", "Mahakiranti", "Nilotic",
               "Mayan", "Northern Pama-Nyungan"]
print("\n  Test languages with ≥5 vocab-observed features per controlled genus:")
for genus in CONTROLLED:
    genus_rows = test_meta[test_meta["genus"] == genus]
    n_ge5 = sum(1 for _, r in genus_rows.iterrows()
                if n_obs_counts.get(r["wals_code"], 0) >= 5)
    print(f"    {genus:30s}: {n_ge5}/{len(genus_rows)} languages")

print(f"\n  Vocabulary-miss blanked features (per model, same count):")
for mname in ["learned", "tcf", "freq"]:
    print(f"    {mname}: {n_vocab_miss_total.get(mname, 0)} features fell back to freq baseline")

print("\n[Done]")
