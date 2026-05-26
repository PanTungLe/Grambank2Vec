#!/usr/bin/env python3
"""
src/sigtyp_retrofit.py — SIGTYP 2020 controlled objective ablation (Appendix E).

Compares UFALNeural (BCE / sigmoid) vs CategoricalNeural (CE / softmax)
under IDENTICAL training conditions:
  - embedding dim 64, dropout 0.5
  - Adam lr=1e-3, weight_decay=0
  - Up to 200 epochs, early stopping on dev NLL, patience 10
  - Seeds 42 43 44 45 46 (five runs each)

Data is downloaded automatically from the UFAL/ST2020 public GitHub if not
already present in --data-dir.

Usage:
    python src/sigtyp_retrofit.py
    python src/sigtyp_retrofit.py --seeds 42,43 --epochs 50   # quick smoke test
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model_ufal import (
    UFALNeural,
    CategoricalNeural,
    build_ufal_vocab,
    build_training_data_ufal,
    transductive_finetune_emb,
)
from model_learned import prepare_categorical

# NOTE: sigtyp_eval.py runs top-level code on import (loads data at /home/user/ST2020),
# so we replicate parse_sigtyp() inline rather than importing it.  The implementation
# below is identical to the one in sigtyp_eval.py (and sigtyp_eval_v4.py).

# ---------------------------------------------------------------------------
# parse_sigtyp — verbatim from sigtyp_eval.py
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
    (" (= ",                                 " (EQUALS "),
]

def _apply_fixes(s: str) -> str:
    for old, new in _SIGTYP_FIXES:
        s = s.replace(old, new)
    return s


def parse_sigtyp(path: str):
    """
    Parse a SIGTYP TSV file.  Identical to the implementation in sigtyp_eval.py.

    Returns
    -------
    meta_df    : DataFrame [wals_code, name, latitude, longitude, genus, family, countrycodes]
    feat_df    : DataFrame rows=languages, cols=lowercase feature names; value = id string or NaN
    blanked    : {wals_code: set of lowercase feature names with value "?"}
    obs_strings: {wals_code: {lowercase feat_name: full "id label" string}}
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

        wals_code = parts[0]
        name      = parts[1]
        try:
            latitude  = float(parts[2])
            longitude = float(parts[3])
        except ValueError:
            latitude = longitude = float("nan")
        genus        = parts[4]
        family       = parts[5]
        countrycodes = parts[6]
        features_field = "\t".join(parts[7:])

        meta_rows.append({
            "wals_code": wals_code, "name": name,
            "latitude":  latitude,  "longitude": longitude,
            "genus":     genus,     "family":    family,
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DATA_BASE = "https://raw.githubusercontent.com/ufal/ST2020/master/data"
_FILES = ["train.csv", "dev.csv", "test_blinded.csv", "test_gold.csv"]

EMBED_DIM = 64
DROPOUT = 0.5
LR = 1e-3
BATCH_SIZE_UFAL = 512          # positives sampled per step (matches train_ufal_model)
STEPS_PER_EPOCH = 1000         # steps per epoch for UFALNeural (matches UFAL paper)
BATCH_SIZE_CAT = 256           # mini-batch size for CategoricalNeural
KMEANS_CLUSTERS = 300          # geo-clusters in UFAL vocab
FINETUNE_STEPS_TEST = 200      # transductive fine-tuning steps for test languages
FINETUNE_LR = 0.05
RESULTS_DIR = REPO_ROOT / "results"


# ===========================================================================
# 1. Data utilities
# ===========================================================================

def _ensure_data(data_dir: Path) -> None:
    """Download SIGTYP 2020 data files from UFAL GitHub if absent."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in _FILES:
        dst = data_dir / fname
        if not dst.exists():
            url = f"{_DATA_BASE}/{fname}"
            print(f"  Downloading {url} ...")
            try:
                urllib.request.urlretrieve(url, str(dst))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download {url}: {exc}\n"
                    f"Please manually place {fname} in {data_dir}."
                ) from exc


def _strip_to_id(full_val_str: str) -> str:
    """'5 SOV' → '5'.  Already-stripped strings pass through."""
    return full_val_str.split()[0] if full_val_str.strip() else full_val_str


def _build_gold_by_lang(gold_obs_strings: dict) -> dict:
    """Strip gold obs_strings to bare ID strings: {wc: {feat_name: '1'}}."""
    return {
        wc: {fn: _strip_to_id(vs) for fn, vs in obs_d.items()}
        for wc, obs_d in gold_obs_strings.items()
    }


# ===========================================================================
# 2. Vocabulary and training data
# ===========================================================================

def build_vocab_from_train(train_feat: pd.DataFrame, train_meta: pd.DataFrame):
    """
    Build shared vocabulary from the TRAIN split only.

    Returns
    -------
    cat_matrix         : (n_train, n_features) int array, -1 = missing
    kept_feature_names : list of str
    feat_to_global_ids : dict  fi → list of global value IDs (WALS-only)
    feat_to_value_names: dict  fi → list of value ID strings ('1','2',...)
    feat_to_fv_ids     : dict  fi → list of UFAL global fv IDs
    metadata_fv_maps   : dict  (genus/family/kmeans metadata maps)
    n_global_fv_ids    : int   total UFAL fv embedding table size
    km                 : fitted KMeans model
    feat_name_to_idx   : dict  feature_name → feature index
    freq_fallback      : dict  feature_name → most-common value string
    """
    feature_cols = list(train_feat.columns)

    cat_matrix, kept_feature_names, feat_to_global_ids, feat_to_value_names = \
        prepare_categorical(train_feat, feature_cols)

    feat_name_to_idx = {name: i for i, name in enumerate(kept_feature_names)}

    # UFAL vocab: WALS + genus + family + geo-cluster
    feat_to_fv_ids, metadata_fv_maps, n_global_fv_ids, km = build_ufal_vocab(
        cat_matrix, kept_feature_names, feat_to_value_names,
        train_meta, kmeans_clusters=KMEANS_CLUSTERS,
    )

    # Frequency fallback (most common value per feature in train)
    freq_fallback = {}
    for fi, name in enumerate(kept_feature_names):
        col = cat_matrix[:, fi]
        obs = col[col >= 0]
        if len(obs) == 0:
            freq_fallback[name] = feat_to_value_names[fi][0]
        else:
            cnt = Counter(obs.tolist())
            freq_fallback[name] = feat_to_value_names[fi][
                cnt.most_common(1)[0][0]]

    return (cat_matrix, kept_feature_names, feat_to_global_ids,
            feat_to_value_names, feat_to_fv_ids, metadata_fv_maps,
            n_global_fv_ids, km, feat_name_to_idx, freq_fallback)


def _cat_matrix_for_split(feat_df: pd.DataFrame,
                           kept_feature_names: list,
                           feat_to_value_names: dict) -> np.ndarray:
    """
    Build a categorical matrix for an arbitrary split (dev / test)
    using the vocabulary derived from train.

    Cells are local value indices (0,1,...) or -1 for missing / OOV.
    """
    n_langs = len(feat_df)
    n_feats = len(kept_feature_names)
    mat = np.full((n_langs, n_feats), -1, dtype=int)

    for fi, fname in enumerate(kept_feature_names):
        if fname not in feat_df.columns:
            continue
        val_to_local = {v: i for i, v in enumerate(feat_to_value_names[fi])}
        for li, val in enumerate(feat_df[fname]):
            if pd.notna(val):
                mat[li, fi] = val_to_local.get(str(val), -1)

    return mat


# ===========================================================================
# 3. Early-stopping training loops
# ===========================================================================

def _dev_finetune_loss(model,
                        model_type: str,
                        dev_meta: pd.DataFrame,
                        dev_obs: dict,
                        feat_name_to_idx: dict,
                        feat_to_value_names: dict,
                        feat_to_fv_ids: dict,
                        embed_dim: int,
                        device: str,
                        finetune_steps: int = 15) -> float:
    """
    Dev early-stopping metric via mini transductive fine-tuning.

    For each dev language:
      1. collect its observed features that are in the training vocabulary
      2. initialise a fresh random embedding (same as test-time inference)
      3. optimise that embedding against the observed features for
         `finetune_steps` Adam steps (model weights FROZEN)
      4. record the final per-feature mean loss

    Returns the mean loss across all dev languages (lower = better).

    WHY this is correct for both objectives
    ----------------------------------------
    The earlier approach used fixed-random dev-lang embeddings that were
    added to the embedding TABLE but never trained.  As the shared
    fv_embeddings improve during training their norms grow (no L2 reg).
    For BCE/sigmoid the cosine normalises the norms away, so the random-
    embedding BCE stays roughly constant — BCE never triggers early stop.
    For CE/softmax larger norms sharpen the softmax; a random lang vector
    increasingly lands on the WRONG peak, so dev CE *increases* as training
    progresses — CE triggers early stop at epoch ~10-15 and never converges.

    Mini transductive fine-tuning is symmetric: the fresh embedding is
    optimised with the same objective used at test time, so "dev NLL after
    N finetune steps" genuinely measures the quality of the shared embedding
    space under both objectives.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    total_loss = 0.0
    n_langs = 0

    for _, row in dev_meta.iterrows():
        wc      = row["wals_code"]
        obs_raw = dev_obs.get(wc, {})

        # Build (fi, vi_local) pairs restricted to training vocabulary
        obs = []
        for fname, vs in obs_raw.items():
            fi = feat_name_to_idx.get(fname)
            if fi is None:
                continue
            val_id = _strip_to_id(vs)
            vlist  = feat_to_value_names.get(fi, [])
            if val_id not in vlist:
                continue
            obs.append((fi, vlist.index(val_id)))

        if not obs:
            continue

        # Fresh random embedding — same initialisation as test inference
        new_emb = nn.Parameter(torch.zeros(1, embed_dim, device=device))
        nn.init.normal_(new_emb, std=0.01)
        opt = torch.optim.Adam([new_emb], lr=FINETUNE_LR, weight_decay=0)

        for _ in range(finetune_steps):
            step_loss = torch.tensor(0.0, device=device)
            for fi, vi_local in obs:
                fi_t = torch.tensor(feat_to_fv_ids[fi],
                                    dtype=torch.long, device=device)
                if model_type == "ufal":
                    le_n = F.normalize(new_emb, dim=-1)
                    fe   = F.normalize(model.fv_embeddings(fi_t), dim=-1)
                    cos  = (le_n.expand(len(fi_t), -1) * fe).sum(-1, keepdim=True)
                    prbs = torch.sigmoid(model.output_layer(cos)).squeeze(-1)
                    tgt  = torch.zeros(len(fi_t), device=device)
                    tgt[vi_local] = 1.0
                    step_loss += F.binary_cross_entropy(prbs, tgt)
                else:   # categorical
                    fe     = model.fv_embeddings(fi_t)
                    logits = (new_emb @ fe.T).squeeze(0)
                    step_loss -= F.log_softmax(logits, dim=0)[vi_local]
            opt.zero_grad()
            (step_loss / len(obs)).backward()
            opt.step()

        # Record final loss on the same observed features (no grad needed)
        with torch.no_grad():
            lang_loss = 0.0
            for fi, vi_local in obs:
                fi_t = torch.tensor(feat_to_fv_ids[fi],
                                    dtype=torch.long, device=device)
                if model_type == "ufal":
                    le_n = F.normalize(new_emb, dim=-1)
                    fe   = F.normalize(model.fv_embeddings(fi_t), dim=-1)
                    cos  = (le_n.expand(len(fi_t), -1) * fe).sum(-1, keepdim=True)
                    prbs = torch.sigmoid(model.output_layer(cos)).squeeze(-1)
                    tgt  = torch.zeros(len(fi_t), device=device)
                    tgt[vi_local] = 1.0
                    lang_loss += F.binary_cross_entropy(prbs, tgt).item()
                else:
                    fe     = model.fv_embeddings(fi_t)
                    logits = (new_emb @ fe.T).squeeze(0)
                    lang_loss -= F.log_softmax(logits, dim=0)[vi_local].item()
            total_loss += lang_loss / len(obs)
            n_langs += 1

    for p in model.parameters():
        p.requires_grad_(True)
    model.train()

    return total_loss / max(n_langs, 1)


def train_ufal_with_es(
        model: UFALNeural,
        train_pos: list,
        dev_evaluator,
        n_global_fv_ids: int,
        n_epochs: int = 200,
        steps_per_epoch: int = STEPS_PER_EPOCH,
        lr: float = LR,
        device: str = "cpu",
        seed: int = 42,
        patience: int = 10,
        verbose: bool = True,
) -> UFALNeural:
    """
    Train UFALNeural (BCE / sigmoid) with early stopping on dev loss.

    Training logic mirrors train_ufal_model() in model_ufal.py:
      50% positive (observed fv_id for that language),
      50% negative (random fv_id NOT in the language's observed set).
    The key difference: epoch-by-epoch control for early stopping,
    weight_decay=0 (no L2 regulariser), and dev_evaluator uses
    mini transductive fine-tuning (symmetric across objectives).

    dev_evaluator: callable(model) -> float  (lower = better)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    criterion = nn.BCELoss()

    # Build per-language positive sets for fast negative rejection
    lang_positives: dict[int, set] = defaultdict(set)
    for li, fv_id in train_pos:
        lang_positives[li].add(fv_id)
    pos_array = np.array(train_pos)
    n_pos = len(pos_array)

    best_dev_nll = float("inf")
    best_state = deepcopy(model.state_dict())
    no_improve = 0
    best_epoch = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            idxs = np.random.randint(0, n_pos, size=BATCH_SIZE_UFAL)
            batch_langs, batch_fvs, batch_labels = [], [], []

            for idx in idxs:
                li, fv_pos = pos_array[idx]
                if np.random.uniform() < 0.5:
                    batch_langs.append(li)
                    batch_fvs.append(fv_pos)
                    batch_labels.append(1.0)
                else:
                    # Up to 20 attempts to find a negative
                    fv_neg = fv_pos
                    for _ in range(20):
                        fv_neg = np.random.randint(1, n_global_fv_ids)
                        if fv_neg not in lang_positives[li]:
                            break
                    batch_langs.append(li)
                    batch_fvs.append(fv_neg)
                    batch_labels.append(0.0)

            li_t  = torch.tensor(batch_langs,  dtype=torch.long,  device=device)
            fv_t  = torch.tensor(batch_fvs,    dtype=torch.long,  device=device)
            lab_t = torch.tensor(batch_labels, dtype=torch.float, device=device)

            probs = model(li_t, fv_t)
            loss  = criterion(probs, lab_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / steps_per_epoch
        dev_nll   = dev_evaluator(model)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1:3d}/{n_epochs}  "
                  f"train={avg_train:.4f}  dev_nll={dev_nll:.4f}")

        if dev_nll < best_dev_nll - 1e-5:
            best_dev_nll = dev_nll
            best_state   = deepcopy(model.state_dict())
            no_improve   = 0
            best_epoch   = epoch + 1
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    ↳ early stop at epoch {epoch+1} "
                          f"(best={best_epoch}, dev_nll={best_dev_nll:.4f})")
                break

    model.load_state_dict(best_state)
    return model


def train_categorical_with_es(
        model: CategoricalNeural,
        cat_matrix_train: np.ndarray,
        dev_evaluator,
        n_epochs: int = 200,
        batch_size: int = BATCH_SIZE_CAT,
        lr: float = LR,
        device: str = "cpu",
        seed: int = 42,
        patience: int = 10,
        verbose: bool = True,
) -> CategoricalNeural:
    """
    Train CategoricalNeural (CE / softmax) with early stopping on dev loss.

    Training logic mirrors train_categorical_model() in model_ufal.py.
    weight_decay=0 (no L2 regulariser) as specified.

    dev_evaluator: callable(model) -> float  (lower = better)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    # Build (lang_idx, feat_idx, local_val_idx) triples from training matrix
    triples = [
        (li, fi, int(cat_matrix_train[li, fi]))
        for li in range(cat_matrix_train.shape[0])
        for fi in range(cat_matrix_train.shape[1])
        if cat_matrix_train[li, fi] >= 0
    ]

    best_dev_nll = float("inf")
    best_state   = deepcopy(model.state_dict())
    no_improve   = 0
    best_epoch   = 0

    for epoch in range(n_epochs):
        model.train()
        perm = np.random.permutation(len(triples))
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, len(triples), batch_size):
            end   = min(start + batch_size, len(triples))
            batch = [triples[perm[i]] for i in range(start, end)]

            li_t = torch.tensor([t[0] for t in batch], dtype=torch.long, device=device)
            fis  = [t[1] for t in batch]
            vi_t = torch.tensor([t[2] for t in batch], dtype=torch.long, device=device)

            loss, l2 = model(li_t, fis, vi_t, device)
            # weight_decay=0 → no L2 term, just CE loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_train = epoch_loss / max(n_batches, 1)
        dev_nll   = dev_evaluator(model)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1:3d}/{n_epochs}  "
                  f"train={avg_train:.4f}  dev_nll={dev_nll:.4f}")

        if dev_nll < best_dev_nll - 1e-5:
            best_dev_nll = dev_nll
            best_state   = deepcopy(model.state_dict())
            no_improve   = 0
            best_epoch   = epoch + 1
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    ↳ early stop at epoch {epoch+1} "
                          f"(best={best_epoch}, dev_nll={best_dev_nll:.4f})")
                break

    model.load_state_dict(best_state)
    return model


# ===========================================================================
# 4. Inference helpers
# ===========================================================================

def predict_features_ufal(model: UFALNeural,
                           new_emb: torch.Tensor,
                           kept_feature_names: list,
                           feat_to_value_names: dict,
                           feat_to_fv_ids: dict,
                           device: str) -> dict:
    """
    Predict all vocabulary features for a test language using its
    transductively fine-tuned embedding.

    Returns {feat_name: value_id_string} (e.g. {'order_of_subject_...': '1'}).
    """
    preds = {}
    model.eval()
    with torch.no_grad():
        for fi, fname in enumerate(kept_feature_names):
            fv_ids = feat_to_fv_ids[fi]
            probs  = model.predict_feature_from_emb(new_emb, fv_ids, device)
            pred_local  = int(np.argmax(probs))
            preds[fname] = feat_to_value_names[fi][pred_local]
    return preds


def predict_features_categorical(model: CategoricalNeural,
                                  new_emb: torch.Tensor,
                                  kept_feature_names: list,
                                  feat_to_value_names: dict,
                                  device: str) -> dict:
    """
    Predict all vocabulary features for a test language.

    Returns {feat_name: value_id_string}.
    """
    preds = {}
    model.eval()
    with torch.no_grad():
        for fi, fname in enumerate(kept_feature_names):
            pred_local, _ = model.predict_from_emb(new_emb, fi, device)
            preds[fname]  = feat_to_value_names[fi][pred_local]
    return preds


def infer_on_split(model,
                   model_type: str,
                   feat_to_fv_ids: dict,
                   split_meta: pd.DataFrame,
                   split_blank: dict,
                   split_obs: dict,
                   feat_name_to_idx: dict,
                   kept_feature_names: list,
                   feat_to_value_names: dict,
                   freq_fallback: dict,
                   embed_dim: int,
                   device: str,
                   finetune_steps: int = FINETUNE_STEPS_TEST) -> dict:
    """
    Transductively fine-tune + predict for every language in a split.

    Observed features (non-"?") are used as fine-tuning signal.
    For features not in vocabulary → freq_fallback.

    Returns preds_by_lang: {wals_code: {feat_name: value_id_string}}
    """
    model.eval()
    preds_by_lang: dict[str, dict] = {}

    for _, row in split_meta.iterrows():
        wc      = row["wals_code"]
        obs_raw = split_obs.get(wc, {})

        # Restrict observed features to training vocabulary, strip labels
        obs_fd = {
            k: _strip_to_id(v)
            for k, v in obs_raw.items()
            if k in feat_name_to_idx
        }

        new_emb = transductive_finetune_emb(
            model, model_type, obs_fd,
            feat_name_to_idx, feat_to_value_names,
            feat_to_fv_ids, embed_dim, device,
            n_steps=finetune_steps, lr=FINETUNE_LR, temperature=1.0,
        )

        if model_type == "ufal":
            lang_preds = predict_features_ufal(
                model, new_emb, kept_feature_names,
                feat_to_value_names, feat_to_fv_ids, device)
        else:   # categorical
            lang_preds = predict_features_categorical(
                model, new_emb, kept_feature_names,
                feat_to_value_names, device)

        # For blanked features outside vocab → frequency fallback
        blanked = split_blank.get(wc, set())
        for fn in blanked:
            if fn not in lang_preds and fn in freq_fallback:
                lang_preds[fn] = freq_fallback[fn]

        preds_by_lang[wc] = lang_preds

    return preds_by_lang


# ===========================================================================
# 5. Macro accuracy (mean of per-language accuracies)
# ===========================================================================

def compute_macro_acc(preds_by_lang: dict,
                       test_meta: pd.DataFrame,
                       test_blank: dict,
                       gold_by_lang: dict,
                       feat_name_to_idx: dict) -> float:
    """
    Macro accuracy = mean of per-language accuracies over test languages
    that have at least one vocabulary-covered blanked feature.

    Per-language accuracy = (# correctly predicted blanked features in vocab)
                            / (# blanked features in vocab for that language).

    This matches the "mean of per-language accuracies" description in
    Appendix E and is the metric used throughout the SIGTYP 2020 literature.
    """
    lang_accs = []

    for _, row in test_meta.iterrows():
        wc      = row["wals_code"]
        blanked = test_blank.get(wc, set())
        pred    = preds_by_lang.get(wc, {})
        gold    = gold_by_lang.get(wc, {})

        correct = total = 0
        for fn in blanked:
            if fn not in feat_name_to_idx:
                continue
            gv = gold.get(fn)
            if gv is None:
                continue
            total   += 1
            pv = pred.get(fn)
            if pv is not None and _strip_to_id(pv) == _strip_to_id(gv):
                correct += 1

        if total > 0:
            lang_accs.append(correct / total)

    return float(np.mean(lang_accs)) if lang_accs else 0.0


# ===========================================================================
# 6. Main
# ===========================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="SIGTYP 2020 objective ablation: BCE vs CE (Appendix E)")
    ap.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data" / "sigtyp_st2020"),
        help="Directory for SIGTYP TSV files (auto-downloaded if absent)",
    )
    ap.add_argument(
        "--seeds",
        default="42,43,44,45,46",
        help="Comma-separated random seeds (default: 42,43,44,45,46)",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Max training epochs (default: 200)",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience in epochs (default: 10)",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu)",
    )
    ap.add_argument(
        "--finetune-steps",
        type=int,
        default=FINETUNE_STEPS_TEST,
        help=f"Transductive fine-tuning steps at test time (default: {FINETUNE_STEPS_TEST})",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-epoch training output",
    )
    return ap.parse_args()


def main():
    args    = parse_args()
    seeds   = [int(s.strip()) for s in args.seeds.split(",")]
    data_dir = Path(args.data_dir)
    device   = args.device

    print("=" * 70)
    print("SIGTYP 2020 — Objective Ablation (Appendix E)")
    print(f"  seeds={seeds}  epochs={args.epochs}  patience={args.patience}")
    print(f"  embed_dim={EMBED_DIM}  dropout={DROPOUT}  lr={LR}")
    print(f"  device={device}  data_dir={data_dir}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Data
    # -----------------------------------------------------------------------
    print("\n[1] Loading SIGTYP data ...")
    _ensure_data(data_dir)

    def _load(fname):
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Required SIGTYP data file not found: {path}\n"
                f"Run without --data-dir to auto-download, or place the file manually."
            )
        return parse_sigtyp(str(path))

    train_meta, train_feat, train_blank, train_obs = _load("train.csv")
    dev_meta,   dev_feat,   dev_blank,   dev_obs   = _load("dev.csv")
    test_meta,  test_feat,  test_blank,  test_obs  = _load("test_blinded.csv")
    gold_meta,  gold_feat,  gold_blank,  gold_obs  = _load("test_gold.csv")

    n_train = len(train_meta)
    n_dev   = len(dev_meta)
    n_test  = len(test_meta)
    print(f"  Train={n_train}  Dev={n_dev}  Test={n_test}")

    # Guard against empty / corrupt files
    if n_train == 0:
        sys.exit(
            f"ERROR: train.csv in {data_dir} contains 0 languages.\n"
            "The file may be corrupt or in the wrong format.\n"
            "Delete it and re-run to trigger an automatic re-download."
        )
    if n_dev == 0:
        sys.exit(
            f"ERROR: dev.csv in {data_dir} contains 0 languages.\n"
            "Delete it and re-run to trigger an automatic re-download."
        )
    if n_test == 0:
        sys.exit(
            f"ERROR: test_blinded.csv in {data_dir} contains 0 languages.\n"
            "Delete it and re-run to trigger an automatic re-download."
        )

    # Gold answers for test: strip to bare IDs
    gold_by_lang = _build_gold_by_lang(gold_obs)

    # -----------------------------------------------------------------------
    # 2. Vocabulary from TRAIN split only
    # -----------------------------------------------------------------------
    print("\n[2] Building vocabulary from TRAIN split ...")
    (cat_matrix_train, kept_feature_names, feat_to_global_ids,
     feat_to_value_names, feat_to_fv_ids, metadata_fv_maps,
     n_global_fv_ids, km, feat_name_to_idx, freq_fallback) = \
        build_vocab_from_train(train_feat, train_meta)

    n_feats = len(kept_feature_names)
    print(f"  {n_feats} features kept, {n_global_fv_ids} UFAL fv IDs")

    # -----------------------------------------------------------------------
    # 3. Build training data structures
    # -----------------------------------------------------------------------
    print("\n[3] Building training data structures ...")
    cluster_id_offset = metadata_fv_maps["cluster_id_offset"]

    # UFAL positive pairs from TRAIN (lang_id in [0, n_train))
    train_pos_ufal = build_training_data_ufal(
        cat_matrix_train, train_meta, feat_to_fv_ids,
        metadata_fv_maps, km, cluster_id_offset, feat_to_value_names,
    )
    print(f"  UFAL train positives: {len(train_pos_ufal)}")
    # Categorical training triples are built inside train_categorical_with_es
    # from cat_matrix_train directly.

    # Dev early-stopping evaluator: mini transductive fine-tuning on dev
    # languages (model frozen, fresh random embedding per language).
    # This is symmetric across BCE and CE — see _dev_finetune_loss docstring.
    DEV_ES_STEPS = 15   # fast: ~15 grad steps to get a stable signal

    # -----------------------------------------------------------------------
    # 4. Multi-seed training and evaluation
    # -----------------------------------------------------------------------
    print("\n[4] Training (5 seeds × 2 models) ...")
    results_bce: list[float] = []
    results_ce:  list[float] = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"\n  ── Seed {seed} ──")

        # Symmetric dev evaluators via mini transductive fine-tuning
        def _make_dev_evaluator(mtype):
            def _eval(m):
                return _dev_finetune_loss(
                    m, mtype, dev_meta, dev_obs,
                    feat_name_to_idx, feat_to_value_names, feat_to_fv_ids,
                    EMBED_DIM, device, finetune_steps=DEV_ES_STEPS)
            return _eval

        dev_evaluator_bce = _make_dev_evaluator("ufal")
        dev_evaluator_ce  = _make_dev_evaluator("categorical")

        # ------------------------------------------------------------------
        # 4a. UFALNeural (BCE / sigmoid)
        # ------------------------------------------------------------------
        print(f"  [BCE] Training UFALNeural ...")
        t0 = time.time()
        model_bce = UFALNeural(
            n_languages    = n_train,
            n_global_fv_ids= n_global_fv_ids,
            embed_dim      = EMBED_DIM,
            dropout_rate   = DROPOUT,
        )
        train_ufal_with_es(
            model_bce,
            train_pos      = train_pos_ufal,
            dev_evaluator  = dev_evaluator_bce,
            n_global_fv_ids = n_global_fv_ids,
            n_epochs   = args.epochs,
            lr         = LR,
            device     = device,
            seed       = seed,
            patience   = args.patience,
            verbose    = not args.quiet,
        )
        t_bce = time.time() - t0
        print(f"  [BCE] Training done in {t_bce:.1f}s, running inference ...")

        preds_bce = infer_on_split(
            model_bce, "ufal", feat_to_fv_ids,
            test_meta, test_blank, test_obs,
            feat_name_to_idx, kept_feature_names,
            feat_to_value_names, freq_fallback,
            EMBED_DIM, device,
            finetune_steps=args.finetune_steps,
        )
        acc_bce = compute_macro_acc(
            preds_bce, test_meta, test_blank, gold_by_lang, feat_name_to_idx)
        results_bce.append(acc_bce)
        print(f"  [BCE] seed={seed}  macro_acc={acc_bce:.4f}")

        # ------------------------------------------------------------------
        # 4b. CategoricalNeural (CE / softmax)
        # ------------------------------------------------------------------
        print(f"  [CE ] Training CategoricalNeural ...")
        t0 = time.time()
        model_ce = CategoricalNeural(
            n_languages     = n_train,
            n_global_fv_ids = n_global_fv_ids,
            feat_to_fv_ids  = feat_to_fv_ids,
            embed_dim       = EMBED_DIM,
            dropout_rate    = DROPOUT,
        )
        train_categorical_with_es(
            model_ce,
            cat_matrix_train = cat_matrix_train,
            dev_evaluator    = dev_evaluator_ce,
            n_epochs         = args.epochs,
            lr               = LR,
            device           = device,
            seed             = seed,
            patience         = args.patience,
            verbose          = not args.quiet,
        )
        t_ce = time.time() - t0
        print(f"  [CE ] Training done in {t_ce:.1f}s, running inference ...")

        preds_ce = infer_on_split(
            model_ce, "categorical", feat_to_fv_ids,
            test_meta, test_blank, test_obs,
            feat_name_to_idx, kept_feature_names,
            feat_to_value_names, freq_fallback,
            EMBED_DIM, device,
            finetune_steps=args.finetune_steps,
        )
        acc_ce = compute_macro_acc(
            preds_ce, test_meta, test_blank, gold_by_lang, feat_name_to_idx)
        results_ce.append(acc_ce)
        print(f"  [CE ] seed={seed}  macro_acc={acc_ce:.4f}")

    # -----------------------------------------------------------------------
    # 5. Aggregate and print Table 7
    # -----------------------------------------------------------------------
    mean_bce = float(np.mean(results_bce))
    std_bce  = float(np.std(results_bce, ddof=1)) if len(results_bce) > 1 else 0.0
    mean_ce  = float(np.mean(results_ce))
    std_ce   = float(np.std(results_ce, ddof=1))  if len(results_ce)  > 1 else 0.0
    delta    = mean_ce - mean_bce

    print()
    print("=" * 70)
    print("Table 7 — SIGTYP 2020 Objective Ablation (Appendix E)")
    print("=" * 70)
    print(f"{'Objective':<22s}  {'Macro acc':>10s}  {'± std':>8s}")
    print("-" * 46)
    print(f"{'Binary (BCE)':<22s}  {mean_bce:>10.3f}  ±{std_bce:.3f}")
    print(f"{'Categorical (CE)':<22s}  {mean_ce:>10.3f}  ±{std_ce:.3f}")
    print("-" * 46)
    sign = "+" if delta >= 0 else ""
    print(f"{'Δ':<22s}  {sign}{delta:>10.3f}")
    print("=" * 70)
    print()
    print(f"  BCE per-seed:  {[round(x, 4) for x in results_bce]}")
    print(f"  CE  per-seed:  {[round(x, 4) for x in results_ce]}")
    print()
    print("Notes:")
    print("  - Macro accuracy = mean of per-language accuracies (test set).")
    print("  - Vocabulary and training data from TRAIN split only.")
    print(f"  - DEV early stopping via mini transductive fine-tuning "
          f"({DEV_ES_STEPS} steps/lang), symmetric across BCE and CE.")
    print("  - Transductive fine-tuning at test time: "
          f"{args.finetune_steps} steps, lr={FINETUNE_LR}.")
    print("  - Both models: embed_dim=64, dropout=0.5, Adam lr=1e-3, "
          "weight_decay=0.")

    # -----------------------------------------------------------------------
    # 6. Save JSON
    # -----------------------------------------------------------------------
    out_path = RESULTS_DIR / "sigtyp_retrofit.json"
    result_dict = {
        "description": "Appendix E — SIGTYP 2020 objective ablation",
        "config": {
            "seeds":          seeds,
            "epochs":         args.epochs,
            "patience":       args.patience,
            "embed_dim":      EMBED_DIM,
            "dropout":        DROPOUT,
            "lr":             LR,
            "weight_decay":   0,
            "steps_per_epoch_ufal": STEPS_PER_EPOCH,
            "batch_size_cat": BATCH_SIZE_CAT,
            "kmeans_clusters": KMEANS_CLUSTERS,
            "finetune_steps": args.finetune_steps,
            "finetune_lr":    FINETUNE_LR,
            "device":         device,
        },
        "bce": {
            "per_seed":  results_bce,
            "mean":      mean_bce,
            "std":       std_bce,
        },
        "ce": {
            "per_seed":  results_ce,
            "mean":      mean_ce,
            "std":       std_ce,
        },
        "delta_ce_minus_bce": delta,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result_dict, fh, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
