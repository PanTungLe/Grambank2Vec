"""
Shared utilities for canonical-model training and analysis.
"""

import os
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int) -> None:
    """Set all relevant seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def wals_code_to_glottocode(wals_repo_dir: str) -> Dict[str, str]:
    """
    Read WALS CLDF languages.csv and return a wals_code → Glottocode map.

    Some WALS languages predate full Glottolog coverage and have no Glottocode.
    Those are simply absent from the returned dict.
    """
    lang_csv = os.path.join(wals_repo_dir, "cldf", "languages.csv")
    if not os.path.exists(lang_csv):
        raise FileNotFoundError(
            f"WALS languages.csv not found at {lang_csv}. "
            f"Ensure --data_path points to the cloned WALS repo.")
    langs = pd.read_csv(lang_csv, low_memory=False)
    langs.columns = [c.strip() for c in langs.columns]

    glot_col = next(
        (c for c in langs.columns if c.strip().lower() == "glottocode"), None)
    if glot_col is None:
        raise ValueError(
            f"No 'Glottocode' column found in {lang_csv}. "
            f"Available columns: {list(langs.columns)}")

    id_col = next(
        (c for c in langs.columns if c.strip().upper() == "ID"), "ID")

    mapping: Dict[str, str] = {}
    for _, row in langs.iterrows():
        wc = str(row[id_col]).strip()
        gc = str(row[glot_col]).strip()
        if gc and gc.lower() not in ("nan", "", "none"):
            mapping[wc] = gc
    return mapping


def build_all_triples_binary(
    binary_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract all observed (lang_idx, binary_col_idx, value) triples.
    NaN cells are skipped.
    """
    langs, feats = np.where(~np.isnan(binary_matrix))
    vals = binary_matrix[langs, feats]
    return (langs.astype(np.int64),
            feats.astype(np.int64),
            vals.astype(np.float32))


def build_all_triples_categorical(
    cat_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract all observed (lang_idx, feat_idx, local_value_idx) triples.
    Cells with value -1 (missing) are skipped.
    """
    langs, feats = np.where(cat_matrix >= 0)
    vals = cat_matrix[langs, feats]
    return (langs.astype(np.int64),
            feats.astype(np.int64),
            vals.astype(np.int64))
