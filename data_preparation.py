"""
Data Preparation for Typological Collaborative Filtering
==========================================================
Handles loading from two specific sources:

1. WALS: https://github.com/cldf-datasets/wals
   CLDF StructureDataset with files:
     - cldf/languages.csv   (ID, Name, Macroarea, Latitude, Longitude, Genus, Family, ...)
     - cldf/values.csv      (ID, Language_ID, Parameter_ID, Value, Code_ID, ...)
     - cldf/codes.csv       (ID, Parameter_ID, Name)
     - cldf/parameters.csv  (ID, Name, ...)

2. ParaBible: https://github.com/LingConLab/parabible/
   Multilingual parallel Bible corpus (1846 translations from cysouw).
   Bible text format: each line is TAB-separated  <verse_id>\\t<text>
   Verse ID format: BBCCCVVV (BB=book, CCC=chapter, VVV=verse)
"""

import os
import re
import glob
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================================
# 1.  WALS Loading from CLDF
# ============================================================================

def load_wals_cldf(
    wals_repo_dir: str,
    min_lang_features: int = 10,
    min_feature_value_langs: int = 10,
) -> Tuple[pd.DataFrame, list]:
    """
    Load WALS directly from the cldf-datasets/wals repository.

    Parameters
    ----------
    wals_repo_dir : str
        Path to the cloned wals repo (containing a cldf/ subdirectory).
    min_lang_features : int
        Drop languages with fewer than this many observed features.
    min_feature_value_langs : int
        Drop feature values observed in fewer than this many languages.

    Returns
    -------
    df : pd.DataFrame  – one row per language, with columns:
        wals_code, name, genus, family, macroarea, feat_<ID> for each feature
    feature_cols : list of str
    """
    cldf_dir = os.path.join(wals_repo_dir, "cldf")

    # --- Load component CSVs ---
    languages = pd.read_csv(os.path.join(cldf_dir, "languages.csv"))
    values = pd.read_csv(os.path.join(cldf_dir, "values.csv"))
    codes = pd.read_csv(os.path.join(cldf_dir, "codes.csv"))
    parameters = pd.read_csv(os.path.join(cldf_dir, "parameters.csv"))

    # Normalise column names (CLDF uses varying capitalisation)
    languages.columns = [c.strip() for c in languages.columns]
    values.columns = [c.strip() for c in values.columns]
    codes.columns = [c.strip() for c in codes.columns]

    # --- Build human-readable value labels ---
    # codes.csv maps Code_ID → human-readable Name
    code_to_name = dict(zip(codes["ID"], codes["Name"]))

    # --- Map values to human-readable labels ---
    values["Value_Label"] = values["Code_ID"].map(code_to_name)

    # --- Pivot: one row per language, one column per feature ---
    values["feat_col"] = "feat_" + values["Parameter_ID"].astype(str)

    wide = values.pivot_table(
        index="Language_ID",
        columns="feat_col",
        values="Value_Label",
        aggfunc="first",
    )
    wide = wide.reset_index().rename(columns={"Language_ID": "wals_code"})

    # --- Merge with language metadata ---
    lang_meta = languages.rename(columns={
        "ID": "wals_code",
        "Name": "name",
    })
    # Handle varying column names across WALS versions
    col_map = {}
    for c in lang_meta.columns:
        cl = c.lower()
        if cl == "genus":
            col_map[c] = "genus"
        elif cl == "family":
            col_map[c] = "family"
        elif cl == "macroarea":
            col_map[c] = "macroarea"
    lang_meta = lang_meta.rename(columns=col_map)

    keep_cols = ["wals_code", "name"]
    for col in ["genus", "family", "macroarea"]:
        if col in lang_meta.columns:
            keep_cols.append(col)
    lang_meta = lang_meta[keep_cols].drop_duplicates(subset="wals_code")

    df = lang_meta.merge(wide, on="wals_code", how="inner")

    # --- Identify feature columns ---
    feature_cols = [c for c in df.columns if c.startswith("feat_")]

    # --- Apply filters ---
    # Filter languages with too few features
    obs_counts = df[feature_cols].notna().sum(axis=1)
    df = df[obs_counts >= min_lang_features].reset_index(drop=True)

    # Filter feature values with too few languages
    for col in feature_cols:
        vc = df[col].value_counts()
        rare_vals = vc[vc < min_feature_value_langs].index
        df.loc[df[col].isin(rare_vals), col] = np.nan

    # Drop features that are now entirely NaN
    all_nan = df[feature_cols].isna().all()
    drop_feats = all_nan[all_nan].index.tolist()
    feature_cols = [c for c in feature_cols if c not in drop_feats]
    df = df.drop(columns=drop_feats)

    print(f"Loaded WALS: {len(df)} languages, {len(feature_cols)} features")
    return df, feature_cols


# ============================================================================
# 2.  ParaBible Corpus Loading
# ============================================================================

def is_latin_cyrillic_greek(text: str, threshold: float = 0.8) -> bool:
    """
    Check whether at least `threshold` fraction of alphabetic characters
    in `text` belong to Latin, Cyrillic, or Greek scripts.
    This implements the paper's filtering criterion.
    """
    if not text:
        return False
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) < 20:
        return False  # too short to judge
    target_count = 0
    for c in alpha_chars:
        name = unicodedata.name(c, "")
        if ("LATIN" in name or "CYRILLIC" in name or "GREEK" in name):
            target_count += 1
    return (target_count / len(alpha_chars)) >= threshold


def load_parabible_texts(
    parabible_dir: str,
    script_filter: bool = True,
    min_tokens: int = 50000,
) -> Dict[str, str]:
    """
    Load Bible translations from the ParaBible export directory.

    The ParaBible project (LingConLab/parabible) stores Bible texts from
    the cysouw Parallel Bible Corpus. After running their export, each
    translation is a file with TAB-separated lines: <verse_id>\\t<text>.

    Alternatively, if you export via their `api_export_csv.py` or
    `db_export_csv.py`, you get CSV files.

    Parameters
    ----------
    parabible_dir : str
        Directory containing Bible text files (one file per translation).
        Accepted formats:
        - .txt files with lines: verse_id<TAB>text
        - .csv files with columns: verse_id, text (or similar)
    script_filter : bool
        If True, only keep translations in Latin, Cyrillic, or Greek
        scripts (matching the paper's methodology).
    min_tokens : int
        Only keep translations with at least this many tokens.

    Returns
    -------
    texts : dict mapping language_id → concatenated text
    """
    texts = {}

    # Look for text files
    patterns = [
        os.path.join(parabible_dir, "*.txt"),
        os.path.join(parabible_dir, "*.csv"),
        os.path.join(parabible_dir, "**", "*.txt"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))

    print(f"Found {len(files)} Bible text files in {parabible_dir}")

    for fpath in files:
        lang_id = os.path.splitext(os.path.basename(fpath))[0]
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            continue

        # Parse: expect TAB-separated verse_id and text
        all_text = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                # verse_id \t text
                all_text.append(parts[1])
            else:
                # Might be CSV or just text
                all_text.append(line)

        full_text = " ".join(all_text)
        tokens = full_text.split()

        if len(tokens) < min_tokens:
            continue

        if script_filter and not is_latin_cyrillic_greek(full_text[:5000]):
            continue

        texts[lang_id] = full_text

    print(f"Loaded {len(texts)} translations "
          f"(after script filter={script_filter}, "
          f"min_tokens={min_tokens})")
    return texts


# ============================================================================
# 3.  Character-level Language Model Data Preparation
# ============================================================================

def prepare_charlm_data(
    texts: Dict[str, str],
    output_dir: str,
    max_chars_per_lang: int = 1_000_000,
) -> str:
    """
    Prepare training data for the character-level LSTM language model
    (Östling & Tiedemann, 2017 architecture).

    Creates one file per language in output_dir, plus a manifest file.
    Each file contains the raw text truncated to max_chars_per_lang.

    Parameters
    ----------
    texts : dict mapping lang_id → text
    output_dir : str
    max_chars_per_lang : int

    Returns
    -------
    manifest_path : str – path to the manifest CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_rows = []

    for lang_id, text in texts.items():
        text_trunc = text[:max_chars_per_lang]
        out_path = os.path.join(output_dir, f"{lang_id}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text_trunc)
        manifest_rows.append({
            "lang_id": lang_id,
            "path": out_path,
            "n_chars": len(text_trunc),
            "n_tokens": len(text_trunc.split()),
        })

    manifest_path = os.path.join(output_dir, "manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Prepared char-LM data for {len(manifest_rows)} languages "
          f"in {output_dir}")
    return manifest_path


# ============================================================================
# 4.  Matching WALS Languages ↔ Bible Languages
# ============================================================================

def match_wals_to_bible(
    df: pd.DataFrame,
    bible_lang_ids: list,
) -> pd.DataFrame:
    """
    Find the intersection of WALS languages and Bible translations.

    This is a heuristic matcher. The paper used an in-house Bible corpus
    where language IDs likely matched directly. With ParaBible (cysouw
    corpus), filenames typically use ISO 639-3 codes or similar.

    Strategy:
    1. Try exact match on wals_code ↔ bible_lang_id
    2. If WALS has ISO 639-3 codes, try matching on those

    You will likely need to manually curate a mapping file for best
    results. This function returns the subset of df that has a match.

    Parameters
    ----------
    df : pd.DataFrame with 'wals_code' column
    bible_lang_ids : list of str

    Returns
    -------
    matched_df : pd.DataFrame  – subset of df with a new column
        'bible_lang_id' indicating the matched Bible translation.
    """
    bible_set = set(bible_lang_ids)
    matches = []

    for _, row in df.iterrows():
        wals_code = str(row["wals_code"]).lower()
        # Try direct match
        if wals_code in bible_set:
            matches.append({**row, "bible_lang_id": wals_code})
        # Try 3-letter prefix (many WALS codes are 3-letter)
        elif wals_code[:3] in bible_set:
            matches.append({**row, "bible_lang_id": wals_code[:3]})

    matched_df = pd.DataFrame(matches)
    print(f"Matched {len(matched_df)} WALS languages to Bible translations "
          f"(out of {len(df)} WALS languages and "
          f"{len(bible_lang_ids)} Bible translations)")
    if len(matched_df) < 50:
        print("WARNING: Few matches found. You likely need a manual mapping "
              "file (wals_code → bible_lang_id). See README for details.")
    return matched_df


# ============================================================================
# 5.  Binarisation (same as before, but included here for completeness)
# ============================================================================

def binarise_wals(
    df: pd.DataFrame,
    feature_cols: list,
) -> Tuple[np.ndarray, list, dict]:
    """
    One-hot encode multi-valued features into a binary matrix.

    Returns
    -------
    binary_matrix : np.ndarray (n_langs × n_binary_features), 0/1/NaN
    binary_col_names : list of str
    feature_groups : dict  original_feat → list of binary col indices
    """
    binary_columns = []
    binary_col_names = []
    feature_groups: Dict[str, List[int]] = {}

    idx = 0
    for col in feature_cols:
        unique_vals = sorted(df[col].dropna().unique())
        group_indices = []
        for val in unique_vals:
            bin_col = np.full(len(df), np.nan)
            observed = df[col].notna()
            bin_col[observed] = (df.loc[observed, col] == val).astype(float).values
            binary_columns.append(bin_col)
            binary_col_names.append(f"{col}={val}")
            group_indices.append(idx)
            idx += 1
        feature_groups[col] = group_indices

    binary_matrix = np.column_stack(binary_columns)
    print(f"Binarised matrix: {binary_matrix.shape[0]} langs × "
          f"{binary_matrix.shape[1]} binary features")
    return binary_matrix, binary_col_names, feature_groups


# ============================================================================
# 6.  End-to-End Data Preparation
# ============================================================================

def prepare_all(
    wals_repo_dir: str,
    parabible_dir: Optional[str] = None,
    charlm_output_dir: str = "charlm_data",
    output_csv: str = "wals_prepared.csv",
) -> dict:
    """
    Full data preparation pipeline.

    Parameters
    ----------
    wals_repo_dir : str
        Path to cloned https://github.com/cldf-datasets/wals
    parabible_dir : str or None
        Path to exported Bible text files from ParaBible.
        If None, only WALS data is prepared (no semi-supervised setup).
    charlm_output_dir : str
        Where to write per-language text files for char-LM training.
    output_csv : str
        Where to save the prepared WALS CSV.

    Returns
    -------
    result : dict with keys:
        'df', 'feature_cols', 'binary_matrix', 'binary_col_names',
        'feature_groups', and optionally 'bible_texts', 'matched_df'
    """
    # --- WALS ---
    df, feature_cols = load_wals_cldf(wals_repo_dir)
    binary_matrix, bin_names, feature_groups = binarise_wals(df, feature_cols)

    df.to_csv(output_csv, index=False)
    print(f"Saved prepared WALS data to {output_csv}")

    result = {
        "df": df,
        "feature_cols": feature_cols,
        "binary_matrix": binary_matrix,
        "binary_col_names": bin_names,
        "feature_groups": feature_groups,
    }

    # --- ParaBible (optional) ---
    if parabible_dir:
        texts = load_parabible_texts(parabible_dir)
        manifest = prepare_charlm_data(texts, charlm_output_dir)
        matched_df = match_wals_to_bible(df, list(texts.keys()))

        result["bible_texts"] = texts
        result["matched_df"] = matched_df
        result["charlm_manifest"] = manifest

    return result


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare data from WALS CLDF repo and ParaBible corpus")
    parser.add_argument("--wals_repo", type=str, required=True,
                        help="Path to cloned cldf-datasets/wals repo")
    parser.add_argument("--parabible_dir", type=str, default=None,
                        help="Path to directory with exported Bible texts")
    parser.add_argument("--charlm_output", type=str, default="charlm_data",
                        help="Output directory for char-LM training data")
    parser.add_argument("--output_csv", type=str, default="wals_prepared.csv",
                        help="Path to save prepared WALS CSV")
    args = parser.parse_args()

    prepare_all(
        wals_repo_dir=args.wals_repo,
        parabible_dir=args.parabible_dir,
        charlm_output_dir=args.charlm_output,
        output_csv=args.output_csv,
    )
