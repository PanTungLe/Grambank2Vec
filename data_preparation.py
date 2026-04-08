"""
Data Preparation for Typological Collaborative Filtering
==========================================================
Handles loading from typological databases and Bible corpora:

1. WALS: https://github.com/cldf-datasets/wals
   CLDF StructureDataset with files:
     - cldf/languages.csv   (ID, Name, Macroarea, Latitude, Longitude, Genus, Family, ...)
     - cldf/values.csv      (ID, Language_ID, Parameter_ID, Value, Code_ID, ...)
     - cldf/codes.csv       (ID, Parameter_ID, Name)
     - cldf/parameters.csv  (ID, Name, ...)

2. Grambank: https://github.com/glottobank/grambank
   CLDF StructureDataset with ~2,400 languages and ~200 features.
   Features are MOSTLY binary (0/1), but some have 3+ values.
   Uses Glottocodes as language IDs. No genus column — genealogical
   groupings come from Glottolog classification or the Family column.

3. eBible: https://github.com/BibleNLP/ebible
   Multilingual parallel Bible corpus (~1079 translations).
   Verse-per-line format: each translation is a plain-text file with one
   verse per line (no verse reference prefix). A companion vref.txt provides
   verse references correlated line-by-line.
   Filenames: <languageCode>-<variant>.txt  (e.g. eng-eng_kjv.txt)

   Also supports legacy ParaBible format (TAB-separated <verse_id>\\t<text>).
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

    # Preserve ISO 639-3 codes for language matching
    iso_col = None
    for c in lang_meta.columns:
        if c.lower() in ("iso639p3code", "iso_codes", "iso639-3"):
            iso_col = c
            break
    if iso_col:
        lang_meta = lang_meta.rename(columns={iso_col: "iso639p3code"})

    keep_cols = ["wals_code", "name"]
    for col in ["genus", "family", "macroarea", "iso639p3code"]:
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
# 1b.  Grambank Loading from CLDF
# ============================================================================

def _extract_glottolog_genus(
    classification: str,
    level: int = 2,
) -> Optional[str]:
    """
    Extract a genus-level grouping from a Glottolog classification string.

    Glottolog classification strings look like:
        "Indo-European/Germanic/West Germanic/..."
    We extract the node at `level` (0-indexed from root).

    For level=1 this gives the top-level sub-family (e.g. "Germanic"),
    which is closest to WALS's genus concept for most families.

    Parameters
    ----------
    classification : str
        Full classification path separated by "/".
    level : int
        Which node to extract (0 = family, 1 = first sub-group, etc.).

    Returns
    -------
    genus : str or None if the classification doesn't have enough depth.
    """
    if not classification or pd.isna(classification):
        return None
    parts = [p.strip() for p in classification.split("/") if p.strip()]
    if len(parts) > level:
        return parts[level]
    # Not deep enough — return the deepest available node
    return parts[-1] if parts else None


def load_glottolog_genus_map(
    glottolog_repo_dir: str,
    genus_level: int = 2,
) -> Dict[str, str]:
    """
    Build a Glottocode → genus mapping from Glottolog's CLDF languages.csv.

    The Glottolog CLDF dataset (https://github.com/glottolog/glottolog-cldf)
    has a languages.csv with columns including Glottocode and
    Classification (a "/" separated genealogical path).

    Supports multiple Glottolog CLDF versions:
    - Versions with a Classification / classification / Classification_Path /
      classification_path column → parse the "/" separated path directly.
    - Versions without a classification column but with Family_ID and
      Parent_ID → reconstruct a shallow hierarchy from the parent chain.
    - Falls back to Family column if neither approach works.

    Parameters
    ----------
    glottolog_repo_dir : str
        Path to the cloned glottolog-cldf repo.
    genus_level : int
        Which level of the classification tree to use as "genus".
        Level 0 = family, 1 = first sub-group (closest to WALS genus), etc.

    Returns
    -------
    genus_map : dict mapping Glottocode → genus string
    """
    import warnings

    cldf_dir = os.path.join(glottolog_repo_dir, "cldf")
    lang_csv = os.path.join(cldf_dir, "languages.csv")

    if not os.path.exists(lang_csv):
        raise FileNotFoundError(
            f"Glottolog languages.csv not found at {lang_csv}. "
            f"Please clone https://github.com/glottolog/glottolog-cldf")

    gl_langs = pd.read_csv(lang_csv, low_memory=False)
    gl_langs.columns = [c.strip() for c in gl_langs.columns]

    # --- Identify the Glottocode column ---
    glottocode_col = None
    for c in gl_langs.columns:
        cl = c.lower()
        if cl in ("id", "glottocode", "language_id"):
            glottocode_col = c
            break
    if glottocode_col is None:
        raise ValueError("Cannot find Glottocode column in Glottolog CSV. "
                         f"Available columns: {list(gl_langs.columns)}")

    # --- Strategy 1: look for a classification path column ---
    class_col = None
    # Try exact and normalised column names across known Glottolog versions
    candidates = ("Classification", "classification",
                  "Classification_Path", "classification_path",
                  "Classification_path", "classificationpath")
    for c in gl_langs.columns:
        if c in candidates or c.lower().replace("_", "").replace(" ", "") in (
                "classification", "classificationpath"):
            class_col = c
            break

    genus_map: Dict[str, str] = {}

    if class_col is not None:
        for _, row in gl_langs.iterrows():
            gc = str(row[glottocode_col]).strip()
            classification = str(row.get(class_col, ""))
            genus = _extract_glottolog_genus(classification, genus_level)
            if genus:
                genus_map[gc] = genus
        print(f"Loaded Glottolog genus map (classification column): "
              f"{len(genus_map)} Glottocodes")
        return genus_map

    # --- Strategy 2: reconstruct hierarchy from Family_ID + Parent_ID ---
    family_id_col = None
    parent_id_col = None
    name_col = None
    for c in gl_langs.columns:
        cl = c.lower()
        if cl in ("family_id", "familyid"):
            family_id_col = c
        elif cl in ("parent_id", "parentid"):
            parent_id_col = c
        elif cl == "name":
            name_col = c

    if family_id_col is not None and parent_id_col is not None:
        print("No classification column found; reconstructing hierarchy "
              "from Family_ID + Parent_ID ...")

        # Build ID → name and ID → parent lookups
        id_to_name: Dict[str, str] = {}
        id_to_parent: Dict[str, str] = {}
        for _, row in gl_langs.iterrows():
            gid = str(row[glottocode_col]).strip()
            parent = str(row.get(parent_id_col, "")).strip()
            nm = str(row.get(name_col, gid)).strip() if name_col else gid
            id_to_name[gid] = nm
            if parent and parent != "nan" and parent != gid:
                id_to_parent[gid] = parent

        # Walk up the parent chain for each language to build a path,
        # then extract the node at the desired genus_level
        for _, row in gl_langs.iterrows():
            gc = str(row[glottocode_col]).strip()
            # Build ancestor chain (child → ... → root)
            chain = []
            cur = gc
            seen = set()
            while cur and cur not in seen:
                seen.add(cur)
                chain.append(id_to_name.get(cur, cur))
                cur = id_to_parent.get(cur)
            # chain is [self, parent, grandparent, ..., root]
            # Reverse to get [root, ..., self] = classification path
            chain.reverse()
            genus = _extract_glottolog_genus("/".join(chain), genus_level)
            if genus:
                genus_map[gc] = genus

        print(f"Loaded Glottolog genus map (Parent_ID hierarchy): "
              f"{len(genus_map)} Glottocodes")
        return genus_map

    # --- Strategy 3: fall back to Family column ---
    family_col = None
    for c in gl_langs.columns:
        if c.lower() == "family":
            family_col = c
            break

    if family_col is not None:
        warnings.warn(
            "Glottolog CSV has no Classification or Parent_ID column. "
            "Falling back to Family column for genus mapping. "
            "This is coarser than classification-based genus and results "
            "may not be comparable to WALS genus-based evaluation.")
        for _, row in gl_langs.iterrows():
            gc = str(row[glottocode_col]).strip()
            family = str(row.get(family_col, "")).strip()
            if family and family != "nan":
                genus_map[gc] = family
        print(f"Loaded Glottolog genus map (Family fallback): "
              f"{len(genus_map)} Glottocodes")
        return genus_map

    # --- Nothing worked ---
    warnings.warn(
        f"Could not extract genus information from Glottolog CSV at "
        f"{lang_csv}. Available columns: {list(gl_langs.columns)}. "
        f"Genus mapping will be empty; genus_source='family' will be "
        f"used as fallback in load_grambank_cldf().")
    return genus_map


def load_grambank_cldf(
    grambank_repo_dir: str,
    min_lang_features: int = 10,
    min_feature_value_langs: int = 10,
    genus_source: str = "glottolog",
    glottolog_repo_dir: Optional[str] = None,
    glottolog_genus_level: int = 2,
) -> Tuple[pd.DataFrame, list]:
    """
    Load Grambank directly from the glottobank/grambank repository.

    Grambank features are MOSTLY binary (0/1), but some have 3+ values.
    Values are treated as categorical string labels (via codes.csv),
    NOT as numeric — this is critical for correct model behaviour.

    For the "genus" equivalent needed by the branch-based evaluation:
      - genus_source="glottolog": Uses Glottolog classification (most accurate,
        requires glottolog_repo_dir to point at glottolog-cldf).
      - genus_source="family": Uses Grambank's Family column directly. This is
        coarser than WALS's genus and results may not be directly comparable.

    Parameters
    ----------
    grambank_repo_dir : str
        Path to the cloned grambank repo (containing a cldf/ subdirectory).
    min_lang_features : int
        Drop languages with fewer than this many observed features.
    min_feature_value_langs : int
        Drop feature values observed in fewer than this many languages.
    genus_source : str
        "glottolog" (default) or "family".
    glottolog_repo_dir : str or None
        Path to cloned glottolog-cldf repo. Required when genus_source="glottolog".
    glottolog_genus_level : int
        Classification tree level for genus extraction (default 2).
        Level 0 = family root, 1 = major sub-group, 2 = sub-sub-group.

    Returns
    -------
    df : pd.DataFrame  – one row per language, with columns:
        glottocode, name, genus, family, macroarea, feat_GB020, ...
    feature_cols : list of str
    """
    cldf_dir = os.path.join(grambank_repo_dir, "cldf")

    # --- Load component CSVs ---
    languages = pd.read_csv(os.path.join(cldf_dir, "languages.csv"))
    values = pd.read_csv(os.path.join(cldf_dir, "values.csv"))
    codes = pd.read_csv(os.path.join(cldf_dir, "codes.csv"))
    parameters = pd.read_csv(os.path.join(cldf_dir, "parameters.csv"))

    # Normalise column names
    languages.columns = [c.strip() for c in languages.columns]
    values.columns = [c.strip() for c in values.columns]
    codes.columns = [c.strip() for c in codes.columns]
    parameters.columns = [c.strip() for c in parameters.columns]

    # --- Build human-readable value labels from codes.csv ---
    # Grambank Code_IDs look like "GB020-1", "GB020-0", etc.
    code_to_name = dict(zip(codes["ID"], codes["Name"]))

    # --- Filter out missing/empty values ---
    # Some Grambank entries have empty strings or special values for missing data
    values = values[values["Value"].notna()].copy()
    values = values[values["Value"].astype(str).str.strip() != ""].copy()
    values = values[values["Value"].astype(str).str.strip() != "?"].copy()
   
    # --- Map values to human-readable labels ---
    # Grambank Value column contains "0", "1", "2", "3" as strings.
    # We use Code_ID → Name mapping for human-readable labels.
    # If Code_ID is missing, fall back to "Value_<raw>" to keep them categorical.
    values["Value_Label"] = values["Code_ID"].map(code_to_name)
    values["Value_Label"] = values["Value_Label"].astype(object)

    # Handle missing Code_IDs: use the raw Value with a prefix
    mask_no_label = values["Value_Label"].isna()
    if mask_no_label.any():
        values.loc[mask_no_label, "Value_Label"] = (
            "value_" + values.loc[mask_no_label, "Value"].astype(str))

    # --- Filter out "?" values (uncertain/not determinable) ---
    # Grambank uses "?" to mark uncertain or indeterminate observations.
    # These must be removed BEFORE the pivot — otherwise they become
    # spurious categorical values like "value_?" during training.
    n_before_q = len(values)
    # Remove rows where the raw Value is literally "?"
    mask_q_raw = values["Value"].astype(str).str.strip() == "?"
    # Safety net: also remove rows where the human-readable label indicates
    # uncertainty, since some codes.csv entries may give "?" a descriptive name
    uncertain_patterns = r"(?i)^(not\s+known|uncertain|not\s+applicable)$"
    mask_q_label = values["Value_Label"].astype(str).str.strip().str.match(
        uncertain_patterns, na=False)
    values = values[~(mask_q_raw | mask_q_label)].copy()
    n_filtered_q = n_before_q - len(values)
    if n_filtered_q > 0:
        print(f"Filtered {n_filtered_q} observations with Value=\"?\" "
              f"(uncertain/not determinable)")

    # --- Pivot: one row per language, one column per feature ---
    values["feat_col"] = "feat_" + values["Parameter_ID"].astype(str)

    wide = values.pivot_table(
        index="Language_ID",
        columns="feat_col",
        values="Value_Label",
        aggfunc="first",
    )
    wide = wide.reset_index().rename(columns={"Language_ID": "glottocode"})

    # --- Merge with language metadata ---
    # Find the Glottocode/ID column in languages.csv
    lang_id_col = None
    for c in languages.columns:
        cl = c.lower()
        if cl in ("id", "glottocode", "language_id"):
            lang_id_col = c
            break
    if lang_id_col is None:
        lang_id_col = languages.columns[0]

    lang_meta = languages.rename(columns={lang_id_col: "glottocode"})

    # Normalise metadata column names
    col_map = {}
    for c in lang_meta.columns:
        cl = c.lower()
        if cl == "name":
            col_map[c] = "name"
        elif cl in ("family", "family_name"):
            col_map[c] = "family"
        elif cl == "macroarea":
            col_map[c] = "macroarea"
    lang_meta = lang_meta.rename(columns=col_map)

    keep_cols = ["glottocode"]
    for col in ["name", "family", "macroarea"]:
        if col in lang_meta.columns:
            keep_cols.append(col)
    lang_meta = lang_meta[keep_cols].drop_duplicates(subset="glottocode")

    df = lang_meta.merge(wide, on="glottocode", how="inner")

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

    # --- Build genus column for branch-based evaluation ---
    if genus_source == "glottolog" and glottolog_repo_dir is not None:
        genus_map = load_glottolog_genus_map(
            glottolog_repo_dir, genus_level=glottolog_genus_level)
        df["genus"] = df["glottocode"].map(genus_map)
        # Languages without a Glottolog genus: fall back to Family
        if "family" in df.columns:
            no_genus = df["genus"].isna()
            df.loc[no_genus, "genus"] = df.loc[no_genus, "family"]
        n_with_genus = df["genus"].notna().sum()
        print(f"Genus from Glottolog: {n_with_genus}/{len(df)} languages "
              f"have genus assignments")
    elif genus_source == "glottolog" and glottolog_repo_dir is None:
        import warnings
        warnings.warn(
            "genus_source='glottolog' but no glottolog_repo_dir provided. "
            "Falling back to genus_source='family'. Results may not be "
            "directly comparable to WALS genus-based evaluation.")
        genus_source = "family"

    if genus_source == "family":
        if "family" in df.columns:
            df["genus"] = df["family"]
        else:
            df["genus"] = "Unknown"
        print(f"WARNING: Using Grambank Family as genus equivalent. "
              f"This is coarser than WALS genus — results may differ.")

    # Drop languages without a genus assignment
    df = df[df["genus"].notna()].reset_index(drop=True)

    print(f"Loaded Grambank: {len(df)} languages, {len(feature_cols)} features")
    return df, feature_cols


# ============================================================================
# 2.  Bible Corpus Loading (eBible / ParaBible)
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


def _extract_iso_from_filename(filename: str) -> str:
    """
    Extract an ISO 639-3-like language code from a Bible text filename.

    Supported naming conventions:
      - eBible:    "eng-eng_kjv.txt", "spa-spaRV1909.txt"
      - ParaBible: "eng.txt", "eng-web.txt", "spa-rvr.txt"
      - Numbered:  "1234.txt" (corpus internal ID)

    Returns the best guess at an ISO 639-3 code.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    # Try to extract a 3-letter code from the start
    match = re.match(r"^([a-z]{2,3})", stem.lower())
    if match:
        return match.group(1)
    return stem.lower()


def _load_bible_dir(
    bible_dir: str,
    script_filter: bool = True,
    min_tokens: int = 50000,
    source_label: str = "Bible",
) -> Dict[str, str]:
    """
    Internal helper: load Bible translations from a single directory.

    Handles both eBible (plain verse-per-line) and legacy ParaBible
    (TAB-separated <verse_id>\\t<text>) formats transparently.

    Parameters
    ----------
    bible_dir : str
        Directory containing Bible text files.
        If an ``corpus/`` subdirectory exists it will be preferred (eBible
        repo layout).
    script_filter : bool
        If True, only keep translations in Latin, Cyrillic, or Greek scripts.
    min_tokens : int
        Only keep translations with at least this many tokens.
    source_label : str
        Label used in log messages (e.g. "eBible", "ParaBible").

    Returns
    -------
    texts : dict mapping language_id → concatenated text
    """
    texts: Dict[str, str] = {}

    # Auto-detect repo root vs corpus directory
    corpus_dir = os.path.join(bible_dir, "corpus")
    if os.path.isdir(corpus_dir):
        search_dir = corpus_dir
    else:
        search_dir = bible_dir

    # Find text files (exclude vref.txt and manifest)
    patterns = [
        os.path.join(search_dir, "*.txt"),
        os.path.join(search_dir, "**", "*.txt"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    exclude_names = {"vref.txt", "manifest.csv"}
    files = [f for f in files if os.path.basename(f) not in exclude_names]
    files = sorted(set(files))

    print(f"[{source_label}] Found {len(files)} Bible text files in {search_dir}")

    # Track ISO codes to handle multiple translations per language
    # (keep the longest one)
    iso_texts: Dict[str, Tuple[str, int]] = {}  # iso -> (text, n_tokens)

    for fpath in files:
        raw_id = os.path.splitext(os.path.basename(fpath))[0]
        iso_code = _extract_iso_from_filename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except (UnicodeDecodeError, OSError):
            continue

        # eBible format: one verse per line, plain text (no reference prefix).
        # Blank lines = missing verses, <range> = grouped verse placeholder.
        all_text = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line == "<range>":
                continue
            # Also handle legacy ParaBible TAB-separated format
            parts = line.split("\t", 1)
            if len(parts) == 2:
                all_text.append(parts[1])
            else:
                all_text.append(line)

        full_text = " ".join(all_text)
        tokens = full_text.split()
        n_tokens = len(tokens)

        if n_tokens < min_tokens:
            continue

        if script_filter and not is_latin_cyrillic_greek(full_text[:5000]):
            continue

        # Keep the longest translation per ISO code
        if iso_code not in iso_texts or n_tokens > iso_texts[iso_code][1]:
            iso_texts[iso_code] = (full_text, n_tokens)

        # Also store by raw filename ID for direct matching
        if raw_id != iso_code:
            texts[raw_id] = full_text

    # Merge ISO-keyed texts
    for iso_code, (text, _) in iso_texts.items():
        texts[iso_code] = text

    print(f"[{source_label}] Loaded {len(texts)} translations "
          f"(after script filter={script_filter}, "
          f"min_tokens={min_tokens})")
    return texts


def load_ebible_texts(
    ebible_dir: str,
    script_filter: bool = True,
    min_tokens: int = 50000,
) -> Dict[str, str]:
    """
    Load Bible translations from the BibleNLP/ebible corpus directory.

    The eBible corpus stores each translation as a plain-text file with
    one verse per line (no verse-reference prefix; blank lines for missing
    verses).  Filenames follow <langCode>-<variant>.txt, e.g.
    eng-eng_kjv.txt, spa-spaRV1909.txt.

    Parameters
    ----------
    ebible_dir : str
        Directory containing Bible text files (one file per translation).
        Can be the eBible repo root (will look in corpus/ subdirectory)
        or the corpus directory itself.
    script_filter : bool
        If True, only keep translations in Latin, Cyrillic, or Greek
        scripts (matching the paper's methodology).
    min_tokens : int
        Only keep translations with at least this many tokens.

    Returns
    -------
    texts : dict mapping language_id → concatenated text
    """
    return _load_bible_dir(ebible_dir, script_filter, min_tokens,
                           source_label="eBible")


def load_parabible_texts(
    parabible_dir: str,
    script_filter: bool = True,
    min_tokens: int = 50000,
) -> Dict[str, str]:
    """
    Load Bible translations from the LingConLab/parabible directory.

    The ParaBible corpus uses plain-text or TAB-separated files with one
    verse per line.  Filenames follow <langCode>.txt or
    <langCode>-<variant>.txt.

    Parameters
    ----------
    parabible_dir : str
        Directory containing ParaBible text files.
    script_filter : bool
        If True, only keep translations in Latin, Cyrillic, or Greek scripts.
    min_tokens : int
        Only keep translations with at least this many tokens.

    Returns
    -------
    texts : dict mapping language_id → concatenated text
    """
    return _load_bible_dir(parabible_dir, script_filter, min_tokens,
                           source_label="ParaBible")


def load_bible_texts(
    bible_source: str = "ebible",
    ebible_dir: Optional[str] = None,
    parabible_dir: Optional[str] = None,
    script_filter: bool = True,
    min_tokens: int = 50000,
) -> Dict[str, str]:
    """
    Load Bible translations from one or both corpus sources.

    Parameters
    ----------
    bible_source : str
        Which corpus to use: ``"ebible"``, ``"parabible"``, or ``"both"``.
    ebible_dir : str or None
        Path to eBible corpus directory (required when source is
        ``"ebible"`` or ``"both"``).
    parabible_dir : str or None
        Path to ParaBible corpus directory (required when source is
        ``"parabible"`` or ``"both"``).
    script_filter : bool
        If True, only keep translations in Latin, Cyrillic, or Greek scripts.
    min_tokens : int
        Only keep translations with at least this many tokens.

    Returns
    -------
    texts : dict mapping language_id → concatenated text.
        When *both* sources are used, translations are merged; for any
        language present in both corpora the longer text is kept.
    """
    texts: Dict[str, str] = {}

    if bible_source in ("ebible", "both"):
        if ebible_dir is None:
            raise ValueError("ebible_dir must be provided when bible_source "
                             f"is '{bible_source}'")
        texts.update(load_ebible_texts(ebible_dir, script_filter, min_tokens))

    if bible_source in ("parabible", "both"):
        if parabible_dir is None:
            raise ValueError("parabible_dir must be provided when "
                             f"bible_source is '{bible_source}'")
        para_texts = load_parabible_texts(parabible_dir, script_filter,
                                          min_tokens)
        if bible_source == "both":
            # Merge: keep the longer text per language
            for lang_id, text in para_texts.items():
                if lang_id not in texts or len(text) > len(texts[lang_id]):
                    texts[lang_id] = text
            print(f"[Merged] {len(texts)} unique translations after combining "
                  f"eBible + ParaBible")
        else:
            texts.update(para_texts)

    if bible_source not in ("ebible", "parabible", "both"):
        raise ValueError(f"Unknown bible_source: '{bible_source}'. "
                         f"Choose from 'ebible', 'parabible', or 'both'.")

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
    manual_mapping_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Find the intersection of WALS languages and Bible translations.

    Strategy (in priority order):
    1. Manual mapping CSV (wals_code,bible_lang_id) if provided
    2. ISO 639-3 code match (WALS iso639p3code ↔ bible filename)
    3. Exact match on wals_code ↔ bible_lang_id
    4. 3-letter prefix match

    Parameters
    ----------
    df : pd.DataFrame with 'wals_code' column and optionally 'iso639p3code'
    bible_lang_ids : list of str
    manual_mapping_csv : str or None
        Path to a CSV file with columns: wals_code, bible_lang_id

    Returns
    -------
    matched_df : pd.DataFrame  – subset of df with a new column
        'bible_lang_id' indicating the matched Bible translation.
    """
    bible_set = set(bible_lang_ids)
    bible_set_lower = {bid.lower() for bid in bible_lang_ids}

    # Build a lookup: lowercase bible_id -> original bible_id
    bible_lower_to_orig = {}
    for bid in bible_lang_ids:
        bible_lower_to_orig[bid.lower()] = bid

    # Load manual mapping if provided
    manual_map = {}
    if manual_mapping_csv and os.path.exists(manual_mapping_csv):
        mapping_df = pd.read_csv(manual_mapping_csv)
        for _, row in mapping_df.iterrows():
            manual_map[str(row["wals_code"]).lower()] = str(row["bible_lang_id"])
        print(f"Loaded {len(manual_map)} manual WALS-Bible mappings")

    matches = []
    matched_wals = set()

    for _, row in df.iterrows():
        wals_code = str(row["wals_code"]).lower()
        bible_id = None

        # 1. Manual mapping
        if wals_code in manual_map:
            candidate = manual_map[wals_code]
            if candidate in bible_set or candidate.lower() in bible_set_lower:
                bible_id = candidate

        # 2. ISO 639-3 match
        if bible_id is None and "iso639p3code" in row.index:
            iso = str(row.get("iso639p3code", "")).lower().strip()
            if iso and iso != "nan" and iso in bible_set_lower:
                bible_id = bible_lower_to_orig.get(iso, iso)

        # 3. Exact match on wals_code
        if bible_id is None and wals_code in bible_set_lower:
            bible_id = bible_lower_to_orig.get(wals_code, wals_code)

        # 4. 3-letter prefix
        if bible_id is None:
            prefix = wals_code[:3]
            if prefix in bible_set_lower:
                bible_id = bible_lower_to_orig.get(prefix, prefix)

        if bible_id is not None and wals_code not in matched_wals:
            matched_wals.add(wals_code)
            matches.append({**row, "bible_lang_id": bible_id})

    matched_df = pd.DataFrame(matches)
    if len(matched_df) > 0:
        matched_df = matched_df.reset_index(drop=True)
    print(f"Matched {len(matched_df)} WALS languages to Bible translations "
          f"(out of {len(df)} WALS languages and "
          f"{len(bible_lang_ids)} Bible translations)")
    if len(matched_df) < 50:
        print("WARNING: Few matches found. Consider providing a manual mapping "
              "CSV (--manual_mapping with columns: wals_code, bible_lang_id).")
    return matched_df


# ============================================================================
# 5.  Binarisation (database-agnostic: works on WALS, Grambank, or any
#     DataFrame with categorical feature columns)
# ============================================================================

def binarise_features(
    df: pd.DataFrame,
    feature_cols: list,
) -> Tuple[np.ndarray, list, dict, dict]:
    """
    Binarise typological features following the paper's protocol (Section 6).

    Works on any DataFrame with categorical feature columns — WALS, Grambank,
    or any other typological database.

    For Grambank: most features are already binary (2-valued, 0/1), so
    binarisation mostly produces single columns rather than one-hot expansions.
    The binary matrix will be closer to the original data than for WALS.
    Features with 3+ values (which exist in Grambank) get proper one-hot
    treatment.

    Protocol:
    - Features with exactly 2 values → single binary column
      (first value alphabetically → 0, second → 1).
    - Features with ≥ 3 values → one-hot encoded into n binary columns.
    - Features with 1 value → dropped (no discriminative information).

    Returns
    -------
    binary_matrix : np.ndarray (n_langs × n_binary_features), 0/1/NaN
    binary_col_names : list of str
    feature_groups : dict  original_feat → list of binary col indices
    feature_value_names : dict  original_feat → list of original value names
        For 2-valued features: [neg_value, pos_value] (0 → neg, 1 → pos)
        For ≥3-valued features: [val_0, val_1, ...] matching feature_groups order
    """
    binary_columns = []
    binary_col_names = []
    feature_groups: Dict[str, List[int]] = {}
    feature_value_names: Dict[str, List[str]] = {}

    idx = 0
    for col in feature_cols:
        unique_vals = sorted(df[col].dropna().unique())
        n_vals = len(unique_vals)

        if n_vals <= 1:
            # Skip: no discriminative information
            continue

        group_indices = []

        if n_vals == 2:
            # Single binary column: first value → 0, second → 1
            bin_col = np.full(len(df), np.nan)
            observed = df[col].notna()
            bin_col[observed] = (df.loc[observed, col] == unique_vals[1]).astype(float).values
            binary_columns.append(bin_col)
            binary_col_names.append(f"{col}")
            group_indices.append(idx)
            idx += 1
            feature_value_names[col] = unique_vals  # [neg_value, pos_value]
        else:
            # One-hot encode (≥ 3 values)
            for val in unique_vals:
                bin_col = np.full(len(df), np.nan)
                observed = df[col].notna()
                bin_col[observed] = (df.loc[observed, col] == val).astype(float).values
                binary_columns.append(bin_col)
                binary_col_names.append(f"{col}={val}")
                group_indices.append(idx)
                idx += 1
            feature_value_names[col] = unique_vals  # [val_0, val_1, ...]

        feature_groups[col] = group_indices

    binary_matrix = np.column_stack(binary_columns)
    print(f"Binarised matrix: {binary_matrix.shape[0]} langs × "
          f"{binary_matrix.shape[1]} binary features")
    return binary_matrix, binary_col_names, feature_groups, feature_value_names


# Backwards compatibility alias
binarise_wals = binarise_features


# ============================================================================
# 6.  Embedding Alignment
# ============================================================================

def align_embeddings(
    lang_embeddings: np.ndarray,
    charlm_lang2idx: dict,
    wals_df: pd.DataFrame,
    matched_df: pd.DataFrame,
) -> np.ndarray:
    """
    Align char-LM language embeddings to match WALS language order.

    The char-LM produces embeddings indexed by Bible language IDs.
    This function creates an embedding matrix where row i corresponds
    to WALS language i (after filtering). Languages without a Bible
    match get a zero embedding.

    Parameters
    ----------
    lang_embeddings : np.ndarray (n_charlm_langs, d)
        Embeddings from the char-LM, ordered by charlm_lang2idx.
    charlm_lang2idx : dict mapping bible_lang_id → index
    wals_df : pd.DataFrame – the WALS dataframe (defines row order)
    matched_df : pd.DataFrame – output of match_wals_to_bible()

    Returns
    -------
    aligned : np.ndarray (n_wals_langs, d) – embeddings aligned to WALS order
    """
    n_wals = len(wals_df)
    d = lang_embeddings.shape[1]
    aligned = np.zeros((n_wals, d), dtype=np.float32)

    # Build WALS code → row index mapping
    wals_code_to_idx = {
        str(code).lower(): i
        for i, code in enumerate(wals_df["wals_code"])
    }

    n_matched = 0
    for _, row in matched_df.iterrows():
        wals_code = str(row["wals_code"]).lower()
        bible_id = str(row["bible_lang_id"])

        wals_idx = wals_code_to_idx.get(wals_code)
        charlm_idx = charlm_lang2idx.get(bible_id)

        if wals_idx is not None and charlm_idx is not None:
            aligned[wals_idx] = lang_embeddings[charlm_idx]
            n_matched += 1

    print(f"Aligned {n_matched}/{n_wals} language embeddings")
    return aligned


# ============================================================================
# 7.  End-to-End Data Preparation
# ============================================================================

def prepare_all(
    wals_repo_dir: str,
    parabible_dir: Optional[str] = None,
    ebible_dir: Optional[str] = None,
    bible_source: str = "ebible",
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
        Path to ParaBible text files.  Also accepted as a fallback when
        *ebible_dir* is None (legacy behaviour).
    ebible_dir : str or None
        Path to eBible corpus directory.  If None and *parabible_dir* is
        set, *parabible_dir* is used for the eBible source (backwards
        compatible).
    bible_source : str
        Which Bible corpus to use: ``"ebible"``, ``"parabible"``, or
        ``"both"``.
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
    binary_matrix, bin_names, feature_groups, feature_value_names = binarise_wals(df, feature_cols)

    df.to_csv(output_csv, index=False)
    print(f"Saved prepared WALS data to {output_csv}")

    result = {
        "df": df,
        "feature_cols": feature_cols,
        "binary_matrix": binary_matrix,
        "binary_col_names": bin_names,
        "feature_groups": feature_groups,
    }

    # --- Bible corpus (optional) ---
    # Backwards compatibility: if only parabible_dir is given, use it as
    # the ebible directory (original behaviour).
    effective_ebible = ebible_dir or parabible_dir
    has_bible = (bible_source in ("ebible", "both") and effective_ebible) or \
                (bible_source in ("parabible", "both") and parabible_dir)
    if has_bible:
        texts = load_bible_texts(
            bible_source=bible_source,
            ebible_dir=effective_ebible,
            parabible_dir=parabible_dir,
        )
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
        description="Prepare data from WALS CLDF repo and Bible corpus")
    parser.add_argument("--wals_repo", type=str, required=True,
                        help="Path to cloned cldf-datasets/wals repo")
    parser.add_argument("--bible_source", type=str, default="ebible",
                        choices=["ebible", "parabible", "both"],
                        help="Bible corpus to use (default: ebible)")
    parser.add_argument("--ebible_dir", type=str, default=None,
                        help="Path to eBible corpus directory")
    parser.add_argument("--parabible_dir", type=str, default=None,
                        help="Path to ParaBible corpus directory")
    parser.add_argument("--charlm_output", type=str, default="charlm_data",
                        help="Output directory for char-LM training data")
    parser.add_argument("--output_csv", type=str, default="wals_prepared.csv",
                        help="Path to save prepared WALS CSV")
    args = parser.parse_args()

    prepare_all(
        wals_repo_dir=args.wals_repo,
        ebible_dir=args.ebible_dir,
        parabible_dir=args.parabible_dir,
        bible_source=args.bible_source,
        charlm_output_dir=args.charlm_output,
        output_csv=args.output_csv,
    )
