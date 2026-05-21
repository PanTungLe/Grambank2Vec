"""
downstream/arc_directions.py
=============================
Extract arc-direction statistics from Universal Dependencies treebanks.

For each language l and each (relation, head_UPOS, dep_UPOS) triple,
compute p(dependent precedes head) = k / n where k = count of arcs where
the dependent token appears before the head token in linear order.

These direction probabilities are the TARGET variable for the downstream task:
predicting gradient UD syntactic behaviour from typological embeddings.

Typological grounding
---------------------
The key UD configurations map onto classic typological parameters:

  (obj,   VERB,  NOUN/PRON)  ↔  VO vs OV        (WALS 83A, Grambank GB065)
  (nsubj, VERB,  NOUN/PRON)  ↔  SV vs VS
  (case,  NOUN,  ADP)        ↔  Pre vs Post      (WALS 85A, Grambank GB068)
  (amod,  NOUN,  ADJ)        ↔  Adj-N vs N-Adj   (WALS 87A, Grambank GB069)
  (nmod,  NOUN,  NOUN/PROPN) ↔  GenN vs NGen     (WALS 86A)
  (aux,   VERB,  AUX)        ↔  Aux order        (WALS 83A proxy)
  (advmod,VERB,  ADV)        ↔  Adverb placement
  (cop,   NOUN,  AUX)        ↔  Copula order

This is the arc-direction entropy approach of Gulordava & Merlo (2015) and
the token-based typology of Östling (2015), applied to a predictive model.

Output
------
  {glottocode}.json  per-language direction statistics
  summary.csv        all languages × all configurations

Usage
-----
    python downstream/arc_directions.py \\
        --manifest downstream_results/ud_data/manifest.json \\
        --out_dir  downstream_results/arc_directions \\
        --split    train
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Target configurations
# Drawn from the most typologically informative (relation, head_upos, dep_upos)
# combinations in UD. Each has a short mnemonic name and a typological gloss.
# ---------------------------------------------------------------------------

# fmt: off
TARGET_CONFIGS = [
    # (deprel, head_upos, dep_upos, name, typological_gloss)
    ("obj",    "VERB",  "NOUN",   "obj_V_N",   "VO order (noun object)"),
    ("obj",    "VERB",  "PRON",   "obj_V_P",   "VO order (pronoun object)"),
    ("nsubj",  "VERB",  "NOUN",   "nsubj_V_N", "SV order (noun subject)"),
    ("nsubj",  "VERB",  "PRON",   "nsubj_V_P", "SV order (pronoun subject)"),
    ("case",   "NOUN",  "ADP",    "case_N_D",  "Adposition order (pre vs post)"),
    ("case",   "PROPN", "ADP",    "case_PN_D", "Adposition order (proper noun)"),
    ("amod",   "NOUN",  "ADJ",    "amod_N_A",  "Adjective-noun order"),
    ("nmod",   "NOUN",  "NOUN",   "nmod_N_N",  "Genitive order (N-N)"),
    ("nmod",   "NOUN",  "PROPN",  "nmod_N_PN", "Genitive order (N-PropN)"),
    ("aux",    "VERB",  "AUX",    "aux_V_A",   "Auxiliary-verb order"),
    ("advmod", "VERB",  "ADV",    "adv_V_D",   "Adverb-verb order"),
    ("advmod", "ADJ",   "ADV",    "adv_J_D",   "Adverb-adjective order"),
    ("cop",    "NOUN",  "AUX",    "cop_N_C",   "Copula order (nominal pred)"),
    ("cop",    "ADJ",   "AUX",    "cop_J_C",   "Copula order (adjectival pred)"),
    ("obl",    "VERB",  "NOUN",   "obl_V_N",   "Oblique argument order"),
    ("xcomp",  "VERB",  "VERB",   "xcomp_V_V", "Complement clause order"),
    ("ccomp",  "VERB",  "VERB",   "ccomp_V_V", "Clausal complement order"),
    ("nsubj",  "ADJ",   "NOUN",   "nsubj_J_N", "Subject-predicate (adj pred)"),
]
# fmt: on

CONFIG_NAMES   = [c[3] for c in TARGET_CONFIGS]   # short mnemonic
CONFIG_GLOSSES = {c[3]: c[4] for c in TARGET_CONFIGS}
MIN_COUNT = 5  # minimum arc count to include a configuration


# ---------------------------------------------------------------------------
# CoNLL-U parser (full, with arc information)
# ---------------------------------------------------------------------------

def parse_conllu_arcs(text: str) -> List[List[Dict]]:
    """
    Parse CoNLL-U text into sentences, each a list of token dicts.

    Each token dict:
        id       int  (1-indexed)
        form     str
        upos     str
        head     int  (0 = root)
        deprel   str
        position int  (= id, for clarity)

    Multi-word tokens and empty nodes are skipped.
    """
    sentences = []
    current = []

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if not line:
            if current:
                sentences.append(current)
                current = []
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        tok_id = parts[0]
        if "-" in tok_id or "." in tok_id:
            continue
        try:
            current.append({
                "id":       int(tok_id),
                "form":     parts[1],
                "upos":     parts[3],
                "head":     int(parts[6]),
                "deprel":   parts[7].split(":")[0],  # strip subtypes
                "position": int(tok_id),
            })
        except (ValueError, IndexError):
            continue

    if current:
        sentences.append(current)
    return sentences


# ---------------------------------------------------------------------------
# Direction extraction
# ---------------------------------------------------------------------------

def extract_directions(
    sentences: List[List[Dict]],
    configs: List[Tuple] = TARGET_CONFIGS,
    min_count: int = MIN_COUNT,
) -> Dict[str, Dict]:
    """
    Extract arc-direction statistics from parsed sentences.

    For each (deprel, head_upos, dep_upos) configuration:
        n_dep_before_head  = count of arcs where dep position < head position
        n_head_before_dep  = count of arcs where head position < dep position
        n_total            = n_dep_before_head + n_head_before_dep
        p_dep_before_head  = n_dep_before_head / n_total

    A value near 1.0 = dependent consistently precedes head  (e.g. OV, postposition)
    A value near 0.0 = head consistently precedes dependent  (e.g. VO, preposition)
    A value near 0.5 = free / no dominant order

    Returns
    -------
    stats : {config_name: {"p": float, "n": int, "k": int}}
            Only includes configs with n >= min_count.
    """
    # Index by (deprel, head_upos, dep_upos) for fast lookup
    config_index: Dict[Tuple, str] = {
        (c[0], c[1], c[2]): c[3] for c in configs
    }

    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # [k, n]

    for sent in sentences:
        id_to_tok = {t["id"]: t for t in sent}

        for tok in sent:
            if tok["head"] == 0:
                continue  # skip root
            head_tok = id_to_tok.get(tok["head"])
            if head_tok is None:
                continue

            key = (tok["deprel"], head_tok["upos"], tok["upos"])
            if key not in config_index:
                continue

            name = config_index[key]
            dep_before_head = int(tok["position"] < head_tok["position"])
            counts[name][0] += dep_before_head   # k: dep before head
            counts[name][1] += 1                  # n: total

    stats = {}
    for name, (k, n) in counts.items():
        if n >= min_count:
            stats[name] = {
                "p": k / n,   # p(dep before head)
                "k": k,
                "n": n,
            }

    return stats


def entropy(p: float) -> float:
    """Binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def extract_language_profile(
    conllu_path: str,
    min_count: int = MIN_COUNT,
) -> Dict:
    """
    Extract full arc-direction profile from a .conllu file.

    Returns a dict with:
        stats       {config_name: {p, k, n}}
        summary     {config_name: p}   (just the probability, for easy loading)
        entropy     {config_name: H}   (word-order freedom)
        n_sentences int
        n_tokens    int
    """
    with open(conllu_path, encoding="utf-8") as f:
        text = f.read()

    sentences = parse_conllu_arcs(text)
    stats = extract_directions(sentences, min_count=min_count)

    return {
        "stats":      stats,
        "summary":    {k: v["p"] for k, v in stats.items()},
        "entropy":    {k: entropy(v["p"]) for k, v in stats.items()},
        "n_sentences": len(sentences),
        "n_tokens":    sum(len(s) for s in sentences),
    }


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_all_profiles(
    manifest: Dict[str, Dict[str, str]],
    out_dir: str,
    split: str = "train",
    min_count: int = MIN_COUNT,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Extract arc-direction profiles for all languages in the manifest.

    Returns
    -------
    df : pd.DataFrame with rows=glottocodes, columns=config_names,
         values=p(dep_before_head). NaN for unobserved configs.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "direction_profiles.csv")
    counts_path  = os.path.join(out_dir, "direction_counts.csv")

    if os.path.exists(summary_path) and not overwrite:
        print(f"Loading existing profiles from {summary_path}")
        return pd.read_csv(summary_path, index_col=0)

    records_p = {}   # glot -> {config: p}
    records_n = {}   # glot -> {config: n}

    for glot, paths in manifest.items():
        if split not in paths:
            continue
        path = paths[split]
        if not os.path.exists(path):
            continue

        print(f"  {glot}: {path}")
        profile = extract_language_profile(path, min_count=min_count)

        # Save individual profile
        lang_path = os.path.join(out_dir, f"{glot}.json")
        if not os.path.exists(lang_path) or overwrite:
            with open(lang_path, "w") as f:
                json.dump(profile, f, indent=2)

        records_p[glot] = profile["summary"]
        records_n[glot] = {k: v["n"] for k, v in profile["stats"].items()}

        n_sents = profile["n_sentences"]
        n_obs = len(profile["stats"])
        print(f"    {n_sents} sentences, {n_obs}/{len(TARGET_CONFIGS)} configs observed")

        # Print key directions
        for name in ["obj_V_N", "case_N_D", "amod_N_A"]:
            if name in profile["summary"]:
                p = profile["summary"][name]
                n = profile["stats"][name]["n"]
                gloss = CONFIG_GLOSSES[name]
                direction = "dep-first" if p > 0.5 else "head-first"
                print(f"    {gloss}: p={p:.3f} ({direction}, n={n})")

    if not records_p:
        print("WARNING: No profiles extracted.")
        return pd.DataFrame()

    df_p = pd.DataFrame(records_p).T  # (n_langs, n_configs)
    df_n = pd.DataFrame(records_n).T

    # Reorder columns to match TARGET_CONFIGS order
    ordered_cols = [c for c in CONFIG_NAMES if c in df_p.columns]
    df_p = df_p[ordered_cols]
    df_n = df_n[[c for c in ordered_cols if c in df_n.columns]]

    df_p.to_csv(summary_path)
    df_n.to_csv(counts_path)

    print(f"\nProfiles saved: {summary_path}")
    print(f"  {len(df_p)} languages × {len(df_p.columns)} configurations")
    print(f"  Mean NaN fraction: {df_p.isna().mean().mean():.3f}")

    # Print summary statistics
    print("\nConfiguration coverage and mean direction:")
    for col in df_p.columns:
        vals = df_p[col].dropna()
        if len(vals) == 0:
            continue
        gloss = CONFIG_GLOSSES.get(col, col)
        print(f"  {col:12s} ({gloss[:35]:35s}): "
              f"n_langs={len(vals):2d}  "
              f"mean_p={vals.mean():.3f}  "
              f"std={vals.std():.3f}")

    return df_p


# ---------------------------------------------------------------------------
# Typological validation: do our extracted directions match WALS?
# ---------------------------------------------------------------------------

def validate_against_wals(
    profiles_df: pd.DataFrame,
    ckpt_dir: str = "checkpoints/wals_learned_s42",
    out_dir: str = ".",
):
    """
    Validate UD-extracted directions against WALS typological predictions.

    For each language, compare:
      WALS 81A argmax (SOV/SVO/VSO) vs UD obj_V_N direction
      WALS 85A argmax (Pre/Post)    vs UD case_N_D direction
      WALS 87A argmax (AdjN/NAdj)   vs UD amod_N_A direction

    Prints a confusion matrix and Spearman correlation.
    """
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from downstream.representations import load_checkpoint
    from scipy.special import softmax
    from scipy.stats import spearmanr

    ckpt = load_checkpoint(ckpt_dir)
    feat2vals = ckpt["feat2values"]
    fv2id = ckpt["featvalue2id"]
    lang2id = ckpt["lang2id"]
    lang_embs = ckpt["lang_embs"]
    fv_embs = ckpt["fv_embs"]
    feat_names = ckpt["feat_names_sorted"]

    print("\n--- Validation: UD directions vs WALS predictions ---")
    for wals_feat, ud_config, description in [
        ("83A", "obj_V_N",   "VO order: WALS 83A vs UD obj direction"),
        ("85A", "case_N_D",  "Adposition: WALS 85A vs UD case direction"),
    ]:
        if wals_feat not in feat_names:
            continue
        if ud_config not in profiles_df.columns:
            continue

        fi = feat_names.index(wals_feat)
        vals = feat2vals[wals_feat]
        gids = [fv2id[f"{wals_feat}={v}"] for v in vals]

        wals_probs, ud_dirs = [], []
        for glot in profiles_df.index:
            if glot not in lang2id:
                continue
            if pd.isna(profiles_df.loc[glot, ud_config]):
                continue
            idx = lang2id[glot]
            le = lang_embs[idx]
            scores = fv_embs[gids] @ le
            probs = softmax(scores)
            # For VO features: p(OV) or p(VO) — depends on feature encoding
            wals_probs.append(float(probs.max()))
            ud_dirs.append(float(profiles_df.loc[glot, ud_config]))

        if len(wals_probs) < 5:
            continue
        rho, p = spearmanr(wals_probs, ud_dirs)
        print(f"  {description}")
        print(f"    n={len(wals_probs)}  Spearman(WALS_conf, UD_p) = {rho:.3f}  (p={p:.4f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Extract UD arc-direction profiles for typological prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True,
                   help="Path to manifest.json from ud_data.py")
    p.add_argument("--out_dir",  required=True,
                   help="Output directory for direction profiles")
    p.add_argument("--split",    default="train",
                   choices=["train", "test"])
    p.add_argument("--min_count", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate", action="store_true",
                   help="Validate against WALS predictions (requires checkpoint)")
    p.add_argument("--checkpoint", default="checkpoints/wals_learned_s42")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    print(f"Extracting arc-direction profiles for {len(manifest)} languages "
          f"(split={args.split}, min_count={args.min_count})")

    df = extract_all_profiles(
        manifest=manifest,
        out_dir=args.out_dir,
        split=args.split,
        min_count=args.min_count,
        overwrite=args.overwrite,
    )

    if args.validate and not df.empty:
        validate_against_wals(df, args.checkpoint, args.out_dir)

    print(f"\nDone. Profiles: {df.shape}")


if __name__ == "__main__":
    main()
