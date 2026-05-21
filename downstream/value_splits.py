"""
downstream/value_splits.py
============================
Generate the three evaluation splits for the direction-prediction task.

Split A — Random language holdout (standard)
    80% train / 20% test, stratified by word-order type.
    Categorical may or may not win; serves as the sanity check.

Split B — Family holdout
    Hold out entire genealogical families (e.g. all Turkic, all Semitic).
    Tests whether typological representations generalise beyond genealogy.
    Categorical should outperform binary here because family-averaged binary
    vectors trivially encode family identity.

Split C — Value holdout (THE CRITICAL SPLIT)
    Hold out all languages with a specific typological value for WALS 81A
    (word order): VOS, OVS, "No dominant order".
    SOV (n=18) and SVO (n=37) are training-only; held-out values are rare.
    This directly tests zero-shot generalisation from the VALUE EMBEDDING
    GEOMETRY — the model must predict syntactic behaviour for a word-order
    type it has never seen during training, using only its position in the
    categorical value space relative to SOV/SVO/VSO.

    Binary one-hot vectors have no training signal for held-out values
    (the bit for VOS/OVS is always 0 in training).
    Categorical embeddings CAN generalise if VOS is embedded close to
    SVO (both are verb-initial or share other typological properties).

Usage
-----
    python downstream/value_splits.py \\
        --checkpoint checkpoints/wals_learned_s42 \\
        --profiles   downstream_results/arc_directions/direction_profiles.csv \\
        --out_dir    downstream_results/value_splits \\
        --family_map downstream_results/ud_data/family_map.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import softmax

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from downstream.representations import load_checkpoint
from downstream.ud_data import GLOTTOCODES_IN_UD, _GLOT_TO_ISO


# ---------------------------------------------------------------------------
# Genealogical family map
# Built from Glottolog families; covers our 59 UD languages.
# ---------------------------------------------------------------------------

# Format: {glottocode: family_name}
# Source: Glottolog v5 (glottolog.org), family-level classification
FAMILY_MAP: Dict[str, str] = {
    # Indo-European
    "stan1293": "Indo-European", "stan1290": "Indo-European",
    "stan1289": "Indo-European", "dani1285": "Indo-European",
    "dutc1256": "Indo-European", "finn1318": "Uralic",
    "ital1282": "Indo-European", "port1283": "Indo-European",
    "swed1254": "Indo-European", "czec1258": "Indo-European",
    "poli1260": "Indo-European", "slov1268": "Indo-European",
    "hung1274": "Uralic",        "latv1249": "Indo-European",
    "lith1251": "Indo-European", "icel1247": "Indo-European",
    "faro1244": "Indo-European", "iris1253": "Indo-European",
    "wels1247": "Indo-European", "bret1244": "Indo-European",
    "nort2671": "Uralic",        "west2354": "Indo-European",
    "bela1254": "Indo-European", "russ1263": "Indo-European",
    "ukra1253": "Indo-European", "hind1269": "Indo-European",
    "mara1378": "Indo-European", "bhoj1244": "Indo-European",
    "pers1301": "Indo-European", "west2369": "Indo-European",
    "nucl1235": "Indo-European", "mode1248": "Indo-European",
    # Turkic
    "nucl1301": "Turkic",        "yaku1245": "Turkic",
    "kaza1248": "Turkic",
    # Dravidian
    "tami1289": "Dravidian",     "telu1262": "Dravidian",
    # Semitic
    "stan1318": "Afro-Asiatic",  "hebr1245": "Afro-Asiatic",
    "malt1254": "Afro-Asiatic",
    # Sino-Tibetan
    "mand1415": "Sino-Tibetan",  "yuec1235": "Sino-Tibetan",
    # Japonic
    "nucl1643": "Japonic",
    # Koreanic
    "kore1280": "Koreanic",
    # Austronesian
    "taga1270": "Austronesian",
    # Uralic (additional)
    "finn1318": "Uralic",        "komi1268": "Uralic",
    "nort2671": "Uralic",
    # Niger-Congo
    "zulu1248": "Niger-Congo",   "swah1253": "Niger-Congo",
    "nucl1347": "Niger-Congo",   "bamb1269": "Niger-Congo",
    "wol1234":  "Niger-Congo",
    # Chukotko-Kamchatkan
    "chuk1273": "Chukotko-Kamchatkan",
    # Tungusic
    "even1259": "Tungusic",
    # Tai-Kadai
    "thai1261": "Tai-Kadai",
    # Austroasiatic
    "viet1252": "Austroasiatic",
    # Afro-Asiatic (Cushitic)
    "soma1255": "Afro-Asiatic",  "beja1238": "Afro-Asiatic",
    # Isolates / small families
    "mund1330": "Munduruku",     "para1311": "Tupian",
    "oloo1241": "Torricelli",    "warl1254": "Pama-Nyungan",
    "nucl1347": "Niger-Congo",
}


# ---------------------------------------------------------------------------
# Word-order value extraction
# ---------------------------------------------------------------------------

def get_predicted_word_order(
    glottocodes: List[str],
    ckpt: Dict,
    feature: str = "81A",
) -> Dict[str, Tuple[str, float]]:
    """
    Get predicted word-order value for each language from the categorical model.

    Returns
    -------
    {glottocode: (predicted_value, confidence)}
    """
    feat_names = ckpt["feat_names_sorted"]
    fv2id      = ckpt["featvalue2id"]
    feat2vals  = ckpt["feat2values"]
    lang2id    = ckpt["lang2id"]
    lang_embs  = ckpt["lang_embs"]
    fv_embs    = ckpt["fv_embs"]

    if feature not in feat_names:
        print(f"WARNING: feature {feature} not found in checkpoint")
        return {}

    fi    = feat_names.index(feature)
    vals  = feat2vals[feature]
    gids  = [fv2id[f"{feature}={v}"] for v in vals]

    result = {}
    for glot in glottocodes:
        if glot not in lang2id:
            continue
        idx   = lang2id[glot]
        le    = lang_embs[idx]
        probs = softmax(fv_embs[gids] @ le)
        pred  = vals[int(probs.argmax())]
        conf  = float(probs.max())
        result[glot] = (pred, conf)

    return result


# ---------------------------------------------------------------------------
# Split A: random language holdout
# ---------------------------------------------------------------------------

def make_split_A(
    glottocodes: List[str],
    test_frac: float = 0.2,
    seed: int = 42,
    word_order_map: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    """
    Random train/test split, stratified by word-order type if available.
    """
    rng = np.random.default_rng(seed)

    if word_order_map:
        # Stratified: sample test_frac from each word-order group
        groups = defaultdict(list)
        for g in glottocodes:
            wo = word_order_map.get(g, "Unknown")
            groups[wo].append(g)

        train, test = [], []
        for wo, langs in groups.items():
            shuffled = list(rng.permutation(langs))
            n_test = max(1, int(len(shuffled) * test_frac))
            test.extend(shuffled[:n_test])
            train.extend(shuffled[n_test:])
    else:
        shuffled = list(rng.permutation(glottocodes))
        n_test   = max(1, int(len(shuffled) * test_frac))
        test     = shuffled[:n_test]
        train    = shuffled[n_test:]

    return {"train": train, "test": test,
            "description": "Random language holdout (stratified by word order)"}


# ---------------------------------------------------------------------------
# Split B: family holdout
# ---------------------------------------------------------------------------

def make_split_B(
    glottocodes: List[str],
    family_map: Dict[str, str],
    held_out_families: Optional[List[str]] = None,
    min_test_size: int = 3,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Hold out entire genealogical families.

    Returns a dict of sub-splits, one per held-out family.
    If held_out_families is None, generate one split per family
    that has >= min_test_size languages.
    """
    # Group glottocodes by family
    family_to_langs: Dict[str, List[str]] = defaultdict(list)
    for g in glottocodes:
        fam = family_map.get(g, "Unknown")
        family_to_langs[fam].append(g)

    if held_out_families is None:
        held_out_families = [f for f, langs in family_to_langs.items()
                              if len(langs) >= min_test_size]

    splits = {}
    for fam in held_out_families:
        test  = family_to_langs.get(fam, [])
        train = [g for g in glottocodes
                 if family_map.get(g, "Unknown") != fam]
        if len(test) < min_test_size or len(train) < 5:
            continue
        splits[f"family_{fam.replace(' ', '_')}"] = {
            "train": train, "test": test,
            "held_out_family": fam,
            "description": f"Family holdout: {fam} ({len(test)} langs)",
        }

    return splits


# ---------------------------------------------------------------------------
# Split C: value holdout (THE CRITICAL SPLIT)
# ---------------------------------------------------------------------------

def make_split_C(
    glottocodes: List[str],
    word_order_map: Dict[str, str],     # glottocode -> 81A value
    held_out_values: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Hold out all languages with specific typological values.

    Default: hold out rare word-order values (VOS, OVS, No dominant order)
    and test whether categorical embeddings can generalise to them.

    This is the DECISIVE split: binary one-hot has no training signal
    for held-out values (their bit is always 0 in training). Categorical
    embeddings can generalise if the value geometry places held-out values
    in a meaningful position relative to training values.

    Returns a dict of sub-splits, one per held-out value (or combination).
    """
    if held_out_values is None:
        # Default: hold out rare values
        # VOS and OVS have very few representatives in our 59-language set
        # "No dominant order" is the most interesting for testing gradient prediction
        held_out_values = [
            ["No dominant order"],                     # Split C1
            ["VOS"],                                    # Split C2 (if any)
            ["OVS"],                                    # Split C3 (if any)
            ["No dominant order", "VOS", "OVS"],       # Split C4: all rare
        ]

    val_to_langs: Dict[str, List[str]] = defaultdict(list)
    for g in glottocodes:
        val = word_order_map.get(g, "Unknown")
        val_to_langs[val].append(g)

    print("Word-order distribution across available languages:")
    for val, langs in sorted(val_to_langs.items(),
                             key=lambda x: len(x[1]), reverse=True):
        print(f"  {val:25s}: {len(langs):2d} languages  "
              f"({', '.join(langs[:5])}{'...' if len(langs) > 5 else ''})")

    splits = {}
    for held_vals in held_out_values:
        test  = [g for g in glottocodes
                 if word_order_map.get(g) in held_vals]
        train = [g for g in glottocodes
                 if word_order_map.get(g) not in held_vals]
        if not test:
            continue
        key = "value_" + "_".join(
            v.replace(" ", "_").replace("dominant", "dom") for v in held_vals)
        splits[key] = {
            "train": train, "test": test,
            "held_out_values": held_vals,
            "n_train": len(train), "n_test": len(test),
            "description": (f"Value holdout: {held_vals} "
                            f"({len(test)} test, {len(train)} train)"),
        }
        print(f"\n  {key}:")
        print(f"    Held-out values: {held_vals}")
        print(f"    Train: {len(train)} langs, Test: {len(test)} langs")
        print(f"    Test languages: {test}")
        if not test:
            print("    WARNING: no test languages for this value set")

    return splits


# ---------------------------------------------------------------------------
# Main split builder
# ---------------------------------------------------------------------------

def build_all_splits(
    profiles_df: pd.DataFrame,
    ckpt_dir: str,
    out_dir: str,
    family_map: Optional[Dict[str, str]] = None,
    seed: int = 42,
) -> Dict:
    """
    Build all three split types and save to out_dir/splits.json.

    Returns the full splits dict.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Languages that have both profiles AND checkpoint coverage
    ckpt = load_checkpoint(ckpt_dir)
    ckpt_glots = set(ckpt["lang2id"].keys())
    glottocodes = [g for g in profiles_df.index if g in ckpt_glots]
    print(f"\nBuilding splits for {len(glottocodes)} languages "
          f"(have both UD profiles and checkpoint)")

    # Get word-order predictions from the model
    wo_map_full = get_predicted_word_order(glottocodes, ckpt, feature="81A")
    word_order_map = {g: v for g, (v, c) in wo_map_full.items()}
    print(f"Word-order predictions: {Counter(word_order_map.values())}")

    # Save word-order map
    with open(os.path.join(out_dir, "word_order_map.json"), "w") as f:
        json.dump(word_order_map, f, indent=2)

    all_splits = {}

    # Split A
    print("\n--- Split A: Random holdout ---")
    split_A = make_split_A(glottocodes, seed=seed,
                            word_order_map=word_order_map)
    all_splits["A_random"] = split_A
    print(f"  Train: {len(split_A['train'])},  Test: {len(split_A['test'])}")

    # Split B
    if family_map is None:
        family_map = FAMILY_MAP
    print("\n--- Split B: Family holdout ---")
    splits_B = make_split_B(glottocodes, family_map,
                             min_test_size=2)
    all_splits.update(splits_B)
    print(f"  {len(splits_B)} family splits generated")

    # Split C
    print("\n--- Split C: Value holdout (critical) ---")
    splits_C = make_split_C(glottocodes, word_order_map)
    all_splits.update(splits_C)
    print(f"  {len(splits_C)} value-holdout splits generated")

    # Save
    splits_path = os.path.join(out_dir, "splits.json")
    with open(splits_path, "w") as f:
        json.dump(all_splits, f, indent=2)
    print(f"\nAll splits saved: {splits_path}")

    # Summary
    print("\nSplit summary:")
    for name, split in all_splits.items():
        desc = split.get("description", "")
        n_tr = len(split.get("train", []))
        n_te = len(split.get("test", []))
        print(f"  {name:<40s}: train={n_tr:2d}  test={n_te:2d}  {desc}")

    # Save family map for use by direction_predictor
    with open(os.path.join(out_dir, "family_map.json"), "w") as f:
        json.dump({g: family_map.get(g, "Unknown") for g in glottocodes}, f, indent=2)

    return all_splits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate evaluation splits for typological direction task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", default="checkpoints/wals_learned_s42")
    p.add_argument("--profiles",   required=True,
                   help="direction_profiles.csv from arc_directions.py")
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--family_map", default=None,
                   help="JSON {glottocode: family} (optional; uses built-in map)")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    import os
    args = parse_args(argv)
    os.chdir(_REPO_ROOT)

    profiles = pd.read_csv(args.profiles, index_col=0)
    print(f"Profiles loaded: {profiles.shape}")

    family_map = None
    if args.family_map and os.path.exists(args.family_map):
        with open(args.family_map) as f:
            family_map = json.load(f)

    splits = build_all_splits(
        profiles_df=profiles,
        ckpt_dir=args.checkpoint,
        out_dir=args.out_dir,
        family_map=family_map,
        seed=args.seed,
    )
    print(f"\nDone. {len(splits)} splits.")


if __name__ == "__main__":
    main()
