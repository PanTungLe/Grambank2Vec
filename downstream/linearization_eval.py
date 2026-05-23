"""
downstream/linearization_eval.py
==================================
Full evaluation pipeline for Typology-Conditioned Dependency Linearization.

For each target language t:
  1. Load t's test treebank (gold linear order)
  2. Build unordered trees (strip positions, keep head/deprel/upos)
  3. Apply each linearizer to predict target-like word order
  4. Evaluate against gold order

Three evaluation conditions:
  SELF: use t's own typological embedding → should reconstruct gold order
  CROSS: use a different language's embedding → measures transferability
  HELD_OUT: t's value was held out from training → cold-value test

Linearizers compared:
  - Random baseline (shuffle each sentence)
  - Source-language majority (use source language's direction profile)
  - Repr A MLP (trained on direct language embeddings)
  - Repr B MLP (trained on expected value embeddings — thesis contribution)
  - Repr C MLP (trained on predicted binary profiles)
  - Majority direction from profiles (no MLP, just per-config thresholds)
  - Gold-profile oracle (upper bound: use true direction profile as predictor)

Usage
-----
    python downstream/linearization_eval.py \\
        --manifest  downstream_results/ud_data/manifest.json \\
        --profiles  downstream_direction/arc_directions/direction_profiles.csv \\
        --splits_file downstream_direction/splits/splits.json \\
        --repr_base downstream_direction/representations \\
        --model_dir downstream_direction/models \\
        --out_dir   downstream_linearization \\
        --device    cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from downstream.arc_directions import parse_conllu_arcs, TARGET_CONFIGS
from downstream.linearizer import (
    Linearizer, MajorityDirectionLinearizer,
    evaluate_linearization, evaluate_per_config,
)


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

class RandomLinearizer:
    """Randomly permutes tokens — sanity floor."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def linearize_sentence(self, tokens):
        ids = [t["id"] for t in tokens]
        return list(self.rng.permutation(ids))

    def linearize_corpus(self, sentences):
        return [self.linearize_sentence(s) for s in sentences]


# ---------------------------------------------------------------------------
# Gold-profile oracle (upper bound)
# ---------------------------------------------------------------------------

class GoldProfileLinearizer(MajorityDirectionLinearizer):
    """
    Uses the language's own UD-extracted direction profile as oracle.
    Upper bound: best possible performance without per-sentence signal.
    """
    def __init__(self, direction_profile: Dict[str, float]):
        super().__init__(direction_profile)
        self._name = "gold_profile_oracle"


# ---------------------------------------------------------------------------
# MLP-based linearizer wrapper
# ---------------------------------------------------------------------------

def build_mlp_linearizer(
    predictor,
    repr_mat: np.ndarray,
    repr_langs: List[str],
    glottocode: str,
    device: str = "cpu",
) -> Optional[Linearizer]:
    """Build a Linearizer for a specific language using a trained MLPPredictor."""
    glot2row = {g: i for i, g in enumerate(repr_langs)}
    if glottocode not in glot2row:
        return None
    lang_repr = repr_mat[glot2row[glottocode]]
    return Linearizer(predictor, lang_repr, device=device)


# ---------------------------------------------------------------------------
# Cross-lingual linearization experiment
# ---------------------------------------------------------------------------

def run_cross_lingual_experiment(
    source_glot: str,
    target_glot: str,
    target_sentences: List[List[Dict]],
    linearizers: Dict[str, object],
    label: str = "",
) -> Dict[str, Dict]:
    """
    Apply all linearizers (built for source_glot) to target sentences,
    evaluate against target gold order.
    """
    results = {}
    for lin_name, lin in linearizers.items():
        orders = lin.linearize_corpus(target_sentences)
        metrics = evaluate_linearization(orders, target_sentences)
        results[lin_name] = metrics
    return results


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_full_evaluation(
    manifest: Dict,
    profiles_df: pd.DataFrame,
    splits: Dict,
    repr_store: Dict,         # {repr_key: {"matrix": np.ndarray, "langs": list}}
    predictors: Dict,         # {repr_key: MLPPredictor} — already trained
    out_dir: str,
    device: str = "cpu",
    max_sentences: int = 500,
) -> pd.DataFrame:
    """
    Full linearization evaluation across all conditions and representations.

    Returns summary DataFrame.
    """
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []

    # Load test treebanks once
    print("Loading treebanks...")
    treebanks: Dict[str, List[List[Dict]]] = {}
    for glot, paths in manifest.items():
        split = "test" if "test" in paths else ("train" if "train" in paths else None)
        if split is None:
            continue
        with open(paths[split], encoding="utf-8") as f:
            text = f.read()
        sents = parse_conllu_arcs(text)
        if max_sentences > 0:
            sents = sents[:max_sentences]
        treebanks[glot] = sents
        print(f"  {glot}: {len(sents)} sentences")

    # ── Condition 1: SELF-linearization ────────────────────────────────────
    # Each language evaluated using its own typological embedding.
    # Tests whether any representation correctly reconstructs self-order.
    print("\n--- Condition 1: Self-linearization ---")

    for repr_key, repr_data in repr_store.items():
        mat, langs = repr_data["matrix"], repr_data["langs"]
        glot2row = {g: i for i, g in enumerate(langs)}
        predictor = predictors.get(repr_key)

        for glot in treebanks:
            if glot not in glot2row:
                continue
            if glot not in profiles_df.index:
                continue

            lang_repr = mat[glot2row[glot]]
            sents     = treebanks[glot]

            # MLP linearizer
            if predictor is not None and predictor.model is not None:
                lin_mlp = Linearizer(predictor, lang_repr, device=device)
                orders  = lin_mlp.linearize_corpus(sents)
                m = evaluate_linearization(orders, sents)
                all_rows.append({
                    "condition":      "self",
                    "repr":           repr_key,
                    "linearizer":     "mlp",
                    "source_glot":    glot,
                    "target_glot":    glot,
                    **m,
                })

            # Gold-profile oracle
            profile = profiles_df.loc[glot].to_dict()
            oracle  = GoldProfileLinearizer(profile)
            orders  = oracle.linearize_corpus(sents)
            m = evaluate_linearization(orders, sents)
            all_rows.append({
                "condition":   "self",
                "repr":        "gold_oracle",
                "linearizer":  "profile",
                "source_glot": glot,
                "target_glot": glot,
                **m,
            })

        # Random baseline (once per target)
        if repr_key == list(repr_store.keys())[0]:
            for glot, sents in treebanks.items():
                rand = RandomLinearizer()
                orders = rand.linearize_corpus(sents)
                m = evaluate_linearization(orders, sents)
                all_rows.append({
                    "condition":   "self",
                    "repr":        "random",
                    "linearizer":  "random",
                    "source_glot": glot,
                    "target_glot": glot,
                    **m,
                })

    # ── Condition 2: VALUE-HOLDOUT cross-lingual linearization ─────────────
    # Train on seen word-order values; apply to held-out word-order languages.
    # This is the critical test: can categorical embeddings predict order
    # for word-order types never seen during training?
    print("\n--- Condition 2: Value-holdout cross-lingual ---")

    for split_name, split_data in splits.items():
        if "value" not in split_name:
            continue
        test_glots  = split_data.get("test", [])
        train_glots = split_data.get("train", [])

        for tgt_glot in test_glots:
            if tgt_glot not in treebanks:
                continue
            tgt_sents = treebanks[tgt_glot]

            for repr_key, repr_data in repr_store.items():
                mat, langs = repr_data["matrix"], repr_data["langs"]
                glot2row = {g: i for i, g in enumerate(langs)}
                predictor = predictors.get(repr_key)

                if tgt_glot not in glot2row:
                    continue
                if predictor is None or predictor.model is None:
                    continue

                lang_repr = mat[glot2row[tgt_glot]]
                lin_mlp   = Linearizer(predictor, lang_repr, device=device)
                orders    = lin_mlp.linearize_corpus(tgt_sents)
                m         = evaluate_linearization(orders, tgt_sents)

                all_rows.append({
                    "condition":    split_name,
                    "repr":         repr_key,
                    "linearizer":   "mlp",
                    "source_glot":  "trained_on_all_train",
                    "target_glot":  tgt_glot,
                    **m,
                })

                # Also evaluate per-config for value-holdout
                per_cfg = evaluate_per_config(orders, tgt_sents)
                per_cfg_path = os.path.join(
                    out_dir, f"per_config_{repr_key}_{split_name}_{tgt_glot}.json")
                with open(per_cfg_path, "w") as f:
                    json.dump(per_cfg, f, indent=2)

    # ── Save results ──────────────────────────────────────────────────────
    if not all_rows:
        print("WARNING: no results generated.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    summary_path = os.path.join(out_dir, "linearization_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nResults saved: {summary_path}  ({len(df)} rows)")

    # ── Print key summaries ──────────────────────────────────────────────
    print("\n--- SELF-linearization: mean Kendall tau per representation ---")
    self_df = df[df["condition"] == "self"]
    if not self_df.empty:
        summary = (self_df.groupby(["repr", "linearizer"])["kendall_tau"]
                          .agg(["mean", "std", "count"])
                          .sort_values("mean", ascending=False))
        print(summary.to_string(float_format="{:.4f}".format))

    print("\n--- VALUE-HOLDOUT: Kendall tau for held-out word-order languages ---")
    val_df = df[df["condition"].str.contains("value", na=False)]
    if not val_df.empty:
        summary_v = (val_df.groupby(["repr", "target_glot", "condition"])
                           ["kendall_tau"].mean()
                           .sort_values(ascending=False))
        print(summary_v.to_string(float_format="{:.4f}".format))

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Full linearization evaluation (TCDL)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest",    required=True)
    p.add_argument("--profiles",    required=True,
                   help="direction_profiles.csv")
    p.add_argument("--splits_file", required=True)
    p.add_argument("--repr_base",   required=True,
                   help="Base directory containing {db}_s{seed}/ subdirs")
    p.add_argument("--model_dir",   default=None,
                   help="Directory with saved MLP predictors "
                        "(run direction task first)")
    p.add_argument("--out_dir",     required=True)
    p.add_argument("--device",      default="cpu", choices=["cpu","cuda","mps"])
    p.add_argument("--reprs",       default="A,B,C",
                   help="Comma-separated repr names to load")
    p.add_argument("--seeds",       default="42")
    p.add_argument("--max_sentences", type=int, default=500)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)
    profiles_df = pd.read_csv(args.profiles, index_col=0)
    with open(args.splits_file) as f:
        splits = json.load(f)

    reprs = [r.strip() for r in args.reprs.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    # Load representations
    repr_store: Dict = {}
    for seed in seeds:
        for db in ["wals", "grambank"]:
            for repr_name in reprs:
                key = f"{db}_s{seed}_{repr_name}"
                npy = os.path.join(args.repr_base, f"{db}_s{seed}",
                                   f"repr_{repr_name}.npy")
                lpath = os.path.join(args.repr_base, f"{db}_s{seed}",
                                     f"repr_{repr_name}_langs.json")
                if not os.path.exists(npy):
                    print(f"  SKIP: {npy} not found")
                    continue
                mat = np.load(npy)
                with open(lpath) as f:
                    langs = json.load(f)
                repr_store[key] = {"matrix": mat, "langs": langs}
                print(f"  Loaded {key}: {mat.shape}")

    # Load predictors (if model_dir provided)
    predictors: Dict = {}
    if args.model_dir and os.path.exists(args.model_dir):
        from downstream.direction_predictor import MLPPredictor
        for key in repr_store:
            # Models would be saved during run_direction_task.py
            # For now, create untrained placeholders
            lang_dim = repr_store[key]["matrix"].shape[1]
            pred = MLPPredictor(lang_dim=lang_dim, device=args.device)
            predictors[key] = pred
            print(f"  Predictor for {key}: lang_dim={lang_dim} (untrained placeholder)")

    if not predictors:
        print("WARNING: No trained predictors found. "
              "Run run_direction_task.py first, or the evaluation will use "
              "only baselines and oracle.")

    df = run_full_evaluation(
        manifest=manifest,
        profiles_df=profiles_df,
        splits=splits,
        repr_store=repr_store,
        predictors=predictors,
        out_dir=args.out_dir,
        device=args.device,
        max_sentences=args.max_sentences,
    )
    print(f"\nDone. {len(df)} result rows.")


if __name__ == "__main__":
    main()
