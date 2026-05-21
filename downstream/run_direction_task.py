"""
downstream/run_direction_task.py
==================================
End-to-end orchestrator for the arc-direction prediction task.

Phases
------
1. Extract UD arc-direction profiles from treebanks
2. Generate evaluation splits (random, family, value-holdout)
3. Extract language representations from trained checkpoints
4. Train and evaluate direction predictors for all representations × splits
5. Compile results table, run statistical tests, generate comparison plots

The critical comparison is Split C (value holdout):
  Does Repr B (categorical expected embedding) generalise to held-out
  word-order values better than Repr C (binary predicted profile)?

Usage
-----
Full run (lab server):

    python downstream/run_direction_task.py \\
        --manifest downstream_results/ud_data/manifest.json \\
        --ckpt_wals   checkpoints/wals_learned_s42 \\
        --ckpt_grambank checkpoints/grambank_learned_s42 \\
        --out_dir downstream_direction \\
        --device cuda

Smoke test (3 languages, CPU, ~2 min):

    python downstream/run_direction_task.py \\
        --manifest /tmp/ud_smoke/manifest.json \\
        --out_dir /tmp/direction_smoke \\
        --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="End-to-end arc-direction prediction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest",      required=True)
    p.add_argument("--ckpt_wals",     default="checkpoints/wals_learned_s42")
    p.add_argument("--ckpt_grambank", default="checkpoints/grambank_learned_s42")
    p.add_argument("--out_dir",       required=True)
    p.add_argument("--device",        default="cpu",
                   choices=["cpu","cuda","mps"])
    p.add_argument("--n_epochs",      type=int, default=100)
    p.add_argument("--patience",      type=int, default=15)
    p.add_argument("--batch_size",    type=int, default=512)
    p.add_argument("--seeds",         default="42",
                   help="Comma-separated seeds (for repr loading and split stability)")
    p.add_argument("--reprs",         default="A,B,C,B_masked30,B_masked50",
                   help="Comma-separated representation names to evaluate")
    p.add_argument("--smoke",         action="store_true",
                   help="3 languages, 5 epochs, CPU — functional test only")
    p.add_argument("--skip_extract",  action="store_true",
                   help="Skip arc direction extraction (use existing profiles)")
    p.add_argument("--skip_splits",   action="store_true",
                   help="Skip split generation")
    p.add_argument("--skip_train",    action="store_true",
                   help="Skip training (only compile existing results)")
    p.add_argument("--overwrite",     action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(_REPO_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.smoke:
        args.n_epochs  = 5
        args.patience  = 3
        args.reprs     = "A,B"
        args.device    = "cpu"
        print("SMOKE TEST: 5 epochs, reprs=A,B, CPU")

    seeds  = [int(s) for s in args.seeds.split(",")]
    reprs  = [r.strip() for r in args.reprs.split(",")]

    dirs = {
        "arc":   os.path.join(args.out_dir, "arc_directions"),
        "splits":os.path.join(args.out_dir, "splits"),
        "repr":  os.path.join(args.out_dir, "representations"),
        "models":os.path.join(args.out_dir, "models"),
        "results":os.path.join(args.out_dir, "results"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1: Extract arc-direction profiles
    # -----------------------------------------------------------------------
    if not args.skip_extract:
        print("\n" + "="*60)
        print("PHASE 1: Extracting arc-direction profiles")
        print("="*60)

        from downstream.arc_directions import extract_all_profiles
        with open(args.manifest) as f:
            manifest = json.load(f)

        profiles_df = extract_all_profiles(
            manifest=manifest,
            out_dir=dirs["arc"],
            split="train",
            overwrite=args.overwrite,
        )
        print(f"Profiles: {profiles_df.shape}")
    else:
        profiles_path = os.path.join(dirs["arc"], "direction_profiles.csv")
        profiles_df = pd.read_csv(profiles_path, index_col=0)
        print(f"Loaded existing profiles: {profiles_df.shape}")

    counts_path  = os.path.join(dirs["arc"], "direction_counts.csv")
    counts_df    = pd.read_csv(counts_path, index_col=0) \
                   if os.path.exists(counts_path) \
                   else pd.DataFrame(index=profiles_df.index,
                                     columns=profiles_df.columns).fillna(1.0)

    # -----------------------------------------------------------------------
    # Phase 2: Generate splits
    # -----------------------------------------------------------------------
    if not args.skip_splits:
        print("\n" + "="*60)
        print("PHASE 2: Generating evaluation splits")
        print("="*60)

        from downstream.value_splits import build_all_splits
        splits = build_all_splits(
            profiles_df=profiles_df,
            ckpt_dir=args.ckpt_wals,
            out_dir=dirs["splits"],
        )
    else:
        with open(os.path.join(dirs["splits"], "splits.json")) as f:
            splits = json.load(f)
        print(f"Loaded existing splits: {list(splits.keys())}")

    # -----------------------------------------------------------------------
    # Phase 3: Extract representations
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 3: Extracting language representations")
    print("="*60)

    from downstream.representations import (
        load_checkpoint, compute_repr_A, compute_repr_B,
        compute_repr_C, compute_repr_B_masked,
    )

    glottocodes = profiles_df.index.tolist()
    repr_store: Dict[str, Dict] = {}   # {repr_key: {matrix, langs}}

    for seed in seeds:
        for db, ckpt_path in [("wals", args.ckpt_wals),
                               ("grambank", args.ckpt_grambank)]:
            ckpt_dir = ckpt_path.replace("s42", f"s{seed}")
            if not os.path.exists(ckpt_dir):
                print(f"  SKIP: {ckpt_dir} not found")
                continue

            repr_dir = os.path.join(dirs["repr"], f"{db}_s{seed}")
            os.makedirs(repr_dir, exist_ok=True)
            ckpt = load_checkpoint(ckpt_dir)
            print(f"\n  [{db} s{seed}]")

            target_glots = [g for g in glottocodes if g in ckpt["lang2id"]]

            for repr_name in reprs:
                key = f"{db}_s{seed}_{repr_name}"
                npy_path = os.path.join(repr_dir, f"repr_{repr_name}.npy")

                if os.path.exists(npy_path) and not args.overwrite:
                    mat = np.load(npy_path)
                    with open(os.path.join(repr_dir,
                              f"repr_{repr_name}_langs.json")) as f:
                        langs = json.load(f)
                    repr_store[key] = {"matrix": mat, "langs": langs}
                    print(f"    {repr_name}: loaded {mat.shape}")
                    continue

                if repr_name == "A":
                    mat, langs = compute_repr_A(ckpt, target_glots)
                elif repr_name == "B":
                    mat, langs = compute_repr_B(ckpt, target_glots)
                elif repr_name == "C":
                    mat, langs = compute_repr_C(ckpt, target_glots)
                elif repr_name.startswith("B_masked"):
                    frac = int(repr_name[8:]) / 100
                    mat, _, langs = compute_repr_B_masked(ckpt, target_glots,
                                                          mask_frac=frac)
                else:
                    print(f"    Unknown repr: {repr_name}, skipping")
                    continue

                np.save(npy_path, mat)
                with open(os.path.join(repr_dir,
                          f"repr_{repr_name}_langs.json"), "w") as f:
                    json.dump(langs, f)
                repr_store[key] = {"matrix": mat, "langs": langs}
                print(f"    {repr_name}: {mat.shape}")

    # -----------------------------------------------------------------------
    # Phase 4: Train and evaluate
    # -----------------------------------------------------------------------
    if not args.skip_train:
        print("\n" + "="*60)
        print("PHASE 4: Training direction predictors")
        print("="*60)

        from downstream.direction_predictor import (
            MLPPredictor, MajorityBaseline, ZeroOrderBaseline,
            FamilyMeanBaseline, evaluate_predictions,
        )

        family_map_path = os.path.join(dirs["splits"], "family_map.json")
        family_map = {}
        if os.path.exists(family_map_path):
            with open(family_map_path) as f:
                family_map = json.load(f)

        all_results: Dict[str, Dict] = {}

        for repr_key, repr_data in repr_store.items():
            mat   = repr_data["matrix"]
            langs = repr_data["langs"]
            repr_name = repr_key.split("_")[-1]
            lang_dim  = mat.shape[1]
            glot2row  = {g: i for i, g in enumerate(langs)}

            print(f"\n  Evaluating: {repr_key}")
            all_results[repr_key] = {}

            for split_name, split_data in splits.items():
                train_g = [g for g in split_data["train"]
                           if g in glot2row and g in profiles_df.index]
                test_g  = [g for g in split_data["test"]
                           if g in glot2row and g in profiles_df.index]

                if len(train_g) < 3 or len(test_g) < 1:
                    continue

                print(f"    {split_name}: train={len(train_g)} test={len(test_g)}")

                # Build aligned repr matrices
                def get_repr(glots):
                    return np.array([mat[glot2row[g]] for g in glots
                                     if g in glot2row])
                train_mat = get_repr(train_g)
                test_mat  = get_repr(test_g)
                all_mat   = np.vstack([train_mat, test_mat])

                split_res = {}

                # Baselines (no torch needed)
                for baseline in [
                    ZeroOrderBaseline(),
                    MajorityBaseline(),
                    FamilyMeanBaseline(family_map) if family_map else None,
                ]:
                    if baseline is None:
                        continue
                    train_prof = profiles_df.loc[[g for g in train_g
                                                  if g in profiles_df.index]]
                    baseline.fit(train_prof, glottocodes=train_g)
                    preds = baseline.predict(test_g, profiles_df)
                    m = evaluate_predictions(preds, profiles_df, counts_df)
                    split_res[baseline.name()] = {
                        "mae":          m.get("mae", np.nan),
                        "brier":        m.get("brier", np.nan),
                        "spearman_rho": m.get("spearman_rho", np.nan),
                        "kl":           m.get("kl", np.nan),
                    }

                # MLP
                try:
                    predictor = MLPPredictor(
                        lang_dim=lang_dim,
                        n_epochs=args.n_epochs,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        device=args.device,
                        repr_name=repr_key,
                    )
                    n_dev = max(1, len(train_g) // 5)
                    predictor.fit(
                        train_glots=train_g[n_dev:],
                        lang_reprs=train_mat[n_dev:],
                        profiles=profiles_df,
                        counts=counts_df,
                        dev_glots=train_g[:n_dev],
                        verbose=False,
                    )
                    preds = predictor.predict(test_g, test_mat, profiles_df)
                    m = evaluate_predictions(preds, profiles_df, counts_df)
                    split_res["mlp"] = {
                        "mae":          m.get("mae", np.nan),
                        "brier":        m.get("brier", np.nan),
                        "spearman_rho": m.get("spearman_rho", np.nan),
                        "kl":           m.get("kl", np.nan),
                        "per_config":   m.get("per_config", {}),
                    }
                    print(f"      mlp: MAE={m.get('mae',0):.4f}  "
                          f"Spearman={m.get('spearman_rho',0):.4f}  "
                          f"KL={m.get('kl',0):.4f}")
                except Exception as e:
                    print(f"      mlp FAILED: {e}")

                all_results[repr_key][split_name] = split_res

        # Save all results
        res_path = os.path.join(dirs["results"], "all_results.json")
        with open(res_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved: {res_path}")

    # -----------------------------------------------------------------------
    # Phase 5: Compile summary table
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("PHASE 5: Compiling results")
    print("="*60)

    res_path = os.path.join(dirs["results"], "all_results.json")
    if not os.path.exists(res_path):
        print("No results file found. Run phases 1-4 first.")
        return

    with open(res_path) as f:
        all_results = json.load(f)

    rows = []
    for repr_key, split_results in all_results.items():
        for split_name, model_results in split_results.items():
            split_type = split_name.split("_")[0]
            for model_name, metrics in model_results.items():
                if model_name == "per_config":
                    continue
                if not isinstance(metrics, dict):
                    continue
                rows.append({
                    "representation": repr_key,
                    "split": split_name,
                    "split_type": split_type,
                    "model": model_name,
                    "mae":          metrics.get("mae", np.nan),
                    "brier":        metrics.get("brier", np.nan),
                    "spearman_rho": metrics.get("spearman_rho", np.nan),
                    "kl":           metrics.get("kl", np.nan),
                })

    if not rows:
        print("No rows to compile.")
        return

    df = pd.DataFrame(rows)
    summary_path = os.path.join(dirs["results"], "summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    # Print the key result: Split C (value holdout), MLP model
    print("\n--- KEY RESULT: Value-holdout splits (Split C), MLP model ---")
    key_df = df[(df["split_type"] == "value") & (df["model"] == "mlp")]
    if not key_df.empty:
        print(key_df[["representation", "split", "mae", "spearman_rho", "kl"]]
              .sort_values("spearman_rho", ascending=False)
              .to_string(index=False, float_format="{:.4f}".format))
    else:
        print("No value-holdout MLP results found yet.")

    print("\n--- Comparison: Repr B vs Repr C on value-holdout splits ---")
    for metric in ["mae", "spearman_rho", "kl"]:
        for split in df["split"].unique():
            if "value" not in split:
                continue
            sub = df[(df["split"] == split) & (df["model"] == "mlp")]
            for db in ["wals", "grambank"]:
                b_row = sub[sub["representation"].str.contains(
                    f"{db}_s42_B$", regex=True)]
                c_row = sub[sub["representation"].str.contains(
                    f"{db}_s42_C", regex=True)]
                if b_row.empty or c_row.empty:
                    continue
                b_val = b_row[metric].values[0]
                c_val = c_row[metric].values[0]
                winner = "B" if (b_val < c_val if metric in ["mae","kl","brier"]
                                 else b_val > c_val) else "C"
                print(f"  {db} {split} {metric}: B={b_val:.4f}  C={c_val:.4f}  "
                      f"Winner={winner}")

    print(f"\n{'='*60}")
    print("DIRECTION TASK COMPLETE")
    print(f"Results directory: {dirs['results']}")


if __name__ == "__main__":
    main()
