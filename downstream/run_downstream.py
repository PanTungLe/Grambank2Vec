"""
downstream/run_downstream.py
=============================
End-to-end orchestrator for the downstream transfer experiment.

Runs the full pipeline in 4 phases:

  Phase 1 — Extract language representations from trained checkpoints
  Phase 2 — Download UD treebanks
  Phase 3 — Build the transfer matrix (biLSTM POS tagging)
  Phase 4 — Compute distances and evaluate ranking correlation

See each module for detailed documentation.

Usage
-----
Full run (lab server with GPU):

    python downstream/run_downstream.py \\
        --ckpt_wals   checkpoints/wals_learned_s42 \\
        --ckpt_grambank checkpoints/grambank_learned_s42 \\
        --out_dir     downstream_results \\
        --device      cuda \\
        --n_epochs    30 \\
        --max_train_sents 5000

Smoke test (3 languages, CPU):

    python downstream/run_downstream.py \\
        --out_dir downstream_smoke \\
        --smoke \\
        --device cpu

Skip phases you've already run:

    python downstream/run_downstream.py \\
        --out_dir downstream_results \\
        --skip_repr     # skip phase 1
        --skip_download # skip phase 2
        --skip_transfer # skip phase 3  (use existing transfer_matrix.csv)
        --device cuda

Run only evaluation (phase 4):

    python downstream/run_downstream.py \\
        --out_dir downstream_results \\
        --only_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="End-to-end downstream transfer evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Checkpoint paths
    p.add_argument("--ckpt_wals", default="checkpoints/wals_learned_s42",
                   help="Path to WALS learned checkpoint")
    p.add_argument("--ckpt_grambank", default="checkpoints/grambank_learned_s42",
                   help="Path to Grambank learned checkpoint")

    # Output
    p.add_argument("--out_dir", required=True,
                   help="Root output directory for all results")

    # Transfer model
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--n_epochs", type=int, default=20,
                   help="Max epochs per source tagger")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max_train_sents", type=int, default=5000,
                   help="Cap source training sentences (0=no cap)")
    p.add_argument("--max_test_sents", type=int, default=500,
                   help="Cap target test sentences (0=no cap)")
    p.add_argument("--save_models", action="store_true")

    # URIEL+
    p.add_argument("--uriel_parquet", default=None,
                   help="Path to URIEL+ parquet file for geo/phylo baselines. "
                        "If not provided, URIEL baselines are skipped.")

    # Pipeline control
    p.add_argument("--skip_repr",     action="store_true",
                   help="Skip phase 1: use existing representations")
    p.add_argument("--skip_download", action="store_true",
                   help="Skip phase 2: use existing UD data")
    p.add_argument("--skip_transfer", action="store_true",
                   help="Skip phase 3: use existing transfer matrix")
    p.add_argument("--only_eval",     action="store_true",
                   help="Only run phase 4 evaluation (phases 1-3 must be done)")
    p.add_argument("--overwrite",     action="store_true",
                   help="Overwrite existing results")
    p.add_argument("--smoke",         action="store_true",
                   help="Smoke test: 3 languages, 3 epochs, CPU")
    p.add_argument("--seeds", default="42",
                   help="Comma-separated seeds to run (e.g. 42,43,44,45,46)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Resolve paths relative to repo root
    os.chdir(_REPO_ROOT)

    # Override settings for smoke test
    if args.smoke:
        args.n_epochs = 3
        args.max_train_sents = 200
        args.max_test_sents  = 100
        args.device = "cpu"
        print("SMOKE TEST mode: 3 languages, 3 epochs, CPU")

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"\n{'='*70}")
    print(f"Grambank2Vec Downstream Transfer Evaluation")
    print(f"Seeds: {seeds}")
    print(f"Output: {args.out_dir}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    # Directories
    dirs = {
        "repr":     os.path.join(args.out_dir, "representations"),
        "ud":       os.path.join(args.out_dir, "ud_data"),
        "transfer": os.path.join(args.out_dir, "transfer"),
        "dist":     os.path.join(args.out_dir, "distances"),
        "results":  os.path.join(args.out_dir, "results"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1: Extract representations
    # -----------------------------------------------------------------------
    if not args.skip_repr and not args.only_eval:
        print("=" * 60)
        print("PHASE 1: Extracting typological representations")
        print("=" * 60)

        from downstream.representations import load_checkpoint, compute_repr_A, \
            compute_repr_B, compute_repr_C, compute_repr_B_masked

        import numpy as np

        for seed in seeds:
            for db, ckpt_path in [("wals", args.ckpt_wals),
                                   ("grambank", args.ckpt_grambank)]:
                ckpt_dir = ckpt_path.replace("s42", f"s{seed}")
                if not os.path.exists(ckpt_dir):
                    print(f"  WARNING: {ckpt_dir} not found, skipping")
                    continue

                repr_dir = os.path.join(dirs["repr"], f"{db}_s{seed}")
                os.makedirs(repr_dir, exist_ok=True)

                print(f"\n  [{db} seed={seed}] Loading checkpoint: {ckpt_dir}")
                ckpt = load_checkpoint(ckpt_dir)
                glottocodes = list(ckpt["lang2id"].keys())
                n_glot = len(glottocodes)
                print(f"  {n_glot} glottocodes in checkpoint")

                from downstream.ud_data import GLOTTOCODES_IN_UD
                if args.smoke:
                    target_glots = ["stan1293", "nucl1301", "tami1289"]
                else:
                    target_glots = [g for g in GLOTTOCODES_IN_UD
                                    if g in ckpt["lang2id"]]
                print(f"  {len(target_glots)} glottocodes overlap with UD")

                # Repr A
                A, found = compute_repr_A(ckpt, target_glots)
                np.save(os.path.join(repr_dir, "repr_A.npy"), A)
                with open(os.path.join(repr_dir, "repr_A_langs.json"), "w") as f:
                    json.dump(found, f)
                print(f"  Repr A: {A.shape}")

                # Repr B
                B, found = compute_repr_B(ckpt, target_glots)
                np.save(os.path.join(repr_dir, "repr_B.npy"), B)
                with open(os.path.join(repr_dir, "repr_B_langs.json"), "w") as f:
                    json.dump(found, f)
                print(f"  Repr B: {B.shape}")

                # Repr C
                C, found = compute_repr_C(ckpt, target_glots)
                np.save(os.path.join(repr_dir, "repr_C.npy"), C)
                with open(os.path.join(repr_dir, "repr_C_langs.json"), "w") as f:
                    json.dump(found, f)
                print(f"  Repr C: {C.shape}")

                # Repr B masked
                for mf in [0.3, 0.5, 0.7]:
                    Bm, Bf, found_m = compute_repr_B_masked(
                        ckpt, target_glots, mask_frac=mf, seed=42)
                    tag = f"B_masked{int(mf*100):02d}"
                    np.save(os.path.join(repr_dir, f"repr_{tag}.npy"), Bm)
                    with open(os.path.join(repr_dir, f"repr_{tag}_langs.json"), "w") as f:
                        json.dump(found_m, f)
                    print(f"  Repr {tag}: {Bm.shape}")

        print("\nPhase 1 complete.")

    # -----------------------------------------------------------------------
    # Phase 2: Download UD treebanks
    # -----------------------------------------------------------------------
    if not args.skip_download and not args.only_eval:
        print("\n" + "=" * 60)
        print("PHASE 2: Downloading UD treebanks")
        print("=" * 60)

        from downstream.ud_data import download_all, GLOTTOCODES_IN_UD

        if args.smoke:
            glottocodes = ["stan1293", "nucl1301", "tami1289"]
        else:
            glottocodes = GLOTTOCODES_IN_UD

        manifest = download_all(
            glottocodes=glottocodes,
            out_dir=dirs["ud"],
            splits=["train", "test"],
            overwrite=args.overwrite,
        )

        manifest_path = os.path.join(dirs["ud"], "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest saved: {manifest_path}")
        n_ok = sum(1 for v in manifest.values()
                   if "train" in v and "test" in v)
        print(f"Languages with train+test: {n_ok}")

        print("\nPhase 2 complete.")

    # -----------------------------------------------------------------------
    # Phase 3: Build transfer matrix
    # -----------------------------------------------------------------------
    if not args.skip_transfer and not args.only_eval:
        print("\n" + "=" * 60)
        print("PHASE 3: Building POS transfer matrix")
        print("=" * 60)

        from downstream.transfer_matrix import build_transfer_matrix

        manifest_path = os.path.join(dirs["ud"], "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        max_train = args.max_train_sents if args.max_train_sents > 0 else None
        max_test  = args.max_test_sents  if args.max_test_sents  > 0 else None

        matrix = build_transfer_matrix(
            manifest=manifest,
            out_dir=dirs["transfer"],
            device=args.device,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            max_train_sents=max_train,
            max_test_sents=max_test,
            save_models=args.save_models,
            overwrite=args.overwrite,
        )
        print("\nPhase 3 complete.")
        print(matrix.head())

    # -----------------------------------------------------------------------
    # Phase 4: Compute distances + evaluate
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 4: Computing distances and evaluating")
    print("=" * 60)

    from downstream.distances import compute_all_distances
    from downstream.evaluate import evaluate_all
    import pandas as pd

    # Load transfer matrix
    matrix_path = os.path.join(dirs["transfer"], "transfer_matrix.csv")
    if not os.path.exists(matrix_path):
        print(f"ERROR: Transfer matrix not found at {matrix_path}")
        print("Run phases 1-3 first, or pass --skip_transfer with an existing matrix.")
        sys.exit(1)

    transfer_matrix = pd.read_csv(matrix_path, index_col=0)
    glottocodes = list(transfer_matrix.index)
    print(f"Transfer matrix: {len(glottocodes)} languages")

    # Save glottocodes for downstream use
    glot_path = os.path.join(dirs["dist"], "glottocodes.json")
    with open(glot_path, "w") as f:
        json.dump(glottocodes, f)

    # Gather all representation directories
    repr_dirs = []
    for seed in seeds:
        for db in ["wals", "grambank"]:
            rdir = os.path.join(dirs["repr"], f"{db}_s{seed}")
            if os.path.exists(rdir):
                repr_dirs.append(rdir)

    if repr_dirs:
        # Use seed 42 representations for main analysis
        main_repr_dirs = [d for d in repr_dirs if "s42" in d]
        if not main_repr_dirs:
            main_repr_dirs = repr_dirs[:2]

        distances = compute_all_distances(
            glottocodes=glottocodes,
            repr_dirs=main_repr_dirs,
            out_dir=dirs["dist"],
            uriel_parquet=args.uriel_parquet,
        )
    else:
        print("WARNING: No representation directories found. "
              "Skipping distance computation. "
              "Run phase 1 first, or check --out_dir.")

    # Load confound control files if present
    sizes_path  = os.path.join(dirs["ud"], "treebank_sizes.json")
    script_path = os.path.join(dirs["ud"], "script_map.json")
    treebank_sizes = json.load(open(sizes_path))  if os.path.exists(sizes_path)  else None
    script_map     = json.load(open(script_path)) if os.path.exists(script_path) else None
    if treebank_sizes:
        print(f"Treebank sizes loaded ({len(treebank_sizes)} entries) — enabling size-residualised Spearman")
    if script_map:
        print(f"Script map loaded ({len(script_map)} entries) — enabling script-controlled Spearman")

    # Evaluate
    if os.path.exists(dirs["dist"]):
        summary = evaluate_all(
            transfer_matrix=transfer_matrix,
            distances_dir=dirs["dist"],
            out_dir=dirs["results"],
            treebank_sizes=treebank_sizes,
            script_map=script_map,
        )
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"Results: {dirs['results']}")
    else:
        print("WARNING: Distances directory not found, skipping evaluation.")

    # -----------------------------------------------------------------------
    # Seed stability (if multiple seeds requested)
    # -----------------------------------------------------------------------
    if len(seeds) > 1 and not args.only_eval:
        print("\n" + "=" * 60)
        print("SEED STABILITY ANALYSIS")
        print("=" * 60)
        print("(Run evaluate.py separately on each seed's repr_dir to compare)")


if __name__ == "__main__":
    main()
