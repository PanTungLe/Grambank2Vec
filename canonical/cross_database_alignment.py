"""
Cross-database feature alignment (RQ3).

Modes
-----
  sanity_check   Phase 1 — test whether the architectural shared-space
                 assumption holds; decide whether Approach A is sound.
  approach_a     Phase 3 — Procrustes-rotated featvalue alignment.
  approach_b     Phase 3 — language-profile correspondence.
  approach_c     Phase 3 — CCA-projected featvalue alignment.
  validate       Phase 4 — validate against gold-standard pairs.
  discover       Phase 5 — novel correspondence discovery.

Usage
-----
  python canonical/cross_database_alignment.py \\
      --mode sanity_check \\
      [--smoke | --full] \\
      [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_geometry import cosine_normalise, load_checkpoint
from utils import seed_everything


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pair(
    ckpt_wals: str,
    ckpt_grambank: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load WALS and Grambank learned checkpoints, validating architecture."""
    for path, label in [(ckpt_wals, "WALS"), (ckpt_grambank, "Grambank")]:
        if not Path(path).exists():
            print(f"[ERROR] {label} checkpoint not found: {path}", file=sys.stderr)
            print("        Train first with canonical/train_canonical.py.",
                  file=sys.stderr)
            sys.exit(1)

    ckpt_w = load_checkpoint(ckpt_wals)
    ckpt_g = load_checkpoint(ckpt_grambank)

    for ckpt, label in [(ckpt_w, "WALS"), (ckpt_g, "Grambank")]:
        if ckpt["architecture"] != "learned":
            raise ValueError(
                f"{label} checkpoint uses architecture={ckpt['architecture']!r}; "
                "cross_database_alignment.py requires the 'learned' architecture."
            )

    return ckpt_w, ckpt_g


def _sample_cosines(
    emb: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_pairs random pairwise cosine similarities from emb (row-normalised)."""
    normed = cosine_normalise(emb)
    n = len(normed)
    i1 = rng.integers(0, n, n_pairs)
    i2 = rng.integers(0, n, n_pairs)
    # Avoid self-pairs when n is large (not critical, but cleaner)
    same = i1 == i2
    i2[same] = (i2[same] + 1) % n
    return (normed[i1] * normed[i2]).sum(axis=1)


# ---------------------------------------------------------------------------
# Phase 1 — Sanity check
# ---------------------------------------------------------------------------

def run_sanity_check(args: argparse.Namespace) -> dict[str, Any]:
    """
    Test the architectural shared-space assumption.

    For Approach A (Procrustes-rotated featvalues) to be geometrically sound,
    language embeddings and feature-value embeddings must live in the same
    metric space within each database: comparable L2 norms and similar cosine
    distributions.

    Decision rule (applied per-database, both must pass):
      SOUND  iff  norm_ratio ∈ [0.5, 2.0]
               AND  Wasserstein-1(lang_cosines, fv_cosines) < 0.2
      else: Approach A is DEMOTED to a robustness check; Approach B becomes
            the headline method for RQ3.
    """
    from scipy.stats import wasserstein_distance

    n_pairs = 1000 if args.smoke else 10_000
    rng = np.random.default_rng(args.seed)

    print(f"Loading checkpoints …")
    ckpt_w, ckpt_g = _load_pair(args.checkpoint_wals, args.checkpoint_grambank)
    print(f"  WALS:     lang_emb={ckpt_w['lang_emb'].shape}  "
          f"fv_emb={ckpt_w['fv_emb'].shape}")
    print(f"  Grambank: lang_emb={ckpt_g['lang_emb'].shape}  "
          f"fv_emb={ckpt_g['fv_emb'].shape}")

    per_db: dict[str, dict[str, Any]] = {}

    for db_name, ckpt in [("wals", ckpt_w), ("grambank", ckpt_g)]:
        lang_emb: np.ndarray = ckpt["lang_emb"]   # (n_langs, d)
        fv_emb: np.ndarray = ckpt["fv_emb"]       # (n_fv, d)

        lang_norms = np.linalg.norm(lang_emb, axis=1)
        fv_norms = np.linalg.norm(fv_emb, axis=1)
        norm_ratio = float(fv_norms.mean() / lang_norms.mean())

        lang_cosines = _sample_cosines(lang_emb, n_pairs, rng)
        fv_cosines = _sample_cosines(fv_emb, n_pairs, rng)
        w1 = float(wasserstein_distance(lang_cosines, fv_cosines))

        print(f"\n  [{db_name.upper()}]")
        print(f"    lang  n={len(lang_emb)}  "
              f"norm_mean={lang_norms.mean():.4f} ± {lang_norms.std():.4f}")
        print(f"    fv    n={len(fv_emb)}   "
              f"norm_mean={fv_norms.mean():.4f} ± {fv_norms.std():.4f}")
        print(f"    norm_ratio (fv/lang) = {norm_ratio:.4f}  "
              f"{'✓ in [0.5,2.0]' if 0.5 <= norm_ratio <= 2.0 else '✗ outside [0.5,2.0]'}")
        print(f"    lang cosine mean = {lang_cosines.mean():.4f}, "
              f"std = {lang_cosines.std():.4f}")
        print(f"    fv   cosine mean = {fv_cosines.mean():.4f}, "
              f"std = {fv_cosines.std():.4f}")
        print(f"    Wasserstein-1 (lang vs fv cosines) = {w1:.4f}  "
              f"{'✓ < 0.2' if w1 < 0.2 else '✗ ≥ 0.2'}")

        per_db[db_name] = {
            "n_languages": int(len(lang_emb)),
            "n_featvalues": int(len(fv_emb)),
            "embed_dim": int(lang_emb.shape[1]),
            "lang_norm_mean": float(lang_norms.mean()),
            "lang_norm_std": float(lang_norms.std()),
            "fv_norm_mean": float(fv_norms.mean()),
            "fv_norm_std": float(fv_norms.std()),
            "norm_ratio": norm_ratio,
            "lang_cosine_mean": float(lang_cosines.mean()),
            "lang_cosine_std": float(lang_cosines.std()),
            "fv_cosine_mean": float(fv_cosines.mean()),
            "fv_cosine_std": float(fv_cosines.std()),
            "wasserstein_lang_vs_fv": w1,
            "norm_ratio_ok": bool(0.5 <= norm_ratio <= 2.0),
            "wasserstein_ok": bool(w1 < 0.2),
        }

    # Decision
    all_norm_ok = all(per_db[db]["norm_ratio_ok"] for db in per_db)
    all_w1_ok = all(per_db[db]["wasserstein_ok"] for db in per_db)
    approach_a_sound = all_norm_ok and all_w1_ok

    verdict = "SOUND" if approach_a_sound else "DEMOTED"
    headline = "A" if approach_a_sound else "B"

    print(f"\n{'='*60}")
    print(f"  Approach A verdict : {verdict}")
    print(f"  Headline method    : Approach {headline}")
    if not approach_a_sound:
        reasons = []
        if not all_norm_ok:
            bad = [db for db in per_db if not per_db[db]["norm_ratio_ok"]]
            reasons.append(f"norm_ratio outside [0.5,2.0] for {bad}")
        if not all_w1_ok:
            bad = [db for db in per_db if not per_db[db]["wasserstein_ok"]]
            reasons.append(f"Wasserstein-1 ≥ 0.2 for {bad}")
        print(f"  Reasons: {'; '.join(reasons)}")
    print(f"{'='*60}")

    return {
        "mode": "sanity_check",
        "per_database": per_db,
        "approach_a_sound": approach_a_sound,
        "approach_a_verdict": verdict,
        "headline_method": headline,
        "decision_criteria": {
            "norm_ratio_bounds": [0.5, 2.0],
            "wasserstein_threshold": 0.2,
            "norm_ratio_ok": all_norm_ok,
            "wasserstein_ok": all_w1_ok,
        },
        "n_pairs_sampled": n_pairs,
        "smoke": args.smoke,
        "checkpoint_wals": str(args.checkpoint_wals),
        "checkpoint_grambank": str(args.checkpoint_grambank),
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Phase 3 — Approach stubs (implemented in later phases)
# ---------------------------------------------------------------------------

def run_approach_a(args: argparse.Namespace) -> dict[str, Any]:
    raise NotImplementedError("Approach A not yet implemented (Phase 3)")


def run_approach_b(args: argparse.Namespace) -> dict[str, Any]:
    raise NotImplementedError("Approach B not yet implemented (Phase 3)")


def run_approach_c(args: argparse.Namespace) -> dict[str, Any]:
    raise NotImplementedError("Approach C not yet implemented (Phase 3)")


# ---------------------------------------------------------------------------
# Phase 4 — Validation stub
# ---------------------------------------------------------------------------

def run_validate(args: argparse.Namespace) -> dict[str, Any]:
    raise NotImplementedError("Validation not yet implemented (Phase 4)")


# ---------------------------------------------------------------------------
# Phase 5 — Discovery stub
# ---------------------------------------------------------------------------

def run_discover(args: argparse.Namespace) -> dict[str, Any]:
    raise NotImplementedError("Discovery not yet implemented (Phase 5)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_MODES = {
    "sanity_check": run_sanity_check,
    "approach_a": run_approach_a,
    "approach_b": run_approach_b,
    "approach_c": run_approach_c,
    "validate": run_validate,
    "discover": run_discover,
}

_DEFAULT_CKPT_WALS = str(_REPO_ROOT / "checkpoints" / "wals_learned_s42")
_DEFAULT_CKPT_GRAMBANK = str(_REPO_ROOT / "checkpoints" / "grambank_learned_s42")
_DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "analysis" / "cross_database_alignment")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-database feature alignment (RQ3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=list(_MODES.keys()),
        required=True,
        help="Which analysis phase to run.",
    )

    scale = p.add_mutually_exclusive_group()
    scale.add_argument("--smoke", action="store_true",
                       help="Run a tiny subset to validate wiring.")
    scale.add_argument("--full", action="store_true",
                       help="Run the production-scale workload (default).")

    p.add_argument("--resume", action="store_true",
                   help="Skip cells whose output already exists.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_wals", default=_DEFAULT_CKPT_WALS,
                   help="Path to the WALS learned checkpoint directory.")
    p.add_argument("--checkpoint_grambank", default=_DEFAULT_CKPT_GRAMBANK,
                   help="Path to the Grambank learned checkpoint directory.")
    p.add_argument("--output_dir", default=_DEFAULT_OUTPUT_DIR,
                   help="Directory for output files.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[cross_database_alignment] mode={args.mode}  "
          f"{'smoke' if args.smoke else 'full'}  seed={args.seed}")

    t0 = time.perf_counter()
    result = _MODES[args.mode](args)
    elapsed = time.perf_counter() - t0

    result["elapsed_seconds"] = round(elapsed, 3)

    # Save mode-specific output
    output_file = out / f"{args.mode}.json"
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved: {output_file}")

    # Save full config for reproducibility
    config = vars(args).copy()
    config["elapsed_seconds"] = result["elapsed_seconds"]
    (out / f"config_{args.mode}.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )
    print(f"Saved: {out / f'config_{args.mode}.json'}")


if __name__ == "__main__":
    main()
