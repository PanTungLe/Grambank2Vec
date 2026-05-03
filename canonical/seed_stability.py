"""
Seed-stability analysis (Phase 6 of the thesis pipeline).

Given K checkpoint directories (one per training seed) for a single
(database, architecture) setting, quantifies how stable the learned
representations are across random initialisations using four metrics:

  Probe A Jaccard@10 — for each curated feature-value target, the Jaccard
    similarity of the top-10 neighbour sets across all C(K,2) seed pairs.
    High Jaccard → the nearest-neighbour structure is reproducible.

  Probe B silhouette stability — global silhouette score per seed; reports
    mean ± std across seeds.

  Probe C residual stability — Greenberg-pair residuals per seed; reports
    mean ± std per pair.

  Procrustes stability — Procrustes disparity for every C(K,2) pair of
    language-embedding matrices (aligned by row index, which is fixed by
    load order and independent of seed).

Outputs
-------
  stability_report.md         Human-readable summary with tables.
  stability_metrics.json      Machine-readable version of all numbers.

Usage
-----
python canonical/seed_stability.py \\
    --checkpoints checkpoints/wals_tcf_s42 checkpoints/wals_tcf_s43 \\
                  checkpoints/wals_tcf_s44 checkpoints/wals_tcf_s45 \\
                  checkpoints/wals_tcf_s46 \\
    --output_dir  analysis/stability_wals_tcf \\
    [--top_k 10] \\
    [--n_baseline 500] \\
    [--seed 0]
"""

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_geometry import (
    DEFAULT_GRAMBANK_TARGETS,
    DEFAULT_WALS_TARGETS,
    GREENBERG_PAIRS_GRAMBANK,
    GREENBERG_PAIRS_WALS,
    load_checkpoint,
    probe_a_nearest_neighbours,
    probe_b_silhouette,
    probe_c_greenberg,
)
from compare_databases import procrustes_align
from utils import seed_everything


# ---------------------------------------------------------------------------
# Probe A — Jaccard@K stability
# ---------------------------------------------------------------------------

def jaccard(s1: set, s2: set) -> float:
    """Jaccard similarity of two sets (returns 1 if both empty)."""
    if not s1 and not s2:
        return 1.0
    union = len(s1 | s2)
    return len(s1 & s2) / union if union > 0 else 0.0


def pairwise_jaccard_top_k(
    checkpoints: list[dict[str, Any]],
    targets: list[str],
    top_k: int = 10,
) -> dict[str, dict[str, float]]:
    """
    For each target, compute mean/std Jaccard@top_k over all C(K,2) seed pairs.

    Returns {target: {"mean_jaccard": ..., "std_jaccard": ..., "n_pairs": ...}}.
    Targets absent from any checkpoint are silently skipped.
    """
    nn_sets: list[dict[str, set[str]]] = []
    for ckpt in checkpoints:
        nn = probe_a_nearest_neighbours(
            ckpt["fv_emb"], ckpt["fv2id"], targets, top_k=top_k)
        nn_sets.append({
            tgt: {r["neighbour"] for r in nbrs}
            for tgt, nbrs in nn.items()
        })

    results: dict[str, dict[str, float]] = {}
    for tgt in targets:
        if not all(tgt in ns for ns in nn_sets):
            continue
        pair_scores = [
            jaccard(nn_sets[i][tgt], nn_sets[j][tgt])
            for i, j in itertools.combinations(range(len(nn_sets)), 2)
        ]
        results[tgt] = {
            "mean_jaccard": float(np.mean(pair_scores)),
            "std_jaccard": float(np.std(pair_scores)),
            "n_pairs": len(pair_scores),
        }
    return results


# ---------------------------------------------------------------------------
# Probe B — Silhouette stability
# ---------------------------------------------------------------------------

def silhouette_stability(
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute global silhouette score for each checkpoint; return mean ± std.
    """
    scores = []
    per_seed = []
    for ckpt in checkpoints:
        result = probe_b_silhouette(ckpt["fv_emb"], ckpt["fv2id"])
        s = result.get("silhouette_global")
        per_seed.append(s)
        if s is not None:
            scores.append(s)

    if not scores:
        return {"mean": None, "std": None, "per_seed": per_seed,
                "note": "No valid silhouette scores."}
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# Probe C — Greenberg residual stability
# ---------------------------------------------------------------------------

def greenberg_stability(
    checkpoints: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    n_baseline: int = 500,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    For each Greenberg pair, compute residual and empirical p-value across
    seeds; return mean ± std.
    """
    pair_names = [p["name"] for p in pairs]
    residuals: dict[str, list[float]] = {n: [] for n in pair_names}
    p_values: dict[str, list[float]] = {n: [] for n in pair_names}
    skipped: dict[str, list[int]] = {n: [] for n in pair_names}

    for si, ckpt in enumerate(checkpoints):
        results = probe_c_greenberg(
            ckpt["fv_emb"], ckpt["fv2id"], pairs,
            n_baseline=n_baseline, seed=seed)
        for entry in results:
            name = entry["name"]
            if "skipped" in entry:
                skipped[name].append(si)
            else:
                residuals[name].append(entry["residual"])
                p_values[name].append(entry["empirical_p_value"])

    summary = []
    for name in pair_names:
        res = residuals[name]
        pvs = p_values[name]
        entry: dict[str, Any] = {"name": name, "n_seeds": len(res)}
        if skipped[name]:
            entry["skipped_seeds"] = skipped[name]
        if res:
            entry.update({
                "residual_mean": float(np.mean(res)),
                "residual_std": float(np.std(res)),
                "p_value_mean": float(np.mean(pvs)),
                "p_value_std": float(np.std(pvs)),
            })
        summary.append(entry)
    return summary


# ---------------------------------------------------------------------------
# Procrustes stability
# ---------------------------------------------------------------------------

def procrustes_stability(
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute Procrustes disparity between every C(K,2) pair of seed
    language-embedding matrices.

    Row i in each lang_emb corresponds to the same language (fixed by load
    order), so no Glottocode alignment is needed.

    Returns a matrix of disparities plus summary statistics.
    """
    n = len(checkpoints)
    disparity_matrix = np.zeros((n, n), dtype=np.float64)
    pair_disparities = []

    for i, j in itertools.combinations(range(n), 2):
        A = checkpoints[i]["lang_emb"]
        B = checkpoints[j]["lang_emb"]
        d, _, _ = procrustes_align(A, B)
        disparity_matrix[i, j] = d
        disparity_matrix[j, i] = d
        pair_disparities.append(d)

    return {
        "mean_disparity": float(np.mean(pair_disparities)),
        "std_disparity": float(np.std(pair_disparities)),
        "min_disparity": float(np.min(pair_disparities)),
        "max_disparity": float(np.max(pair_disparities)),
        "n_pairs": len(pair_disparities),
        "disparity_matrix": disparity_matrix.tolist(),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt(v: float | None, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if v is not None else "N/A"


def generate_report(
    checkpoints: list[dict[str, Any]],
    checkpoint_dirs: list[str],
    jaccard_results: dict[str, dict[str, float]],
    silhouette_stats: dict[str, Any],
    greenberg_stats: list[dict[str, Any]],
    procrustes_stats: dict[str, Any],
    top_k: int,
) -> str:
    """Build the Markdown stability report."""
    db = checkpoints[0]["database"]
    arch = checkpoints[0]["architecture"]
    K = len(checkpoints)
    n_pairs = K * (K - 1) // 2

    lines = [
        f"# Seed Stability Report — {db.upper()} {arch.upper()}",
        "",
        f"**Settings:** database={db}, architecture={arch}, "
        f"K={K} seeds, {n_pairs} pairwise comparisons",
        "",
        "**Checkpoints:**",
    ]
    for d in checkpoint_dirs:
        lines.append(f"  - `{d}`")
    lines += ["", "---", ""]

    # --- Probe A ---
    lines += [
        f"## Probe A — Nearest-Neighbour Jaccard@{top_k}",
        "",
        "Jaccard similarity of the top-K neighbour sets across all seed pairs.  "
        "Values near 1 indicate the neighbour structure is stable across random initialisations.",
        "",
        f"| Feature-value | Mean Jaccard@{top_k} | Std |",
        "|---------------|---------------------|-----|",
    ]
    if jaccard_results:
        for tgt, stats in sorted(jaccard_results.items(),
                                  key=lambda kv: -kv[1]["mean_jaccard"]):
            lines.append(
                f"| {tgt} | {_fmt(stats['mean_jaccard'], 3)} "
                f"| {_fmt(stats['std_jaccard'], 3)} |"
            )
        all_means = [v["mean_jaccard"] for v in jaccard_results.values()]
        lines += [
            "",
            f"**Overall mean Jaccard@{top_k}:** {np.mean(all_means):.3f} "
            f"± {np.std(all_means):.3f}",
        ]
    else:
        lines.append("| (no targets found in any checkpoint) | — | — |")
    lines += ["", "---", ""]

    # --- Probe B ---
    lines += [
        "## Probe B — Silhouette Stability",
        "",
        "Global silhouette score (metric=cosine) per seed.  "
        "Negative values are expected — see Phase 4 notes.",
        "",
        "| Seed | Silhouette |",
        "|------|------------|",
    ]
    for si, s in enumerate(silhouette_stats.get("per_seed", [])):
        lines.append(f"| {si} (`s{42 + si}`) | {_fmt(s, 4)} |")
    lines += [
        "",
        f"**Mean:** {_fmt(silhouette_stats.get('mean'), 4)}  "
        f"**Std:** {_fmt(silhouette_stats.get('std'), 4)}",
        "",
        "---",
        "",
    ]

    # --- Probe C ---
    lines += [
        "## Probe C — Greenberg Residual Stability",
        "",
        "Residual = ||(a_pos − a_neg) − (b_pos − b_neg)||₂.  "
        "Low residual (low empirical p) → universal supported.",
        "",
        "| Universal | Residual mean ± std | p-value mean ± std |",
        "|-----------|--------------------|--------------------|",
    ]
    for entry in greenberg_stats:
        if "residual_mean" in entry:
            lines.append(
                f"| {entry['name']} "
                f"| {_fmt(entry['residual_mean'], 3)} ± {_fmt(entry['residual_std'], 3)} "
                f"| {_fmt(entry['p_value_mean'], 3)} ± {_fmt(entry['p_value_std'], 3)} |"
            )
        else:
            lines.append(f"| {entry['name']} | SKIPPED | — |")
    lines += ["", "---", ""]

    # --- Procrustes ---
    lines += [
        "## Procrustes Stability (language embeddings)",
        "",
        "Standardised Procrustes disparity between every pair of seed "
        "language-embedding matrices.  Lower → more similar geometry.  "
        "Values near 0 indicate the latent language space is stable "
        "across random initialisations.",
        "",
        f"**Mean disparity:** {_fmt(procrustes_stats['mean_disparity'], 4)}  "
        f"± {_fmt(procrustes_stats['std_disparity'], 4)}  "
        f"(range [{_fmt(procrustes_stats['min_disparity'], 4)}, "
        f"{_fmt(procrustes_stats['max_disparity'], 4)}])",
        "",
        "**Pairwise disparity matrix:**",
        "",
        "```",
    ]
    mat = np.array(procrustes_stats["disparity_matrix"])
    lines.append("     " + "  ".join(f"s{42+j}" for j in range(K)))
    for i in range(K):
        row = "  ".join(f"{mat[i,j]:.4f}" for j in range(K))
        lines.append(f"s{42+i}  {row}")
    lines += [
        "```",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    all_jac = [v["mean_jaccard"] for v in jaccard_results.values()]
    lines.append(
        f"| Mean Jaccard@{top_k} (Probe A) "
        f"| {np.mean(all_jac):.3f} ± {np.std(all_jac):.3f} |"
        if all_jac else
        f"| Mean Jaccard@{top_k} (Probe A) | N/A |"
    )
    lines.append(
        f"| Silhouette mean ± std (Probe B) "
        f"| {_fmt(silhouette_stats.get('mean'), 4)} "
        f"± {_fmt(silhouette_stats.get('std'), 4)} |"
    )
    lines.append(
        f"| Procrustes disparity mean ± std "
        f"| {_fmt(procrustes_stats['mean_disparity'], 4)} "
        f"± {_fmt(procrustes_stats['std_disparity'], 4)} |"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Seed-stability analysis for a canonical-model setting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="Space-separated list of checkpoint directories (one per seed).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_baseline", type=int, default=500,
                   help="Baseline permutations for Probe C.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for Probe C baseline RNG.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load checkpoints ---
    missing = [d for d in args.checkpoints if not Path(d).exists()]
    if missing:
        print("[ERROR] The following checkpoint directories do not exist:")
        for m in missing:
            print(f"  {m}")
        print("Train them first with train_canonical.py, then re-run.")
        raise SystemExit(1)

    print(f"Loading {len(args.checkpoints)} checkpoints…")
    checkpoints = []
    for cdir in args.checkpoints:
        ckpt = load_checkpoint(cdir)
        print(f"  {cdir}: {ckpt['database']} {ckpt['architecture']} "
              f"lang={ckpt['lang_emb'].shape}  fv={ckpt['fv_emb'].shape}")
        checkpoints.append(ckpt)

    db = checkpoints[0]["database"]
    arch = checkpoints[0]["architecture"]

    # Verify all same database + architecture
    for ckpt in checkpoints[1:]:
        if ckpt["database"] != db or ckpt["architecture"] != arch:
            raise ValueError(
                f"Mixed checkpoints: expected ({db}, {arch}), "
                f"got ({ckpt['database']}, {ckpt['architecture']})")

    targets = DEFAULT_WALS_TARGETS if db == "wals" else DEFAULT_GRAMBANK_TARGETS
    pairs = GREENBERG_PAIRS_WALS if db == "wals" else GREENBERG_PAIRS_GRAMBANK

    # --- Probe A ---
    print(f"\n[Probe A] Jaccard@{args.top_k} across "
          f"{len(checkpoints)*(len(checkpoints)-1)//2} seed pairs…")
    jaccard_results = pairwise_jaccard_top_k(checkpoints, targets, args.top_k)
    if jaccard_results:
        all_means = [v["mean_jaccard"] for v in jaccard_results.values()]
        print(f"  {len(jaccard_results)} targets found.  "
              f"Mean Jaccard@{args.top_k} = {np.mean(all_means):.3f} "
              f"± {np.std(all_means):.3f}")
    else:
        print("  No targets found in checkpoints.")

    # --- Probe B ---
    print("\n[Probe B] Silhouette stability…")
    silhouette_stats = silhouette_stability(checkpoints)
    print(f"  Mean silhouette = {silhouette_stats.get('mean', 'N/A'):.4f} "
          f"± {silhouette_stats.get('std', 0.0):.4f}")

    # --- Probe C ---
    print(f"\n[Probe C] Greenberg residual stability (n_baseline={args.n_baseline})…")
    greenberg_stats = greenberg_stability(
        checkpoints, pairs, n_baseline=args.n_baseline, seed=args.seed)
    for entry in greenberg_stats:
        if "residual_mean" in entry:
            print(f"  {entry['name']}: "
                  f"residual={entry['residual_mean']:.3f} ± {entry['residual_std']:.3f}, "
                  f"p={entry['p_value_mean']:.3f} ± {entry['p_value_std']:.3f}")
        else:
            print(f"  {entry['name']}: SKIPPED")

    # --- Procrustes ---
    print("\n[Procrustes] Pairwise lang-embedding stability…")
    proc_stats = procrustes_stability(checkpoints)
    print(f"  Mean disparity = {proc_stats['mean_disparity']:.4f} "
          f"± {proc_stats['std_disparity']:.4f}")

    # --- Write outputs ---
    report = generate_report(
        checkpoints, args.checkpoints,
        jaccard_results, silhouette_stats, greenberg_stats, proc_stats,
        top_k=args.top_k,
    )
    (out / "stability_report.md").write_text(report)

    metrics = {
        "database": db,
        "architecture": arch,
        "k_seeds": len(checkpoints),
        "checkpoints": args.checkpoints,
        "probe_a_jaccard": jaccard_results,
        "probe_b_silhouette": silhouette_stats,
        "probe_c_greenberg": greenberg_stats,
        "procrustes_stability": proc_stats,
    }
    (out / "stability_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nArtifacts written to {out}/")


if __name__ == "__main__":
    main()
