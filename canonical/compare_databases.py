"""
Cross-database comparison (Phase 5 of the thesis pipeline).

Aligns two canonical-training checkpoints (typically WALS vs Grambank, or two
seeds of the same database) on their shared-language Glottocode intersection
and measures latent-space similarity via:

  Step 1 — Glottocode intersection of shared languages.
  Step 2 — Orthogonal Procrustes alignment on shared-language embeddings.
  Step 3 — Permutation test for Procrustes disparity (default 500 perms).
  Step 4 — CCA component correlations (n_components = min(10, embed_dim)).
  Step 5 — Family-preservation probe (requires --family_csv).

Outputs
-------
  comparison_summary.json       All numeric results.
  procrustes_null_distribution.png
  cca_correlations.png

Usage
-----
python canonical/compare_databases.py \\
    --checkpoint_a checkpoints/wals_learned_s42 \\
    --checkpoint_b checkpoints/grambank_learned_s42 \\
    --output_dir  analysis/wals_vs_grambank_learned \\
    [--family_csv  data/lang_families.csv] \\
    [--n_perm 500] \\
    [--n_cca  10] \\
    [--top_k  10] \\
    [--seed   42]

NOTE: Grambank checkpoints require a local clone of the Grambank CLDF repo
(training blocked by network during development).  The script will exit with a
clear message if a checkpoint directory does not exist.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import procrustes as scipy_procrustes

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_geometry import cosine_normalise, load_checkpoint
from utils import seed_everything


# ---------------------------------------------------------------------------
# Step 1 — Glottocode intersection
# ---------------------------------------------------------------------------

def build_shared_index(
    lang2id_a: dict[str, int],
    lang2id_b: dict[str, int],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Intersect two Glottocode-keyed lang2id dicts.

    Returns:
      glottocodes — sorted list of shared Glottocodes
      idx_a       — int64 array of row indices into embedding matrix A
      idx_b       — int64 array of row indices into embedding matrix B
    """
    shared = sorted(set(lang2id_a) & set(lang2id_b))
    idx_a = np.array([lang2id_a[g] for g in shared], dtype=np.int64)
    idx_b = np.array([lang2id_b[g] for g in shared], dtype=np.int64)
    return shared, idx_a, idx_b


# ---------------------------------------------------------------------------
# Step 2 — Procrustes alignment
# ---------------------------------------------------------------------------

def procrustes_align(
    A: np.ndarray,
    B: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Scipy standardised Procrustes: centre + scale both matrices to unit
    Frobenius norm, then find the optimal orthogonal rotation of B onto A.

    Returns (disparity, A_std, B_aligned) where
      disparity = ||A_std - B_aligned||_F^2   (lower → more similar)
    """
    mtx1, mtx2, disparity = scipy_procrustes(
        A.astype(np.float64), B.astype(np.float64))
    return float(disparity), mtx1, mtx2


# ---------------------------------------------------------------------------
# Step 3 — Permutation test
# ---------------------------------------------------------------------------

def permutation_test_procrustes(
    A: np.ndarray,
    B: np.ndarray,
    n_perm: int = 500,
    seed: int = 42,
) -> tuple[float, np.ndarray, float]:
    """
    Test whether observed Procrustes disparity is lower than chance.

    Each permutation shuffles the rows of B before alignment (equivalent to
    a random language correspondence).

    Returns (observed_disparity, null_array, p_value) where
      p_value = fraction of null disparities ≤ observed
      (low p ⇒ alignment is better than random ⇒ spaces are similar).
    """
    rng = np.random.default_rng(seed)
    observed, _, _ = procrustes_align(A, B)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(len(B))
        null[i], _, _ = procrustes_align(A, B[perm])
    p_value = float((null <= observed).mean())
    return observed, null, p_value


# ---------------------------------------------------------------------------
# Step 4 — CCA
# ---------------------------------------------------------------------------

def cca_correlations(
    A: np.ndarray,
    B: np.ndarray,
    n_components: int = 10,
) -> list[float]:
    """
    Fit CCA on (A, B) and return per-component canonical correlations.

    n_components is capped at min(n_components, d_A, d_B, n_samples − 1).
    """
    from sklearn.cross_decomposition import CCA

    n_comp = min(n_components, A.shape[1], B.shape[1], A.shape[0] - 1)
    cca = CCA(n_components=n_comp, max_iter=2000)
    cca.fit(A, B)
    A_t, B_t = cca.transform(A, B)
    return [float(np.corrcoef(A_t[:, i], B_t[:, i])[0, 1])
            for i in range(n_comp)]


# ---------------------------------------------------------------------------
# Step 5 — Family-preservation probe
# ---------------------------------------------------------------------------

def load_family_map(family_csv: str) -> dict[str, str]:
    """
    Read a CSV with columns 'glottocode' and 'family_id'.
    Returns {glottocode: family_id}.
    """
    import pandas as pd
    df = pd.read_csv(family_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    if "glottocode" not in df.columns or "family_id" not in df.columns:
        raise ValueError(
            f"family_csv must have 'glottocode' and 'family_id' columns; "
            f"got {list(df.columns)}"
        )
    return dict(zip(df["glottocode"].astype(str), df["family_id"].astype(str)))


def family_preservation_probe(
    glottocodes: list[str],
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    family_map: dict[str, str],
    top_k: int = 10,
    n_baseline: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """
    For each language with a known family (and ≥2 same-family members in the
    shared set), compute the fraction of top-K cosine-nearest neighbours that
    share the family label.  Performed separately for emb_a and emb_b.
    A random-permutation baseline (shuffle family labels, fixed NN) calibrates
    the expected score under the null of no family clustering.

    Returns a dict with scores, baseline stats, and one-sided p-values.
    """
    from collections import Counter

    families = [family_map.get(g) for g in glottocodes]
    fam_counts = Counter(f for f in families if f is not None)
    valid = [f is not None and fam_counts[f] >= 2 for f in families]
    valid_mask = np.array(valid, dtype=bool)

    if valid_mask.sum() < 2:
        return {"note": "Fewer than 2 languages with ≥2 same-family members."}

    def _precompute_top_k(emb: np.ndarray) -> np.ndarray:
        normed = cosine_normalise(emb)
        n = len(normed)
        result = np.empty((n, top_k), dtype=np.int64)
        for i in range(n):
            sims = normed @ normed[i]
            sims[i] = -np.inf
            result[i] = np.argsort(-sims)[:top_k]
        return result

    top_k_a = _precompute_top_k(emb_a)
    top_k_b = _precompute_top_k(emb_b)

    def _score(top_k_nbrs: np.ndarray, fam_labels: list) -> float:
        fracs = []
        for i in np.where(valid_mask)[0]:
            my_fam = fam_labels[i]
            same = sum(1 for j in top_k_nbrs[i] if fam_labels[j] == my_fam)
            fracs.append(same / top_k)
        return float(np.mean(fracs))

    score_a = _score(top_k_a, families)
    score_b = _score(top_k_b, families)

    rng = np.random.default_rng(seed)
    families_arr = list(families)
    baseline = np.array([
        _score(top_k_a, list(rng.permutation(families_arr)))
        for _ in range(n_baseline)
    ])

    return {
        "n_valid_languages": int(valid_mask.sum()),
        "n_families": len({families[i] for i in np.where(valid_mask)[0]}),
        "score_a": score_a,
        "score_b": score_b,
        "baseline_mean": float(baseline.mean()),
        "baseline_p95": float(np.percentile(baseline, 95)),
        "p_value_a": float((baseline >= score_a).mean()),
        "p_value_b": float((baseline >= score_b).mean()),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_procrustes_null(
    observed: float,
    null: np.ndarray,
    p_value: float,
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null, bins=40, color="steelblue", alpha=0.7, label="Null (permuted)")
    ax.axvline(observed, color="crimson", linewidth=2,
               label=f"Observed = {observed:.4f}  (p = {p_value:.3f})")
    ax.set_xlabel("Procrustes disparity")
    ax.set_ylabel("Count")
    ax.set_title("Procrustes permutation test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_cca(corrs: list[float], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(corrs)), 4))
    xs = list(range(1, len(corrs) + 1))
    ax.bar(xs, corrs, color="steelblue", alpha=0.8)
    ax.set_xlabel("CCA component")
    ax.set_ylabel("Canonical correlation")
    ax.set_title("CCA: per-component canonical correlations")
    ax.set_xticks(xs)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-database latent-space comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint_a", required=True,
                   help="First checkpoint directory (e.g. checkpoints/wals_learned_s42)")
    p.add_argument("--checkpoint_b", required=True,
                   help="Second checkpoint directory (e.g. checkpoints/grambank_learned_s42)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--family_csv", default=None,
                   help="Optional CSV with columns 'glottocode' and 'family_id' "
                        "for family-preservation probe (Step 5).")
    p.add_argument("--n_perm", type=int, default=500,
                   help="Procrustes permutation-test replicates.")
    p.add_argument("--n_cca", type=int, default=10,
                   help="Maximum CCA components.")
    p.add_argument("--top_k", type=int, default=10,
                   help="Nearest-neighbour k for family-preservation probe.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load checkpoints ---
    for attr, label in [("checkpoint_a", "A"), ("checkpoint_b", "B")]:
        cdir = Path(getattr(args, attr))
        if not cdir.exists():
            print(f"[ERROR] Checkpoint {label} not found: {cdir}")
            print("        Train first with train_canonical.py, then re-run.")
            sys.exit(1)

    print(f"Loading checkpoint A: {args.checkpoint_a}")
    ckpt_a = load_checkpoint(args.checkpoint_a)
    print(f"  database={ckpt_a['database']}  architecture={ckpt_a['architecture']}")
    print(f"  lang_emb={ckpt_a['lang_emb'].shape}")

    print(f"Loading checkpoint B: {args.checkpoint_b}")
    ckpt_b = load_checkpoint(args.checkpoint_b)
    print(f"  database={ckpt_b['database']}  architecture={ckpt_b['architecture']}")
    print(f"  lang_emb={ckpt_b['lang_emb'].shape}")

    # --- Step 1: Glottocode intersection ---
    print("\n[Step 1] Building Glottocode intersection…")
    glottocodes, idx_a, idx_b = build_shared_index(
        ckpt_a["lang2id"], ckpt_b["lang2id"])
    n_shared = len(glottocodes)
    print(f"  A: {len(ckpt_a['lang2id'])} languages")
    print(f"  B: {len(ckpt_b['lang2id'])} languages")
    print(f"  Shared: {n_shared} languages")

    if n_shared < 10:
        print("[WARNING] Fewer than 10 shared languages — results will be unreliable.")

    emb_a = ckpt_a["lang_emb"][idx_a]
    emb_b = ckpt_b["lang_emb"][idx_b]

    # --- Step 2–3: Procrustes + permutation test ---
    print(f"\n[Step 2–3] Procrustes alignment + permutation test "
          f"(n_perm={args.n_perm})…")
    observed, null, p_value = permutation_test_procrustes(
        emb_a, emb_b, n_perm=args.n_perm, seed=args.seed)
    print(f"  Observed disparity : {observed:.6f}")
    print(f"  Null mean          : {null.mean():.6f}")
    print(f"  p-value            : {p_value:.4f}  "
          f"({'significant' if p_value < 0.05 else 'not significant'} at α=0.05)")
    _plot_procrustes_null(
        observed, null, p_value,
        str(out / "procrustes_null_distribution.png"))

    # --- Step 4: CCA ---
    print(f"\n[Step 4] CCA (max {args.n_cca} components)…")
    corrs = cca_correlations(emb_a, emb_b, n_components=args.n_cca)
    print(f"  Canonical correlations: {[round(c, 3) for c in corrs]}")
    _plot_cca(corrs, str(out / "cca_correlations.png"))

    # --- Step 5: Family-preservation probe ---
    family_results: dict[str, Any] | None = None
    if args.family_csv:
        print(f"\n[Step 5] Family-preservation probe (top_k={args.top_k})…")
        fmap = load_family_map(args.family_csv)
        family_results = family_preservation_probe(
            glottocodes, emb_a, emb_b, fmap,
            top_k=args.top_k, n_baseline=args.n_perm, seed=args.seed)
        print(f"  n_valid_languages  : {family_results.get('n_valid_languages', '—')}")
        print(f"  n_families         : {family_results.get('n_families', '—')}")
        print(f"  Score A            : {family_results.get('score_a', '—'):.4f}")
        print(f"  Score B            : {family_results.get('score_b', '—'):.4f}")
        print(f"  Baseline mean      : {family_results.get('baseline_mean', '—'):.4f}")
        print(f"  p-value A          : {family_results.get('p_value_a', '—'):.4f}")
        print(f"  p-value B          : {family_results.get('p_value_b', '—'):.4f}")
    else:
        print("\n[Step 5] Skipped (no --family_csv provided).")

    # --- Save summary ---
    summary: dict[str, Any] = {
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "database_a": ckpt_a["database"],
        "database_b": ckpt_b["database"],
        "architecture_a": ckpt_a["architecture"],
        "architecture_b": ckpt_b["architecture"],
        "n_langs_a": len(ckpt_a["lang2id"]),
        "n_langs_b": len(ckpt_b["lang2id"]),
        "n_shared": n_shared,
        "shared_glottocodes_sample": glottocodes[:10],
        "procrustes": {
            "observed_disparity": observed,
            "null_mean": float(null.mean()),
            "null_p5": float(np.percentile(null, 5)),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": p_value,
            "n_perm": args.n_perm,
            "interpretation": (
                "Low p-value (< 0.05) means the two embedding spaces are "
                "more similar than chance — latent typological structure is "
                "shared across databases."
            ),
        },
        "cca": {
            "n_components": len(corrs),
            "canonical_correlations": corrs,
            "mean_correlation": float(np.mean(corrs)),
        },
        "family_probe": family_results,
        "seed": args.seed,
    }
    (out / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nArtifacts written to {out}/")


if __name__ == "__main__":
    main()
