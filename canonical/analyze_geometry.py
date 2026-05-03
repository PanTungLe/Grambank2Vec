"""
Geometry probes for canonical embeddings (Phase 4 of the thesis pipeline).

Implements three RQ2-second-clause probes on a checkpoint produced by
canonical/train_canonical.py:

  Probe A — Nearest-neighbour qualitative inspection.
    Top-K cosine neighbours for a curated list of feature-value targets.

  Probe B — Silhouette score by feature membership.
    Group every featvalue id by its parent feature, compute
    silhouette_score(emb, feature_labels, metric="cosine") plus a
    per-feature breakdown.

  Probe C — Typological-universal analogy probes.
    Vector-arithmetic residual tests for Greenberg's universals,
    compared against a permuted-quadruple null distribution.

Both T-CF and Learned checkpoints are supported.  For T-CF, the probe
operates on per-binary-column embeddings (the only objects available in
that architecture); the output annotation makes this clear so downstream
comparisons aren't confused.

Usage
-----
python canonical/analyze_geometry.py \\
    --checkpoint_dir checkpoints/wals_learned_s42 \\
    --output_dir analysis/wals_learned_s42 \\
    --top_k 10
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import seed_everything


# ---------------------------------------------------------------------------
# Default curated targets (RQ2 second clause)
# ---------------------------------------------------------------------------

# WALS feature-value targets used in Probe A. Targets that don't exist in a
# checkpoint's featvalue2id.json are silently skipped.
DEFAULT_WALS_TARGETS: list[str] = [
    # 81A — Order of Subject, Object and Verb
    "81A=SOV", "81A=SVO", "81A=VSO", "81A=VOS", "81A=OVS", "81A=OSV",
    # 85A — Order of Adposition and Noun Phrase
    "85A=Postpositions", "85A=Prepositions",
    # 55A — Numeral Classifiers
    "55A=Absent", "55A=Obligatory", "55A=Optional",
    # 13A — Tone
    "13A=No tones", "13A=Simple tone system", "13A=Complex tone system",
]

# Grambank targets — populated when we have access to the Grambank checkpoint.
# Word-order, adposition, classifier, tone analogues exist as GB### IDs;
# include a placeholder list so the script doesn't crash on Grambank input
# (the silent-skip behaviour will drop unknown keys).
DEFAULT_GRAMBANK_TARGETS: list[str] = [
    # GB070 — VO order (binary 0/1) — the binarycol/featvalue label is "GB070=1"
    "GB070=1", "GB070=0",
    # GB071 — OV order
    "GB071=1", "GB071=0",
    # GB022 — Postpositions
    "GB022=1", "GB022=0",
    # GB023 — Prepositions
    "GB023=1", "GB023=0",
    # GB082 — Numeral classifiers
    "GB082=1", "GB082=0",
    # GB319 — Lexical tone (binary)
    "GB319=1", "GB319=0",
]


# Probe C universal pairs
# Format: (label, [(positive_target_a, negative_target_a),
#                  (positive_target_b, negative_target_b)])
# Tests:  emb[posA] - emb[negA] ≈ emb[posB] - emb[negB]
# Greenberg U4: SOV ↔ Postpositions ; VSO ↔ Prepositions
GREENBERG_PAIRS_WALS: list[dict[str, Any]] = [
    {
        "name": "Greenberg-U4 (SOV/Post vs VSO/Prep)",
        "a_pos": "81A=SOV", "a_neg": "85A=Postpositions",
        "b_pos": "81A=VSO", "b_neg": "85A=Prepositions",
    },
    {
        "name": "Word-order/Adposition (SVO/Prep vs SOV/Post)",
        "a_pos": "81A=SVO", "a_neg": "85A=Prepositions",
        "b_pos": "81A=SOV", "b_neg": "85A=Postpositions",
    },
    {
        "name": "Tone vs Classifier (SimpleTone/NoClass vs NoTone/ObligClass)",
        "a_pos": "13A=Simple tone system", "a_neg": "55A=Absent",
        "b_pos": "13A=No tones", "b_neg": "55A=Obligatory",
    },
]

GREENBERG_PAIRS_GRAMBANK: list[dict[str, Any]] = [
    {
        "name": "Word-order/Adposition (OV/Post vs VO/Prep)",
        "a_pos": "GB071=1", "a_neg": "GB022=1",
        "b_pos": "GB070=1", "b_neg": "GB023=1",
    },
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_dir: str) -> dict[str, Any]:
    """
    Load a canonical-training checkpoint directory.

    Returns a dict containing:
      architecture: "tcf" | "learned"
      database:     "wals" | "grambank"
      lang_emb:     (n_langs, d) numpy array
      fv_emb:       (n_fv, d) numpy array — featvalue or binarycol embeddings
      fv2id:        Dict[str, int] — "81A=SOV" → row index
      lang2id:      Dict[str, int] — Glottocode → row index (canonical)
      lang2id_full: Dict[str, int] — native ID → row index
      feat2values:  Dict[str, list[str]]
      config:       full training config
    """
    cdir = Path(checkpoint_dir)
    config = json.loads((cdir / "config.json").read_text())
    arch = config["architecture"]
    db = config["database"]

    lang_emb = np.load(cdir / "lang_embeddings.npy")
    if arch == "learned":
        fv_emb = np.load(cdir / "featvalue_embeddings.npy")
        fv2id = json.loads((cdir / "featvalue2id.json").read_text())
    else:
        fv_emb = np.load(cdir / "binarycol_embeddings.npy")
        fv2id = json.loads((cdir / "binarycol2id.json").read_text())

    lang2id = json.loads((cdir / "lang2id.json").read_text())
    lang2id_full = json.loads((cdir / "lang2id_full.json").read_text())
    feat2values = json.loads((cdir / "feat2values.json").read_text())

    return dict(
        architecture=arch, database=db,
        lang_emb=lang_emb, fv_emb=fv_emb,
        fv2id=fv2id, lang2id=lang2id, lang2id_full=lang2id_full,
        feat2values=feat2values, config=config,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_normalise(M: np.ndarray) -> np.ndarray:
    """Row-wise unit-normalise; rows of zero norm become zero rows."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return M / norms


def feature_id_of(featvalue_label: str) -> str:
    """Strip everything after '=' to get the parent feature ID.

    For T-CF 2-valued features (no '=' suffix), the label IS the feature ID.
    """
    return featvalue_label.split("=", 1)[0]


# ---------------------------------------------------------------------------
# Probe A — Nearest neighbours
# ---------------------------------------------------------------------------

def probe_a_nearest_neighbours(
    fv_emb: np.ndarray,
    fv2id: dict[str, int],
    targets: list[str],
    top_k: int = 10,
) -> dict[str, list[dict[str, float]]]:
    """For each curated target, return the top-K cosine-nearest neighbours."""
    id2fv = {i: name for name, i in fv2id.items()}
    normed = cosine_normalise(fv_emb)
    results: dict[str, list[dict[str, float]]] = {}

    for tgt in targets:
        if tgt not in fv2id:
            continue
        i = fv2id[tgt]
        sims = normed @ normed[i]
        sims[i] = -np.inf
        top = np.argsort(-sims)[:top_k]
        results[tgt] = [
            {"neighbour": id2fv[int(j)], "cosine_sim": float(sims[int(j)])}
            for j in top
        ]
    return results


# ---------------------------------------------------------------------------
# Probe B — Silhouette by feature membership
# ---------------------------------------------------------------------------

def probe_b_silhouette(
    fv_emb: np.ndarray,
    fv2id: dict[str, int],
) -> dict[str, Any]:
    """
    Silhouette score over all featvalues, using parent-feature as the cluster
    label.  Returns the global score plus a per-feature breakdown.

    Features represented by only one value-embedding are excluded (silhouette
    undefined for singleton clusters).
    """
    # Order rows by the fv2id index so we work in embedding-row order
    n = fv_emb.shape[0]
    id2fv = {i: name for name, i in fv2id.items()}

    feat_labels = np.array([feature_id_of(id2fv[i]) for i in range(n)])
    # Drop rows whose parent feature has only one value (silhouette needs k≥2)
    unique_feats, counts = np.unique(feat_labels, return_counts=True)
    multi_feats = set(unique_feats[counts >= 2])
    keep_mask = np.array([f in multi_feats for f in feat_labels])

    if keep_mask.sum() < 2 or len(multi_feats) < 2:
        return {
            "silhouette_global": None,
            "n_used": int(keep_mask.sum()),
            "note": "Insufficient multi-value features for silhouette.",
            "per_feature": {},
        }

    M = cosine_normalise(fv_emb[keep_mask])
    labels = feat_labels[keep_mask]
    global_score = float(silhouette_score(M, labels, metric="cosine"))
    sample_scores = silhouette_samples(M, labels, metric="cosine")

    per_feature: dict[str, dict[str, float]] = {}
    for f in sorted(multi_feats):
        idx = np.where(labels == f)[0]
        per_feature[f] = {
            "n_values": int(len(idx)),
            "mean_silhouette": float(sample_scores[idx].mean()),
        }
    return {
        "silhouette_global": global_score,
        "n_used": int(keep_mask.sum()),
        "n_features_used": len(multi_feats),
        "per_feature": per_feature,
    }


# ---------------------------------------------------------------------------
# Probe C — Greenberg analogy probes
# ---------------------------------------------------------------------------

def _residual(emb: np.ndarray, a_pos: int, a_neg: int,
              b_pos: int, b_neg: int) -> float:
    """L2 norm of (a_pos - a_neg) - (b_pos - b_neg)."""
    diff = (emb[a_pos] - emb[a_neg]) - (emb[b_pos] - emb[b_neg])
    return float(np.linalg.norm(diff))


def probe_c_greenberg(
    fv_emb: np.ndarray,
    fv2id: dict[str, int],
    pairs: list[dict[str, Any]],
    n_baseline: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    Vector-arithmetic residual test for typological-universal pairs.

    For each pair, compute:
        observed = ||(a_pos - a_neg) - (b_pos - b_neg)||
        baseline = same residual but with random quadruples drawn from
                   ALL featvalue rows.
    Report observed vs baseline mean / 5th-percentile / empirical p.

    Pairs that reference featvalues missing from the checkpoint are skipped
    (their entry in the result has a "skipped" key with the reason).
    """
    rng = np.random.default_rng(seed)
    n = fv_emb.shape[0]
    results: list[dict[str, Any]] = []

    for spec in pairs:
        missing = [k for k in ("a_pos", "a_neg", "b_pos", "b_neg")
                   if spec[k] not in fv2id]
        if missing:
            results.append({
                "name": spec["name"],
                "skipped": f"missing featvalues: {[spec[k] for k in missing]}",
            })
            continue

        a_pos = fv2id[spec["a_pos"]]
        a_neg = fv2id[spec["a_neg"]]
        b_pos = fv2id[spec["b_pos"]]
        b_neg = fv2id[spec["b_neg"]]

        observed = _residual(fv_emb, a_pos, a_neg, b_pos, b_neg)

        baseline = np.empty(n_baseline, dtype=np.float64)
        for k in range(n_baseline):
            ap, an, bp, bn = rng.integers(0, n, size=4)
            baseline[k] = _residual(fv_emb, ap, an, bp, bn)

        empirical_p = float((baseline <= observed).mean())
        results.append({
            "name": spec["name"],
            "a_pos": spec["a_pos"], "a_neg": spec["a_neg"],
            "b_pos": spec["b_pos"], "b_neg": spec["b_neg"],
            "residual": observed,
            "baseline_mean": float(baseline.mean()),
            "baseline_p5": float(np.percentile(baseline, 5)),
            "baseline_median": float(np.median(baseline)),
            "empirical_p_value": empirical_p,
            "n_baseline": n_baseline,
        })
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Geometry probes for a canonical-training checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--targets_file", default=None,
        help="Optional JSON file with a list of featvalue labels for Probe A. "
             "Falls back to a built-in default list per database.")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_baseline", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def load_targets(
    targets_file: str | None,
    database: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return (probe_a_targets, probe_c_pairs) appropriate for the database."""
    if targets_file:
        loaded = json.loads(Path(targets_file).read_text())
        # Allow either a flat list (Probe A only) or {"a": [...], "c": [...]}
        if isinstance(loaded, list):
            return loaded, (GREENBERG_PAIRS_WALS if database == "wals"
                            else GREENBERG_PAIRS_GRAMBANK)
        return loaded.get("a", []), loaded.get("c", [])

    if database == "wals":
        return DEFAULT_WALS_TARGETS, GREENBERG_PAIRS_WALS
    return DEFAULT_GRAMBANK_TARGETS, GREENBERG_PAIRS_GRAMBANK


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint_dir}")
    ckpt = load_checkpoint(args.checkpoint_dir)
    arch = ckpt["architecture"]
    db = ckpt["database"]
    print(f"  database={db}  architecture={arch}  "
          f"fv_emb={ckpt['fv_emb'].shape}  lang_emb={ckpt['lang_emb'].shape}")

    targets, pairs = load_targets(args.targets_file, db)

    # --- Probe A ---
    print(f"\n[Probe A] Nearest neighbours for {len(targets)} curated targets")
    nn_results = probe_a_nearest_neighbours(
        ckpt["fv_emb"], ckpt["fv2id"], targets, top_k=args.top_k)
    found = list(nn_results.keys())
    skipped = [t for t in targets if t not in nn_results]
    print(f"  found {len(found)}, skipped {len(skipped)} (not in checkpoint)")
    if skipped:
        print(f"  skipped: {skipped[:10]}{'...' if len(skipped)>10 else ''}")

    annotation = ("T-CF: nearest neighbours are over per-binary-column "
                  "embeddings; 2-valued features only have a '1' column."
                  if arch == "tcf" else
                  "Learned: nearest neighbours are over per-featvalue "
                  "embeddings (one row per (feature, value) pair).")
    nn_payload = {
        "_annotation": annotation,
        "_architecture": arch,
        "_database": db,
        "results": nn_results,
    }
    (out / "nearest_neighbours.json").write_text(
        json.dumps(nn_payload, indent=2, ensure_ascii=False))

    # --- Probe B ---
    print(f"\n[Probe B] Silhouette by feature membership")
    sil = probe_b_silhouette(ckpt["fv_emb"], ckpt["fv2id"])
    sil_payload = {
        "_annotation": annotation,
        "_architecture": arch,
        "_database": db,
        **sil,
    }
    (out / "silhouette.json").write_text(json.dumps(sil_payload, indent=2))
    print(f"  global silhouette: {sil['silhouette_global']}")
    print(f"  n_features used: {sil.get('n_features_used', 0)}")
    if sil.get("per_feature"):
        # Top/bottom 5 features by mean silhouette
        feats = sorted(sil["per_feature"].items(),
                       key=lambda kv: -kv[1]["mean_silhouette"])
        print(f"  Top 5 best-clustered features:")
        for f, info in feats[:5]:
            print(f"    {f:>6}  silh={info['mean_silhouette']:+.3f}  "
                  f"n_values={info['n_values']}")
        print(f"  Bottom 5 worst-clustered features:")
        for f, info in feats[-5:]:
            print(f"    {f:>6}  silh={info['mean_silhouette']:+.3f}  "
                  f"n_values={info['n_values']}")

    # --- Probe C ---
    print(f"\n[Probe C] Greenberg-style analogy probes "
          f"(n_baseline={args.n_baseline})")
    analogies = probe_c_greenberg(
        ckpt["fv_emb"], ckpt["fv2id"], pairs,
        n_baseline=args.n_baseline, seed=args.seed)
    analogies_payload = {
        "_annotation": annotation,
        "_architecture": arch,
        "_database": db,
        "results": analogies,
    }
    (out / "analogies.json").write_text(json.dumps(analogies_payload, indent=2))
    for a in analogies:
        if "skipped" in a:
            print(f"  [skipped] {a['name']}: {a['skipped']}")
            continue
        print(f"  {a['name']}")
        print(f"    residual = {a['residual']:.4f}")
        print(f"    baseline mean = {a['baseline_mean']:.4f}, "
              f"p5 = {a['baseline_p5']:.4f}, median = {a['baseline_median']:.4f}")
        print(f"    empirical p = {a['empirical_p_value']:.4f}  "
              f"(low p ⇒ residual is unusually small ⇒ universal supported)")

    print(f"\nArtifacts in {out}/")


if __name__ == "__main__":
    main()
