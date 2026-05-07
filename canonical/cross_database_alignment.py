"""
Cross-database feature alignment (RQ3).

Modes
-----
  sanity_check   Phase 1 — test whether the architectural shared-space
                 assumption holds; decide whether Approach A is sound.
  approach_a     Phase 3 — Procrustes-rotated featvalue alignment.
  approach_b     Phase 3 — language-profile correspondence (headline method).
  approach_c     Phase 3 — CCA-projected featvalue alignment.
  validate       Phase 4 — validate against gold-standard pairs.
  discover       Phase 5 — novel correspondence discovery.

Usage
-----
  python canonical/cross_database_alignment.py \\
      --mode {sanity_check|approach_a|approach_b|approach_c|validate|discover} \\
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
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CKPT_WALS = str(_REPO_ROOT / "checkpoints" / "wals_learned_s42")
_DEFAULT_CKPT_GRAMBANK = str(_REPO_ROOT / "checkpoints" / "grambank_learned_s42")
_DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "analysis" / "cross_database_alignment")
_VALIDATION_PAIRS_FILE = str(
    _REPO_ROOT / "analysis" / "cross_database_alignment" / "validation_pairs.json"
)

# Maximum time (seconds) before Approach B warns / halts
_APPROACH_B_WARN_SEC = 30 * 60
_APPROACH_B_HALT_SEC = 2 * 60 * 60


# ---------------------------------------------------------------------------
# Shared loading helpers
# ---------------------------------------------------------------------------

def _load_pair(
    ckpt_wals: str,
    ckpt_grambank: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load WALS and Grambank learned checkpoints, validate architecture."""
    for path, label in [(ckpt_wals, "WALS"), (ckpt_grambank, "Grambank")]:
        if not Path(path).exists():
            print(f"[ERROR] {label} checkpoint not found: {path}", file=sys.stderr)
            sys.exit(1)
    ckpt_w = load_checkpoint(ckpt_wals)
    ckpt_g = load_checkpoint(ckpt_grambank)
    for ckpt, label in [(ckpt_w, "WALS"), (ckpt_g, "Grambank")]:
        if ckpt["architecture"] != "learned":
            raise ValueError(
                f"{label} checkpoint architecture={ckpt['architecture']!r}; "
                "cross_database_alignment.py requires 'learned'."
            )
    return ckpt_w, ckpt_g


def _build_shared_index(
    lang2id_a: dict[str, int],
    lang2id_b: dict[str, int],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Sorted Glottocode intersection → (glottocodes, idx_a, idx_b)."""
    shared = sorted(set(lang2id_a) & set(lang2id_b))
    idx_a = np.array([lang2id_a[g] for g in shared], dtype=np.int64)
    idx_b = np.array([lang2id_b[g] for g in shared], dtype=np.int64)
    return shared, idx_a, idx_b


def _build_feat_global_ids(
    fv2id: dict[str, int],
    feat2values: dict[str, list],
) -> dict[str, list[int]]:
    """
    For each feature, return the list of global embedding indices of its values
    in the order they appear in feat2values.

    Returns {feature_name: [global_id, ...]}
    """
    feat_global: dict[str, list[int]] = {}
    for feat, vals in feat2values.items():
        gids = []
        for v in vals:
            label = f"{feat}={v}"
            if label in fv2id:
                gids.append(fv2id[label])
        if gids:
            feat_global[feat] = gids
    return feat_global


def _compute_profiles(
    lang_emb: np.ndarray,
    fv_emb: np.ndarray,
    fv2id: dict[str, int],
    feat2values: dict[str, list],
) -> np.ndarray:
    """
    Compute language-profile matrix of shape (n_fv, n_shared).

    Row i = softmax probability vector over the shared languages for the
    featvalue whose global embedding index is i.

    P(v | lang_l, feat_f) = softmax over {fv_emb[g] : g in feat_global[f]}.

    Featvalues not belonging to any retained feature (filtered) have a
    zero-profile row.
    """
    from scipy.special import softmax as scipy_softmax

    n_fv = len(fv2id)
    n_shared = lang_emb.shape[0]
    profiles = np.zeros((n_fv, n_shared), dtype=np.float32)

    feat_global = _build_feat_global_ids(fv2id, feat2values)

    for feat, gids in feat_global.items():
        val_embs = fv_emb[gids]                         # (K_f, d)
        scores = lang_emb @ val_embs.T                  # (n_shared, K_f)
        probs = scipy_softmax(scores, axis=1)            # (n_shared, K_f)
        for local_k, gid in enumerate(gids):
            profiles[gid] = probs[:, local_k]

    return profiles


def _top_k_per_wals(
    sim_matrix: np.ndarray,
    wals_fv2id: dict[str, int],
    gb_fv2id: dict[str, int],
    top_k: int = 100,
    smoke_limit: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    For each WALS featvalue (rows of sim_matrix), return the top-k Grambank
    featvalues (columns of sim_matrix) by similarity.

    sim_matrix shape: (n_fv_wals, n_fv_gb)
    """
    id2gb = {v: k for k, v in gb_fv2id.items()}
    id2wals = {v: k for k, v in wals_fv2id.items()}

    results: dict[str, list[dict[str, Any]]] = {}
    n_wals = sim_matrix.shape[0]

    wals_row_range = range(n_wals)
    if smoke_limit:
        # In smoke mode, only query a fixed small set of featvalues
        rng = np.random.default_rng(42)
        indices = rng.choice(n_wals, min(smoke_limit, n_wals), replace=False)
        wals_row_range = sorted(indices)

    for wals_idx in wals_row_range:
        if wals_idx not in id2wals:
            continue
        wals_label = id2wals[wals_idx]
        row = sim_matrix[wals_idx]
        top_indices = np.argsort(-row)[:top_k]
        top_list = []
        for rank, gb_idx in enumerate(top_indices, start=1):
            if gb_idx in id2gb:
                top_list.append({
                    "grambank": id2gb[int(gb_idx)],
                    "sim": float(row[int(gb_idx)]),
                    "rank": rank,
                })
        results[wals_label] = top_list

    return results


def _sample_cosines(
    emb: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    normed = cosine_normalise(emb)
    n = len(normed)
    i1 = rng.integers(0, n, n_pairs)
    i2 = rng.integers(0, n, n_pairs)
    same = i1 == i2
    i2[same] = (i2[same] + 1) % n
    return (normed[i1] * normed[i2]).sum(axis=1)


# ---------------------------------------------------------------------------
# Phase 1 — Sanity check
# ---------------------------------------------------------------------------

def run_sanity_check(args: argparse.Namespace) -> dict[str, Any]:
    """
    Test the architectural shared-space assumption.

    Decision rule (applied per-database, both must pass):
      SOUND  iff  norm_ratio ∈ [0.5, 2.0]
               AND  Wasserstein-1(lang_cosines, fv_cosines) < 0.2
      else: Approach A is DEMOTED to robustness check; Approach B = headline.
    """
    from scipy.stats import wasserstein_distance

    n_pairs = 1000 if args.smoke else 10_000
    rng = np.random.default_rng(args.seed)

    print("Loading checkpoints …")
    ckpt_w, ckpt_g = _load_pair(args.checkpoint_wals, args.checkpoint_grambank)
    print(f"  WALS:     lang_emb={ckpt_w['lang_emb'].shape}  "
          f"fv_emb={ckpt_w['fv_emb'].shape}")
    print(f"  Grambank: lang_emb={ckpt_g['lang_emb'].shape}  "
          f"fv_emb={ckpt_g['fv_emb'].shape}")

    per_db: dict[str, dict[str, Any]] = {}
    for db_name, ckpt in [("wals", ckpt_w), ("grambank", ckpt_g)]:
        lang_emb = ckpt["lang_emb"]
        fv_emb = ckpt["fv_emb"]
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
# Phase 3 — Approach A: Procrustes-rotated featvalues
# ---------------------------------------------------------------------------

def run_approach_a(args: argparse.Namespace) -> dict[str, Any]:
    """
    Apply the language-space Procrustes rotation R (WALS → Grambank) to
    WALS featvalue embeddings, then find top-100 cosine-nearest Grambank
    featvalues for each WALS featvalue.

    Note: Phase 1 DEMOTED Approach A (norm_ratio outside [0.5,2.0]; large
    Wasserstein-1).  Results are retained as a robustness check.
    """
    from scipy.linalg import orthogonal_procrustes

    smoke_limit = 20 if args.smoke else None

    print("Loading checkpoints …")
    ckpt_w, ckpt_g = _load_pair(args.checkpoint_wals, args.checkpoint_grambank)

    shared, idx_w, idx_g = _build_shared_index(
        ckpt_w["lang2id"], ckpt_g["lang2id"])
    n_shared = len(shared)
    print(f"  Shared languages: {n_shared}")

    L_W = ckpt_w["lang_emb"][idx_w]   # (n_shared, 64)
    L_G = ckpt_g["lang_emb"][idx_g]   # (n_shared, 64)

    # Orthogonal Procrustes: find R minimising ||L_G - L_W @ R||_F
    # so that L_W @ R ≈ L_G  →  rotate WALS space to Grambank space
    print("Computing Procrustes rotation …")
    R, scale = orthogonal_procrustes(L_G, L_W)
    print(f"  Procrustes scale = {scale:.4f}")

    F_W = ckpt_w["fv_emb"]             # (640, 64)
    F_G = ckpt_g["fv_emb"]             # (392, 64)

    # Rotate WALS featvalue embeddings into Grambank language space
    F_W_rotated = F_W @ R              # (640, 64)

    # Cosine-normalise both
    F_W_norm = cosine_normalise(F_W_rotated)
    F_G_norm = cosine_normalise(F_G)

    # Similarity matrix: (n_fv_w, n_fv_g)
    print("Computing cosine similarity matrix …")
    sim_matrix = F_W_norm @ F_G_norm.T

    n_fv_w = F_W.shape[0]
    top_k = min(100, F_G.shape[0])

    print(f"  Extracting top-{top_k} per WALS featvalue "
          f"({'smoke subset' if smoke_limit else 'all'}) …")
    results = _top_k_per_wals(
        sim_matrix, ckpt_w["fv2id"], ckpt_g["fv2id"],
        top_k=top_k, smoke_limit=smoke_limit,
    )

    n_queried = len(results)
    mean_top1_sim = float(np.mean([r[0]["sim"] for r in results.values() if r]))

    print(f"  Featvalues queried: {n_queried}")
    print(f"  Mean top-1 cosine:  {mean_top1_sim:.4f}")

    return {
        "mode": "approach_a",
        "results": results,
        "n_featvalues_queried": n_queried,
        "n_fv_wals": n_fv_w,
        "n_fv_grambank": int(F_G.shape[0]),
        "n_shared_languages": n_shared,
        "top_k": top_k,
        "mean_top1_cosine": mean_top1_sim,
        "procrustes_scale": float(scale),
        "smoke": args.smoke,
        "note": (
            "Approach A is DEMOTED (Phase 1 sanity check failed). "
            "Results included as robustness check only."
        ),
    }


# ---------------------------------------------------------------------------
# Phase 3 — Approach B: Language-profile correspondence (headline method)
# ---------------------------------------------------------------------------

def run_approach_b(args: argparse.Namespace) -> dict[str, Any]:
    """
    For each featvalue, build a language-profile vector of predicted
    probabilities over the 1015 shared Glottocodes, then compute cross-
    database cosine similarity between WALS and Grambank profiles.

    This is the headline method for RQ3 (Approach A was demoted by Phase 1).
    """
    smoke_limit = 20 if args.smoke else None

    t0 = time.perf_counter()
    print("Loading checkpoints …")
    ckpt_w, ckpt_g = _load_pair(args.checkpoint_wals, args.checkpoint_grambank)

    shared, idx_w, idx_g = _build_shared_index(
        ckpt_w["lang2id"], ckpt_g["lang2id"])
    n_shared = len(shared)
    print(f"  Shared languages: {n_shared}")

    L_W = ckpt_w["lang_emb"][idx_w].astype(np.float32)
    L_G = ckpt_g["lang_emb"][idx_g].astype(np.float32)
    F_W = ckpt_w["fv_emb"].astype(np.float32)
    F_G = ckpt_g["fv_emb"].astype(np.float32)

    print("Computing WALS language profiles …")
    wals_profiles = _compute_profiles(
        L_W, F_W, ckpt_w["fv2id"], ckpt_w["feat2values"])
    t_wals = time.perf_counter() - t0
    print(f"  WALS profiles shape: {wals_profiles.shape}  "
          f"({t_wals:.1f}s)")

    print("Computing Grambank language profiles …")
    t1 = time.perf_counter()
    gb_profiles = _compute_profiles(
        L_G, F_G, ckpt_g["fv2id"], ckpt_g["feat2values"])
    t_gb = time.perf_counter() - t1
    print(f"  Grambank profiles shape: {gb_profiles.shape}  "
          f"({t_gb:.1f}s)")

    elapsed_profiles = time.perf_counter() - t0
    if elapsed_profiles > _APPROACH_B_WARN_SEC:
        print(f"[WARNING] Approach B profile computation took "
              f"{elapsed_profiles/60:.1f} min (> 30 min threshold).")
    if elapsed_profiles > _APPROACH_B_HALT_SEC:
        print("[HALT] Approach B exceeded 2-hour time limit.", file=sys.stderr)
        sys.exit(1)

    print("Computing cosine similarity matrix (WALS profiles × Grambank profiles) …")
    wals_norm = cosine_normalise(wals_profiles)
    gb_norm = cosine_normalise(gb_profiles)
    sim_matrix = wals_norm @ gb_norm.T     # (n_fv_w, n_fv_g)

    top_k = min(100, F_G.shape[0])
    print(f"  Extracting top-{top_k} per WALS featvalue "
          f"({'smoke subset' if smoke_limit else 'all'}) …")
    results = _top_k_per_wals(
        sim_matrix, ckpt_w["fv2id"], ckpt_g["fv2id"],
        top_k=top_k, smoke_limit=smoke_limit,
    )

    n_queried = len(results)
    mean_top1_sim = float(np.mean([r[0]["sim"] for r in results.values() if r]))

    print(f"  Featvalues queried: {n_queried}")
    print(f"  Mean top-1 cosine:  {mean_top1_sim:.4f}")

    return {
        "mode": "approach_b",
        "results": results,
        "n_featvalues_queried": n_queried,
        "n_fv_wals": int(F_W.shape[0]),
        "n_fv_grambank": int(F_G.shape[0]),
        "n_shared_languages": n_shared,
        "top_k": top_k,
        "mean_top1_cosine": mean_top1_sim,
        "smoke": args.smoke,
        "note": "Approach B is the HEADLINE method (Approach A demoted by Phase 1).",
    }


# ---------------------------------------------------------------------------
# Phase 3 — Approach C: CCA-projected featvalues
# ---------------------------------------------------------------------------

def run_approach_c(args: argparse.Namespace) -> dict[str, Any]:
    """
    Fit CCA on shared language embeddings, project both featvalue sets
    onto the joint 10-component basis, and compute cross-database cosines
    in the projected space.
    """
    from sklearn.cross_decomposition import CCA

    smoke_limit = 20 if args.smoke else None

    print("Loading checkpoints …")
    ckpt_w, ckpt_g = _load_pair(args.checkpoint_wals, args.checkpoint_grambank)

    shared, idx_w, idx_g = _build_shared_index(
        ckpt_w["lang2id"], ckpt_g["lang2id"])
    n_shared = len(shared)
    print(f"  Shared languages: {n_shared}")

    L_W = ckpt_w["lang_emb"][idx_w]
    L_G = ckpt_g["lang_emb"][idx_g]
    F_W = ckpt_w["fv_emb"]
    F_G = ckpt_g["fv_emb"]

    n_comp = min(10, L_W.shape[1], L_G.shape[1], n_shared - 1)
    print(f"Fitting CCA (n_components={n_comp}) on shared language embeddings …")
    cca = CCA(n_components=n_comp, max_iter=2000)
    cca.fit(L_W, L_G)

    # CCA x_rotations_ and y_rotations_ are the (d, n_comp) matrices used by
    # transform(), with centering applied internally via x_mean_/y_mean_.
    mu_W = cca._x_mean    # (64,)
    mu_G = cca._y_mean    # (64,)
    C_W = cca.x_rotations_   # (64, n_comp)
    C_G = cca.y_rotations_   # (64, n_comp)

    print(f"  CCA fitted.  x_rotations shape: {C_W.shape}")

    # Project featvalue embeddings using language-space CCA rotation
    # (valid if featvalues live in the same space as languages — demoted
    # by Phase 1, but included as a robustness check)
    F_W_proj = (F_W - mu_W) @ C_W    # (640, n_comp)
    F_G_proj = (F_G - mu_G) @ C_G    # (392, n_comp)

    # Cosine similarity in CCA space
    F_W_norm = cosine_normalise(F_W_proj)
    F_G_norm = cosine_normalise(F_G_proj)
    sim_matrix = F_W_norm @ F_G_norm.T   # (640, 392)

    top_k = min(100, F_G.shape[0])
    print(f"  Extracting top-{top_k} per WALS featvalue "
          f"({'smoke subset' if smoke_limit else 'all'}) …")
    results = _top_k_per_wals(
        sim_matrix, ckpt_w["fv2id"], ckpt_g["fv2id"],
        top_k=top_k, smoke_limit=smoke_limit,
    )

    n_queried = len(results)
    mean_top1_sim = float(np.mean([r[0]["sim"] for r in results.values() if r]))

    print(f"  Featvalues queried: {n_queried}")
    print(f"  Mean top-1 cosine:  {mean_top1_sim:.4f}")

    return {
        "mode": "approach_c",
        "results": results,
        "n_featvalues_queried": n_queried,
        "n_fv_wals": int(F_W.shape[0]),
        "n_fv_grambank": int(F_G.shape[0]),
        "n_shared_languages": n_shared,
        "n_cca_components": n_comp,
        "top_k": top_k,
        "mean_top1_cosine": mean_top1_sim,
        "smoke": args.smoke,
    }


# ---------------------------------------------------------------------------
# Phase 4 — Validate against gold standard
# ---------------------------------------------------------------------------

def _rank_in_approach(
    approach_results: dict[str, list[dict[str, Any]]],
    wals_fv: str,
    gb_fv: str,
) -> tuple[int | None, float | None]:
    """
    Return (rank, sim) of gb_fv in approach_results[wals_fv], or (None, None)
    if wals_fv not queried or gb_fv not in top-100.
    """
    if wals_fv not in approach_results:
        return None, None
    for entry in approach_results[wals_fv]:
        if entry["grambank"] == gb_fv:
            return entry["rank"], entry["sim"]
    return None, None


def _recall_and_mrr(
    pairs: list[dict[str, Any]],
    approach_results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    Compute top-{1,3,5,10} recall and MRR over a list of pairs.

    Each pair has 'wals' (single WALS featvalue) and 'grambank' (list of
    target Grambank featvalues).  A hit at rank k means at least one of the
    grambank targets appears at rank ≤ k.
    """
    reciprocal_ranks: list[float] = []
    hit_at: dict[int, int] = {1: 0, 3: 0, 5: 0, 10: 0}
    n_total = 0

    for pair in pairs:
        wals_fv = pair["wals"]
        gb_targets = pair["grambank"]
        if wals_fv not in approach_results:
            continue
        n_total += 1

        best_rank: int | None = None
        for gb_fv in gb_targets:
            rank, _ = _rank_in_approach(approach_results, wals_fv, gb_fv)
            if rank is not None:
                best_rank = rank if best_rank is None else min(best_rank, rank)

        if best_rank is None:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / best_rank)
            for k in hit_at:
                if best_rank <= k:
                    hit_at[k] += 1

    if n_total == 0:
        return {"n_pairs": 0, "note": "No queryable pairs"}

    return {
        "n_pairs": n_total,
        "recall_at_1":  round(hit_at[1]  / n_total, 4),
        "recall_at_3":  round(hit_at[3]  / n_total, 4),
        "recall_at_5":  round(hit_at[5]  / n_total, 4),
        "recall_at_10": round(hit_at[10] / n_total, 4),
        "mrr":          round(float(np.mean(reciprocal_ranks)), 4),
    }


def run_validate(args: argparse.Namespace) -> dict[str, Any]:
    """
    Evaluate all three approaches against the gold-standard pairs.

    Headline metrics use high-confidence pairs only.
    Medium-confidence pairs are reported separately for inspection.

    Acceptance: at least one approach achieves top-10 recall ≥ 0.5 on
    high-confidence pairs.  If none, HALT.
    """
    out = Path(args.output_dir)

    # --- Load approach results ---
    approach_files = {
        "A": out / "approach_a.json",
        "B": out / "approach_b.json",
        "C": out / "approach_c.json",
    }
    missing = [k for k, f in approach_files.items() if not f.exists()]
    if missing:
        print(
            f"[ERROR] Missing approach results: {missing}. "
            "Run --mode approach_{a,b,c} first.",
            file=sys.stderr,
        )
        sys.exit(1)

    approach_data: dict[str, dict[str, Any]] = {}
    for label, fpath in approach_files.items():
        d = json.loads(fpath.read_text())
        approach_data[label] = d["results"]
    print(f"Loaded approach results: {list(approach_data.keys())}")

    # --- Load gold standard ---
    vp_file = Path(_VALIDATION_PAIRS_FILE)
    if not vp_file.exists():
        print("[ERROR] validation_pairs.json not found. "
              "Run build_validation_pairs.py first.", file=sys.stderr)
        sys.exit(1)
    vp = json.loads(vp_file.read_text())
    high_pairs = [p for p in vp["pairs"] if p["confidence"] == "high"]
    med_pairs  = [p for p in vp["pairs"] if p["confidence"] == "medium"]
    low_pairs  = [p for p in vp["pairs"] if p["confidence"] == "low"]
    print(f"Gold standard: {len(high_pairs)} high / {len(med_pairs)} medium / "
          f"{len(low_pairs)} low")

    # --- Compute metrics per approach ---
    results_by_approach: dict[str, dict[str, Any]] = {}
    for label, approach_res in approach_data.items():
        results_by_approach[label] = {
            "high": _recall_and_mrr(high_pairs, approach_res),
            "medium": _recall_and_mrr(med_pairs, approach_res),
            "low": _recall_and_mrr(low_pairs, approach_res),
        }

    # --- Summary table ---
    header = f"{'Approach':>8} | {'top-1':>6} | {'top-3':>6} | {'top-5':>6} | {'top-10':>7} | {'MRR':>6}"
    print(f"\n{'='*60}")
    print(f"  HEADLINE METRICS (high-confidence pairs)")
    print(f"  {header}")
    print(f"  {'-'*60}")
    for label in ["A", "B", "C"]:
        m = results_by_approach[label]["high"]
        print(f"  {'Approach '+label:>8} | "
              f"{m.get('recall_at_1',0):>6.3f} | "
              f"{m.get('recall_at_3',0):>6.3f} | "
              f"{m.get('recall_at_5',0):>6.3f} | "
              f"{m.get('recall_at_10',0):>7.3f} | "
              f"{m.get('mrr',0):>6.3f}")
    print(f"\n  INSPECTION (medium-confidence pairs, n={len(med_pairs)})")
    print(f"  {header}")
    print(f"  {'-'*60}")
    for label in ["A", "B", "C"]:
        m = results_by_approach[label]["medium"]
        print(f"  {'Approach '+label:>8} | "
              f"{m.get('recall_at_1',0):>6.3f} | "
              f"{m.get('recall_at_3',0):>6.3f} | "
              f"{m.get('recall_at_5',0):>6.3f} | "
              f"{m.get('recall_at_10',0):>7.3f} | "
              f"{m.get('mrr',0):>6.3f}")
    print(f"{'='*60}")

    # --- Acceptance check ---
    max_top10 = max(
        results_by_approach[lbl]["high"].get("recall_at_10", 0.0)
        for lbl in results_by_approach
    )
    if max_top10 < 0.5:
        print(
            f"\n[HALT] Acceptance criterion NOT MET: best top-10 recall = "
            f"{max_top10:.3f} < 0.5.  The model has not learned meaningful "
            "cross-database correspondences.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(f"\n[OK] Acceptance criterion met: best top-10 recall = {max_top10:.3f} ≥ 0.5")

    return {
        "mode": "validate",
        "results_by_approach": results_by_approach,
        "acceptance": {
            "criterion": "top-10 recall ≥ 0.5 on high-confidence pairs",
            "best_top10_recall": float(max_top10),
            "passed": True,
        },
        "n_high": len(high_pairs),
        "n_medium": len(med_pairs),
        "n_low": len(low_pairs),
        "smoke": args.smoke,
    }


# ---------------------------------------------------------------------------
# Phase 5 — Discovery component
# ---------------------------------------------------------------------------

_WALS_CHAPTERS: dict[str, str] = {
    "1": "tone", "2": "tone", "3": "tone", "4": "tone", "5": "tone",
    "6": "tone", "7": "tone", "8": "tone", "9": "tone", "10": "tone",
    "11": "tone", "12": "tone", "13": "tone", "14": "tone",
    "15": "morphology", "16": "morphology", "17": "morphology",
    "18": "morphology", "19": "morphology", "20": "morphology",
    "21": "morphology", "22": "morphology", "23": "morphology",
    "24": "morphology", "25": "morphology", "26": "morphology",
    "27": "morphology", "28": "morphology", "29": "morphology",
    "30": "morphology", "31": "morphology", "32": "morphology",
    "33": "morphology", "34": "morphology",
    "35": "nominal-categories", "36": "nominal-categories",
    "37": "nominal-categories", "38": "nominal-categories",
    "39": "nominal-categories", "40": "nominal-categories",
    "41": "nominal-categories", "42": "nominal-categories",
    "43": "nominal-categories", "44": "nominal-categories",
    "45": "nominal-categories", "46": "nominal-categories",
    "47": "nominal-categories", "48": "nominal-categories",
    "49": "nominal-categories", "50": "nominal-categories",
    "51": "nominal-categories", "52": "nominal-categories",
    "53": "nominal-categories", "54": "nominal-categories",
    "55": "nominal-categories",
    "56": "nominal-syntax", "57": "nominal-syntax", "58": "nominal-syntax",
    "59": "nominal-syntax", "60": "nominal-syntax", "61": "nominal-syntax",
    "62": "nominal-syntax", "63": "nominal-syntax", "64": "nominal-syntax",
    "65": "nominal-syntax",
    "66": "verbal-categories", "67": "verbal-categories",
    "68": "verbal-categories", "69": "verbal-categories",
    "70": "verbal-categories", "71": "verbal-categories",
    "72": "verbal-categories", "73": "verbal-categories",
    "74": "verbal-categories", "75": "verbal-categories",
    "76": "verbal-categories", "77": "verbal-categories",
    "78": "verbal-categories", "79": "verbal-categories",
    "80": "verbal-categories",
    "81": "word-order", "82": "word-order", "83": "word-order",
    "84": "word-order", "85": "word-order", "86": "word-order",
    "87": "word-order", "88": "word-order", "89": "word-order",
    "90": "word-order", "91": "word-order", "92": "word-order",
    "93": "word-order", "94": "word-order", "95": "word-order",
    "96": "word-order", "97": "word-order",
    "98": "complex-sentences", "99": "complex-sentences",
    "100": "complex-sentences", "101": "complex-sentences",
    "102": "complex-sentences", "103": "complex-sentences",
    "104": "complex-sentences", "105": "complex-sentences",
    "106": "complex-sentences", "107": "complex-sentences",
    "108": "complex-sentences", "109": "complex-sentences",
    "110": "complex-sentences", "111": "complex-sentences",
    "112": "complex-sentences",
    "113": "lexicon", "114": "lexicon", "115": "lexicon",
    "116": "lexicon", "117": "lexicon", "118": "lexicon",
    "119": "lexicon", "120": "lexicon", "121": "lexicon",
    "122": "lexicon", "123": "lexicon", "124": "lexicon",
    "125": "lexicon", "126": "lexicon", "127": "lexicon",
    "128": "lexicon", "129": "lexicon", "130": "lexicon",
    "131": "lexicon", "132": "lexicon", "133": "lexicon",
    "134": "lexicon", "135": "lexicon", "136": "lexicon",
    "137": "lexicon", "138": "lexicon", "139": "lexicon",
    "140": "lexicon", "141": "lexicon", "142": "lexicon",
    "143": "lexicon", "144": "lexicon",
}


def _wals_chapter(wals_fv: str) -> str:
    """'81A=SOV' → 'word-order'; unknown → 'other'."""
    import re
    m = re.match(r"^(\d+)", wals_fv)
    if m:
        return _WALS_CHAPTERS.get(m.group(1), "other")
    return "other"


def _domain_label(chapter: str) -> str:
    """Map chapter group to RQ3 discovery domain."""
    domain_map = {
        "word-order": "word-order",
        "nominal-categories": "nominal-categories",
        "nominal-syntax": "nominal-syntax",
        "morphology": "morphology",
        "verbal-categories": "verbal-categories",
        "tone": "tone",
        "complex-sentences": "complex-sentences",
        "lexicon": "lexicon",
    }
    return domain_map.get(chapter, "other")


def run_discover(args: argparse.Namespace) -> dict[str, Any]:
    """
    For every WALS featvalue NOT in the gold standard, find the intersection
    of Approach A and B top-5, rank by mean rank across both, and report the
    top 50 novel candidate correspondences.

    Also generates novel_correspondences.md grouping the top 20 by domain.
    """
    out = Path(args.output_dir)

    # Load approach results
    for mode_name in ("approach_a", "approach_b"):
        fpath = out / f"{mode_name}.json"
        if not fpath.exists():
            print(f"[ERROR] {fpath} not found. Run --mode {mode_name} first.",
                  file=sys.stderr)
            sys.exit(1)

    a_data = json.loads((out / "approach_a.json").read_text())["results"]
    b_data = json.loads((out / "approach_b.json").read_text())["results"]

    # Load gold standard to exclude known pairs
    vp = json.loads(Path(_VALIDATION_PAIRS_FILE).read_text())
    known_wals = {p["wals"] for p in vp["pairs"]}

    # Compute smoke limit
    smoke_limit = 30 if args.smoke else None

    candidates: list[dict[str, Any]] = []
    wals_fvs = [w for w in b_data if w not in known_wals]

    if smoke_limit:
        rng = np.random.default_rng(args.seed)
        wals_fvs = list(rng.choice(wals_fvs, min(smoke_limit, len(wals_fvs)),
                                    replace=False))

    for wals_fv in wals_fvs:
        if wals_fv not in a_data or wals_fv not in b_data:
            continue

        top5_a = {r["grambank"]: r for r in a_data[wals_fv][:5]}
        top5_b = {r["grambank"]: r for r in b_data[wals_fv][:5]}
        intersection = set(top5_a.keys()) & set(top5_b.keys())

        for gb_fv in intersection:
            rank_a = top5_a[gb_fv]["rank"]
            sim_a  = top5_a[gb_fv]["sim"]
            rank_b = top5_b[gb_fv]["rank"]
            sim_b  = top5_b[gb_fv]["sim"]
            mean_rank = (rank_a + rank_b) / 2.0
            chapter = _wals_chapter(wals_fv)
            domain = _domain_label(chapter)

            candidates.append({
                "wals": wals_fv,
                "grambank": gb_fv,
                "rank_a": rank_a,
                "rank_b": rank_b,
                "sim_a": round(sim_a, 4),
                "sim_b": round(sim_b, 4),
                "mean_rank": mean_rank,
                "wals_chapter": chapter,
                "domain": domain,
            })

    # Sort by mean_rank (lower = better)
    candidates.sort(key=lambda x: x["mean_rank"])
    top50 = candidates[:50]

    # Save JSON
    novel_json_path = out / "novel_correspondences.json"
    novel_json_path.write_text(
        json.dumps({"candidates": top50, "total_novel_candidates": len(candidates)},
                   indent=2, ensure_ascii=False)
    )

    # Generate Markdown summary for top 20
    top20 = top50[:20]

    # Group by domain
    from collections import defaultdict
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in top20:
        by_domain[c["domain"]].append(c)

    md_lines = [
        "# Novel WALS ↔ Grambank Correspondences — Top 20",
        "",
        "Candidates are WALS featvalues absent from the gold standard whose "
        "Approach A and Approach B top-5 lists share at least one Grambank "
        "featvalue.  Ranked by mean rank across both approaches.",
        "",
        "_Require expert validation before use as research claims._",
        "",
    ]
    rank_counter = 1
    for domain in sorted(by_domain.keys()):
        domain_items = sorted(by_domain[domain], key=lambda x: x["mean_rank"])
        md_lines.append(f"## {domain.replace('-', ' ').title()}")
        md_lines.append("")
        md_lines.append(
            "| # | WALS | Grambank | rank_A | rank_B | sim_A | sim_B |"
        )
        md_lines.append("|---|------|----------|--------|--------|-------|-------|")
        for c in domain_items:
            md_lines.append(
                f"| {rank_counter} | `{c['wals']}` | `{c['grambank']}` | "
                f"{c['rank_a']} | {c['rank_b']} | "
                f"{c['sim_a']:.3f} | {c['sim_b']:.3f} |"
            )
            rank_counter += 1
        md_lines.append("")

    md_path = out / "novel_correspondences.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    n_queried = len(wals_fvs)
    n_with_intersection = len({c["wals"] for c in candidates})

    print(f"  WALS featvalues queried (not in gold std): {n_queried}")
    print(f"  With A∩B top-5 intersection:              {n_with_intersection}")
    print(f"  Total candidate pairs:                    {len(candidates)}")
    print(f"  Top 50 saved to novel_correspondences.json")
    print(f"  Top 20 Markdown saved to novel_correspondences.md")

    return {
        "mode": "discover",
        "n_queried_wals_fvs": n_queried,
        "n_with_intersection": n_with_intersection,
        "total_candidates": len(candidates),
        "top50": top50,
        "smoke": args.smoke,
    }



# ---------------------------------------------------------------------------
# Phase 5 — Tier-1 discovery: B∩C intersection
# ---------------------------------------------------------------------------

def run_discover_bc(args: argparse.Namespace) -> dict[str, Any]:
    """
    Tier-1 discovery: WALS featvalues absent from the gold standard whose
    Approach B top-5 AND Approach C top-5 share at least one Grambank
    featvalue.  The B∩C intersection requires two independent methods to
    agree, giving higher-confidence candidates than either alone.

    Outputs novel_correspondences_tier1.json and updates the two-tier
    novel_correspondences.md (combined with Tier-2 from discover_b.json).
    """
    out = Path(args.output_dir)

    for mode_name in ("approach_b", "approach_c"):
        fpath = out / f"{mode_name}.json"
        if not fpath.exists():
            print(f"[ERROR] {fpath} not found. Run --mode {mode_name} first.",
                  file=sys.stderr)
            sys.exit(1)

    b_data = json.loads((out / "approach_b.json").read_text())["results"]
    c_data = json.loads((out / "approach_c.json").read_text())["results"]

    vp = json.loads(Path(_VALIDATION_PAIRS_FILE).read_text())
    known_wals = {p["wals"] for p in vp["pairs"]}

    smoke_limit = 30 if args.smoke else None
    wals_fvs = [w for w in b_data if w not in known_wals and w in c_data]
    if smoke_limit:
        rng = np.random.default_rng(args.seed)
        wals_fvs = list(rng.choice(wals_fvs, min(smoke_limit, len(wals_fvs)),
                                    replace=False))

    candidates: list[dict[str, Any]] = []
    for wals_fv in wals_fvs:
        top5_b = {r["grambank"]: r for r in b_data[wals_fv][:5]}
        top5_c = {r["grambank"]: r for r in c_data[wals_fv][:5]}
        intersection = set(top5_b.keys()) & set(top5_c.keys())

        for gb_fv in intersection:
            rank_b = top5_b[gb_fv]["rank"]
            sim_b  = top5_b[gb_fv]["sim"]
            rank_c = top5_c[gb_fv]["rank"]
            sim_c  = top5_c[gb_fv]["sim"]
            mean_rank = (rank_b + rank_c) / 2.0
            chapter = _wals_chapter(wals_fv)
            candidates.append({
                "wals": wals_fv,
                "grambank": gb_fv,
                "rank_b": rank_b,
                "rank_c": rank_c,
                "sim_b": round(sim_b, 4),
                "sim_c": round(sim_c, 4),
                "mean_rank": mean_rank,
                "wals_chapter": chapter,
                "domain": _domain_label(chapter),
                "tier": 1,
            })

    candidates.sort(key=lambda x: x["mean_rank"])
    top50 = candidates[:50]

    json_path = out / "novel_correspondences_tier1.json"
    json_path.write_text(
        json.dumps({"tier": 1, "method": "B∩C",
                    "candidates": top50,
                    "total_candidates": len(candidates)},
                   indent=2, ensure_ascii=False)
    )

    n_queried = len(wals_fvs)
    n_with = len({c["wals"] for c in candidates})
    print(f"  [Tier 1 B∩C] WALS queried: {n_queried}  "
          f"with intersection: {n_with}  total pairs: {len(candidates)}")

    _write_two_tier_md(out)

    return {
        "mode": "discover_bc",
        "tier": 1,
        "method": "B∩C",
        "n_queried_wals_fvs": n_queried,
        "n_with_intersection": n_with,
        "total_candidates": len(candidates),
        "top50": top50,
        "smoke": args.smoke,
    }


# ---------------------------------------------------------------------------
# Phase 5 — Tier-2 discovery: pure-B top-5
# ---------------------------------------------------------------------------

def run_discover_b(args: argparse.Namespace) -> dict[str, Any]:
    """
    Tier-2 discovery: WALS featvalues absent from the gold standard with at
    least one hit in Approach B's top-5.  Uses only Approach B (the headline
    language-profile method) without requiring a second method to agree.
    Broader coverage than Tier 1, lower confidence — suitable for exploratory
    hypothesis generation.

    Outputs novel_correspondences_tier2.json and updates the two-tier
    novel_correspondences.md (combined with Tier-1 from discover_bc.json).
    """
    out = Path(args.output_dir)

    fpath = out / "approach_b.json"
    if not fpath.exists():
        print(f"[ERROR] {fpath} not found. Run --mode approach_b first.",
              file=sys.stderr)
        sys.exit(1)

    b_data = json.loads(fpath.read_text())["results"]

    vp = json.loads(Path(_VALIDATION_PAIRS_FILE).read_text())
    known_wals = {p["wals"] for p in vp["pairs"]}

    smoke_limit = 30 if args.smoke else None
    wals_fvs = [w for w in b_data if w not in known_wals]
    if smoke_limit:
        rng = np.random.default_rng(args.seed)
        wals_fvs = list(rng.choice(wals_fvs, min(smoke_limit, len(wals_fvs)),
                                    replace=False))

    candidates: list[dict[str, Any]] = []
    for wals_fv in wals_fvs:
        top5_b = b_data[wals_fv][:5]
        for entry in top5_b:
            chapter = _wals_chapter(wals_fv)
            candidates.append({
                "wals": wals_fv,
                "grambank": entry["grambank"],
                "rank_b": entry["rank"],
                "sim_b": round(entry["sim"], 4),
                "mean_rank": float(entry["rank"]),
                "wals_chapter": chapter,
                "domain": _domain_label(chapter),
                "tier": 2,
            })

    candidates.sort(key=lambda x: (x["mean_rank"], -x["sim_b"]))
    top50 = candidates[:50]

    json_path = out / "novel_correspondences_tier2.json"
    json_path.write_text(
        json.dumps({"tier": 2, "method": "pure-B",
                    "candidates": top50,
                    "total_candidates": len(candidates)},
                   indent=2, ensure_ascii=False)
    )

    n_queried = len(wals_fvs)
    print(f"  [Tier 2 pure-B] WALS queried: {n_queried}  "
          f"total top-5 entries: {len(candidates)}")

    _write_two_tier_md(out)

    return {
        "mode": "discover_b",
        "tier": 2,
        "method": "pure-B",
        "n_queried_wals_fvs": n_queried,
        "total_candidates": len(candidates),
        "top50": top50,
        "smoke": args.smoke,
    }


def _write_two_tier_md(out: Path) -> None:
    """
    Write a fresh two-tier novel_correspondences.md combining Tier-1 (B∩C)
    and Tier-2 (pure-B).  Both tier JSON files must exist; missing tiers are
    silently skipped so each discover mode can call this independently.
    """
    from collections import defaultdict

    def _tier_section(title: str, candidates: list[dict], tier: int) -> list[str]:
        top20 = candidates[:20]
        by_domain: dict[str, list] = defaultdict(list)
        for c in top20:
            by_domain[c["domain"]].append(c)

        lines = [f"## {title}", ""]
        rank_counter = 1
        for domain in sorted(by_domain.keys()):
            items = sorted(by_domain[domain], key=lambda x: x["mean_rank"])
            lines.append(f"### {domain.replace('-', ' ').title()}")
            lines.append("")
            if tier == 1:
                lines.append("| # | WALS | Grambank | rank_B | rank_C | sim_B | sim_C |")
                lines.append("|---|------|----------|--------|--------|-------|-------|")
            else:
                lines.append("| # | WALS | Grambank | rank_B | sim_B |")
                lines.append("|---|------|----------|--------|-------|")
            for c in items:
                if tier == 1:
                    row = (f"| {rank_counter} | `{c['wals']}` | `{c['grambank']}` | "
                           f"{c.get('rank_b','—')} | {c.get('rank_c','—')} | "
                           f"{c.get('sim_b',0.0):.3f} | {c.get('sim_c',0.0):.3f} |")
                else:
                    row = (f"| {rank_counter} | `{c['wals']}` | `{c['grambank']}` | "
                           f"{c.get('rank_b','—')} | {c.get('sim_b',0.0):.3f} |")
                lines.append(row)
                rank_counter += 1
            lines.append("")
        return lines

    md = [
        "# Novel WALS ↔ Grambank Correspondences — Two-Tier Discovery",
        "",
        "Candidates are WALS feature-values **absent from the gold standard** "
        "that appear in the top-5 of one or more alignment approaches.",
        "",
        "**Tier 1 (B∩C, higher confidence):** Both the language-profile method "
        "(Approach B) and the CCA-projection method (Approach C) independently "
        "place the same Grambank feature-value in their top-5 lists.  "
        "The two methods use different mathematical machinery, so agreement "
        "between them raises confidence.",
        "",
        "**Tier 2 (pure-B, exploratory):** Only Approach B (headline method) "
        "lists the Grambank feature-value in its top-5.  Broader coverage "
        "than Tier 1; requires expert validation before use as research claims.",
        "",
        "_All candidates require expert validation before use as research claims._",
        "",
    ]

    t1_path = out / "novel_correspondences_tier1.json"
    t2_path = out / "novel_correspondences_tier2.json"

    if t1_path.exists():
        t1 = json.loads(t1_path.read_text())
        md += _tier_section("Tier 1 — B∩C Agreement (Top 20)", t1["candidates"], 1)

    if t2_path.exists():
        t2 = json.loads(t2_path.read_text())
        md += _tier_section("Tier 2 — Pure-B Exploratory (Top 20)", t2["candidates"], 2)

    md_path = out / "novel_correspondences.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"  Written two-tier novel_correspondences.md")


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
    "discover_bc": run_discover_bc,
    "discover_b": run_discover_b,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-database feature alignment (RQ3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=list(_MODES.keys()), required=True,
        help="Which analysis phase to run.",
    )
    scale = p.add_mutually_exclusive_group()
    scale.add_argument("--smoke", action="store_true",
                       help="Run a tiny subset to validate wiring.")
    scale.add_argument("--full", action="store_true",
                       help="Run the production-scale workload (default).")
    p.add_argument("--resume", action="store_true",
                   help="Skip if output already exists.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_wals", default=_DEFAULT_CKPT_WALS)
    p.add_argument("--checkpoint_grambank", default=_DEFAULT_CKPT_GRAMBANK)
    p.add_argument("--output_dir", default=_DEFAULT_OUTPUT_DIR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    output_file = out / f"{args.mode}.json"
    if args.resume and output_file.exists():
        print(f"[resume] {output_file} already exists — skipping.")
        return

    print(f"[cross_database_alignment] mode={args.mode}  "
          f"{'smoke' if args.smoke else 'full'}  seed={args.seed}")

    t0 = time.perf_counter()
    result = _MODES[args.mode](args)
    elapsed = time.perf_counter() - t0

    result["elapsed_seconds"] = round(elapsed, 3)

    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved: {output_file}  ({elapsed:.1f}s)")

    config = vars(args).copy()
    config["elapsed_seconds"] = result["elapsed_seconds"]
    (out / f"config_{args.mode}.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
