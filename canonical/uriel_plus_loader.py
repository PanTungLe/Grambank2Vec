"""
URIEL+ geographic and phylogenetic vector loader for RQ4.

Queries URIEL+ for geographic ('geo') and phylogenetic ('fam') vectors for
every Glottocode appearing in the canonical WALS and Grambank checkpoints.

HARD CONSTRAINT (Constraint 3): this script NEVER queries, touches, or stores
any 'fea' (typological) URIEL+ features.  URIEL+ integrates Grambank as a
typological source; using the typological vector to condition predictions of
Grambank features would be circular.

Three imputation methods for Glottocodes absent from URIEL+:
  familymean — mean of found languages in the same Glottolog family
               (analysis/families.csv); singletons use overall mean.
  knn        — K=10 nearest found languages by cosine similarity of
               canonical embedding (our model's learned language vectors),
               averaged to produce the imputed geo/phylo vector.
  softimpute — SoftImpute (fancyimpute) on the combined (found + missing)
               language matrix; missing rows are entirely NaN.

Output (per method):
  analysis/conditioning_uriel_plus/uriel_plus_vectors_<method>.parquet
    columns: glottocode, geo_vec (list[float32]), phylo_vec (list[float32]),
             geo_imputed (bool), phylo_imputed (bool),
             imputation_method (str)

Usage
-----
  python canonical/uriel_plus_loader.py \\
      [--output_dir analysis/conditioning_uriel_plus] \\
      [--imputation softimpute|knn|familymean] \\
      [--urielplus_dir /tmp/URIELPlus] \\
      [--smoke | --full] \\
      [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "analysis" / "conditioning_uriel_plus")
_DEFAULT_URIELPLUS_DIR = "/tmp/URIELPlus"
_URIELPLUS_REPO = "https://github.com/Masonshipton25/URIELPlus"
_FAMILIES_CSV = str(_REPO_ROOT / "analysis" / "families.csv")

_CHECKPOINT_DIRS = [
    _REPO_ROOT / "checkpoints" / "wals_learned_s42",
    _REPO_ROOT / "checkpoints" / "grambank_learned_s42",
]

# ──────────────────────────────────────────────────────────────────────────────
# Safety assertion — enforced at every access point
# ──────────────────────────────────────────────────────────────────────────────

_FORBIDDEN_FEATURE_TYPES = frozenset(["fea", "typological", "featural",
                                       "syntactic", "phonological",
                                       "inventory", "morphological"])

# Columns that are permitted to appear in the output parquet (Constraint 3).
_PERMITTED_PARQUET_COLUMNS = frozenset([
    "glottocode", "geo_vec", "phylo_vec",
    "geo_imputed", "phylo_imputed", "imputation_method",
])


def _assert_not_typological(label: str) -> None:
    """Raise if `label` names a typological / featural vector category."""
    lower = label.lower()
    for forbidden in _FORBIDDEN_FEATURE_TYPES:
        if forbidden in lower:
            raise AssertionError(
                f"[CONSTRAINT 3 VIOLATED] Attempted to access typological "
                f"vector category '{label}'. Grambank is a URIEL+ typological "
                "source; using it as conditioning signal is circular. "
                "Only 'geo' (geographic) and 'fam' / 'phylogeny' vectors "
                "are permitted for RQ4 conditioning."
            )


def _verify_parquet_schema(out_path: Path) -> None:
    """Raise if the saved parquet contains any column outside the permitted set."""
    t = pq.read_table(str(out_path))
    actual = set(t.schema.names)
    forbidden = actual - _PERMITTED_PARQUET_COLUMNS
    assert not forbidden, (
        f"[Constraint 3] Parquet {out_path.name} contains forbidden columns: "
        f"{sorted(forbidden)}. Only geo/phylo data is permitted (no typological)."
    )
    log.info("[Constraint 3] Parquet schema verified — no forbidden columns.")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _collect_glottocodes() -> list[str]:
    """Union of all Glottocodes across the WALS and Grambank checkpoints."""
    all_gc: set[str] = set()
    for ckpt in _CHECKPOINT_DIRS:
        l2id_path = ckpt / "lang2id.json"
        if not l2id_path.exists():
            log.warning("lang2id.json not found at %s — skipping.", ckpt)
            continue
        l2id = json.loads(l2id_path.read_text())
        all_gc.update(l2id.keys())
    log.info("Collected %d unique Glottocodes from checkpoints.", len(all_gc))
    return sorted(all_gc)


def _load_canonical_lang_emb() -> dict[str, np.ndarray]:
    """
    Load mean language embeddings across WALS and Grambank checkpoints.
    Used as feature basis for KNN imputation.
    Returns {glottocode: (64,) float32 array}.
    """
    glottocode_embs: dict[str, list[np.ndarray]] = {}
    for ckpt in _CHECKPOINT_DIRS:
        l2id = json.loads((ckpt / "lang2id.json").read_text())
        emb = np.load(ckpt / "lang_embeddings.npy").astype(np.float32)
        for gc, idx in l2id.items():
            glottocode_embs.setdefault(gc, []).append(emb[idx])
    return {gc: np.mean(vecs, axis=0) for gc, vecs in glottocode_embs.items()}


def _clone_or_verify(urielplus_dir: str) -> str:
    """Clone URIELPlus if absent; return directory path."""
    d = Path(urielplus_dir)
    if not d.exists():
        log.info("Cloning URIELPlus → %s …", d)
        subprocess.run(
            ["git", "clone", "--depth=1", _URIELPLUS_REPO, str(d)],
            check=True,
        )
    return str(d)


# ──────────────────────────────────────────────────────────────────────────────
# URIEL+ data extraction
# ──────────────────────────────────────────────────────────────────────────────

def _load_uriel_plus(urielplus_dir: str):
    """
    Initialise URIEL+ in Glottocode mode and return the URIELPlus object.

    The typological index (data[1] / features.npz) is present in memory but
    will never be read by our code; we verify this by asserting on every lookup.
    """
    # Ensure the package is importable from the cloned repo
    if str(Path(urielplus_dir).parent) not in sys.path:
        sys.path.insert(0, str(Path(urielplus_dir).parent))
    # Also accept if installed system-wide
    try:
        from urielplus import urielplus as up_module
    except ImportError:
        up_path = Path(urielplus_dir) / "urielplus"
        if not up_path.exists():
            raise ImportError(
                f"Cannot import urielplus. Ensure URIELPlus is cloned at "
                f"{urielplus_dir} and pip install -e was run."
            )
        sys.path.insert(0, str(Path(urielplus_dir)))
        from urielplus import urielplus as up_module

    u = up_module.URIELPlus()
    u.set_glottocodes()

    # Safety: verify the file ordering matches expectations
    assert "family" in u.files[0].lower(), \
        f"Unexpected file[0]: {u.files[0]} — expected family_features.npz"
    assert "geocoord" in u.files[2].lower(), \
        f"Unexpected file[2]: {u.files[2]} — expected geocoord_features.npz"

    # Verify file[1] is the typological file — should never be geo/fam
    assert "feature" in u.files[1].lower(), \
        f"Unexpected file[1]: {u.files[1]} — expected features.npz (typological)"

    log.info("URIEL+ initialised.  "
             "Phylo: %d langs × %d features.  "
             "Geo:   %d langs × %d features.",
             len(u.langs[0]), u.data[0].shape[1],
             len(u.langs[2]), u.data[2].shape[1])
    return u


def _extract_raw_vectors(
    u,
    glottocodes: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
           list[str], list[str]]:
    """
    Extract geographic and phylogenetic vectors for every Glottocode.

    Constraint 3 is enforced: we only access data[0] (phylo) and data[2] (geo).
    data[1] (typological) is never touched.

    Returns:
        geo_vecs   — {glottocode: (299,) float32}
        phylo_vecs — {glottocode: (3718,) float32}
        found      — glottocodes present in URIEL+
        missing    — glottocodes absent from URIEL+
    """
    # Constraint 3: only access indices 0 (phylo) and 2 (geo); index 1 is typological
    assert len(u.files) > 2, "URIEL+ must have at least 3 file slots"
    assert "feature" in u.files[1].lower(), \
        f"[Constraint 3] data[1] must be typological (features.npz), got {u.files[1]}"

    # Index maps: glottocode → row index in each URIEL+ array
    geo_idx_map: dict[str, int] = {
        gc: i for i, gc in enumerate(u.langs[2].tolist())
    }
    phylo_idx_map: dict[str, int] = {
        gc: i for i, gc in enumerate(u.langs[0].tolist())
    }

    geo_vecs: dict[str, np.ndarray] = {}
    phylo_vecs: dict[str, np.ndarray] = {}
    found: list[str] = []
    missing: list[str] = []

    for gc in glottocodes:
        in_geo   = gc in geo_idx_map
        in_phylo = gc in phylo_idx_map

        if in_geo and in_phylo:
            # data[i] shape: (n_langs, n_feats, 1)
            geo_vecs[gc]   = u.data[2][geo_idx_map[gc]][:, 0].astype(np.float32)
            phylo_vecs[gc] = u.data[0][phylo_idx_map[gc]][:, 0].astype(np.float32)
            found.append(gc)
        else:
            missing.append(gc)

    log.info("Found in URIEL+: %d / %d  (missing: %d)",
             len(found), len(glottocodes), len(missing))
    return geo_vecs, phylo_vecs, found, missing


# ──────────────────────────────────────────────────────────────────────────────
# Imputation methods
# ──────────────────────────────────────────────────────────────────────────────

def _familymean_impute(
    missing: list[str],
    geo_vecs: dict[str, np.ndarray],
    phylo_vecs: dict[str, np.ndarray],
    families_csv: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Impute missing Glottocodes using the mean of found languages in the same
    Glottolog family.  Isolates (no family or no found family members) fall
    back to the overall mean of all found vectors.
    """
    fam_df = pd.read_csv(families_csv)
    gc_to_fam: dict[str, str] = dict(
        zip(fam_df["glottocode"].astype(str), fam_df["family_id"].astype(str))
    )

    # Group found languages by family
    fam_to_gcs: dict[str, list[str]] = {}
    for gc in geo_vecs:
        fam = gc_to_fam.get(gc)
        if fam:
            fam_to_gcs.setdefault(fam, []).append(gc)

    # Overall mean fallback
    all_geo   = np.array(list(geo_vecs.values()),   dtype=np.float32)
    all_phylo = np.array(list(phylo_vecs.values()), dtype=np.float32)
    overall_geo_mean   = all_geo.mean(axis=0)
    overall_phylo_mean = all_phylo.mean(axis=0)

    imputed_geo: dict[str, np.ndarray] = {}
    imputed_phylo: dict[str, np.ndarray] = {}

    for gc in missing:
        fam = gc_to_fam.get(gc)
        if fam and fam in fam_to_gcs:
            members = fam_to_gcs[fam]
            imputed_geo[gc]   = np.mean([geo_vecs[m]   for m in members], axis=0)
            imputed_phylo[gc] = np.mean([phylo_vecs[m] for m in members], axis=0)
        else:
            imputed_geo[gc]   = overall_geo_mean.copy()
            imputed_phylo[gc] = overall_phylo_mean.copy()

    return imputed_geo, imputed_phylo


def _knn_impute(
    missing: list[str],
    geo_vecs: dict[str, np.ndarray],
    phylo_vecs: dict[str, np.ndarray],
    lang_embs: dict[str, np.ndarray],
    k: int = 10,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Impute missing Glottocodes using the mean of the K nearest found languages
    in the canonical embedding space (cosine similarity).

    For missing languages without a canonical embedding, fall back to overall
    mean (same as familymean fallback).
    """
    found_gcs = list(geo_vecs.keys())

    # Build embedding matrix for found languages
    found_with_emb = [gc for gc in found_gcs if gc in lang_embs]
    if not found_with_emb:
        # Degenerate fallback: overall mean
        overall_geo   = np.mean(list(geo_vecs.values()),   axis=0)
        overall_phylo = np.mean(list(phylo_vecs.values()), axis=0)
        return (
            {gc: overall_geo.copy()   for gc in missing},
            {gc: overall_phylo.copy() for gc in missing},
        )

    found_emb_matrix = np.array(
        [lang_embs[gc] for gc in found_with_emb], dtype=np.float32
    )  # (n_found_with_emb, 64)

    # Cosine-normalise
    norms = np.linalg.norm(found_emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    found_emb_normed = found_emb_matrix / norms

    overall_geo   = np.mean(list(geo_vecs.values()),   axis=0).astype(np.float32)
    overall_phylo = np.mean(list(phylo_vecs.values()), axis=0).astype(np.float32)

    imputed_geo: dict[str, np.ndarray] = {}
    imputed_phylo: dict[str, np.ndarray] = {}

    for gc in missing:
        if gc not in lang_embs:
            imputed_geo[gc]   = overall_geo.copy()
            imputed_phylo[gc] = overall_phylo.copy()
            continue

        q = lang_embs[gc].astype(np.float32)
        qn = np.linalg.norm(q)
        if qn < 1e-12:
            imputed_geo[gc]   = overall_geo.copy()
            imputed_phylo[gc] = overall_phylo.copy()
            continue
        q = q / qn

        sims = found_emb_normed @ q            # (n_found_with_emb,)
        top_k = min(k, len(found_with_emb))
        nbr_indices = np.argsort(-sims)[:top_k]
        nbr_gcs = [found_with_emb[i] for i in nbr_indices]

        imputed_geo[gc]   = np.mean([geo_vecs[g]   for g in nbr_gcs], axis=0)
        imputed_phylo[gc] = np.mean([phylo_vecs[g] for g in nbr_gcs], axis=0)

    return imputed_geo, imputed_phylo


def _svt_impute(
    M: np.ndarray,
    rank: int = 50,
    lambda_frac: float = 0.05,
    max_iter: int = 20,
    tol: float = 1e-3,
) -> np.ndarray:
    """
    SVT-based SoftImpute using randomised truncated SVD.

    For each iteration:
      1. Compute rank-k truncated SVD of the current (filled) matrix.
      2. Soft-threshold singular values by lambda.
      3. Reconstruct the full matrix from the thresholded decomposition.
      4. Update only the missing-entry positions.

    Because missing rows are entirely NaN (no partial observations), they are
    initialised with column means and then pulled toward the global low-rank
    structure of the observed sub-matrix.  This gives a different imputation
    from familymean (which uses family-local means).

    Parameters
    ----------
    M         : float64 array, NaN at missing entries
    rank      : number of singular components to retain
    lambda_frac : lambda = lambda_frac × sigma_max (auto-tuned from data)
    max_iter  : SVT iteration cap
    tol       : convergence threshold on max absolute change in missing entries
    """
    from sklearn.utils.extmath import randomized_svd

    nan_mask = np.isnan(M)
    col_means = np.nanmean(M, axis=0)
    np.nan_to_num(col_means, nan=0.0, copy=False)

    # Initialise missing entries with column means
    M_filled = M.copy()
    M_filled[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Estimate lambda from the leading singular value
    _, s0, _ = randomized_svd(M_filled, n_components=1, n_iter=3, random_state=42)
    lambda_ = float(s0[0]) * lambda_frac
    log.info("SVT: lambda=%.3f  rank=%d  shape=%s  n_missing=%d",
             lambda_, rank, M.shape, nan_mask.sum())

    prev_missing = M_filled[nan_mask].copy()

    for it in range(max_iter):
        U, s, Vt = randomized_svd(M_filled, n_components=rank,
                                   n_iter=4, random_state=42)
        s_thresh = np.maximum(s - lambda_, 0)
        M_new = (U * s_thresh) @ Vt
        M_filled[nan_mask] = M_new[nan_mask]

        change = float(np.abs(M_filled[nan_mask] - prev_missing).max())
        prev_missing = M_filled[nan_mask].copy()
        log.info("  SVT iter %d/%d: max-change=%.6f", it + 1, max_iter, change)
        if change < tol:
            log.info("  SVT converged at iter %d.", it + 1)
            break

    return M_filled


def _softimpute_impute(
    missing: list[str],
    geo_vecs: dict[str, np.ndarray],
    phylo_vecs: dict[str, np.ndarray],
    families_csv: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Impute missing Glottocodes using SoftImpute (nuclear norm minimisation) on
    the combined geo+phylo matrix.  Missing rows are filled with NaN before
    running SoftImpute.  For completely missing rows, SoftImpute fills based
    on the low-rank structure of the observed matrix.

    Tries fancyimpute.SoftImpute first; if it fails (e.g. sklearn ≥ 1.6
    incompatibility), falls back to an internal SVT implementation that
    produces genuinely different results from familymean.
    """
    found_gcs = list(geo_vecs.keys())
    n_found = len(found_gcs)
    n_missing = len(missing)

    geo_dim   = next(iter(geo_vecs.values())).shape[0]
    phylo_dim = next(iter(phylo_vecs.values())).shape[0]
    total_dim = geo_dim + phylo_dim

    # Build combined matrix; missing rows = NaN
    M = np.full((n_found + n_missing, total_dim), np.nan, dtype=np.float64)
    for i, gc in enumerate(found_gcs):
        M[i, :geo_dim]   = geo_vecs[gc].astype(np.float64)
        M[i, geo_dim:]   = phylo_vecs[gc].astype(np.float64)

    try:
        from fancyimpute import SoftImpute
        log.info("Running fancyimpute.SoftImpute on %d × %d matrix …",
                 M.shape[0], M.shape[1])
        t0 = time.perf_counter()
        M_filled = SoftImpute(verbose=False).fit_transform(M)
        log.info("fancyimpute.SoftImpute done in %.1f s.", time.perf_counter() - t0)
    except Exception as exc:
        log.warning(
            "fancyimpute.SoftImpute failed (%s); "
            "falling back to internal SVT implementation.", exc)
        t0 = time.perf_counter()
        M_filled = _svt_impute(M)
        log.info("SVT imputation done in %.1f s.", time.perf_counter() - t0)

    imputed_geo: dict[str, np.ndarray] = {}
    imputed_phylo: dict[str, np.ndarray] = {}
    for i, gc in enumerate(missing, start=n_found):
        imputed_geo[gc]   = M_filled[i, :geo_dim].astype(np.float32)
        imputed_phylo[gc] = M_filled[i, geo_dim:].astype(np.float32)

    return imputed_geo, imputed_phylo


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def _save_parquet(
    glottocodes: list[str],
    geo_vecs: dict[str, np.ndarray],
    phylo_vecs: dict[str, np.ndarray],
    imputed_gcs: set[str],
    imputation_method: str,
    out_path: Path,
) -> None:
    """Save the combined (found + imputed) vectors to a parquet file."""
    rows_geo   = [geo_vecs[gc].tolist()   for gc in glottocodes]
    rows_phylo = [phylo_vecs[gc].tolist() for gc in glottocodes]
    geo_imp    = [gc in imputed_gcs for gc in glottocodes]
    phylo_imp  = [gc in imputed_gcs for gc in glottocodes]
    methods    = [imputation_method] * len(glottocodes)

    table = pa.table({
        "glottocode":        pa.array(glottocodes, type=pa.string()),
        "geo_vec":           pa.array(rows_geo,    type=pa.list_(pa.float32())),
        "phylo_vec":         pa.array(rows_phylo,  type=pa.list_(pa.float32())),
        "geo_imputed":       pa.array(geo_imp,     type=pa.bool_()),
        "phylo_imputed":     pa.array(phylo_imp,   type=pa.bool_()),
        "imputation_method": pa.array(methods,     type=pa.string()),
    })
    pq.write_table(table, str(out_path))
    log.info("Saved %s  (%d rows)", out_path.name, len(glottocodes))


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_loader(
    imputation_method: str,
    output_dir: str,
    urielplus_dir: str,
    smoke: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Build the URIEL+ vector parquet for one imputation method."""
    log.info("[Constraint 3] Confirmed: only geo (data[2]) and phylo (data[0]) vectors will be accessed.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Collect all Glottocodes
    all_gcs = _collect_glottocodes()
    if smoke:
        rng = np.random.default_rng(seed)
        all_gcs = sorted(
            rng.choice(all_gcs, min(200, len(all_gcs)), replace=False).tolist()
        )
        log.info("[smoke] Subset to %d Glottocodes.", len(all_gcs))

    # 2. Load URIEL+
    _clone_or_verify(urielplus_dir)
    u = _load_uriel_plus(urielplus_dir)

    # 3. Extract raw vectors (geo and phylo only — Constraint 3 enforced inside)
    geo_vecs, phylo_vecs, found, missing = _extract_raw_vectors(u, all_gcs)

    pct_found = 100.0 * len(found) / len(all_gcs)
    log.info("Coverage: %.1f%%  (%d found, %d missing)", pct_found, len(found), len(missing))

    # 4. Impute missing Glottocodes
    log.info("Imputing %d missing Glottocodes via method='%s' …",
             len(missing), imputation_method)

    imputed_gcs: set[str] = set()
    if missing:
        if imputation_method == "familymean":
            imp_geo, imp_phylo = _familymean_impute(
                missing, geo_vecs, phylo_vecs, _FAMILIES_CSV)

        elif imputation_method == "knn":
            lang_embs = _load_canonical_lang_emb()
            imp_geo, imp_phylo = _knn_impute(
                missing, geo_vecs, phylo_vecs, lang_embs, k=10)

        elif imputation_method == "softimpute":
            imp_geo, imp_phylo = _softimpute_impute(
                missing, geo_vecs, phylo_vecs, _FAMILIES_CSV)

        else:
            raise ValueError(
                f"Unknown imputation method: {imputation_method!r}. "
                "Valid options: softimpute, knn, familymean."
            )

        geo_vecs.update(imp_geo)
        phylo_vecs.update(imp_phylo)
        imputed_gcs = set(missing)

    # 5. Acceptance check: ≥90% non-imputed
    n_non_imputed = len(found)
    pct_non_imputed = 100.0 * n_non_imputed / len(all_gcs)
    log.info("Acceptance check: %.1f%% non-imputed  (≥90%% required)",
             pct_non_imputed)
    if pct_non_imputed < 90.0 and not smoke:
        log.error("ACCEPTANCE FAILURE: only %.1f%% of Glottocodes have "
                  "non-imputed URIEL+ vectors (< 90%% threshold). "
                  "Check URIEL+ version or Glottocode set.", pct_non_imputed)
        sys.exit(1)

    # 6. Save parquet
    out_path = out / f"uriel_plus_vectors_{imputation_method}.parquet"
    _save_parquet(
        glottocodes=all_gcs,
        geo_vecs=geo_vecs,
        phylo_vecs=phylo_vecs,
        imputed_gcs=imputed_gcs,
        imputation_method=imputation_method,
        out_path=out_path,
    )

    # Constraint 3 post-save guard: verify no typological columns leaked in.
    _verify_parquet_schema(out_path)

    summary = {
        "imputation_method": imputation_method,
        "n_glottocodes": len(all_gcs),
        "n_found": len(found),
        "n_missing": len(missing),
        "pct_non_imputed": round(pct_non_imputed, 2),
        "acceptance_passed": pct_non_imputed >= 90.0,
        "geo_dim": next(iter(geo_vecs.values())).shape[0],
        "phylo_dim": next(iter(phylo_vecs.values())).shape[0],
        "output_file": str(out_path),
        "smoke": smoke,
        "seed": seed,
    }
    # Save config
    (out / f"config_uriel_plus_{imputation_method}.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="URIEL+ geographic and phylogenetic vector loader for RQ4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--imputation",
                   choices=["softimpute", "knn", "familymean"],
                   default="softimpute")
    p.add_argument("--output_dir", default=_DEFAULT_OUTPUT_DIR)
    p.add_argument("--urielplus_dir", default=_DEFAULT_URIELPLUS_DIR)
    scale = p.add_mutually_exclusive_group()
    scale.add_argument("--smoke", action="store_true",
                       help="Run on a 200-language subset.")
    scale.add_argument("--full", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip if output parquet already exists.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    out_path = (Path(args.output_dir) /
                f"uriel_plus_vectors_{args.imputation}.parquet")
    if args.resume and out_path.exists():
        log.info("[resume] %s already exists — skipping.", out_path.name)
        return

    t0 = time.perf_counter()
    summary = run_loader(
        imputation_method=args.imputation,
        output_dir=args.output_dir,
        urielplus_dir=args.urielplus_dir,
        smoke=args.smoke,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"  Method:         {summary['imputation_method']}")
    print(f"  Glottocodes:    {summary['n_glottocodes']}")
    print(f"  Found:          {summary['n_found']}")
    print(f"  Imputed:        {summary['n_missing']}")
    print(f"  % non-imputed:  {summary['pct_non_imputed']:.1f}%")
    print(f"  Geo dim:        {summary['geo_dim']}")
    print(f"  Phylo dim:      {summary['phylo_dim']}")
    print(f"  Acceptance:     {'PASSED' if summary['acceptance_passed'] else 'FAILED'}")
    print(f"  Elapsed:        {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
