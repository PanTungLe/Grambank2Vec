"""
confound_baselines.py — Genealogical confound diagnostics for Table 2.

Three baselines that address the question: how much of Obs-ϕ's retrieval
success is genuine cross-database signal, and how much is genealogical/areal
confounding?

Baseline 1 — Obs-Jaccard (Tanimoto)
    Positive-only co-occurrence without marginal correction.  Uses only the
    (1,1) cell relative to the union; discards (0,0) co-absences.  Ablates
    the absence-informative property of ϕ.  Expected gap ϕ > Jaccard ≈ cosine
    confirms that (0,0) cell contributes real signal — and since absences are
    as genealogically structured as presences, a gap here is not a genealogical
    artefact.

Baseline 2 — Family-prevalence cosine
    Collapses each 1015-language profile to a vector over Glottolog families:
    each dimension is mean prevalence within that family over coded members.
    Alignment by cosine of these family-level vectors.  If Fam-prevalence ≈
    Obs-ϕ, genealogy explains ϕ.  If Obs-ϕ >> Fam-prevalence, there is
    genuine within-family cross-database signal that requires language-level
    resolution to recover.

Baseline 3 — URIEL+ phylogenetic centroid cosine
    For each feature-value, computes the centroid of URIEL+ phylogenetic
    (family-membership) binary vectors over languages that carry that value.
    Pure external genealogy signal, independent of both WALS and Grambank.
    Cross-validates Baseline 2: if Fam-prevalence ≈ URIEL-phylo, both
    genealogy signals agree.  If URIEL-phylo < Fam-prevalence, URIEL+'s
    coarser family coverage penalises the phylogenetic signal.

Reference baselines (ϕ and cosine) are computed in the same script to
produce a self-contained table for paper insertion.

Usage
-----
    python canonical/confound_baselines.py \\
        --wals_values    /path/to/wals/values.csv \\
        --wals_languages /path/to/wals/languages.csv \\
        --wals_codes     /path/to/wals/codes.csv \\
        --grambank_dir   /path/to/grambank/cldf \\
        --families_csv   analysis/families.csv \\
        --uriel_npz      /path/to/URIELPlus/.../family_features.npz \\
        --uriel_gmap     /path/to/URIELPlus/.../uriel_glottocode_map.csv \\
        [--output_dir    analysis/confound_baselines] \\
        [--min_co_obs    5] \\
        [--min_family_langs 2] \\
        [--smoke]

Output
------
    analysis/confound_baselines/results.json
    analysis/confound_baselines/table2_supplement.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_geometry import load_checkpoint

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_WALS_VALUES    = "/tmp/wals_values.csv"
_WALS_LANGS     = "/tmp/wals_languages.csv"
_WALS_CODES     = "/tmp/wals_codes.csv"
_GB_CLDF_DIR    = "/tmp/grambank/cldf"
_FAMILIES_CSV   = str(_REPO_ROOT / "analysis" / "families.csv")
_URIEL_NPZ      = "/tmp/URIELPlus/urielplus/database/original_uriel/family_features.npz"
_URIEL_GMAP     = "/tmp/URIELPlus/urielplus/database/urielplus_csvs/uriel_glottocode_map.csv"
_OUTPUT_DIR     = str(_REPO_ROOT / "analysis" / "confound_baselines")
_CKPT_WALS      = str(_REPO_ROOT / "checkpoints" / "wals_learned_s42")
_CKPT_GRAMBANK  = str(_REPO_ROOT / "checkpoints" / "grambank_learned_s42")

# ---------------------------------------------------------------------------
# Benchmark (78-pair reconstruction from tcf_retrieval_5seeds)
# ---------------------------------------------------------------------------

from canonical.tcf_retrieval_5seeds import _BENCHMARK_PAIRS as _BENCHMARK_PAIRS_RAW  # type: ignore

BENCHMARK: list[tuple[str, list[str], str]] = list(_BENCHMARK_PAIRS_RAW)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cnorm(X: np.ndarray) -> np.ndarray:
    """Row-normalise to unit cosine norm.  Zero rows stay zero."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe  = np.where(norms < 1e-8, 1.0, norms)
    return X / safe


def _eval_retrieval(
    wals_queries:  list[str],
    gb_candidates: list[str],
    sim_matrix:    np.ndarray,          # (n_wals, n_gb)
    wals2row:      dict[str, int],
    gb2col:        dict[str, int],
    benchmark:     list[tuple[str, list[str], str]],
) -> dict[str, Any]:
    """
    Compute R@1/5/10 and MRR over the benchmark.

    Returns aggregate metrics plus per-domain breakdown.
    Missing queries (wals_fv not in wals2row) contribute RR=0 and are not
    counted in n_pairs but are flagged in skipped_queries.
    """
    hit_total: dict[int, int] = {1: 0, 5: 0, 10: 0}
    rr_list:   list[float]    = []
    n_eval = 0
    per_pair:   list[dict[str, Any]] = []
    skipped:    list[str]            = []

    # Per-domain accumulators
    domain_hit:  dict[str, dict[int, int]] = {}
    domain_rr:   dict[str, list[float]]    = {}
    domain_n:    dict[str, int]            = {}

    for wals_fv, gb_targets, domain in benchmark:
        if wals_fv not in wals2row:
            skipped.append(wals_fv)
            continue

        resolved = [t for t in gb_targets if t in gb2col]
        if not resolved:
            skipped.append(f"{wals_fv}→{gb_targets}(unresolvable)")
            continue

        row_idx = wals2row[wals_fv]
        sims    = sim_matrix[row_idx]
        ranked  = [gb_candidates[i] for i in np.argsort(-sims)]

        best_rank: int | None = None
        for t in resolved:
            try:
                rk = ranked.index(t) + 1
            except ValueError:
                continue
            if best_rank is None or rk < best_rank:
                best_rank = rk

        rr = 1.0 / best_rank if best_rank else 0.0
        n_eval += 1
        rr_list.append(rr)

        for k in hit_total:
            if best_rank and best_rank <= k:
                hit_total[k] += 1

        # Domain accumulators
        if domain not in domain_hit:
            domain_hit[domain] = {1: 0, 5: 0, 10: 0}
            domain_rr[domain]  = []
            domain_n[domain]   = 0
        domain_n[domain] += 1
        domain_rr[domain].append(rr)
        if best_rank:
            for k in domain_hit[domain]:
                if best_rank <= k:
                    domain_hit[domain][k] += 1

        per_pair.append({
            "wals": wals_fv, "gb_targets": gb_targets, "domain": domain,
            "best_rank": best_rank, "rr": round(rr, 4),
        })

    if n_eval == 0:
        return {"n_pairs": 0, "recall_at_1": 0.0, "recall_at_5": 0.0,
                "recall_at_10": 0.0, "mrr": 0.0, "per_domain": {}, "per_pair": []}

    per_domain: dict[str, dict[str, float]] = {}
    for d in sorted(domain_hit):
        nd = domain_n[d]
        per_domain[d] = {
            "n":           nd,
            "recall_at_1":  round(domain_hit[d][1]  / nd, 4),
            "recall_at_5":  round(domain_hit[d][5]  / nd, 4),
            "recall_at_10": round(domain_hit[d][10] / nd, 4),
            "mrr":          round(float(np.mean(domain_rr[d])), 4),
        }

    return {
        "n_pairs":      n_eval,
        "recall_at_1":  round(hit_total[1]  / n_eval, 4),
        "recall_at_5":  round(hit_total[5]  / n_eval, 4),
        "recall_at_10": round(hit_total[10] / n_eval, 4),
        "mrr":          round(float(np.mean(rr_list)), 4),
        "skipped":      skipped,
        "per_domain":   per_domain,
        "per_pair":     per_pair,
    }


# ---------------------------------------------------------------------------
# Vectorised similarity kernels
# (fixed WALS profile against a stacked GB matrix)
# ---------------------------------------------------------------------------

def _batch_jaccard(
    wals_prof: np.ndarray,   # (n_shared,) int8: -1=NaN, 0=absent, 1=present
    gb_mat:    np.ndarray,   # (n_gb, n_shared) int8
    min_co_obs: int = 5,
) -> np.ndarray:
    """
    Vectorised Tanimoto/Jaccard over all GB candidates for one WALS query.

    Jaccard(a, b) = |{co-obs: a=1 AND b=1}| / |{co-obs: a=1 OR b=1}|
    where co-obs = both features coded (neither is -1).
    Returns (n_gb,) float32; pairs with < min_co_obs co-obs or empty union
    get 0.0.
    """
    # co-observed mask: (n_gb, n_shared)
    co = (wals_prof[np.newaxis, :] >= 0) & (gb_mat >= 0)
    n_co = co.sum(axis=1)                    # (n_gb,)

    a1 = (wals_prof == 1)                    # (n_shared,) bool
    b1 = (gb_mat == 1)                       # (n_gb, n_shared) bool

    inter = (a1[np.newaxis, :] & b1 & co).sum(axis=1).astype(np.float32)
    union = ((a1[np.newaxis, :] | b1) & co).sum(axis=1).astype(np.float32)

    result = np.zeros(gb_mat.shape[0], dtype=np.float32)
    valid  = (n_co >= min_co_obs) & (union > 0)
    result[valid] = inter[valid] / union[valid]
    return result


def _batch_phi(
    wals_prof: np.ndarray,   # (n_shared,) int8
    gb_mat:    np.ndarray,   # (n_gb, n_shared) int8
    min_co_obs: int = 10,
) -> np.ndarray:
    """
    Vectorised Matthews/ϕ correlation for one WALS query against all GB
    candidates.  Only uses co-observed language positions.
    Returns (n_gb,) float32; insufficiently observed pairs get 0.0.
    """
    co  = (wals_prof[np.newaxis, :] >= 0) & (gb_mat >= 0)  # (n_gb, n_shared)
    n_co = co.sum(axis=1)

    co_f = co.astype(np.float32)
    a = (wals_prof == 1).astype(np.float32)         # (n_shared,)
    b = (gb_mat    == 1).astype(np.float32)         # (n_gb, n_shared)

    TP = (a[np.newaxis, :] *      b  * co_f).sum(axis=1)
    TN = ((1 - a)[np.newaxis, :] * (1 - b) * co_f).sum(axis=1)
    FP = ((1 - a)[np.newaxis, :] *      b  * co_f).sum(axis=1)
    FN = (a[np.newaxis, :] * (1 - b) * co_f).sum(axis=1)

    denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    result = np.zeros(gb_mat.shape[0], dtype=np.float32)
    valid  = (n_co >= min_co_obs) & (denom > 0)
    result[valid] = ((TP[valid] * TN[valid] - FP[valid] * FN[valid])
                     / denom[valid]).astype(np.float32)
    return result


def _batch_cosine(
    wals_prof: np.ndarray,   # (n_shared,) float or int8
    gb_mat:    np.ndarray,   # (n_gb, n_shared)
    min_co_obs: int = 5,
) -> np.ndarray:
    """
    Pairwise-deleted cosine for one WALS query against all GB candidates.
    Uses only co-observed language positions.
    Returns (n_gb,) float32.
    """
    co   = (wals_prof[np.newaxis, :] >= 0) & (gb_mat >= 0)
    n_co = co.sum(axis=1)

    wp   = wals_prof.astype(np.float32)
    gp   = gb_mat.astype(np.float32)
    co_f = co.astype(np.float32)

    dot  = (wp[np.newaxis, :] * gp * co_f).sum(axis=1)
    na   = np.sqrt((wp ** 2 * co_f).sum(axis=1))
    nb   = np.sqrt((gp ** 2 * co_f).sum(axis=1))

    result = np.zeros(gb_mat.shape[0], dtype=np.float32)
    valid  = (n_co >= min_co_obs) & (na > 1e-8) & (nb > 1e-8)
    result[valid] = dot[valid] / (na[valid] * nb[valid])
    return result


# ---------------------------------------------------------------------------
# Step 1: Build observed binary profiles from raw CLDF
# ---------------------------------------------------------------------------

def build_observed_profiles(
    wals_values_csv:  str,
    wals_langs_csv:   str,
    wals_codes_csv:   str,
    gb_cldf_dir:      str,
    shared_glottocodes: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Return observed binary profiles in {-1, 0, 1} int8 for every feature-value
    in each database, indexed by position in shared_glottocodes.

    Encoding:
      -1  feature not observed for that language (NaN)
       0  feature observed but this value absent (coded absence)
       1  feature observed and this value present

    WALS  → label:  "{Parameter_ID}={Name}"   e.g. "81A=SOV"
    Grambank → label: "{Parameter_ID}={Value}" e.g. "GB133=1"

    The coded-absence fill is vectorised via per-parameter groupby.
    """
    gc2idx: dict[str, int] = {gc: i for i, gc in enumerate(shared_glottocodes)}
    n      = len(shared_glottocodes)
    shared_set = set(shared_glottocodes)

    # ── WALS ─────────────────────────────────────────────────────────────────
    print("  Loading WALS CLDF …")
    wals_langs   = (pd.read_csv(wals_langs_csv, usecols=["ID", "Glottocode"])
                      .dropna(subset=["Glottocode"]))
    wid2gc       = dict(zip(wals_langs["ID"], wals_langs["Glottocode"]))

    wals_codes   = pd.read_csv(wals_codes_csv, usecols=["ID", "Name"])
    codeid2name  = dict(zip(wals_codes["ID"], wals_codes["Name"]))

    wals_raw = pd.read_csv(wals_values_csv,
                           usecols=["Language_ID", "Parameter_ID", "Code_ID"])
    wals_raw["gc"]    = wals_raw["Language_ID"].map(wid2gc)
    wals_raw["vname"] = wals_raw["Code_ID"].map(codeid2name)
    wals_raw          = wals_raw.dropna(subset=["gc", "vname"])
    wals_raw          = wals_raw[wals_raw["gc"].isin(shared_set)]
    wals_raw["pos"]   = wals_raw["gc"].map(gc2idx)
    wals_raw["fv"]    = wals_raw["Parameter_ID"] + "=" + wals_raw["vname"]

    # Build profiles: initialise to -1 (NaN)
    all_wals_fvs = wals_raw["fv"].unique()
    wals_profiles: dict[str, np.ndarray] = {
        fv: np.full(n, -1, dtype=np.int8) for fv in all_wals_fvs
    }

    # Set presences
    for fv, grp in wals_raw.groupby("fv"):
        wals_profiles[fv][grp["pos"].to_numpy()] = 1

    # Set coded absences: for each parameter, positions where any sibling
    # value is observed get 0 on all other siblings.
    print("  Filling WALS coded absences …")
    param2fvs: dict[str, list[str]] = {}
    for fv in all_wals_fvs:
        param = fv.split("=", 1)[0]
        param2fvs.setdefault(param, []).append(fv)

    wals_obs = wals_raw[["Parameter_ID", "pos", "fv"]].drop_duplicates()
    for param_id, param_grp in wals_obs.groupby("Parameter_ID"):
        siblings = param2fvs.get(param_id, [])
        if len(siblings) <= 1:
            continue
        for _, obs_row in param_grp.iterrows():
            pv = int(obs_row["pos"])
            obs_fv = obs_row["fv"]
            for sib in siblings:
                if sib != obs_fv and wals_profiles[sib][pv] == -1:
                    wals_profiles[sib][pv] = 0

    print(f"  WALS: {len(wals_profiles)} feature-value profiles built.")

    # ── Grambank ──────────────────────────────────────────────────────────────
    print("  Loading Grambank CLDF …")
    gb_raw = pd.read_csv(Path(gb_cldf_dir) / "values.csv",
                         usecols=["Language_ID", "Parameter_ID", "Value"])
    gb_raw = gb_raw[gb_raw["Value"] != "?"]
    gb_raw["Value"]    = gb_raw["Value"].astype(str)
    gb_raw             = gb_raw[gb_raw["Language_ID"].isin(shared_set)]
    gb_raw["pos"]      = gb_raw["Language_ID"].map(gc2idx)
    gb_raw["fv"]       = gb_raw["Parameter_ID"] + "=" + gb_raw["Value"]

    all_gb_fvs = gb_raw["fv"].unique()
    gb_profiles: dict[str, np.ndarray] = {
        fv: np.full(n, -1, dtype=np.int8) for fv in all_gb_fvs
    }

    for fv, grp in gb_raw.groupby("fv"):
        gb_profiles[fv][grp["pos"].to_numpy()] = 1

    print("  Filling Grambank coded absences …")
    param2fvs_gb: dict[str, list[str]] = {}
    for fv in all_gb_fvs:
        param = fv.split("=", 1)[0]
        param2fvs_gb.setdefault(param, []).append(fv)

    gb_obs = gb_raw[["Parameter_ID", "pos", "fv"]].drop_duplicates()
    for param_id, param_grp in gb_obs.groupby("Parameter_ID"):
        siblings = param2fvs_gb.get(param_id, [])
        if len(siblings) <= 1:
            continue
        for _, obs_row in param_grp.iterrows():
            pv = int(obs_row["pos"])
            obs_fv = obs_row["fv"]
            for sib in siblings:
                if sib != obs_fv and gb_profiles[sib][pv] == -1:
                    gb_profiles[sib][pv] = 0

    print(f"  Grambank: {len(gb_profiles)} feature-value profiles built.")
    return wals_profiles, gb_profiles


# ---------------------------------------------------------------------------
# Shared retrieval runner (factored out: builds sim_matrix via a kernel fn)
# ---------------------------------------------------------------------------

def _run_similarity_retrieval(
    label:        str,
    wals_profiles: dict[str, np.ndarray],
    gb_profiles:   dict[str, np.ndarray],
    kernel_fn,                             # (wals_prof, gb_mat) -> (n_gb,) scores
    benchmark:     list[tuple[str, list[str], str]],
    progress:      bool = True,
) -> dict[str, Any]:
    """
    Generic retrieval runner.

    For each benchmark WALS query, computes similarity against ALL Grambank
    feature-values using kernel_fn, then evaluates rank metrics.  Rankings are
    over the full Grambank candidate pool (not just benchmark targets).
    """
    bm_wals   = {w for w, _, _ in benchmark}
    wals_keys = sorted(bm_wals & set(wals_profiles))
    all_gb    = sorted(gb_profiles.keys())

    wals2row  = {k: i for i, k in enumerate(wals_keys)}
    gb2col    = {k: i for i, k in enumerate(all_gb)}

    n_w = len(wals_keys)
    n_g = len(all_gb)
    sim_matrix = np.zeros((n_w, n_g), dtype=np.float32)

    # Stack GB profiles into a matrix for vectorised kernels
    gb_mat = np.stack([gb_profiles[k] for k in all_gb])  # (n_gb, n_shared)

    for i, wk in enumerate(wals_keys):
        if progress and i % 5 == 0:
            print(f"    {i:>3}/{n_w} WALS fvs …", end="\r", flush=True)
        sim_matrix[i] = kernel_fn(wals_profiles[wk], gb_mat)

    if progress:
        print(f"    {n_w}/{n_w} WALS fvs done.    ")

    metrics = _eval_retrieval(wals_keys, all_gb, sim_matrix,
                              wals2row, gb2col, benchmark)
    print(f"  {label}: R@1={metrics['recall_at_1']:.3f}  "
          f"R@5={metrics['recall_at_5']:.3f}  "
          f"R@10={metrics['recall_at_10']:.3f}  "
          f"MRR={metrics['mrr']:.3f}  "
          f"(n={metrics['n_pairs']})")
    return metrics


# ---------------------------------------------------------------------------
# Reference Baseline 0a: Observed cosine (matches paper Table 2 "Obs cos")
# ---------------------------------------------------------------------------

def run_obs_cosine(
    wals_profiles: dict[str, np.ndarray],
    gb_profiles:   dict[str, np.ndarray],
    benchmark:     list[tuple[str, list[str], str]],
    min_co_obs:    int = 5,
) -> dict[str, Any]:
    print("\n[Reference 0a] Observed cosine (paper Table 2) …")
    return _run_similarity_retrieval(
        "Obs cos",
        wals_profiles, gb_profiles,
        lambda wp, gm: _batch_cosine(wp, gm, min_co_obs),
        benchmark,
    )


# ---------------------------------------------------------------------------
# Reference Baseline 0b: Observed ϕ (matches paper Table 2 "Obs ϕ")
# ---------------------------------------------------------------------------

def run_phi(
    wals_profiles: dict[str, np.ndarray],
    gb_profiles:   dict[str, np.ndarray],
    benchmark:     list[tuple[str, list[str], str]],
    min_co_obs:    int = 10,
) -> dict[str, Any]:
    """
    Matthews/ϕ correlation over observed binary profiles.

    This is the headline anchor-layer method from Table 2 (R@10=0.819,
    MRR=0.694).  Running it here lets the confound table be self-contained.
    """
    print("\n[Reference 0b] Observed ϕ (paper Table 2) …")
    return _run_similarity_retrieval(
        "Obs ϕ",
        wals_profiles, gb_profiles,
        lambda wp, gm: _batch_phi(wp, gm, min_co_obs),
        benchmark,
    )


# ---------------------------------------------------------------------------
# Baseline 1: Obs-Jaccard (vectorised)
# ---------------------------------------------------------------------------

def run_jaccard(
    wals_profiles: dict[str, np.ndarray],
    gb_profiles:   dict[str, np.ndarray],
    benchmark:     list[tuple[str, list[str], str]],
    min_co_obs:    int = 5,
) -> dict[str, Any]:
    """
    Tanimoto/Jaccard on raw observed {0,1} binary profiles.

    Uses only the (1,1) cell relative to the union; discards (0,0)
    co-absences.  Ablates the absence-informative property of ϕ.

    The gap Obs-ϕ > Obs-Jaccard quantifies how much information absence
    contributes to cross-database anchoring.  Because absences are as
    genealogically structured as presences, a positive gap is not an
    artefact of genealogical bias.
    """
    print("\n[Baseline 1] Obs-Jaccard …")
    return _run_similarity_retrieval(
        "Obs Jaccard",
        wals_profiles, gb_profiles,
        lambda wp, gm: _batch_jaccard(wp, gm, min_co_obs),
        benchmark,
    )


# ---------------------------------------------------------------------------
# Baseline 2: Family-prevalence cosine (vectorised)
# ---------------------------------------------------------------------------

def _build_family_matrix(
    shared_glottocodes: list[str],
    families_csv: str,
    min_family_langs: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the (n_shared, n_families) indicator matrix used for family-level
    aggregation.  Returns (indicator, family_labels).

    indicator[i, f] = 1 if language i belongs to family f, else 0.
    Families with fewer than min_family_langs shared members are excluded
    (they contribute near-zero signal and inflate dimensionality).
    """
    fam_df = pd.read_csv(families_csv, usecols=["glottocode", "family_id"])
    gc2fam: dict[str, str] = dict(zip(fam_df["glottocode"], fam_df["family_id"]))

    fam_at_pos = np.array(
        [gc2fam.get(gc, f"iso:{gc}") for gc in shared_glottocodes]
    )  # (n_shared,), one family label per position

    # Count members per family
    unique_fams, fam_counts = np.unique(fam_at_pos, return_counts=True)
    keep = unique_fams[fam_counts >= min_family_langs]
    fam2idx = {f: i for i, f in enumerate(keep)}

    n = len(shared_glottocodes)
    indicator = np.zeros((n, len(keep)), dtype=np.float32)
    for pos, fam in enumerate(fam_at_pos):
        if fam in fam2idx:
            indicator[pos, fam2idx[fam]] = 1.0

    return indicator, list(keep)


def _profiles_to_family_vectors(
    profiles:    dict[str, np.ndarray],
    indicator:   np.ndarray,          # (n_shared, n_families)
    min_family_langs: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert per-language profiles to per-family prevalence vectors.

    For each feature-value fv and family f:
        prevalence(fv, f) = sum(profile[fv] == 1 AND member_of[f])
                            / sum(profile[fv] >= 0 AND member_of[f])

    Families with fewer than min_family_langs coded members for a given fv
    receive 0 prevalence (not NaN, to avoid distorting cosine norms).

    Returns (matrix, keys) where matrix is (n_fvs, n_families) float32.
    Fully vectorised: no Python-level loops over languages or families.
    """
    keys = sorted(profiles.keys())
    n_fv = len(keys)
    n_fam = indicator.shape[1]

    # Stack all profiles: (n_fv, n_shared) int8
    prof_mat = np.stack([profiles[k] for k in keys])

    # coded: (n_fv, n_shared) bool — feature is coded (not NaN)
    coded   = (prof_mat >= 0).astype(np.float32)
    present = (prof_mat == 1).astype(np.float32)

    # Denominator: (n_fv, n_fam) — coded observations per family per fv
    denom = coded @ indicator              # (n_fv, n_fam)

    # Numerator: (n_fv, n_fam) — positive observations per family per fv
    numer = present @ indicator            # (n_fv, n_fam)

    # Prevalence, with masking for insufficient observations
    valid   = denom >= float(min_family_langs)
    prev    = np.where(valid, numer / np.where(denom > 0, denom, 1.0), 0.0)

    return prev.astype(np.float32), keys


def run_family_prevalence(
    wals_profiles:      dict[str, np.ndarray],
    gb_profiles:        dict[str, np.ndarray],
    shared_glottocodes: list[str],
    families_csv:       str,
    benchmark:          list[tuple[str, list[str], str]],
    min_family_langs:   int = 2,
) -> dict[str, Any]:
    """
    Coarsen observed profiles to Glottolog family-level prevalence vectors,
    then rank by cosine similarity.

    Diagnostic interpretation:
      Obs-ϕ >> Fam-prevalence → language-level cross-database signal beyond
                                 what family membership alone recovers.
      Obs-ϕ ≈  Fam-prevalence → ϕ's signal is predominantly genealogical;
                                 disclose as confound in §Limitations.
    """
    print("\n[Baseline 2] Family-prevalence cosine …")

    indicator, fam_labels = _build_family_matrix(
        shared_glottocodes, families_csv, min_family_langs
    )
    print(f"  {len(fam_labels)} families with ≥{min_family_langs} shared members.")

    print("  Building WALS family-prevalence vectors …")
    wals_fv_mat, wals_keys_all = _profiles_to_family_vectors(
        wals_profiles, indicator, min_family_langs
    )

    print("  Building Grambank family-prevalence vectors …")
    gb_fv_mat, gb_keys_all = _profiles_to_family_vectors(
        gb_profiles, indicator, min_family_langs
    )

    # Filter to benchmark queries
    bm_wals  = {w for w, _, _ in benchmark}
    wals_idx = [i for i, k in enumerate(wals_keys_all) if k in bm_wals]
    wals_keys = [wals_keys_all[i] for i in wals_idx]
    W_sub = wals_fv_mat[wals_idx]           # (n_bm_wals, n_fam)

    # Cosine similarity: (n_bm_wals, n_gb)
    sim_matrix = _cnorm(W_sub) @ _cnorm(gb_fv_mat).T

    wals2row  = {k: i for i, k in enumerate(wals_keys)}
    gb2col    = {k: i for i, k in enumerate(gb_keys_all)}

    metrics = _eval_retrieval(wals_keys, gb_keys_all, sim_matrix,
                              wals2row, gb2col, benchmark)
    metrics["n_families"] = len(fam_labels)
    print(f"  Fam-prevalence: R@1={metrics['recall_at_1']:.3f}  "
          f"R@5={metrics['recall_at_5']:.3f}  "
          f"R@10={metrics['recall_at_10']:.3f}  "
          f"MRR={metrics['mrr']:.3f}  "
          f"(n={metrics['n_pairs']})")
    return metrics


# ---------------------------------------------------------------------------
# Baseline 3: URIEL+ phylogenetic centroid cosine
# ---------------------------------------------------------------------------

def _load_uriel_phylo(
    uriel_npz:     str,
    uriel_gmap_csv: str,
    shared_glottocodes: list[str],
) -> tuple[np.ndarray, int]:
    """
    Load URIEL+ phylogenetic (family-membership) binary vectors aligned to
    shared_glottocodes.  Returns (aligned_matrix, coverage_count).

    The URIEL+ family_features.npz contains:
        "langs": 1-D array of URIEL language codes (7970,)
        "data":  3-D array (n_langs, n_features, n_sources) float32

    We use source index 0 (the original URIEL values, before aggregation).
    Coverage is ~99% for the shared set; absent languages get zero vectors
    (honest fallback, same convention as RQ4 conditioning).
    """
    fam_data    = np.load(uriel_npz)
    uriel_langs = list(fam_data["langs"])
    code2idx    = {c: i for i, c in enumerate(uriel_langs)}

    raw = fam_data["data"]
    # Handle 2-D and 3-D layouts
    if raw.ndim == 3:
        phylo_full = raw[:, :, 0].astype(np.float32)    # (n_langs, n_feat)
    elif raw.ndim == 2:
        phylo_full = raw.astype(np.float32)
    else:
        raise ValueError(f"Unexpected URIEL NPZ shape: {raw.shape}")

    dim = phylo_full.shape[1]

    # Glottocode → URIEL row index via the provided mapping
    gmap        = pd.read_csv(uriel_gmap_csv, usecols=["code", "glottocode"])
    gc2uriel: dict[str, int] = {}
    for _, row in gmap.iterrows():
        gc = row["glottocode"]
        uc = row["code"]
        if pd.isna(gc) or uc not in code2idx:
            continue
        gc2uriel[str(gc)] = code2idx[uc]

    n_shared = len(shared_glottocodes)
    aligned  = np.zeros((n_shared, dim), dtype=np.float32)
    covered  = 0
    for pos, gc in enumerate(shared_glottocodes):
        ui = gc2uriel.get(gc)
        if ui is not None:
            aligned[pos] = phylo_full[ui]
            covered += 1

    return aligned, covered


def run_uriel_phylo_centroid(
    wals_profiles:       dict[str, np.ndarray],
    gb_profiles:         dict[str, np.ndarray],
    shared_glottocodes:  list[str],
    uriel_npz:           str,
    uriel_gmap_csv:      str,
    benchmark:           list[tuple[str, list[str], str]],
    min_pos_langs:       int = 5,
) -> dict[str, Any]:
    """
    Rank Grambank values by cosine similarity of URIEL+ phylogenetic centroids.

    For each feature-value, the centroid is the mean URIEL+ phylogenetic vector
    over languages that carry that value.  This is a pure external-genealogy
    signal, independent of the typological databases being compared.

    Cross-validates Baseline 2:
      URIEL-phylo ≈ Fam-prevalence → both genealogy signals agree.
      URIEL-phylo < Fam-prevalence → URIEL+'s coarser/sparser family coverage
                                      penalises the signal.
    """
    print("\n[Baseline 3] URIEL+ phylogenetic centroid …")

    phylo_aligned, covered = _load_uriel_phylo(
        uriel_npz, uriel_gmap_csv, shared_glottocodes
    )
    n_shared = len(shared_glottocodes)
    n_dim    = phylo_aligned.shape[1]
    print(f"  URIEL+ coverage: {covered}/{n_shared} shared languages "
          f"({100*covered/n_shared:.1f}%)  dim={n_dim}")

    def _centroid(profile: np.ndarray) -> np.ndarray:
        """Mean phylo vector over languages where profile == 1."""
        mask = profile == 1
        if mask.sum() < min_pos_langs:
            return np.zeros(n_dim, dtype=np.float32)
        return phylo_aligned[mask].mean(axis=0)

    # Benchmark WALS queries only; all GB candidates
    bm_wals     = {w for w, _, _ in benchmark}
    wals_keys   = sorted(bm_wals & set(wals_profiles))
    all_gb_keys = sorted(gb_profiles.keys())
    wals2row    = {k: i for i, k in enumerate(wals_keys)}
    gb2col      = {k: i for i, k in enumerate(all_gb_keys)}

    print("  Building WALS phylo centroids …")
    W_mat = np.stack([_centroid(wals_profiles[k]) for k in wals_keys])

    print("  Building Grambank phylo centroids …")
    G_mat = np.stack([_centroid(gb_profiles[k]) for k in all_gb_keys])

    sim_matrix = _cnorm(W_mat) @ _cnorm(G_mat).T

    metrics = _eval_retrieval(wals_keys, all_gb_keys, sim_matrix,
                              wals2row, gb2col, benchmark)
    metrics["uriel_coverage"] = round(covered / n_shared, 4)
    metrics["phylo_dim"]      = n_dim
    print(f"  URIEL+ phylo cent.: R@1={metrics['recall_at_1']:.3f}  "
          f"R@5={metrics['recall_at_5']:.3f}  "
          f"R@10={metrics['recall_at_10']:.3f}  "
          f"MRR={metrics['mrr']:.3f}  "
          f"(n={metrics['n_pairs']})")
    return metrics


# ---------------------------------------------------------------------------
# Output: Table 2 supplement (self-contained for paper insertion)
# ---------------------------------------------------------------------------

def print_table(
    results:    dict[str, dict[str, Any]],
    ref_labels: set[str] | None = None,
) -> str:
    """
    Format the complete confound-diagnostic table for paper insertion.

    ref_labels: method labels to annotate as '(ref)' in the table.
    """
    ref_labels = ref_labels or {"Obs cos (ref)", "Obs ϕ (ref)"}

    # ── Aggregate table ───────────────────────────────────────────────────────
    sep  = "─" * 64
    rows = [
        "TABLE 2 SUPPLEMENT — Genealogical confound diagnostics",
        sep,
        f"{'Method':<22} {'R@1':>6} {'R@5':>6} {'R@10':>7} {'MRR':>6}  n",
        sep,
    ]
    for label, m in results.items():
        tag = "  ← reference" if label in ref_labels else ""
        rows.append(
            f"  {label:<20} "
            f"{m.get('recall_at_1',0):>6.3f} "
            f"{m.get('recall_at_5',0):>6.3f} "
            f"{m.get('recall_at_10',0):>7.3f} "
            f"{m.get('mrr',0):>6.3f} "
            f"{m.get('n_pairs','—'):>4}{tag}"
        )
    rows.append(sep)

    # ── Per-domain breakdown (non-reference rows) ─────────────────────────────
    # Collect all domains across all results
    all_domains: set[str] = set()
    for m in results.values():
        all_domains.update(m.get("per_domain", {}).keys())

    if all_domains:
        rows.append("")
        rows.append("PER-DOMAIN R@10")
        rows.append(sep)
        header_dom = f"  {'Domain':<22}" + "".join(
            f"  {lbl[:10]:<10}" for lbl in results
        )
        rows.append(header_dom)
        rows.append(sep)
        for domain in sorted(all_domains):
            row = f"  {domain:<22}"
            for m in results.values():
                pd_m = m.get("per_domain", {}).get(domain, {})
                v = pd_m.get("recall_at_10", "—")
                row += f"  {v if isinstance(v,str) else f'{v:.3f}':<10}"
            rows.append(row)
        rows.append(sep)

    # ── Interpretation guide ──────────────────────────────────────────────────
    rows += [
        "",
        "Interpretation guide:",
        "  Obs-ϕ > Obs-Jaccard > Obs-cos  →  (0,0) cell adds genuine signal.",
        "                                     Not a genealogical artefact because",
        "                                     absences are equally genealogically",
        "                                     structured.",
        "  Obs-ϕ >> Fam-prevalence        →  Language-level resolution adds",
        "                                     within-family cross-database signal.",
        "  Obs-ϕ ≈  Fam-prevalence        →  ϕ is predominantly genealogical;",
        "                                     disclose as confound in §Limitations.",
        "  URIEL-phylo ≈ Fam-prevalence   →  Two independent genealogy signals agree.",
        "  URIEL-phylo < Fam-prevalence   →  URIEL+ coarser coverage penalises",
        "                                     the external genealogy signal.",
    ]

    table = "\n".join(rows)
    print("\n" + table)
    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genealogical confound diagnostics for Table 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--wals_values",     default=_WALS_VALUES)
    p.add_argument("--wals_languages",  default=_WALS_LANGS)
    p.add_argument("--wals_codes",      default=_WALS_CODES)
    p.add_argument("--grambank_dir",    default=_GB_CLDF_DIR)
    p.add_argument("--families_csv",    default=_FAMILIES_CSV)
    p.add_argument("--uriel_npz",       default=_URIEL_NPZ)
    p.add_argument("--uriel_gmap",      default=_URIEL_GMAP)
    p.add_argument("--output_dir",      default=_OUTPUT_DIR)
    p.add_argument("--min_co_obs",      type=int, default=5,
                   help="Min co-observed languages for Jaccard/cosine/ϕ.")
    p.add_argument("--min_co_obs_phi",  type=int, default=10,
                   help="Min co-observed languages for ϕ (higher = more reliable).")
    p.add_argument("--min_family_langs", type=int, default=2,
                   help="Min observed languages per family (Baseline 2).")
    p.add_argument("--skip_uriel",      action="store_true",
                   help="Skip Baseline 3 (e.g. URIEL NPZ not available).")
    p.add_argument("--smoke",           action="store_true",
                   help="Run on first 15 benchmark pairs only.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    benchmark = BENCHMARK[:15] if args.smoke else BENCHMARK
    print(f"[confound_baselines]  {len(benchmark)} pairs  "
          f"{'(smoke)' if args.smoke else '(full)'}")

    # ── Shared Glottocodes from checkpoints ───────────────────────────────────
    print("\nLoading checkpoints to get shared Glottocodes …")
    ck_w = load_checkpoint(_CKPT_WALS)
    ck_g = load_checkpoint(_CKPT_GRAMBANK)
    shared_glottocodes = sorted(
        set(ck_w["lang2id"].keys()) & set(ck_g["lang2id"].keys())
    )
    print(f"  Shared Glottocodes: {len(shared_glottocodes)}")

    # ── Build observed profiles ───────────────────────────────────────────────
    print("\nBuilding observed binary profiles from CLDF …")
    wals_profiles, gb_profiles = build_observed_profiles(
        args.wals_values, args.wals_languages, args.wals_codes,
        args.grambank_dir, shared_glottocodes,
    )

    # ── Run all methods ───────────────────────────────────────────────────────
    results_ordered: dict[str, dict[str, Any]] = {}

    results_ordered["Obs cos (ref)"] = run_obs_cosine(
        wals_profiles, gb_profiles, benchmark, min_co_obs=args.min_co_obs
    )
    results_ordered["Obs ϕ (ref)"] = run_phi(
        wals_profiles, gb_profiles, benchmark, min_co_obs=args.min_co_obs_phi
    )
    results_ordered["Obs Jaccard"] = run_jaccard(
        wals_profiles, gb_profiles, benchmark, min_co_obs=args.min_co_obs
    )
    results_ordered["Fam-prevalence cos"] = run_family_prevalence(
        wals_profiles, gb_profiles, shared_glottocodes,
        args.families_csv, benchmark, min_family_langs=args.min_family_langs,
    )
    if not args.skip_uriel:
        results_ordered["URIEL+ phylo cent."] = run_uriel_phylo_centroid(
            wals_profiles, gb_profiles, shared_glottocodes,
            args.uriel_npz, args.uriel_gmap, benchmark,
        )
    else:
        print("\n[Baseline 3] URIEL+ skipped (--skip_uriel).")

    # ── Print + save ──────────────────────────────────────────────────────────
    ref_labels = {"Obs cos (ref)", "Obs ϕ (ref)"}
    table_str  = print_table(results_ordered, ref_labels=ref_labels)

    full_output = {
        "benchmark_size":     len(benchmark),
        "n_shared_langs":     len(shared_glottocodes),
        "min_co_obs":         args.min_co_obs,
        "min_co_obs_phi":     args.min_co_obs_phi,
        "min_family_langs":   args.min_family_langs,
        "smoke":              args.smoke,
        "baselines":          results_ordered,
    }

    (out / "results.json").write_text(
        json.dumps(full_output, indent=2, default=lambda x: float(x)
                   if hasattr(x, "__float__") else str(x)),
        encoding="utf-8",
    )
    (out / "table2_supplement.txt").write_text(table_str, encoding="utf-8")
    print(f"\nSaved: {out}/results.json  and  {out}/table2_supplement.txt")


if __name__ == "__main__":
    main()
