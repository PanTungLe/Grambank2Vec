"""
T-CF / Learned 5-seed cross-database retrieval — Table 2 component.

Benchmark v2 (74 pairs, 22 domains) — source: crosswalk_gold.json.
All pairs are Tier A, documentation-derived.  Two annotation errors from v1
removed (39A=No inclusive/exclusive→GB028=1 and 118A=Nonverbal→GB068=1 both
had mcc≈−0.74, perm_p=1.000; confirmed inversions).  82A corrected:
SV→GB130=1, VS→GB130=2 (v1 had SV→GB130=3, VS→GB130=1).

Evaluable pair counts (seed 42 checkpoint)
-------------------------------------------
  Learned  70 / 74   4 WALS values below the 10-language training threshold
                      are unresolvable in all models:
                        90A=Correlative, 41A=Four-way contrast,
                        44A=3rd person non-singular only,
                        59A=More than five classes.

  T-CF     54 / 74   3 of the 4 above are also missing from T-CF (41A=Three-way
                      contrast is present because 41A is binary in T-CF).
                      17 additional pairs have only '=0' GB targets, which are
                      unresolvable in T-CF (no negated profile stored for binary
                      features).  Those 17 pairs isolate a structural limitation
                      of yes/no encoding: absent correspondences cannot be queried.

Soft-matching (T-CF)
--------------------
  T-CF encodes binary WALS/Grambank features as bare keys ('107A', 'GB028')
  rather than typed values ('107A=Present', 'GB028=1').  Soft-match applied:
    WALS: '107A=Present' → '107A'   (T-CF profile = P(Present | lang))
    GB  : 'GB028=1'     → 'GB028'  (same semantic justification)
    GB  : 'GB028=0'     → unresolvable (no negated profile)
  Implemented in _resolve_wals_key_tcf and _resolve_gb_targets_tcf.

Usage
-----
  python canonical/tcf_retrieval_5seeds.py [--output_dir analysis/tcf_retrieval]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import softmax as sp_softmax

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

def load_checkpoint(checkpoint_dir):
    import json, numpy as np
    from pathlib import Path as _P
    cdir = _P(checkpoint_dir)
    config = json.loads((cdir / 'config.json').read_text())
    arch = config['architecture']
    lang_emb = np.load(cdir / 'lang_embeddings.npy')
    if arch == 'learned':
        fv_emb = np.load(cdir / 'featvalue_embeddings.npy')
        fv2id  = json.loads((cdir / 'featvalue2id.json').read_text())
    else:
        fv_emb = np.load(cdir / 'binarycol_embeddings.npy')
        fv2id  = json.loads((cdir / 'binarycol2id.json').read_text())
    lang2id = json.loads((cdir / 'lang2id.json').read_text())
    lang2id_full = json.loads((cdir / 'lang2id_full.json').read_text())
    feat2values = json.loads((cdir / 'feat2values.json').read_text())
    return dict(architecture=arch, database=config['database'],
                lang_emb=lang_emb, fv_emb=fv_emb, fv2id=fv2id,
                lang2id=lang2id, lang2id_full=lang2id_full,
                feat2values=feat2values, config=config)
def seed_everything(seed):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

_SEEDS = [42, 43, 44, 45, 46]
_DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "analysis" / "tcf_retrieval")


# ---------------------------------------------------------------------------
# Benchmark v2: 74 Tier-A WALS–Grambank value pairs, 22 domains.
# Source: crosswalk_gold.json — documentation-driven annotation, all Tier A.
# All pairs verified against WALS and Grambank CLDF parameter definitions.
#
# Changes from v1 (73 pairs):
#   CORRECTED  82A=SV  → GB130=1 (was GB130=3), 82A=VS → GB130=2 (was GB130=1)
#   CORRECTED  118A=Nonverbal encoding → GB068=0 only (GB068=1 row was a
#              copy-paste error confirmed by mcc=−0.582, perm_p=1.000)
#   CORRECTED  39A=No inclusive/exclusive → GB028=0 only (GB028=1 row was a
#              logical inversion confirmed by mcc=−0.741, perm_p=1.000)
#   DROPPED    81A (SOV/SVO/VSO/VOS) — reclassified as PROJECTION / Tier B
#   DROPPED    83A=VO — reclassified as PARTIAL / Tier B
#   DROPPED    109A (applicatives), 117A (pred-possession), 51A (case),
#              66A (TAM), 63A=identical — all Tier B in adjudicated crosswalk
#   DROPPED    37A/38A values without standalone GB feature (definite affix,
#              indefinite affix, indefinite same as 'one')
#   DROPPED    41A=Two-way contrast → GB035=0 (not a positive GB value)
#   ADDED      101A (pro-drop), 105A (ditransitive), 111A (causative),
#              45A (politeness), 55A (classifiers), 71A (imperative-neg),
#              77A (evidentiality), 79A (suppletion), 121A (comparison v2),
#              90A=Internally headed / Correlative, 44A=3rd-person non-singular
#   WEAK-STAT  6 pairs with perm_p > 0.5 retained on definitional grounds
#              (all DIRECTIONAL; presence_vs_dominance caveat explains
#               negative MCC — see crosswalk_adjudicated.csv perm_p column).
# ---------------------------------------------------------------------------
_BENCHMARK_PAIRS: list[tuple[str, list[str], str]] = [

    # ── Articles (37A) ────────────────────────────────────────────────────────
    ("37A=Definite word distinct from demonstrative",   ["GB020=1"], "articles"),
    ("37A=Demonstrative word used as definite article", ["GB020=1"], "articles"),  # WEAK-STAT
    ("37A=No definite or indefinite article",           ["GB020=0"], "articles"),

    # ── Articles (38A) ────────────────────────────────────────────────────────
    ("38A=Indefinite word distinct from 'one'",         ["GB021=1"], "articles"),
    ("38A=No indefinite, but definite article",         ["GB021=0"], "articles"),
    ("38A=No definite or indefinite article",           ["GB021=0"], "articles"),

    # ── Inclusive/exclusive (39A) ─────────────────────────────────────────────
    ("39A=Inclusive/exclusive",                         ["GB028=1"], "incl-excl"),
    ("39A=No inclusive/exclusive",                      ["GB028=0"], "incl-excl"),

    # ── Demonstrative distance contrasts (41A) ────────────────────────────────
    ("41A=Three-way contrast",                          ["GB035=1"], "dem-distance"),
    ("41A=Four-way contrast",                           ["GB035=1"], "dem-distance"),

    # ── Gender distinctions in pronouns (44A) ─────────────────────────────────
    ("44A=In 3rd person + 1st and/or 2nd person",      ["GB030=1"], "gender"),
    ("44A=3rd person only, but also non-singular",      ["GB030=1"], "gender"),
    ("44A=3rd person singular only",                    ["GB030=1"], "gender"),
    ("44A=3rd person non-singular only",                ["GB030=1"], "gender"),
    ("44A=No gender distinctions",                      ["GB030=0"], "gender"),

    # ── Politeness (45A) ──────────────────────────────────────────────────────
    ("45A=No politeness distinction",                   ["GB415=0"], "politeness"),
    ("45A=Binary politeness distinction",               ["GB415=1"], "politeness"),
    ("45A=Multiple politeness distinctions",            ["GB415=1"], "politeness"),

    # ── Numeral classifiers (55A) ─────────────────────────────────────────────
    ("55A=Absent",                                      ["GB057=0"], "classifiers"),
    ("55A=Optional",                                    ["GB057=1"], "classifiers"),
    ("55A=Obligatory",                                  ["GB057=1"], "classifiers"),

    # ── Possessive classifiers (59A) ──────────────────────────────────────────
    ("59A=No possessive classification",                ["GB058=0"], "possession"),
    ("59A=Two classes",                                 ["GB058=1"], "possession"),
    ("59A=Three to five classes",                       ["GB058=1"], "possession"),  # WEAK-STAT
    ("59A=More than five classes",                      ["GB058=1"], "possession"),

    # ── Nominal conjunction / comitative (63A) ────────────────────────────────
    ("63A='And' different from 'with'",                 ["GB027=1"], "nominal-syntax"),

    # ── Inflectional future (67A) ─────────────────────────────────────────────
    ("67A=Inflectional future exists",                  ["GB084=1"], "infl-future"),
    ("67A=No inflectional future",                      ["GB084=0"], "infl-future"),

    # ── Imperative / negation interaction (71A) ───────────────────────────────
    ("71A=Normal imperative + special negative",        ["GB139=1"], "imperative-neg"),
    ("71A=Special imperative + special negative",       ["GB139=1"], "imperative-neg"),
    ("71A=Special imperative + normal negative",        ["GB139=1"], "imperative-neg"),  # WEAK-STAT

    # ── Evidentiality (77A) ───────────────────────────────────────────────────
    ("77A=No grammatical evidentials",                  ["GB322=0"], "evidentiality"),
    ("77A=Indirect only",                               ["GB323=1"], "evidentiality"),
    ("77A=Direct and indirect",                         ["GB322=1"], "evidentiality"),

    # ── Verb suppletion (79A) ─────────────────────────────────────────────────
    ("79A=Tense and aspect",                            ["GB110=1"], "suppletion"),

    # ── Intransitive clause order (82A) ───────────────────────────────────────
    # CORRECTED: GB130=1=SV, GB130=2=VS (old had SV→3, VS→1)
    ("82A=SV",                                          ["GB130=1"], "word-order"),
    ("82A=VS",                                          ["GB130=2"], "word-order"),

    # ── OV order (83A) ────────────────────────────────────────────────────────
    # 83A=VO dropped (PARTIAL/Tier B). Only OV→GB133=1 kept.
    ("83A=OV",                                          ["GB133=1"], "word-order"),

    # ── Adpositions (85A) ─────────────────────────────────────────────────────
    ("85A=Prepositions",                                ["GB074=1"], "word-order"),
    ("85A=Postpositions",                               ["GB075=1"], "word-order"),
    ("85A=No adpositions",                              ["GB074=0", "GB075=0"], "word-order"),

    # ── Possessor/genitive order (86A) ────────────────────────────────────────
    ("86A=Genitive-Noun",                               ["GB065=1"], "word-order"),
    ("86A=Noun-Genitive",                               ["GB065=2"], "word-order"),

    # ── Adjective-noun order (87A) ────────────────────────────────────────────
    ("87A=Adjective-Noun",                              ["GB193=1"], "word-order"),
    ("87A=Noun-Adjective",                              ["GB193=2"], "word-order"),

    # ── Demonstrative order (88A) ─────────────────────────────────────────────
    ("88A=Demonstrative-Noun",                          ["GB025=1"], "word-order"),
    ("88A=Noun-Demonstrative",                          ["GB025=2"], "word-order"),

    # ── Numeral-noun order (89A) ──────────────────────────────────────────────
    ("89A=Numeral-Noun",                                ["GB024=1"], "word-order"),
    ("89A=Noun-Numeral",                                ["GB024=2"], "word-order"),

    # ── Relative clause order (90A) ───────────────────────────────────────────
    ("90A=Noun-Relative clause",                        ["GB327=1"], "rel-clause"),
    ("90A=Relative clause-Noun",                        ["GB328=1"], "rel-clause"),
    ("90A=Internally headed",                           ["GB329=1"], "rel-clause"),
    ("90A=Correlative",                                 ["GB330=1"], "rel-clause"),

    # ── Pro-drop (101A) ───────────────────────────────────────────────────────
    ("101A=Optional pronouns in subject position",      ["GB522=1"], "pro-drop"),
    ("101A=Obligatory pronouns in subject position",    ["GB522=0"], "pro-drop"),

    # ── Ditransitive constructions (105A) ─────────────────────────────────────
    ("105A=Double-object construction",                 ["GB105=1"], "ditransitive"),
    ("105A=Indirect-object construction",               ["GB105=0"], "ditransitive"),

    # ── Passive (107A) ────────────────────────────────────────────────────────
    ("107A=Present",                                    ["GB147=1"], "passive"),
    ("107A=Absent",                                     ["GB147=0"], "passive"),

    # ── Morphological causative (111A) ────────────────────────────────────────
    ("111A=Morphological but no compound",              ["GB155=1"], "causative"),
    ("111A=Neither",                                    ["GB155=0"], "causative"),

    # ── Predicative adjectives (118A) ─────────────────────────────────────────
    # CORRECTED: Nonverbal→GB068=0 only; GB068=1 row was an annotation error
    ("118A=Verbal encoding",                            ["GB068=1"], "pred-adj"),
    ("118A=Nonverbal encoding",                         ["GB068=0"], "pred-adj"),

    # ── Comparison strategy (121A) ────────────────────────────────────────────
    ("121A=Exceed",                                     ["GB265=1"], "comparison"),
    ("121A=Locational",                                 ["GB266=1"], "comparison"),
    ("121A=Conjoined",                                  ["GB270=1"], "comparison"),
    ("121A=Particle",                                   ["GB273=1"], "comparison"),

    # ── Numeral bases (131A) ──────────────────────────────────────────────────
    ("131A=Decimal",                                    ["GB333=1"], "numeral-bases"),
    ("131A=Hybrid vigesimal-decimal",                   ["GB333=1"], "numeral-bases"),  # WEAK-STAT
    ("131A=Pure vigesimal",                             ["GB335=1"], "numeral-bases"),

    # ── Associative plural (36A) ──────────────────────────────────────────────
    ("36A=No associative plural",                       ["GB046=0"], "assoc-plural"),
    ("36A=Unique affixal associative plural",           ["GB046=1"], "assoc-plural"),  # WEAK-STAT
    ("36A=Unique periphrastic associative plural",      ["GB046=1"], "assoc-plural"),  # WEAK-STAT
    ("36A=Associative same as additive plural",         ["GB046=1"], "assoc-plural"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cnorm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n < 1e-8, 1.0, n)


def _build_shared_index(
    lang2id_a: dict[str, int], lang2id_b: dict[str, int]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    shared = sorted(set(lang2id_a) & set(lang2id_b))
    return (shared,
            np.array([lang2id_a[g] for g in shared], dtype=np.int64),
            np.array([lang2id_b[g] for g in shared], dtype=np.int64))


# ---------------------------------------------------------------------------
# Soft-matching helpers for T-CF's bare-name encoding
# ---------------------------------------------------------------------------

def _resolve_wals_key_tcf(fv_label: str, fv2id: dict[str, int]) -> str | None:
    """
    Resolve a typed WALS fv label to a T-CF fv2id key.

    T-CF stores multi-valued WALS features as 'feat=val' (identical to Learned)
    and binary WALS features as bare 'feat'.  For a benchmark label like
    '107A=Present' that isn't in T-CF's fv2id, fall back to '107A' if that
    bare key is present.  Semantically valid: the T-CF sigmoid profile for a
    binary WALS feature encodes P(positive_value | lang).
    """
    if fv_label in fv2id:
        return fv_label
    bare = fv_label.split("=", 1)[0]
    if bare in fv2id:
        return bare
    return None


def _resolve_gb_targets_tcf(
    targets: list[str], fv2id: dict[str, int]
) -> list[str]:
    """
    Resolve a list of typed GB targets to T-CF fv2id keys.

    T-CF stores binary Grambank features as bare 'GB_x' and multi-valued ones
    as 'GB_x=k'.  For a typed target like 'GB028=1', fall back to bare 'GB028'
    if present (same semantic justification as for WALS).  'GB028=0' (negative
    direction) remains unresolvable — T-CF stores no negated profile.
    Returns the deduplicated list of resolved keys (may be shorter than input).
    """
    resolved = []
    seen = set()
    for t in targets:
        if t in fv2id:
            key = t
        elif t.endswith("=0"):
            continue           # no negated T-CF profile
        else:
            base = t.rsplit("=", 1)[0]
            key = base if base in fv2id else None
        if key and key not in seen:
            resolved.append(key)
            seen.add(key)
    return resolved


# ---------------------------------------------------------------------------
# T-CF retrieval  (Experiment A — matches paper)
# ---------------------------------------------------------------------------

def _tcf_retrieval(
    benchmark: list[tuple[str, list[str], str]],
    ck_w: dict, ck_g: dict,
    idx_w: np.ndarray, idx_g: np.ndarray,
) -> dict[str, Any]:
    """
    T-CF predicted-profile cross-database retrieval with soft-matching.

    WALS queries:
      Exact match first; if missing, try bare feature key for binary WALS
      features (see _resolve_wals_key_tcf).  If neither resolves, skip.

    GB candidates: all entries in T-CF gb_fv2id (bare names for binary,
                   '=k' for multi-valued).
    GB target resolution:
      Exact match first; for '=1' binary targets fall back to bare 'GB_x';
      '=0' targets are unresolvable (no negated profile stored).
      See _resolve_gb_targets_tcf.

    A pair is counted in n_eval only if the WALS query resolves and at least
    one GB target resolves.  Pairs with no resolving target contribute RR=0
    to the MRR but are still counted in n_eval (the WALS side resolved).
    """
    L_W = ck_w["lang_emb"][idx_w].astype(np.float32)
    L_G = ck_g["lang_emb"][idx_g].astype(np.float32)
    F_W = ck_w["fv_emb"].astype(np.float32)
    F_G = ck_g["fv_emb"].astype(np.float32)

    wals_profs = sigmoid(L_W @ F_W.T).T.astype(np.float32)   # (n_fv_w, n_shared)
    wals_fv2id = ck_w["fv2id"]

    gb_profs   = sigmoid(L_G @ F_G.T).T.astype(np.float32)   # (n_fv_g, n_shared)
    gb_fv2id   = ck_g["fv2id"]
    gb_labels  = sorted(gb_fv2id, key=lambda k: gb_fv2id[k])
    gb_norm    = _cnorm(gb_profs)

    hit: dict[int, int] = {1: 0, 5: 0, 10: 0}
    rr_list: list[float] = []
    n_eval = 0
    per_pair = []

    for wals_fv, gb_targets, domain in benchmark:
        wals_key = _resolve_wals_key_tcf(wals_fv, wals_fv2id)
        if wals_key is None:
            continue                          # WALS fv unresolvable; skip

        wq      = wals_profs[wals_fv2id[wals_key]]
        wq_norm = wq / max(float(np.linalg.norm(wq)), 1e-8)
        sims    = gb_norm @ wq_norm           # (n_fv_g,)
        ranked  = [gb_labels[i] for i in np.argsort(-sims)]

        resolved = _resolve_gb_targets_tcf(gb_targets, gb_fv2id)

        best_rank: int | None = None
        best_hit:  str | None = None
        for t in resolved:
            rk = ranked.index(t) + 1
            if best_rank is None or rk < best_rank:
                best_rank = rk
                best_hit  = t

        rr = 1.0 / best_rank if best_rank else 0.0
        rr_list.append(rr)
        n_eval += 1
        if best_rank:
            for k in hit:
                if best_rank <= k:
                    hit[k] += 1
        per_pair.append({
            "wals": wals_fv, "wals_key_used": wals_key,
            "targets": gb_targets, "resolved_targets": resolved,
            "domain": domain,
            "best_rank": best_rank, "best_hit": best_hit,
            "rr": round(rr, 4),
        })

    if n_eval == 0:
        return {"n_pairs": 0}

    return {
        "n_pairs":      n_eval,
        "recall_at_1":  round(hit[1]  / n_eval, 4),
        "recall_at_5":  round(hit[5]  / n_eval, 4),
        "recall_at_10": round(hit[10] / n_eval, 4),
        "mrr":          round(float(np.mean(rr_list)), 4),
        "per_pair":     per_pair,
    }


# ---------------------------------------------------------------------------
# Learned retrieval  (Experiment D — softmax profiles, 392 candidates)
# ---------------------------------------------------------------------------

def _learned_retrieval(
    benchmark: list[tuple[str, list[str], str]],
    ck_w: dict, ck_g: dict,
    idx_w: np.ndarray, idx_g: np.ndarray,
) -> dict[str, Any]:
    L_W = ck_w["lang_emb"][idx_w].astype(np.float32)
    L_G = ck_g["lang_emb"][idx_g].astype(np.float32)
    F_W = ck_w["fv_emb"].astype(np.float32)
    F_G = ck_g["fv_emb"].astype(np.float32)

    def _softmax_profiles(L, F, fv2id, feat2values):
        profs: dict[str, np.ndarray] = {}
        for feat, vals in feat2values.items():
            gids, lbls = [], []
            for v in vals:
                lbl = f"{feat}={v}"
                if lbl in fv2id:
                    gids.append(fv2id[lbl]); lbls.append(lbl)
            if not gids:
                continue
            scores = L @ F[gids].T          # (n_shared, K)
            p = sp_softmax(scores, axis=1)  # (n_shared, K)
            for k, lbl in enumerate(lbls):
                profs[lbl] = p[:, k].astype(np.float32)
        return profs

    wals_profs = _softmax_profiles(L_W, F_W, ck_w["fv2id"], ck_w["feat2values"])
    gb_profs   = _softmax_profiles(L_G, F_G, ck_g["fv2id"], ck_g["feat2values"])

    gb_labels = sorted(gb_profs.keys())
    gb_mat    = _cnorm(np.stack([gb_profs[l] for l in gb_labels]))  # (392, n_s)
    wals_fv2label = {l: i for i, l in enumerate(gb_labels)}          # not used

    hit = {1: 0, 5: 0, 10: 0}
    rr_list: list[float] = []
    n_eval = 0
    per_pair = []

    for wals_fv, gb_targets, domain in benchmark:
        if wals_fv not in wals_profs:
            continue

        wq = wals_profs[wals_fv]
        wq_norm = wq / max(float(np.linalg.norm(wq)), 1e-8)
        sims   = gb_mat @ wq_norm
        ranked = [gb_labels[i] for i in np.argsort(-sims)]

        resolved = [t for t in gb_targets if t in gb_profs]
        best_rank: int | None = None
        best_hit: str | None = None
        for t in resolved:
            rk = ranked.index(t) + 1
            if best_rank is None or rk < best_rank:
                best_rank = rk; best_hit = t

        rr = 1.0 / best_rank if best_rank else 0.0
        rr_list.append(rr)
        n_eval += 1
        if best_rank:
            for k in hit:
                if best_rank <= k:
                    hit[k] += 1
        per_pair.append({
            "wals": wals_fv, "targets": gb_targets, "domain": domain,
            "best_rank": best_rank, "best_hit": best_hit,
            "rr": round(rr, 4),
        })

    if n_eval == 0:
        return {"n_pairs": 0}

    return {
        "n_pairs":      n_eval,
        "recall_at_1":  round(hit[1]  / n_eval, 4),
        "recall_at_5":  round(hit[5]  / n_eval, 4),
        "recall_at_10": round(hit[10] / n_eval, 4),
        "mrr":          round(float(np.mean(rr_list)), 4),
        "per_pair":     per_pair,
    }


# ---------------------------------------------------------------------------
# Single-seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int, repo: Path, bm) -> dict[str, Any]:
    ck_w_t = load_checkpoint(str(repo / f"checkpoints/wals_tcf_s{seed}"))
    ck_g_t = load_checkpoint(str(repo / f"checkpoints/grambank_tcf_s{seed}"))
    ck_w_l = load_checkpoint(str(repo / f"checkpoints/wals_learned_s{seed}"))
    ck_g_l = load_checkpoint(str(repo / f"checkpoints/grambank_learned_s{seed}"))

    shared_t, idx_wt, idx_gt = _build_shared_index(ck_w_t["lang2id"], ck_g_t["lang2id"])
    shared_l, idx_wl, idx_gl = _build_shared_index(ck_w_l["lang2id"], ck_g_l["lang2id"])
    print(f"  s{seed}: shared_tcf={len(shared_t)}  shared_learned={len(shared_l)}")

    m_tcf = _tcf_retrieval(bm, ck_w_t, ck_g_t, idx_wt, idx_gt)
    m_lea = _learned_retrieval(bm, ck_w_l, ck_g_l, idx_wl, idx_gl)
    return {"tcf": m_tcf, "learned": m_lea}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_SEEDS_USED: list[int] = []

def _agg(results, key):
    metrics = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"]
    vals = {m: [r[key][m] for r in results] for m in metrics}
    return {
        "mean":     {m: round(float(np.mean(v)), 4) for m, v in vals.items()},
        "std":      {m: round(float(np.std(v, ddof=1)), 4) for m, v in vals.items()},
        "per_seed": {_SEEDS_USED[i]: {m: v[i] for m, v in vals.items()}
                     for i in range(len(results))},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=_DEFAULT_OUTPUT_DIR)
    p.add_argument("--seeds", type=int, nargs="+", default=_SEEDS)
    return p.parse_args(argv)


def main(argv=None):
    global _SEEDS_USED
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bm = _BENCHMARK_PAIRS
    _SEEDS_USED = args.seeds
    n_domains = len(set(d for _, _, d in bm))
    print(f"Benchmark: {len(bm)} pairs  |  {n_domains} domains")
    print(f"Method: T-CF sigma profiles × raw 206-candidate pool (no synthetic =0)\n")

    per_seed = []
    for seed in _SEEDS_USED:
        seed_everything(seed)
        r = run_seed(seed, _REPO_ROOT, bm)
        per_seed.append(r)
        m = r["tcf"]; l = r["learned"]
        print(f"  s{seed} T-CF:    R@1={m['recall_at_1']:.3f}  "
              f"R@5={m['recall_at_5']:.3f}  R@10={m['recall_at_10']:.3f}  "
              f"MRR={m['mrr']:.3f}  (n={m['n_pairs']})")
        print(f"  s{seed} Learned: R@1={l['recall_at_1']:.3f}  "
              f"R@5={l['recall_at_5']:.3f}  R@10={l['recall_at_10']:.3f}  "
              f"MRR={l['mrr']:.3f}  (n={l['n_pairs']})")

    agg_t = _agg(per_seed, "tcf")
    agg_l = _agg(per_seed, "learned")

    print("\n" + "=" * 70)
    print("  5-SEED AGGREGATE")
    print("=" * 70)
    for label, agg in [("T-CF", agg_t), ("Learned", agg_l)]:
        m = agg["mean"]; s = agg["std"]
        print(f"  {label:<10}  "
              f"R@1={m['recall_at_1']:.3f}±{s['recall_at_1']:.3f}  "
              f"R@5={m['recall_at_5']:.3f}±{s['recall_at_5']:.3f}  "
              f"R@10={m['recall_at_10']:.3f}±{s['recall_at_10']:.3f}  "
              f"MRR={m['mrr']:.3f}±{s['mrr']:.3f}")
    print("=" * 70)
    print(f"\nPaper Table 2 target:  T-CF R@10=0.108 MRR=0.062  "
          f"| Learned R@10=0.289±.013 MRR=0.155±.006")
    print(f"(gap vs paper = benchmark reconstruction: 68/78 vs 83 evaluable pairs)")

    out_path = out / "tcf_5seed_retrieval.json"
    out_path.write_text(json.dumps({
        "method":           "tcf_sigma_profiles_raw206_candidates",
        "benchmark_size":   len(bm),
        "benchmark_domains": list(set(d for _,_,d in bm)),
        "seeds":            _SEEDS_USED,
        "per_seed":         per_seed,
        "aggregated":       {"tcf": agg_t, "learned": agg_l},
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
