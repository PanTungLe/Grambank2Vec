"""
probe_verkerk.py — Verkerk et al. (2026) Universals × Embedding Geometry
=========================================================================

Replaces probe_tier_a.py as the primary universals geometry probe.

Source: TestingLinguisticUniversals / SI Data 1 / results.txt
        (Greenhill et al. GitHub; Verkerk et al. 2026, Nat. Hum. Behav.)

Design decision — formula aggregation (Difference 1 vs Tier-A):
  Tier-A decomposed compound universals into atomic single-token pairs
  (manual curation). Verkerk formulas are inherently compound (74% have
  >1 token on at least one side), so per-token decomposition would change
  the universal being tested. We instead apply structured aggregation that
  respects the logical form:

  Antecedent side:
    AND / mixed / single  → centroid of all ant embeddings
    pure OR               → min over disjuncts (conservative: every trigger
                            must point toward the consequent)

  Consequent side:
    single                → direct cosine against ant representation
    OR  (existential)     → max over con token cosines against ant repr.
    AND (universal)       → min over con token cosines against ant repr.
    mixed                 → centroid (practical fallback)

  This is strictly better than pure centroid (avoids OR-consequent inflation)
  and does not require manual curation.

Three metrics:
  M1   Structured implication cosine (above).
  M2   Anti-cosine: cos(ant_repr, neg_con_repr), where neg_con_repr is
       the centroid of all negation-token embeddings for each con token
       (i.e., the centroid of every other value of the same feature).
       For binary Grambank features, neg = the opposite pole exactly.
  M3   Analogy residual: ||(ant_repr − ant_neg_repr) − (con_repr − con_neg_repr)||
       Tested against a cross-feature null of 10,000 random quadruples.

Usage:
  python canonical/probe_verkerk.py
  python canonical/probe_verkerk.py --seeds 42,43,44,45,46 --n_null_cos 30000
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Default path; overridden at runtime by --universals_tsv.
# Expects data/ka_grambank_formulas.tsv (columns: row, KA_number, variant,
# formal_formula, domain, impl_grade, SPAPHY_SIG, supported, short_label).
_UNIVERSALS_TSV = _REPO_ROOT / "data" / "ka_grambank_formulas.tsv"

_TOKEN_RE = re.compile(r"GB\d+\s*:\s*\d+")


# ---------------------------------------------------------------------------
# Formula parser with connective detection
# ---------------------------------------------------------------------------

def _tokens(s: str) -> list[str]:
    return [t.replace(" ", "").replace(":", "=") for t in _TOKEN_RE.findall(s)]


def _detect_connective(s: str) -> str:
    """'single' | 'and' | 'or' | 'mixed'"""
    toks = _TOKEN_RE.findall(s)
    if len(toks) <= 1:
        return "single"
    stripped = _TOKEN_RE.sub("X", s)
    has_or  = "|" in stripped
    has_and = "&" in stripped
    if has_or and not has_and:
        return "or"
    if has_and and not has_or:
        return "and"
    return "mixed"


def parse_formula(formula: str) -> dict[str, Any] | None:
    """
    Returns dict with:
      ant_tokens, ant_connective,
      con_tokens, con_connective,
      formula_type   (simple | and_ant | or_ant | mixed_ant | complex_con)
    or None if not testable.
    """
    formula = re.sub(r"(?<![A-Z\d])193\s*:\s*1", "GB193:1", formula)
    if "sum(" in formula.lower() or ">" not in formula:
        return None
    parts = formula.split(">", 1)
    ant_str, con_str = parts[0].strip(), parts[1].strip()
    ant_toks = _tokens(ant_str)
    con_toks = _tokens(con_str)
    if not ant_toks or not con_toks:
        return None
    ant_conn = _detect_connective(ant_str)
    con_conn = _detect_connective(con_str)
    if len(ant_toks) == 1 and len(con_toks) == 1:
        ftype = "simple"
    elif ant_conn == "and" and len(ant_toks) > 1:
        ftype = "and_ant"
    elif ant_conn == "or":
        ftype = "or_ant"
    else:
        ftype = "mixed_ant" if con_conn == "single" else "complex_con"
    return dict(
        ant_tokens=ant_toks,
        ant_connective=ant_conn,
        con_tokens=con_toks,
        con_connective=con_conn,
        formula_type=ftype,
    )


# ---------------------------------------------------------------------------
# Negation lookup
# ---------------------------------------------------------------------------

def build_negation_map(
    fv2id: dict[str, int],
    feat2values: dict[str, list],
) -> dict[str, list[int]]:
    """
    For each featvalue label 'GBxxx=y', return the indices of every other
    value of the same feature that exists in fv2id.
    For binary features: exactly one negation index.
    For multi-valued: centroid of all other values.
    """
    neg_map: dict[str, list[int]] = {}
    for label, idx in fv2id.items():
        m = re.match(r"(GB\d+)=(\d+)", label)
        if not m:
            # T-CF bare key like "GB074": no negation entry needed
            continue
        feat, val = m.group(1), int(m.group(2))
        all_vals = feat2values.get(feat, [])
        neg_ids = []
        for v in all_vals:
            if v != val:
                neg_label = f"{feat}={v}"
                if neg_label in fv2id:
                    neg_ids.append(fv2id[neg_label])
        neg_map[label] = neg_ids
    return neg_map


# ---------------------------------------------------------------------------
# Resolve key (handles T-CF bare-key scheme)
# ---------------------------------------------------------------------------

def resolve(fv2id: dict[str, int], label: str) -> int | None:
    if label in fv2id:
        return fv2id[label]
    if label.endswith("=1"):
        bare = label[:-2]
        if bare in fv2id:
            return fv2id[bare]
    return None


# ---------------------------------------------------------------------------
# Cosine helpers
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def centroid(vecs: list[np.ndarray]) -> np.ndarray:
    return np.stack(vecs, axis=0).mean(axis=0)


def neg_centroid(
    fv_emb: np.ndarray,
    tokens: list[str],
    neg_map: dict[str, list[int]],
) -> np.ndarray | None:
    """Centroid of all negation embeddings for a list of tokens."""
    neg_vecs = []
    for tok in tokens:
        for nid in neg_map.get(tok, []):
            neg_vecs.append(fv_emb[nid])
    if not neg_vecs:
        return None
    return centroid(neg_vecs)


# ---------------------------------------------------------------------------
# Structured implication cosine (M1)
# ---------------------------------------------------------------------------

def structured_m1(
    fv_emb: np.ndarray,
    ant_vecs: list[np.ndarray],
    ant_conn: str,
    con_vecs: list[np.ndarray],
    con_conn: str,
) -> float:
    """
    Compute implication cosine with logical-form-aware aggregation.

    Antecedent representation:
      single / and / mixed → centroid
      or                   → min over disjuncts (conservative)

    Consequent test:
      single               → cosine(ant_repr, con_vec)
      or   (existential)   → max(cosine(ant_repr, ci) for ci in con_vecs)
      and  (universal)     → min(cosine(ant_repr, ci) for ci in con_vecs)
      mixed                → cosine(ant_repr, centroid(con_vecs))
    """
    if ant_conn == "or" and len(ant_vecs) > 1:
        # Compute consequent-side score for each disjunct separately, take min
        scores = []
        for av in ant_vecs:
            scores.append(_con_score(av, con_vecs, con_conn))
        valid = [s for s in scores if not np.isnan(s)]
        return float(min(valid)) if valid else float("nan")
    else:
        # AND / mixed / single: centroid as antecedent repr
        ant_repr = centroid(ant_vecs)
        return _con_score(ant_repr, con_vecs, con_conn)


def _con_score(
    ant_repr: np.ndarray,
    con_vecs: list[np.ndarray],
    con_conn: str,
) -> float:
    if len(con_vecs) == 1 or con_conn == "single":
        return cosine(ant_repr, con_vecs[0])
    elif con_conn == "or":
        scores = [cosine(ant_repr, cv) for cv in con_vecs]
        valid = [s for s in scores if not np.isnan(s)]
        return float(max(valid)) if valid else float("nan")
    elif con_conn == "and":
        scores = [cosine(ant_repr, cv) for cv in con_vecs]
        valid = [s for s in scores if not np.isnan(s)]
        return float(min(valid)) if valid else float("nan")
    else:  # mixed
        return cosine(ant_repr, centroid(con_vecs))


# ---------------------------------------------------------------------------
# Null distributions
# ---------------------------------------------------------------------------

def null_cosine(
    fv_emb: np.ndarray,
    fv2id: dict[str, int],
    n: int = 30_000,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    feat_to_rows: dict[str, list[int]] = {}
    for label, idx in fv2id.items():
        m = re.match(r"(GB\d+)", label)
        if m:
            feat_to_rows.setdefault(m.group(1), []).append(idx)
    feats = list(feat_to_rows.keys())
    sims: list[float] = []
    while len(sims) < n:
        fa, fb = rng.choice(len(feats), size=2, replace=False)
        ia = rng.choice(feat_to_rows[feats[fa]])
        ib = rng.choice(feat_to_rows[feats[fb]])
        c = cosine(fv_emb[ia], fv_emb[ib])
        if not np.isnan(c):
            sims.append(c)
    return np.array(sims[:n])


def null_analogy(
    fv_emb: np.ndarray,
    fv2id: dict[str, int],
    neg_map: dict[str, list[int]],
    n: int = 10_000,
    seed: int = 0,
) -> np.ndarray:
    """
    Null distribution for the analogy residual.
    Each sample: pick random tokens a, b (different features),
    compute ||(emb[a] - neg[a]) - (emb[b] - neg[b])||.
    Only samples where both tokens have negation entries are used.
    """
    rng = np.random.default_rng(seed)
    # Tokens that have at least one negation
    neg_tokens = [(label, idx) for label, idx in fv2id.items()
                  if neg_map.get(label)]
    feat_to_neg: dict[str, list[tuple[str, int]]] = {}
    for label, idx in neg_tokens:
        m = re.match(r"(GB\d+)", label)
        if m:
            feat_to_neg.setdefault(m.group(1), []).append((label, idx))
    feats = [f for f, v in feat_to_neg.items() if len(v) >= 1]
    residuals: list[float] = []
    while len(residuals) < n:
        fa, fb = rng.choice(len(feats), size=2, replace=False)
        la, ia = feat_to_neg[feats[fa]][rng.integers(len(feat_to_neg[feats[fa]]))]
        lb, ib = feat_to_neg[feats[fb]][rng.integers(len(feat_to_neg[feats[fb]]))]
        neg_a = neg_centroid(fv_emb, [la], neg_map)
        neg_b = neg_centroid(fv_emb, [lb], neg_map)
        if neg_a is None or neg_b is None:
            continue
        diff_a = fv_emb[ia] - neg_a
        diff_b = fv_emb[ib] - neg_b
        residuals.append(float(np.linalg.norm(diff_a - diff_b)))
    return np.array(residuals[:n])


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_ckpt(ckpt_dir: Path) -> dict[str, Any]:
    config = json.loads((ckpt_dir / "config.json").read_text())
    arch = config["architecture"]
    if arch == "learned":
        fv_emb = np.load(ckpt_dir / "featvalue_embeddings.npy")
        fv2id  = json.loads((ckpt_dir / "featvalue2id.json").read_text())
    else:
        fv_emb = np.load(ckpt_dir / "binarycol_embeddings.npy")
        fv2id  = json.loads((ckpt_dir / "binarycol2id.json").read_text())
    feat2values = json.loads((ckpt_dir / "feat2values.json").read_text())
    return dict(arch=arch, db=config["database"],
                fv_emb=fv_emb, fv2id=fv2id, feat2values=feat2values)


# ---------------------------------------------------------------------------
# Load universals
# ---------------------------------------------------------------------------

def load_universals(path: Path) -> list[dict[str, Any]]:
    """Load universals from ka_grambank_formulas.tsv.

    Expected columns (tab-separated, with header):
      row, KA_number, variant, formal_formula, domain,
      impl_grade, SPAPHY_SIG, supported, short_label
    """
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        tsv_rows = list(reader)
    records = []
    for r in tsv_rows:
        formula_raw = r.get("formal_formula", "").strip()
        if not formula_raw:
            continue
        parsed = parse_formula(formula_raw)
        # spaphy_sig: "yes"/"no" in TSV; supported: "SIG"/"NOT SIG"
        rec: dict[str, Any] = {
            "ka_num":      r.get("KA_number", "").strip(),
            "variant":     r.get("variant", "").strip(),
            "shorter":     r.get("short_label", "").strip(),
            "domain":      r.get("domain", "").strip(),
            "formula_raw": formula_raw,
            "impl_grade":  r.get("impl_grade", "").strip(),
            "spaphy_sig":  r.get("SPAPHY_SIG", "").strip(),
            "supported":   r.get("supported", "").strip(),
        }
        if parsed is None:
            rec["status"] = "not_testable"
        else:
            rec.update(parsed)
            rec["status"] = "pending"
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Per-seed probe
# ---------------------------------------------------------------------------

def probe_one_seed(
    universals: list[dict],
    fv_emb: np.ndarray,
    fv2id: dict[str, int],
    feat2values: dict[str, list],
    neg_map: dict[str, list[int]],
    null_cos: np.ndarray,
    null_ana: np.ndarray,
) -> list[dict]:
    null_mean = float(null_cos.mean())
    null_std  = float(null_cos.std())

    results = []
    for u in universals:
        rec = dict(u)
        if u["status"] == "not_testable":
            results.append(rec); continue

        # Resolve ant tokens
        ant_ids = [resolve(fv2id, t) for t in u["ant_tokens"]]
        con_ids = [resolve(fv2id, t) for t in u["con_tokens"]]
        ant_valid = [(t, i) for t, i in zip(u["ant_tokens"], ant_ids) if i is not None]
        con_valid = [(t, i) for t, i in zip(u["con_tokens"], con_ids) if i is not None]

        if not ant_valid or not con_valid:
            rec["status"] = "missing_fv"
            rec["missing_ant"] = [t for t, i in zip(u["ant_tokens"], ant_ids) if i is None]
            rec["missing_con"] = [t for t, i in zip(u["con_tokens"], con_ids) if i is None]
            results.append(rec); continue

        ant_vecs_all = [fv_emb[i] for _, i in ant_valid]
        con_vecs_all = [fv_emb[i] for _, i in con_valid]
        ant_toks_valid = [t for t, _ in ant_valid]
        con_toks_valid = [t for t, _ in con_valid]

        # ── M1: structured implication cosine ─────────────────────────────
        m1 = structured_m1(
            fv_emb,
            ant_vecs_all,
            u["ant_connective"],
            con_vecs_all,
            u["con_connective"],
        )
        m1_z = (m1 - null_mean) / (null_std + 1e-12) if not np.isnan(m1) else float("nan")
        m1_p = float((null_cos >= m1).mean())          if not np.isnan(m1) else float("nan")

        # Antecedent representation for M2/M3 (same logic as structured_m1)
        if u["ant_connective"] == "or" and len(ant_vecs_all) > 1:
            # For anti-cosine we use centroid (the "average trigger direction")
            ant_repr = centroid(ant_vecs_all)
        else:
            ant_repr = centroid(ant_vecs_all)

        # ── M2: anti-cosine ────────────────────────────────────────────────
        # neg_con = centroid of all negation embeddings for con tokens
        neg_con_vecs = []
        for t in con_toks_valid:
            for nid in neg_map.get(t, []):
                neg_con_vecs.append(fv_emb[nid])
        if neg_con_vecs:
            neg_con_repr = centroid(neg_con_vecs)
            cos_anti = cosine(ant_repr, neg_con_repr)
            impl_margin = (m1 - cos_anti) if not np.isnan(m1) else float("nan")
        else:
            cos_anti = float("nan")
            impl_margin = float("nan")

        # ── M3: analogy residual ────────────────────────────────────────────
        # Direction: (ant_repr − ant_neg_repr) and (con_repr − con_neg_repr)
        neg_ant_vecs = []
        for t in ant_toks_valid:
            for nid in neg_map.get(t, []):
                neg_ant_vecs.append(fv_emb[nid])
        con_repr = centroid(con_vecs_all)  # always centroid for analogy
        neg_con_repr_ana = centroid(neg_con_vecs) if neg_con_vecs else None

        if neg_ant_vecs and neg_con_repr_ana is not None:
            ant_neg_repr = centroid(neg_ant_vecs)
            dir_ant = ant_repr - ant_neg_repr
            dir_con = con_repr - neg_con_repr_ana
            ana_res = float(np.linalg.norm(dir_ant - dir_con))
            # empirical p = proportion of null residuals <= observed (smaller = better)
            ana_p = float((null_ana <= ana_res).mean())
        else:
            ana_res = float("nan")
            ana_p   = float("nan")

        rec.update(
            status="ok",
            null_mean=null_mean,
            null_std=null_std,
            null_ana_mean=float(null_ana.mean()),
            null_ana_median=float(np.median(null_ana)),
            n_ant=len(ant_valid),
            n_con=len(con_valid),
            m1=round(m1, 5),
            m1_z=round(m1_z, 4),
            m1_p=round(m1_p, 5),
            m1_above_null=bool(not np.isnan(m1) and m1 > null_mean),
            cos_anti=round(cos_anti, 5) if not np.isnan(cos_anti) else None,
            impl_margin=round(impl_margin, 5) if not np.isnan(impl_margin) else None,
            ana_res=round(ana_res, 5) if not np.isnan(ana_res) else None,
            ana_p=round(ana_p, 5) if not np.isnan(ana_p) else None,
        )
        results.append(rec)
    return results


# ---------------------------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------------------------

def aggregate(all_results: list[list[dict]]) -> list[dict]:
    n_univs = len(all_results[0])
    agg = []
    scalar_fields = ("m1", "m1_z", "m1_p",
                     "cos_anti", "impl_margin",
                     "ana_res", "ana_p")
    for i in range(n_univs):
        rows = [all_results[s][i] for s in range(len(all_results))]
        base = {k: v for k, v in rows[0].items() if k not in scalar_fields}
        for f in scalar_fields:
            vals = [r.get(f) for r in rows
                    if r.get(f) is not None
                    and not (isinstance(r.get(f), float) and np.isnan(r.get(f)))]
            base[f"{f}_mean"] = round(float(np.mean(vals)), 5) if vals else None
            base[f"{f}_std"]  = round(float(np.std(vals)),  5) if vals else None
        agg.append(base)
    return agg


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(v, fmt="+.3f"):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "  —  "
    return format(v, fmt)


def report(agg: list[dict], tag: str) -> str:
    from scipy import stats as sp

    ok  = [r for r in agg if r.get("status") == "ok"]
    nt  = [r for r in agg if r.get("status") == "not_testable"]
    mf  = [r for r in agg if r.get("status") == "missing_fv"]

    null_m = ok[0]["null_mean"]    if ok else 0.0
    null_s = ok[0]["null_std"]     if ok else 1.0
    ana_m  = ok[0]["null_ana_mean"]    if ok else 0.0
    ana_med= ok[0]["null_ana_median"]  if ok else 0.0

    def m1(r):  return r.get("m1_mean")
    def mg(r):  return r.get("impl_margin_mean")
    def ca(r):  return r.get("cos_anti_mean")
    def ar(r):  return r.get("ana_res_mean")
    def ap(r):  return r.get("ana_p_mean")

    def cat(r):
        if r.get("supported","").strip() == "SIG":  return "full"
        if r.get("spaphy_sig","").strip() == "yes":  return "brms"
        return "none"

    from collections import defaultdict
    groups = defaultdict(list)
    for r in ok: groups[cat(r)].append(r)

    def vals(grp, fn):
        return [fn(r) for r in grp if fn(r) is not None]

    L = []
    sep = "=" * 80
    L += [sep,
          f"  MODEL: {tag}",
          f"  191 universals: {len(ok)} tested | {len(nt)} not testable (sum/no->) | {len(mf)} missing fv",
          f"  Null cosine: mean={null_m:+.4f}  sd={null_s:.4f}  (n=30,000 cross-feature pairs)",
          f"  Null analogy: mean={ana_m:.4f}  median={ana_med:.4f}  (n=10,000 quadruples)",
          sep, ""]

    # ── Global ──────────────────────────────────────────────────────────────
    all_m1 = vals(ok, m1)
    all_mg = vals(ok, mg)
    all_ca = vals(ok, ca)
    n_ab = sum(1 for v in all_m1 if v > null_m)
    n_pos= sum(1 for v in all_m1 if v > 0)
    L += [
        f"  Global (n={len(ok)}):",
        f"    M1 (impl cos):   mean={np.mean(all_m1):+.4f}  above_null={n_ab}/{len(all_m1)} ({100*n_ab/max(len(all_m1),1):.0f}%)",
        f"    cos_anti:        mean={np.mean(all_ca):+.4f}",
        f"    impl_margin:     mean={np.mean(all_mg):+.4f}",
        "",
    ]

    # ── Three groups ─────────────────────────────────────────────────────────
    for name, key, n_paper in [
        ("Full (BRMS+BT)",  "full", 60),
        ("BRMS only",       "brms", 29),
        ("Unsupported",     "none", 102),
    ]:
        grp = groups[key]
        gm1 = vals(grp, m1); gmg = vals(grp, mg); gca = vals(grp, ca)
        gab = sum(1 for v in gm1 if v > null_m)
        L.append(f"  {name:18s} (n_paper={n_paper}, n_tested={len(grp)})")
        L.append(f"    M1={np.mean(gm1) if gm1 else float('nan'):+.4f}  "
                 f"cos_anti={np.mean(gca) if gca else float('nan'):+.4f}  "
                 f"margin={np.mean(gmg) if gmg else float('nan'):+.4f}  "
                 f"above_null={gab}/{len(gm1)}")

    fv = vals(groups["full"], m1)
    bv = vals(groups["brms"], m1)
    nv = vals(groups["none"], m1)
    if fv and nv:
        _, p_fn = sp.mannwhitneyu(fv, nv, alternative="greater")
        _, p_fb = sp.mannwhitneyu(fv, bv, alternative="greater")
        _, p_bn = sp.mannwhitneyu(bv, nv, alternative="greater")
        L += ["",
              f"  MWU (one-sided): full>none p={p_fn:.4f}  full>brms p={p_fb:.4f}  brms>none p={p_bn:.4f}"]
    L.append("")

    # ── By domain ────────────────────────────────────────────────────────────
    L.append("  Per domain (M1 | margin | full vs none Δ):")
    for dom in ("narrow word order","broad word order","hierarchy","other"):
        grp = [r for r in ok if r.get("domain","").lower().strip()==dom]
        if not grp: continue
        gm  = vals(grp, m1); gg = vals(grp, mg)
        gab = sum(1 for v in gm if v > null_m)
        s_  = vals([r for r in grp if cat(r)=="full"], m1)
        n_  = vals([r for r in grp if cat(r)=="none"], m1)
        Δ   = f"{np.mean(s_)-np.mean(n_):+.3f}" if s_ and n_ else "  —  "
        L.append(f"    {dom:25s}  n={len(grp):3d}  M1={np.mean(gm):+.3f}  "
                 f"margin={np.mean(gg) if gg else float('nan'):+.3f}  "
                 f"above={gab}/{len(gm)}  Δ(F-N)={Δ}")
    L.append("")

    # ── By formula type ───────────────────────────────────────────────────────
    L.append("  Per formula type:")
    for ftype in ("simple","and_ant","or_ant","mixed_ant","complex_con"):
        grp = [r for r in ok if r.get("formula_type","") == ftype]
        if not grp: continue
        gm = vals(grp, m1); gg = vals(grp, mg)
        gab = sum(1 for v in gm if v > null_m)
        L.append(f"    {ftype:15s}  n={len(grp):3d}  M1={np.mean(gm):+.3f}  "
                 f"margin={np.mean(gg) if gg else float('nan'):+.3f}  above={gab}/{len(gm)}")
    L.append("")

    # ── Analogy residual ──────────────────────────────────────────────────────
    ana_ok = [r for r in ok if ar(r) is not None]
    if ana_ok:
        ana_sig = [r for r in ana_ok if ap(r) is not None and ap(r) < 0.05]
        L += [
            f"  Analogy residual (n={len(ana_ok)} tested; null median={ana_med:.4f}):",
            f"    Significant (p<0.05): {len(ana_sig)}",
        ]
        for r in sorted(ana_sig, key=lambda r: ap(r)):
            L.append(f"    {r['ka_num'][:22]:22s} {r['variant']:>2}  "
                     f"res={ar(r):+.3f}  p={ap(r):.4f}  M1={m1(r):+.3f}  "
                     f"{r.get('shorter','')[:35]}")
    L.append("")

    # ── Full table: SPAPHY SIG ────────────────────────────────────────────────
    L.append("  ALL SPAPHY SIG universals, sorted by M1 desc:")
    L.append(f"  {'KA':22s} {'v':>2}  {'M1':>7}  {'anti':>7}  {'marg':>7}  "
             f"{'M1_z':>5}  {'ana_p':>6}  {'dom':20s}  {'grade':10s}")
    sig_sorted = sorted(
        [r for r in ok if r.get("spaphy_sig")=="yes" and m1(r) is not None],
        key=lambda r: -m1(r)
    )
    for r in sig_sorted:
        z = (m1(r) - null_m) / (null_s + 1e-12)
        L.append(f"  {r['ka_num'][:22]:22s} {r['variant']:>2}  "
                 f"{_f(m1(r)):>7}  {_f(ca(r)):>7}  {_f(mg(r)):>7}  "
                 f"{z:+.1f}   {_f(ap(r),'.3f'):>6}  "
                 f"{r['domain'][:20]:20s}  {r.get('impl_grade','')[:10]:10s}")
    L.append("")

    # ── Failures ──────────────────────────────────────────────────────────────
    miss = [r for r in ok if r.get("spaphy_sig")=="yes"
            and m1(r) is not None and m1(r) < null_m]
    L.append(f"  SPAPHY SIG but M1 below null (n={len(miss)}):")
    for r in sorted(miss, key=lambda r: m1(r)):
        L.append(f"    {r['ka_num'][:22]:22s}  M1={m1(r):+.4f}  "
                 f"{r.get('shorter','')[:40]}")
    L.append("")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_model(db, arch, seeds, ckpt_root, out_dir, universals,
              n_null_cos=30_000, n_null_ana=10_000):
    tag = f"{db}_{arch}"
    print(f"\n{'─'*65}\n  {tag}  seeds={seeds}\n{'─'*65}")
    if db != "grambank":
        print("  Skipping — Verkerk formulas are Grambank-only."); return

    seed0_dir = ckpt_root / f"{db}_{arch}_s{seeds[0]}"
    if not seed0_dir.exists():
        print(f"  [skip] {seed0_dir} not found"); return

    ckpt0 = load_ckpt(seed0_dir)
    print(f"  fv_emb: {ckpt0['fv_emb'].shape}")
    neg_map0 = build_negation_map(ckpt0["fv2id"], ckpt0["feat2values"])
    print(f"  Building null cosine (n={n_null_cos})…")
    null_cos = null_cosine(ckpt0["fv_emb"], ckpt0["fv2id"], n=n_null_cos)
    print(f"  null_cos: mean={null_cos.mean():+.4f}  sd={null_cos.std():.4f}")
    print(f"  Building null analogy (n={n_null_ana})…")
    null_ana = null_analogy(ckpt0["fv_emb"], ckpt0["fv2id"], neg_map0, n=n_null_ana)
    print(f"  null_ana: mean={null_ana.mean():.4f}  median={np.median(null_ana):.4f}")

    all_results = []
    for s in seeds:
        cdir = ckpt_root / f"{db}_{arch}_s{s}"
        if not cdir.exists(): print(f"  skip {cdir}"); continue
        ckpt = load_ckpt(cdir)
        neg_map = build_negation_map(ckpt["fv2id"], ckpt["feat2values"])
        sr = probe_one_seed(universals, ckpt["fv_emb"], ckpt["fv2id"],
                            ckpt["feat2values"], neg_map,
                            null_cos, null_ana)
        n_ok = sum(1 for r in sr if r["status"] == "ok")
        print(f"  seed {s}: {n_ok}/{len(universals)} ok")
        all_results.append(sr)

    if not all_results: print("  No seeds."); return
    agg = aggregate(all_results)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{tag}_verkerk_results.json"
    json_path.write_text(json.dumps({"model": tag, "results": agg}, indent=2))
    rpt = report(agg, tag)
    print(rpt)
    (out_dir / f"{tag}_verkerk_report.txt").write_text(rpt)
    print(f"  Saved: {json_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_root", default=None)
    p.add_argument("--universals_tsv",
                   default=str(_UNIVERSALS_TSV),
                   help="Path to ka_grambank_formulas.tsv.")
    p.add_argument("--output_dir", default="analysis/verkerk_universals")
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--databases", default="grambank")
    p.add_argument("--architectures", default="learned,tcf")
    p.add_argument("--n_null_cos", type=int, default=30_000)
    p.add_argument("--n_null_ana", type=int, default=10_000)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    ckpt_root = (Path(args.checkpoint_root) if args.checkpoint_root
                 else _REPO_ROOT / "checkpoints")
    out_dir = (_REPO_ROOT / args.output_dir
               if not Path(args.output_dir).is_absolute()
               else Path(args.output_dir))

    universals = load_universals(Path(args.universals_tsv))
    n_ok = sum(1 for u in universals if u["status"] == "pending")
    n_nt = sum(1 for u in universals if u["status"] == "not_testable")
    print(f"Loaded {len(universals)} universals: {n_ok} testable, {n_nt} not testable")

    for db in args.databases.split(","):
        for arch in args.architectures.split(","):
            run_model(db, arch, seeds, ckpt_root, out_dir,
                      universals, n_null_cos=args.n_null_cos,
                      n_null_ana=args.n_null_ana)
    print(f"\nDone. Output in {out_dir}/")


if __name__ == "__main__":
    main()
