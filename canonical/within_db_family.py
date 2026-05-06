"""
Within-database family-preservation probe across seeds.

For each (database, architecture) canonical setting, runs the family-
preservation probe on the FULL Glottocoded language set (not the WALS/Grambank
shared subset) for seeds 42–46 and aggregates score mean ± std.

Output: ``analysis/within_db_family/results.json``

Usage:
    python canonical/within_db_family.py \
        --checkpoints_dir checkpoints \
        --family_csv analysis/families.csv \
        --output_dir analysis/within_db_family
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "canonical"))
from compare_databases import cosine_normalise, load_family_map  # type: ignore


def within_db_family_probe(
    glottocodes: list[str],
    emb: np.ndarray,
    family_map: dict[str, str],
    top_k: int = 10,
    n_baseline: int = 500,
    seed: int = 42,
) -> dict:
    families = [family_map.get(g) for g in glottocodes]
    fam_counts = Counter(f for f in families if f is not None)
    valid = [f is not None and fam_counts[f] >= 2 for f in families]
    valid_mask = np.array(valid, dtype=bool)
    if valid_mask.sum() < 2:
        return {"note": "Fewer than 2 valid languages."}

    normed = cosine_normalise(emb)
    n = len(normed)
    top_k_nbrs = np.empty((n, top_k), dtype=np.int64)
    for i in range(n):
        sims = normed @ normed[i]
        sims[i] = -np.inf
        top_k_nbrs[i] = np.argsort(-sims)[:top_k]

    def _score(fam_labels: list) -> float:
        fracs = []
        for i in np.where(valid_mask)[0]:
            my = fam_labels[i]
            same = sum(1 for j in top_k_nbrs[i] if fam_labels[j] == my)
            fracs.append(same / top_k)
        return float(np.mean(fracs))

    score = _score(families)
    rng = np.random.default_rng(seed)
    baseline = np.array([
        _score(list(rng.permutation(families))) for _ in range(n_baseline)
    ])
    return {
        "n_valid_languages": int(valid_mask.sum()),
        "n_total_languages": int(len(glottocodes)),
        "n_families": int(len({families[i] for i in np.where(valid_mask)[0]})),
        "score": score,
        "baseline_mean": float(baseline.mean()),
        "baseline_p95": float(np.percentile(baseline, 95)),
        "p_value": float((baseline >= score).mean()),
    }


def load_glottocoded_lang_emb(checkpoint_dir: Path):
    lang_emb = np.load(checkpoint_dir / "lang_embeddings.npy")
    with open(checkpoint_dir / "lang2id.json") as f:
        lang2id = json.load(f)
    items = sorted(lang2id.items(), key=lambda kv: kv[1])
    glottocodes = [gc for gc, _ in items]
    rows = [idx for _, idx in items]
    return lang_emb[rows], glottocodes


def aggregate(per_seed: list[dict]) -> dict:
    scores = [r["score"] for r in per_seed]
    baselines = [r["baseline_mean"] for r in per_seed]
    return {
        "n_seeds": len(per_seed),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "baseline_mean": float(np.mean(baselines)),
        "n_valid_languages": per_seed[0]["n_valid_languages"],
        "n_total_languages": per_seed[0]["n_total_languages"],
        "n_families": per_seed[0]["n_families"],
        "per_seed": per_seed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_dir", type=Path,
                   default=REPO_ROOT / "checkpoints")
    p.add_argument("--family_csv", type=Path,
                   default=REPO_ROOT / "analysis" / "families.csv")
    p.add_argument("--output_dir", type=Path,
                   default=REPO_ROOT / "analysis" / "within_db_family")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 43, 44, 45, 46])
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_baseline", type=int, default=500)
    args = p.parse_args()

    family_map = load_family_map(args.family_csv)
    print(f"Loaded {len(family_map)} glottocodes / "
          f"{len(set(family_map.values()))} families\n")

    settings = [
        ("wals", "learned"),
        ("wals", "tcf"),
        ("grambank", "learned"),
        ("grambank", "tcf"),
    ]

    results = {}
    for db, arch in settings:
        key = f"{db}_{arch}"
        per_seed = []
        for s in args.seeds:
            ckpt = args.checkpoints_dir / f"{db}_{arch}_s{s}"
            if not ckpt.exists():
                print(f"  MISSING {ckpt}")
                continue
            emb, gcodes = load_glottocoded_lang_emb(ckpt)
            r = within_db_family_probe(
                gcodes, emb, family_map,
                top_k=args.top_k, n_baseline=args.n_baseline, seed=s,
            )
            r["seed"] = s
            per_seed.append(r)
            print(f"  {key} s{s}: score={r['score']:.4f}  "
                  f"baseline={r['baseline_mean']:.4f}  p={r['p_value']:.3f}")
        if per_seed:
            agg = {"setting": key, **aggregate(per_seed)}
            results[key] = agg
            print(f"  → {key}: {agg['score_mean']:.4f} ± {agg['score_std']:.4f}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Written to {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
