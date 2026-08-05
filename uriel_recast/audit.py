"""Phase 1 -- build the union matrix and audit group coherence.

Writes ``results/audit_coherence.csv``: for every group, on three views (WALS source
only, SSWL source only, union), how many languages observe the group at all and what
share of those carry zero, exactly one, or two-or-more active flags.

Under a correct categorical reading every observed instance would carry zero or one
active flag. Anything in ``pct_multi`` is a structurally impossible profile that the
independent-binary-flag encoding permits.
"""

import argparse
import os

import numpy as np
import pandas as pd

from groups import GROUPS, resolve

NPZ = "/tmp/URIELPlus/urielplus/database/original_uriel/features.npz"
HERE = os.path.dirname(os.path.abspath(__file__))


def load_raw(path=NPZ):
    d = np.load(path, allow_pickle=True)
    return d["data"], d["feats"], d["langs"], d["sources"]


def load_union(path=NPZ):
    """URIEL+'s 'U' aggregation: observed if any source observes, value = logical OR.

    Languages with no observation anywhere are dropped. Gate 1a pins the result at
    shape (3357, 289) with observed density 0.3922.
    """
    data, feats, langs, sources = load_raw(path)
    obs = (data != -1)
    any_obs = obs.any(2)
    pos = ((data == 1) & obs).any(2)
    X = np.where(any_obs, pos.astype(np.float32), np.nan)
    keep = any_obs.any(1)
    return X[keep].astype(np.float64), feats, langs[keep], keep


def source_view(data, keep, src_idx):
    """Single-source view over the same language set as the union matrix."""
    sl = data[:, :, src_idx]
    obs = sl != -1
    X = np.where(obs, (sl == 1).astype(np.float32), np.nan)
    return X[keep].astype(np.float64)


def coherence_rows(X, group_cols, view):
    rows = []
    for gname, cols in group_cols.items():
        sub = X[:, cols]
        obs = np.isfinite(sub)
        seen = obs.any(1)
        act = (obs & (sub == 1)).sum(1)[seen]
        n = int(seen.sum())
        if n == 0:
            rows.append(dict(group=gname, kind=GROUPS[gname]["kind"],
                             wals=GROUPS[gname]["wals"] or "", view=view,
                             n_members=len(cols), n_langs_observed=0,
                             pct_zero=np.nan, pct_one=np.nan, pct_multi=np.nan,
                             max_active=0, n_multi=0))
            continue
        rows.append(dict(
            group=gname, kind=GROUPS[gname]["kind"],
            wals=GROUPS[gname]["wals"] or "", view=view,
            n_members=len(cols), n_langs_observed=n,
            pct_zero=100.0 * float((act == 0).mean()),
            pct_one=100.0 * float((act == 1).mean()),
            pct_multi=100.0 * float((act >= 2).mean()),
            max_active=int(act.max()),
            n_multi=int((act >= 2).sum()),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "audit_coherence.csv"))
    args = ap.parse_args()

    data, feats, langs, sources = load_raw()
    X, feats, langs_kept, keep = load_union()
    group_cols, grouped_idx, ungrouped_idx = resolve(feats)

    dens = float(np.isfinite(X).mean())
    print(f"[Gate 1a] union shape {X.shape} (want (3357, 289)) "
          f"-> {'PASS' if X.shape == (3357, 289) else 'FAIL'}")
    print(f"[Gate 1a] observed density {dens:.6f} -> {dens:.4f} (want 0.3922) "
          f"-> {'PASS' if abs(dens - 0.3922) < 5e-5 else 'FAIL'}")
    print(f"          {len(group_cols)} groups over {len(grouped_idx)} of {len(feats)} features; "
          f"{len(ungrouped_idx)} ungrouped")

    src = {s: i for i, s in enumerate(sources)}
    views = {
        "wals": source_view(data, keep, src["WALS"]),
        "sswl": source_view(data, keep, src["SSWL"]),
        "union": X,
    }

    rows = []
    for vname, VX in views.items():
        rows += coherence_rows(VX, group_cols, vname)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"          wrote {args.out}  ({len(df)} rows)")

    # ---- Gate 1b -----------------------------------------------------------
    def cell(group, view, col):
        return float(df[(df.group == group) & (df.view == view)][col].iloc[0])

    checks = [
        ("basic_order wals pct_one", cell("basic_order", "wals", "pct_one"), 94.4),
        ("basic_order sswl pct_multi", cell("basic_order", "sswl", "pct_multi"), 44.7),
        ("basic_order sswl max_active", cell("basic_order", "sswl", "max_active"), 6),
        ("basic_order union pct_multi", cell("basic_order", "union", "pct_multi"), 10.1),
        ("basic_order union max_active", cell("basic_order", "union", "max_active"), 6),
        ("vowel_quality_inventory union pct_multi",
         cell("vowel_quality_inventory", "union", "pct_multi"), 9.6),
        ("vowel_quality_inventory union max_active",
         cell("vowel_quality_inventory", "union", "max_active"), 5),
    ]
    print("\n[Gate 1b]")
    ok = True
    for name, got, want in checks:
        if "max_active" in name:
            good = int(got) == int(want)
            print(f"  {name:44s} got {int(got)}  want {int(want)}  {'PASS' if good else 'FAIL'}")
        else:
            good = abs(round(got, 1) - want) < 0.05
            print(f"  {name:44s} got {got:.1f}  want {want:.1f}  {'PASS' if good else 'FAIL'}")
        ok &= good
    print(f"  Gate 1b overall: {'PASS' if ok else 'FAIL'}")

    # Co-observation: how often is a group only partially observed? This is what
    # licenses the 'observed at all' convention used for gold classes in Phase 3.
    part = []
    for gname, cols in group_cols.items():
        sub = np.isfinite(X[:, cols])
        seen = sub.any(1)
        if seen.sum():
            full = sub[seen].all(1).mean()
            part.append((gname, float(full)))
    fr = np.array([p[1] for p in part])
    print(f"\n[co-observation] share of observed group instances that are FULLY observed: "
          f"min {fr.min():.4f}  median {np.median(fr):.4f}  mean {fr.mean():.4f}")
    worst = sorted(part, key=lambda t: t[1])[:3]
    print("                 lowest: " + ", ".join(f"{g} {v:.4f}" for g, v in worst))


if __name__ == "__main__":
    main()
