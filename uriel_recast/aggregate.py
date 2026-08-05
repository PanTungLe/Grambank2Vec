"""Phase 5 -- aggregate the test runs, paired contrasts, and the ordinal probe.

Reads ``results/results_all.csv`` (+ ``results/results_all_Wvowel.npz``) and writes
``results/summary_means.csv``, ``results/paired_contrasts.csv`` and
``results/ordinal_probe.csv``.

Paired two-sided t-tests are taken across the five test seeds, per protocol and per
config, for each of the three Bin/Cat contrasts on all four metrics. Five paired
observations is a small sample; the t statistics are reported as the protocol asks
for them, but the win count out of five carries at least as much weight as p.
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

from groups import GROUPS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
METRICS = ["cell_F1", "group_acc", "exact_profile", "validity"]
FAMILIES = ["softimpute", "knn", "mf"]


def paired(df, config, protocol, family, metric):
    b = df[(df.config == config) & (df.protocol == protocol) &
           (df.arm == f"{family}_bin")].sort_values("seed")
    c = df[(df.config == config) & (df.protocol == protocol) &
           (df.arm == f"{family}_cat")].sort_values("seed")
    if len(b) == 0 or len(c) == 0 or len(b) != len(c):
        return None
    if not np.array_equal(b.seed.values, c.seed.values):
        raise ValueError("seed mismatch between bin and cat arms")
    xb, xc = b[metric].to_numpy(float), c[metric].to_numpy(float)
    d = xc - xb
    if np.allclose(d, d[0]) and np.isclose(d.std(ddof=1), 0.0):
        # Constant delta (happens when validity is exactly 1.0 in every cat seed):
        # the t statistic is undefined, so report it as such rather than as infinity.
        t, p = np.nan, np.nan
    else:
        t, p = stats.ttest_rel(xc, xb)
    return dict(
        config=config, protocol=protocol, family=family, metric=metric,
        bin_mean=xb.mean(), bin_sd=xb.std(ddof=1),
        cat_mean=xc.mean(), cat_sd=xc.std(ddof=1),
        delta=d.mean(), delta_sd=d.std(ddof=1),
        wins=int((d > 0).sum()), n_seeds=len(d),
        t=float(t) if t == t else np.nan, p=float(p) if p == p else np.nan,
    )


def ordinal_probe(npz_path, scale):
    """Spearman rho between embedding distance and inventory-size difference.

    Takes the eight ``vowel_quality_inventory`` value columns out of ``W``,
    L2-normalises each, computes pairwise cosine distance over the 28 pairs, and
    correlates against the absolute difference in inventory size. One rho per arm
    per seed. A positive rho means the embedding geometry recovers the ordinal
    scale that the flat binary encoding throws away.
    """
    if not os.path.exists(npz_path):
        return pd.DataFrame()
    z = np.load(npz_path)
    sc = np.asarray(scale, dtype=float)
    iu = np.triu_indices(len(sc), k=1)
    gap = np.abs(sc[:, None] - sc[None, :])[iu]
    rows = []
    for key in z.files:
        tag, arm, protocol, seed = key.split("|")
        W = z[key]                                   # (rank, 8)
        Wn = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
        cos = Wn.T @ Wn
        dist = (1.0 - cos)[iu]
        rho, p = stats.spearmanr(dist, gap)
        rows.append(dict(config=tag, arm=arm, protocol=protocol, seed=int(seed),
                         rho=float(rho), p=float(p), n_pairs=len(gap)))
    return pd.DataFrame(rows).sort_values(["config", "protocol", "arm", "seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(RESULTS, "results_all.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.results)

    # ---- means per arm -----------------------------------------------------
    means = (df.groupby(["config", "protocol", "arm"])[METRICS]
               .agg(["mean", "std"]).round(6))
    means.columns = [f"{m}_{s}" for m, s in means.columns]
    means = means.reset_index()
    means.to_csv(os.path.join(RESULTS, "summary_means.csv"), index=False)

    for config in sorted(df.config.unique()):
        for protocol in sorted(df.protocol.unique()):
            sub = means[(means.config == config) & (means.protocol == protocol)]
            if sub.empty:
                continue
            print(f"\n=== config {config} | {protocol} protocol | mean over seeds ===")
            print(f"{'arm':16s} {'cell F1':>9s} {'group acc':>10s} {'exact':>9s} {'validity':>9s}")
            for _, r in sub.iterrows():
                print(f"{r['arm']:16s} {r['cell_F1_mean']:9.4f} {r['group_acc_mean']:10.4f} "
                      f"{r['exact_profile_mean']:9.4f} {r['validity_mean']:9.4f}")

    # ---- paired contrasts --------------------------------------------------
    rows = []
    for config, protocol, family, metric in itertools.product(
            sorted(df.config.unique()), sorted(df.protocol.unique()), FAMILIES, METRICS):
        r = paired(df, config, protocol, family, metric)
        if r:
            rows.append(r)
    pc = pd.DataFrame(rows)
    pc.to_csv(os.path.join(RESULTS, "paired_contrasts.csv"), index=False)

    for config in sorted(pc.config.unique()):
        for protocol in sorted(pc.protocol.unique()):
            sub = pc[(pc.config == config) & (pc.protocol == protocol)]
            if sub.empty:
                continue
            print(f"\n=== config {config} | {protocol} protocol | Cat - Bin ===")
            print(f"{'family':11s} {'metric':15s} {'bin':>8s} {'cat':>8s} "
                  f"{'delta':>9s} {'wins':>5s} {'t':>9s} {'p':>9s}")
            for _, r in sub.iterrows():
                t = "n/a" if r.t != r.t else f"{r.t:9.3f}"
                p = "n/a" if r.p != r.p else f"{r.p:9.5f}"
                print(f"{r.family:11s} {r.metric:15s} {r.bin_mean:8.4f} {r.cat_mean:8.4f} "
                      f"{r.delta:+9.4f} {r.wins:>3d}/{r.n_seeds:<2d} {t} {p}")

    # ---- ordinal probe -----------------------------------------------------
    op = ordinal_probe(args.results.replace(".csv", "_Wvowel.npz"),
                       GROUPS["vowel_quality_inventory"]["scale"])
    if op.empty:
        print("\n[ordinal probe] no W_vowel arrays found")
        return
    op.to_csv(os.path.join(RESULTS, "ordinal_probe.csv"), index=False)
    print("\n=== ordinal probe (vowel inventory size vs embedding distance) ===")
    for (config, protocol), g in op.groupby(["config", "protocol"]):
        b = g[g.arm == "mf_bin"].sort_values("seed")
        c = g[g.arm == "mf_cat"].sort_values("seed")
        if len(b) != len(c) or len(b) == 0:
            continue
        d = c.rho.to_numpy() - b.rho.to_numpy()
        t, p = stats.ttest_rel(c.rho.to_numpy(), b.rho.to_numpy())
        print(f"\n  config {config} | {protocol}")
        print(f"    mf_cat rho {c.rho.mean():+.4f} +/- {c.rho.std(ddof=1):.4f}   "
              f"per-seed {[round(v, 4) for v in c.rho]}")
        print(f"    mf_bin rho {b.rho.mean():+.4f} +/- {b.rho.std(ddof=1):.4f}   "
              f"per-seed {[round(v, 4) for v in b.rho]}")
        print(f"    paired delta {d.mean():+.4f}  {(d > 0).sum()}/{len(d)} seeds  "
              f"t={t:.3f}  p={p:.5f}")


if __name__ == "__main__":
    main()
