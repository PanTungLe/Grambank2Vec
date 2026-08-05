"""Phase 3/4 -- masking, evaluation, arm dispatch, CLI.

Two masking protocols at 20%:

  ``group``  for each group, pick 20% of the languages that observe it and NaN out
             *all* member cells. Realistic: the language loses its word order.
  ``cell``   pick the same way, but NaN out one randomly chosen observed member cell.
             This is URIEL+'s own protocol, and it leaks -- knowing ``S_SVO = 1``
             determines the other five members of ``basic_order``.

The scoring rule that matters (and the one that is easy to get wrong): cell-level
metrics score **only held-out positions**, while group-level metrics score the
**completed profile**, in which cells still observed after masking keep their observed
values and only masked positions come from the model. Scoring every cell of a masked
group under the ``cell`` protocol inflates the binary baseline and penalises the
categorical decoder, because a naive decoder overwrites observed values with its own
argmax. ``run.py sanity`` checks the fix: under the ``group`` protocol nothing in a
masked group is observed, so constrained and naive decoding must agree bit-for-bit.
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from audit import load_union
from groups import GROUPS, resolve
from models import (bin_decode, cat_decode_constrained, cat_decode_naive, fit_mf,
                    knn_fill, mf_group_softmax, mf_scores, softimpute)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Selected in Phase 4 on validation seeds 901/902, on cell_F1.
DEFAULTS = dict(shrink=0.5, k=50, tau=0.35, rank=64, l2=0.02, n_iter=1200,
                strict_gold=False)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def make_mask(X, group_cols, protocol, seed, frac=0.2):
    """Return ``(X_masked, heldout, picks)``.

    ``heldout`` marks cells that were observed in ``X`` and are NaN in ``X_masked``.
    ``picks`` maps each group to the language indices selected for masking.
    """
    rng = np.random.default_rng(seed)
    Xm = X.copy()
    heldout = np.zeros(X.shape, dtype=bool)
    picks = {}
    for gname, cols in group_cols.items():
        cols = np.asarray(cols)
        obs = np.isfinite(X[:, cols])
        seen = np.where(obs.any(1))[0]
        n_pick = int(round(frac * seen.size))
        if n_pick == 0:
            picks[gname] = np.array([], dtype=np.int64)
            continue
        sel = np.sort(rng.choice(seen, size=n_pick, replace=False))
        if protocol == "group":
            blk = np.isfinite(X[np.ix_(sel, cols)])
            rr, cc = np.where(blk)
            heldout[sel[rr], cols[cc]] = True
            Xm[np.ix_(sel, cols)] = np.nan
        elif protocol == "cell":
            for r in sel:
                avail = cols[np.isfinite(X[r, cols])]
                c = avail[rng.integers(avail.size)]
                Xm[r, c] = np.nan
                heldout[r, c] = True
        else:
            raise ValueError(f"unknown protocol {protocol!r}")
        picks[gname] = sel
    return Xm, heldout, picks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(X, Xc, group_cols, heldout, picks):
    """Four metrics over the masked evaluation set.

    ``cell_F1``        micro P/R over held-out member cells only.
    ``group_acc``      predicted class == gold class, over masked group instances whose
                       gold is well formed (zero or one active flag). A prediction with
                       two or more active flags counts as wrong.
    ``exact_profile``  predicted binary vector == gold vector, over masked instances
                       whose member cells are *all* observed in the gold. This includes
                       multi-active golds, which the categorical arm cannot represent.
    ``validity``       share of predicted profiles with at most one active flag.
    """
    gold_h = X[heldout]
    pred_h = Xc[heldout]
    tp = float(((pred_h == 1) & (gold_h == 1)).sum())
    fp = float(((pred_h == 1) & (gold_h == 0)).sum())
    fn = float(((pred_h == 0) & (gold_h == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    acc_n = acc_d = ex_n = ex_d = val_n = val_d = 0
    for gname, cols in group_cols.items():
        rows = picks[gname]
        if rows.size == 0:
            continue
        cols = np.asarray(cols)
        K = cols.size
        g = X[np.ix_(rows, cols)]
        obs = np.isfinite(g)
        g0 = np.nan_to_num(g, nan=0.0)
        gold_act = (obs & (g == 1)).sum(1)

        p = Xc[np.ix_(rows, cols)]
        pred_act = p.sum(1).astype(np.int64)

        gold_cls = np.where(gold_act == 1, np.argmax(g0 == 1, axis=1), K)
        pred_cls = np.where(pred_act == 1, np.argmax(p == 1, axis=1),
                            np.where(pred_act == 0, K, -1))
        wf = gold_act <= 1
        acc_n += int((gold_cls[wf] == pred_cls[wf]).sum())
        acc_d += int(wf.sum())

        full = obs.all(1)
        ex_n += int((p[full] == g0[full]).all(1).sum())
        ex_d += int(full.sum())

        val_n += int((pred_act <= 1).sum())
        val_d += int(rows.size)

    return dict(
        cell_F1=f1, cell_P=prec, cell_R=rec,
        group_acc=acc_n / acc_d if acc_d else np.nan,
        exact_profile=ex_n / ex_d if ex_d else np.nan,
        validity=val_n / val_d if val_d else np.nan,
        n_heldout_cells=int(heldout.sum()), n_group_inst=val_d,
        n_wellformed=acc_d, n_fullobs=ex_d,
    )


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

class SeedCache:
    """Caches the mask and the deterministic completions for one (protocol, seed).

    The SoftImpute solve is computed once and read two ways by the ``bin`` and ``cat``
    arms: that is the same completion under two decoders, and recomputing it would be
    both wasteful and wrong.
    """

    def __init__(self, X, group_cols, protocol, seed):
        self.X, self.group_cols = X, group_cols
        self.Xm, self.heldout, self.picks = make_mask(X, group_cols, protocol, seed)
        self._si, self._knn = {}, {}

    def softimpute(self, shrink):
        if shrink not in self._si:
            self._si[shrink] = softimpute(self.Xm, shrink)
        return self._si[shrink]

    def knn(self, k):
        if k not in self._knn:
            self._knn[k] = knn_fill(self.Xm, k)
        return self._knn[k]


def complete_bin(scores, Xm):
    return bin_decode(scores, Xm)


def complete_cat(scores, Xm, group_cols, tau):
    base = bin_decode(scores, Xm)
    return cat_decode_constrained(scores, Xm, group_cols, tau=tau, base=base)


def run_arm(arm, cache, group_cols, ungrouped_idx, seed, hp):
    """Return ``(completed_matrix, extras)`` for one arm."""
    Xm = cache.Xm
    extras = {}
    if arm.startswith("softimpute"):
        S = cache.softimpute(hp["shrink"])
        Xc = complete_bin(S, Xm) if arm.endswith("_bin") else \
            complete_cat(S, Xm, group_cols, hp["tau"])
    elif arm.startswith("knn"):
        S = cache.knn(hp["k"])
        Xc = complete_bin(S, Xm) if arm.endswith("_bin") else \
            complete_cat(S, Xm, group_cols, hp["tau"])
    elif arm.startswith("mf"):
        mode = "bin" if arm.endswith("_bin") else "cat"
        model = fit_mf(Xm, group_cols, ungrouped_idx, mode=mode, rank=hp["rank"],
                       l2=hp["l2"], n_iter=hp["n_iter"], seed=seed,
                       loss_norm=hp.get("loss_norm", "sum"),
                       strict_gold=hp.get("strict_gold", False))
        P, Z = mf_scores(model)
        if mode == "bin":
            Xc = complete_bin(P, Xm)
        else:
            base = bin_decode(P, Xm)
            gs = mf_group_softmax(model, Z)
            scores = P.copy()
            none_scores = {}
            for gname, cols in group_cols.items():
                mp, npb = gs[gname]
                scores[:, cols] = mp
                none_scores[gname] = npb
            Xc = cat_decode_constrained(scores, Xm, group_cols,
                                        none_scores=none_scores, base=base)
        # Vowel-inventory value columns of W, for the Phase 5 ordinal probe.
        vcols = group_cols["vowel_quality_inventory"]
        extras["W_vowel"] = model["W"][:, vcols].copy()
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return Xc, extras


ARMS = ["softimpute_bin", "softimpute_cat", "knn_bin", "knn_cat", "mf_bin", "mf_cat"]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def load():
    X, feats, langs, keep = load_union()
    group_cols, grouped_idx, ungrouped_idx = resolve(feats)
    return X, feats, group_cols, ungrouped_idx


def cmd_test(args):
    X, feats, group_cols, ungrouped_idx = load()
    hp = dict(DEFAULTS)
    for key in ("shrink", "k", "tau", "rank", "l2", "n_iter", "strict_gold"):
        v = getattr(args, key, None)
        if v is not None:
            hp[key] = v
    rows, wv = [], {}
    for protocol in args.protocols:
        for seed in args.seeds:
            t0 = time.time()
            cache = SeedCache(X, group_cols, protocol, seed)
            for arm in args.arms:
                Xc, extras = run_arm(arm, cache, group_cols, ungrouped_idx, seed, hp)
                m = evaluate(X, Xc, group_cols, cache.heldout, cache.picks)
                m.update(arm=arm, protocol=protocol, seed=seed, config=args.tag, **hp)
                rows.append(m)
                if "W_vowel" in extras:
                    wv[f"{args.tag}|{arm}|{protocol}|{seed}"] = extras["W_vowel"]
                print(f"  {protocol:6s} seed {seed} {arm:16s} "
                      f"F1 {m['cell_F1']:.4f}  acc {m['group_acc']:.4f}  "
                      f"exact {m['exact_profile']:.4f}  val {m['validity']:.4f}")
            print(f"  -- {protocol} seed {seed} done in {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, args.out)
    if args.append and os.path.exists(out):
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    if wv:
        wout = os.path.join(RESULTS, args.out.replace(".csv", "_Wvowel.npz"))
        if args.append and os.path.exists(wout):
            old = dict(np.load(wout))
            old.update(wv)
            wv = old
        np.savez(wout, **wv)
        print(f"wrote {wout} ({len(wv)} arrays)")


def cmd_sanity(args):
    """Group-protocol numbers must be identical under naive and constrained decoding."""
    X, feats, group_cols, ungrouped_idx = load()
    hp = dict(DEFAULTS)
    seed = args.seeds[0]
    cache = SeedCache(X, group_cols, "group", seed)
    S = cache.softimpute(hp["shrink"])
    base = bin_decode(S, cache.Xm)
    con = cat_decode_constrained(S, cache.Xm, group_cols, tau=hp["tau"], base=base)
    nai = cat_decode_naive(S, group_cols, tau=hp["tau"], base=base)
    gcols = np.concatenate([np.asarray(c) for c in group_cols.values()])
    same = np.array_equal(con[:, gcols], nai[:, gcols])
    mc = evaluate(X, con, group_cols, cache.heldout, cache.picks)
    mn = evaluate(X, nai, group_cols, cache.heldout, cache.picks)
    ident = all(mc[k] == mn[k] for k in ("cell_F1", "group_acc", "exact_profile", "validity"))
    print(f"[sanity] group protocol, seed {seed}")
    print(f"  member-column matrices identical: {same}")
    print(f"  constrained  F1 {mc['cell_F1']:.6f} acc {mc['group_acc']:.6f} "
          f"exact {mc['exact_profile']:.6f} val {mc['validity']:.6f}")
    print(f"  naive        F1 {mn['cell_F1']:.6f} acc {mn['group_acc']:.6f} "
          f"exact {mn['exact_profile']:.6f} val {mn['validity']:.6f}")
    print(f"  metrics bit-identical: {ident} -> {'PASS' if ident else 'FAIL'}")

    # And under the cell protocol they must differ -- that is the whole point.
    cache2 = SeedCache(X, group_cols, "cell", seed)
    S2 = cache2.softimpute(hp["shrink"])
    b2 = bin_decode(S2, cache2.Xm)
    c2 = cat_decode_constrained(S2, cache2.Xm, group_cols, tau=hp["tau"], base=b2)
    n2 = cat_decode_naive(S2, group_cols, tau=hp["tau"], base=b2)
    m2c = evaluate(X, c2, group_cols, cache2.heldout, cache2.picks)
    m2n = evaluate(X, n2, group_cols, cache2.heldout, cache2.picks)
    print(f"[sanity] cell protocol, seed {seed} (constrained must beat naive)")
    print(f"  constrained  F1 {m2c['cell_F1']:.6f} acc {m2c['group_acc']:.6f} "
          f"exact {m2c['exact_profile']:.6f}")
    print(f"  naive        F1 {m2n['cell_F1']:.6f} acc {m2n['group_acc']:.6f} "
          f"exact {m2n['exact_profile']:.6f}")


def cmd_tune(args):
    X, feats, group_cols, ungrouped_idx = load()
    rows = []
    caches = {(p, s): SeedCache(X, group_cols, p, s)
              for p in args.protocols for s in args.seeds}

    def rec(**kw):
        rows.append(kw)
        print("  " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in kw.items()}))

    if args.what in ("all", "softimpute"):
        print("[tune] SoftImpute shrink")
        for shrink in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            for (p, s), c in caches.items():
                S = c.softimpute(shrink)
                m = evaluate(X, complete_bin(S, c.Xm), group_cols, c.heldout, c.picks)
                rec(sweep="softimpute", protocol=p, seed=s, shrink=shrink,
                    cell_F1=m["cell_F1"], group_acc=m["group_acc"],
                    exact_profile=m["exact_profile"], validity=m["validity"])

    if args.what in ("all", "knn"):
        print("[tune] KNN k")
        for k in [5, 10, 25, 50, 100, 200]:
            for (p, s), c in caches.items():
                S = c.knn(k)
                m = evaluate(X, complete_bin(S, c.Xm), group_cols, c.heldout, c.picks)
                rec(sweep="knn", protocol=p, seed=s, k=k,
                    cell_F1=m["cell_F1"], group_acc=m["group_acc"],
                    exact_profile=m["exact_profile"], validity=m["validity"])

    if args.what in ("all", "tau"):
        print("[tune] decode tau (on the selected SoftImpute solve)")
        for tau in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
            for (p, s), c in caches.items():
                S = c.softimpute(args.shrink)
                m = evaluate(X, complete_cat(S, c.Xm, group_cols, tau),
                             group_cols, c.heldout, c.picks)
                rec(sweep="tau", protocol=p, seed=s, tau=tau, shrink=args.shrink,
                    cell_F1=m["cell_F1"], group_acc=m["group_acc"],
                    exact_profile=m["exact_profile"], validity=m["validity"])

    if args.what in ("all", "mf"):
        print("[tune] MF rank x l2")
        for rank, l2 in itertools.product(args.ranks, args.l2s):
            for (p, s), c in caches.items():
                for mode, arm in (("bin", "mf_bin"), ("cat", "mf_cat")):
                    hp = dict(DEFAULTS, rank=rank, l2=l2, n_iter=args.n_iter,
                              loss_norm=args.loss_norm)
                    Xc, _ = run_arm(arm, c, group_cols, ungrouped_idx, s, hp)
                    m = evaluate(X, Xc, group_cols, c.heldout, c.picks)
                    rec(sweep="mf", protocol=p, seed=s, arm=arm, rank=rank, l2=l2,
                        n_iter=args.n_iter, loss_norm=args.loss_norm,
                        cell_F1=m["cell_F1"], group_acc=m["group_acc"],
                        exact_profile=m["exact_profile"], validity=m["validity"])

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, args.out)
    if args.append and os.path.exists(out):
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("test")
    t.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    t.add_argument("--protocols", nargs="+", default=["group", "cell"])
    t.add_argument("--arms", nargs="+", default=ARMS)
    t.add_argument("--out", default="results_all.csv")
    t.add_argument("--append", action="store_true")
    t.add_argument("--tag", default="A", help="config label recorded in the output CSV")
    for key, typ in (("shrink", float), ("k", int), ("tau", float),
                     ("rank", int), ("l2", float), ("n_iter", int)):
        t.add_argument(f"--{key}", type=typ, default=None)
    t.add_argument("--strict_gold", action="store_true", default=None,
                   help="drop ambiguous all-zero-observed group instances from the CE loss")
    t.set_defaults(func=cmd_test)

    v = sub.add_parser("tune")
    v.add_argument("--seeds", type=int, nargs="+", default=[901, 902])
    v.add_argument("--protocols", nargs="+", default=["group"])
    v.add_argument("--what", default="all",
                   choices=["all", "softimpute", "knn", "tau", "mf"])
    v.add_argument("--shrink", type=float, default=0.5)
    v.add_argument("--n_iter", type=int, default=1200)
    v.add_argument("--ranks", type=int, nargs="+", default=[32, 64])
    v.add_argument("--l2s", type=float, nargs="+", default=[0.005, 0.02, 0.08])
    v.add_argument("--loss_norm", default="sum", choices=["sum", "mean"])
    v.add_argument("--out", default="tuning.csv")
    v.add_argument("--append", action="store_true")
    v.set_defaults(func=cmd_tune)

    s = sub.add_parser("sanity")
    s.add_argument("--seeds", type=int, nargs="+", default=[42])
    s.set_defaults(func=cmd_sanity)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
