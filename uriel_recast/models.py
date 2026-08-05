"""Completion arms and categorical decoding.

Four completion arms over the URIEL union matrix (NaN = missing, else 0/1):

  ``softimpute``  Mazumder, Hastie & Tibshirani (2010), matching URIEL+'s config.
  ``knn_fill``    cosine-similarity neighbour averaging over the zero-filled matrix.
  ``fit_mf``      logistic matrix factorisation, binary or categorical loss.
  ``cat_decode_constrained``  the drop-in patch: read any continuous completion as a
                  per-group categorical choice, respecting cells that are still observed.

All randomness flows through an explicit ``numpy.random.default_rng(seed)``; there are
no global ``np.random`` calls in this module.
"""

import numpy as np


# ---------------------------------------------------------------------------
# SoftImpute
# ---------------------------------------------------------------------------

def softimpute(X, shrink, max_iter=200, tol=1e-4):
    """Soft-thresholded SVD imputation.

    Column-mean initial fill, then iterate: SVD, soft-threshold the singular values
    by ``shrink``, clip the reconstruction to [0, 1], restore observed cells. Stops
    when the relative Frobenius change in the working matrix drops below ``tol`` or
    after ``max_iter`` iterations.

    Returns the clipped low-rank reconstruction (*not* the observation-restored
    matrix), so callers see the model's opinion about every cell. Restoring observed
    values is the caller's job -- see ``cat_decode_constrained`` and the evaluation
    in ``run.py``, which both need to know which cells the model actually supplied.
    """
    obs = np.isfinite(X)
    with np.errstate(invalid="ignore"):
        colmean = np.nanmean(np.where(obs, X, np.nan), axis=0)
    # A column with no observation at all cannot have a mean; an unobserved binary
    # flag defaults to 0. With 3357 languages and 20% masking this never triggers.
    colmean = np.where(np.isfinite(colmean), colmean, 0.0)

    Z = np.where(obs, np.nan_to_num(X), colmean[None, :]).astype(np.float64)
    Zhat = Z
    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        s_thr = np.maximum(s - shrink, 0.0)
        Zhat = np.clip((U * s_thr) @ Vt, 0.0, 1.0)
        Znew = np.where(obs, Z, Zhat)
        denom = np.linalg.norm(Z)
        change = np.linalg.norm(Znew - Z) / max(denom, 1e-12)
        Z = Znew
        if change < tol:
            break
    return Zhat


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------

def knn_fill(X, k, block=256):
    """Cosine-similarity KNN completion.

    Similarity is computed over the zero-filled observed matrix; self-similarity is
    set to -inf so a language is never its own neighbour. Each cell is a
    similarity-weighted mean over the top-``k`` neighbours *that observe it*, with
    weights clipped at zero (a negative cosine contributes nothing rather than
    voting against). Cells no neighbour observes fall back to the column mean.

    Predicts every cell, including observed ones, so the return value is directly
    comparable to ``softimpute``'s reconstruction.
    """
    obs = np.isfinite(X)
    Z = np.where(obs, np.nan_to_num(X), 0.0).astype(np.float64)
    M = obs.astype(np.float64)

    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Zn = Z / np.maximum(norms, 1e-12)
    S = Zn @ Zn.T
    np.fill_diagonal(S, -np.inf)

    n = X.shape[0]
    kk = min(k, n - 1)
    # Top-k neighbours per row.
    part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]

    with np.errstate(invalid="ignore"):
        colmean = np.nanmean(np.where(obs, X, np.nan), axis=0)
    colmean = np.where(np.isfinite(colmean), colmean, 0.0)

    out = np.empty_like(Z)
    for start in range(0, n, block):
        stop = min(start + block, n)
        idx = part[start:stop]                       # (b, kk)
        w = np.take_along_axis(S[start:stop], idx, axis=1)
        w = np.clip(w, 0.0, None)                    # (b, kk)
        V = Z[idx]                                   # (b, kk, D)
        Mk = M[idx]                                  # (b, kk, D)
        num = np.einsum("bk,bkd->bd", w, V * Mk)
        den = np.einsum("bk,bkd->bd", w, Mk)
        blk = np.where(den > 0, num / np.maximum(den, 1e-12), colmean[None, :])
        out[start:stop] = blk
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Matrix factorisation
# ---------------------------------------------------------------------------

def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def group_gold(X, group_cols, strict=False):
    """Per-group gold classes under the 'observed at all' convention.

    A (language, group) instance participates when at least one member cell is
    observed. Its active count is the number of member cells that are observed and
    equal to 1. Gold is the index of the single active member if exactly one is
    active, ``len(cols)`` (NONE) if none is active, and the instance is *dropped*
    if two or more are active, since no single class represents that state.

    ``strict=True`` additionally drops *ambiguous* instances: those whose observed
    members are all 0 while some member cell is missing. Such an instance cannot
    distinguish "genuinely no member active" from "the active member is one of the
    cells that is missing", so labelling it NONE is a guess.

    This is not a cosmetic distinction. Under the ``cell`` masking protocol the
    lenient rule mislabels 6.8% of training instances as NONE, and those instances
    are precisely the held-out evaluation positions -- the model is trained to
    predict NONE exactly where it is about to be scored. See REPLICATION.md.

    Returns ``{group: (rows, gold)}`` with ``rows`` the participating language
    indices and ``gold`` the class index for each.
    """
    out = {}
    for gname, cols in group_cols.items():
        sub = X[:, cols]
        obs = np.isfinite(sub)
        act = (obs & (sub == 1)).sum(1)
        seen = obs.any(1)
        keep = seen & (act <= 1)
        if strict:
            keep &= ~((act == 0) & ~obs.all(1))
        rows = np.where(keep)[0]
        gold = np.full(rows.shape[0], len(cols), dtype=np.int64)
        one = act[rows] == 1
        if one.any():
            gold[one] = np.argmax(np.nan_to_num(sub[rows[one]], nan=0.0) == 1, axis=1)
        out[gname] = (rows, gold)
    return out


def fit_mf(X, group_cols, ungrouped_idx, mode, rank, l2, n_iter, seed,
           lr=0.05, betas=(0.9, 0.999), eps=1e-8, loss_norm="sum",
           strict_gold=False, verbose=False):
    """Logistic matrix factorisation, binary or categorical loss.

    Logits are ``Z = U @ W + b`` with ``U`` of shape ``(n, rank)`` and ``W`` of shape
    ``(rank, D + n_groups)``. The ``n_groups`` trailing columns are per-group NONE
    logits. They exist in *both* arms so the two arms have identical parameter
    counts; in the binary arm they simply receive no data gradient.

    ``mode='bin'``  sigmoid + BCE on all D columns, masked to observed cells.
    ``mode='cat'``  sigmoid + BCE on ungrouped columns only, plus for each group a
                    softmax over ``[member columns, NONE column]`` with cross-entropy
                    against the gold class from ``group_gold``.

    ``loss_norm='sum'`` sums the data loss over supervised units (observed cells for
    BCE, group instances for CE) and adds ``l2 * (||U||^2 + ||W||^2)``. This is the
    standard regularised-MF objective and is the regime in which the swept ``l2``
    values are neither negligible nor crushing; ``loss_norm='mean'`` divides the data
    term by the unit count and is provided for the sensitivity check reported in
    REPLICATION.md. ``b`` is unregularised in both.
    """
    n, D = X.shape
    gnames = list(group_cols.keys())
    n_groups = len(gnames)
    out_dim = D + n_groups

    rng = np.random.default_rng(seed)
    U = 0.05 * rng.standard_normal((n, rank))
    W = 0.05 * rng.standard_normal((rank, out_dim))
    b = np.zeros(out_dim)

    obs = np.isfinite(X)
    Y = np.nan_to_num(X, nan=0.0)

    if mode == "bin":
        mask = obs.astype(np.float64)
        n_units = float(mask.sum())
    elif mode == "cat":
        ung = np.asarray(ungrouped_idx, dtype=np.int64)
        mask_u = obs[:, ung].astype(np.float64)
        golds = group_gold(X, group_cols, strict=strict_gold)
        # Per group: participating rows, the logit columns [members..., NONE], gold.
        gspec = []
        for gi, gname in enumerate(gnames):
            rows, gold = golds[gname]
            cols = np.asarray(list(group_cols[gname]) + [D + gi], dtype=np.int64)
            gspec.append((rows, cols, gold))
        n_units = float(mask_u.sum() + sum(len(r) for r, _, _ in gspec))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    scale = 1.0 / n_units if loss_norm == "mean" else 1.0

    mU = np.zeros_like(U); vU = np.zeros_like(U)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    b1, b2 = betas

    for t in range(1, n_iter + 1):
        Z = U @ W + b
        G = np.zeros_like(Z)

        if mode == "bin":
            P = _sigmoid(Z[:, :D])
            G[:, :D] = (P - Y) * mask
        else:
            Pu = _sigmoid(Z[:, ung])
            G[np.ix_(np.arange(n), ung)] += (Pu - Y[:, ung]) * mask_u
            for rows, cols, gold in gspec:
                if rows.size == 0:
                    continue
                zl = Z[np.ix_(rows, cols)]
                zl = zl - zl.max(axis=1, keepdims=True)
                ez = np.exp(zl)
                p = ez / ez.sum(axis=1, keepdims=True)
                p[np.arange(rows.size), gold] -= 1.0
                G[np.ix_(rows, cols)] += p

        G *= scale
        gU = G @ W.T + 2.0 * l2 * U
        gW = U.T @ G + 2.0 * l2 * W
        gb = G.sum(axis=0)

        for prm, g, m, v in ((U, gU, mU, vU), (W, gW, mW, vW), (b, gb, mb, vb)):
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * (g * g)
            prm -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)

        if verbose and (t % 200 == 0 or t == 1):
            print(f"    iter {t:5d}  |U|={np.abs(U).mean():.4f} |W|={np.abs(W).mean():.4f}")

    return {"U": U, "W": W, "b": b, "D": D, "gnames": gnames,
            "group_cols": group_cols, "mode": mode}


def mf_scores(model):
    """Sigmoid probabilities over the D feature columns, plus the raw logit matrix."""
    Z = model["U"] @ model["W"] + model["b"]
    return _sigmoid(Z[:, :model["D"]]), Z


def mf_group_softmax(model, Z=None):
    """Per-group softmax over ``[member columns, NONE]``.

    Returns ``{group: (member_probs, none_prob)}``.
    """
    if Z is None:
        Z = model["U"] @ model["W"] + model["b"]
    D = model["D"]
    out = {}
    for gi, gname in enumerate(model["gnames"]):
        cols = list(model["group_cols"][gname]) + [D + gi]
        zl = Z[:, cols]
        zl = zl - zl.max(axis=1, keepdims=True)
        ez = np.exp(zl)
        p = ez / ez.sum(axis=1, keepdims=True)
        out[gname] = (p[:, :-1], p[:, -1])
    return out


# ---------------------------------------------------------------------------
# Categorical decoding
# ---------------------------------------------------------------------------

def cat_decode_naive(scores, group_cols, tau, base=None):
    """Per group: argmax over *all* member scores, all-zeros if the best is below tau.

    Only correct when a whole group is masked -- it overwrites cells that are still
    observed with its own argmax. Kept for the Phase 3 contrast; production paths use
    ``cat_decode_constrained``.
    """
    out = np.zeros_like(scores) if base is None else base.copy()
    for cols in group_cols.values():
        sub = scores[:, cols]
        best = sub.argmax(1)
        win = sub.max(1) >= tau
        blk = np.zeros_like(sub)
        blk[np.arange(sub.shape[0]), best] = 1.0
        out[:, cols] = np.where(win[:, None], blk, 0.0)
    return out


def cat_decode_constrained(scores, X_known, group_cols, tau=None,
                           none_scores=None, base=None):
    """Constrained categorical decode.

    ``X_known`` is the matrix *after* masking: finite entries are cells the model is
    allowed to see. For each (language, group):

      * cells still observed keep their observed value;
      * if an observed member cell is already 1 the group's value is known, so every
        free cell in that group goes to 0;
      * otherwise argmax over the *free* cells only, against NONE -- either a fixed
        threshold ``tau`` or, when ``none_scores`` supplies a per-group NONE score
        (the MF arm's own NONE logit), against that score.

    Free cells are everything not observed after masking, which covers both held-out
    cells and cells the union matrix never observed in the first place.
    """
    if (tau is None) == (none_scores is None):
        raise ValueError("supply exactly one of tau or none_scores")

    known = np.isfinite(X_known)
    out = np.zeros_like(scores) if base is None else base.copy()
    for gname, cols in group_cols.items():
        cols = np.asarray(cols)
        sub = scores[:, cols]
        kn = known[:, cols]
        kv = np.where(kn, np.nan_to_num(X_known[:, cols]), 0.0)

        free = ~kn
        settled = (kn & (kv == 1)).any(1)              # an observed member is already 1

        # Argmax restricted to free cells.
        masked_scores = np.where(free, sub, -np.inf)
        best = masked_scores.argmax(1)
        best_val = masked_scores.max(1)
        thr = tau if none_scores is None else none_scores[gname]
        win = np.isfinite(best_val) & (best_val >= thr)

        blk = np.zeros_like(sub)
        rows = np.where(win & ~settled)[0]
        blk[rows, best[rows]] = 1.0
        # Cells still observed keep their observed value.
        blk = np.where(kn, kv, blk)
        out[:, cols] = blk
    return out


def bin_decode(scores, X_known, thresh=0.5):
    """Binary completion: threshold at ``thresh``, but keep cells still observed."""
    known = np.isfinite(X_known)
    out = (scores >= thresh).astype(np.float64)
    return np.where(known, np.nan_to_num(X_known), out)
