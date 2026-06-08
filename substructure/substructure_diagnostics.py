"""
src/substructure_diagnostics.py
================================
Substructure-recovery diagnostics for WALS feature-value embeddings.

Five diagnostic functions, each suited to one of the annotation structure types
defined in data/substructure_annotation.json:

    ordinal_spearman    — for type == "ordinal"
    triplet_accuracy    — for type == "presence_subtype"
    lattice_r2          — for type == "boolean_lattice"
    cross_product_probe — for type == "permutation"
    nn_purity           — for type == "strategy_family"

All functions expect *within-feature-centred* embeddings:

    E_centred = E_raw - E_raw.mean(axis=0)

This removes the feature-level offset so that geometry reflects within-feature
value contrasts rather than between-feature identity.  Use load_feature_embeddings()
to obtain centred embeddings directly from a checkpoint directory.

Results from running these diagnostics on all 70 annotated features across seeds
42–46 are summarised in the _results_summary block of substructure_annotation.json.

Key empirical findings (WALS, seed 42, Learned vs T-CF):
  - 22A Verb Synthesis:  Learned ρ = +0.52, T-CF ρ = -0.45  (sign reversal)
  - 37A Definite Articles: triplet_acc Learned = 1.00, T-CF = 0.25
  - 84A Object-Oblique-Verb: cross_product_probe verb_pos Learned = 1.00, T-CF = 0.55
  - 102A Verbal Person Marking: lattice_r2 Learned = -0.82, T-CF = -2.30
  - Strategy families (98A–121A): nn_purity = 0.00 in BOTH architectures
    (paradigmatic-competitor effect: same-feature values are pushed apart by the
    softmax objective; typological structure lives cross-feature, not within-feature)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut


# ── types ──────────────────────────────────────────────────────────────────────

ValueIndex = dict[str, int]   # value_string → row index into E
DiagResult = dict[str, Any]   # returned by every diagnostic


# ── embedding utilities ────────────────────────────────────────────────────────

def center(E: np.ndarray) -> np.ndarray:
    """Remove the feature-level offset from a value-embedding matrix."""
    return E - E.mean(axis=0, keepdims=True)


def cosine_distance_matrix(E: np.ndarray) -> np.ndarray:
    """(n, n) cosine distance matrix for an (n, d) embedding matrix.

    Note: distances can exceed 1.0 for centred embeddings when normalised vectors
    have negative dot products.  This is expected and does not affect rankings.
    """
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    En = E / norms
    return 1.0 - En @ En.T


def load_feature_embeddings(
    feat_id: str,
    arch: str = "learned",
    seed: int = 42,
    checkpoint_dir: str | Path = "checkpoints",
) -> tuple[np.ndarray, ValueIndex]:
    """Load and centre the embedding rows for a single WALS feature.

    Parameters
    ----------
    feat_id : str
        WALS feature identifier, e.g. ``"81A"``.
    arch : str
        Architecture name: ``"learned"`` (softmax categorical) or ``"tcf"``
        (binary Bernoulli T-CF baseline).
    seed : int
        Training seed (42–46 in the Grambank2Vec training runs).
    checkpoint_dir : path-like
        Path to the directory containing ``wals_{arch}_s{seed}/`` subdirs.

    Returns
    -------
    E : np.ndarray, shape (n_values, embed_dim)
        Within-feature-centred embedding matrix, one row per value that is
        present in the checkpoint's vocabulary.
    value_index : ValueIndex
        ``{value_string: row_index}`` mapping into E.

    Raises
    ------
    FileNotFoundError
        If the checkpoint directory does not exist.
    KeyError
        If no values for ``feat_id`` are found in the vocabulary.
    """
    ckpt = Path(checkpoint_dir) / f"wals_{arch}_s{seed}"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if arch == "learned":
        emb = np.load(ckpt / "featvalue_embeddings.npy")
        k2id: dict[str, int] = json.load(open(ckpt / "featvalue2id.json"))
    else:
        emb = np.load(ckpt / "binarycol_embeddings.npy")
        k2id = json.load(open(ckpt / "binarycol2id.json"))

    prefix = feat_id + "="
    rows, labels = [], []
    for key, idx in k2id.items():
        if key.startswith(prefix):
            rows.append(emb[idx])
            labels.append(key[len(prefix):])

    if not rows:
        raise KeyError(f"No values found for feature {feat_id!r} in {ckpt}")

    E_raw = np.stack(rows)
    E = center(E_raw)
    value_index = {v: i for i, v in enumerate(labels)}
    return E, value_index


def load_annotation(path: str | Path = "data/substructure_annotation.json") -> dict:
    """Load the feature annotation file."""
    return json.load(open(path, encoding="utf-8"))


# ── diagnostic 1: ordinal_spearman ────────────────────────────────────────────

def ordinal_spearman(
    E: np.ndarray,
    value_index: ValueIndex,
    rank_dict: dict[str, int],
) -> DiagResult:
    """Spearman ρ between gold ordinal rank-distance and cosine distance.

    Applicable to features of type ``"ordinal"`` in the annotation file, where
    ``rank_dict`` maps each value string to its integer rank (0 = smallest/least).

    The gold distance between values i and j is |rank_i - rank_j|.  A positive ρ
    means the embedding places higher-ranked values farther from lower-ranked ones,
    i.e. the ordinal geometry is recovered.

    Parameters
    ----------
    E : np.ndarray, shape (n, d)
        Within-feature-centred value embeddings.
    value_index : ValueIndex
        ``{value_string: row_index}`` mapping into E.
    rank_dict : dict
        ``{value_string: rank_int}`` from the annotation's ``"values"`` field.

    Returns
    -------
    dict with keys:
        spearman_rho : float
        p_value      : float
        n_values     : int   (values present in both value_index and rank_dict)
    """
    items = [(v, r) for v, r in rank_dict.items() if v in value_index]
    n = len(items)
    if n < 3:
        return {"spearman_rho": float("nan"), "p_value": float("nan"), "n_values": n}

    idxs  = [value_index[v] for v, _ in items]
    ranks = np.array([r for _, r in items])

    D_emb  = cosine_distance_matrix(E[idxs])
    D_gold = squareform(pdist(ranks.reshape(-1, 1), "cityblock"))

    iu = np.triu_indices_from(D_emb, k=1)
    rho, p = spearmanr(D_emb[iu], D_gold[iu])
    return {"spearman_rho": float(rho), "p_value": float(p), "n_values": n}


# ── diagnostic 2: triplet_accuracy ────────────────────────────────────────────

def triplet_accuracy(
    E: np.ndarray,
    value_index: ValueIndex,
    triplets: list[tuple[str, str, str]],
) -> DiagResult:
    """Proportion of triplets where anchor is closer to the positive than the negative.

    Applicable to features of type ``"presence_subtype"`` (and optionally
    ``"permutation"`` when a ``"triplets"`` field is present).  Each triplet is
    ``(anchor, near, far)``: the near value should be closer to anchor than far.

    The canonical use case is features like WALS 13A Tone, where the expected
    geometry is d(Simple, Complex) < d(No tones, Simple).

    Parameters
    ----------
    E : np.ndarray, shape (n, d)
        Within-feature-centred value embeddings.
    value_index : ValueIndex
        ``{value_string: row_index}`` mapping into E.
    triplets : list of (anchor, near, far) tuples
        From the annotation's ``"triplets"`` field.

    Returns
    -------
    dict with keys:
        accuracy    : float   (NaN if no eligible triplets)
        n_correct   : int
        n_evaluated : int     (triplets where all three values are present)
        n_skipped   : int     (triplets with a missing value)
    """
    D = cosine_distance_matrix(E)
    correct = 0
    evaluated = 0
    skipped = 0

    for anchor, near, far in triplets:
        if anchor not in value_index or near not in value_index or far not in value_index:
            skipped += 1
            continue
        ia, in_, if_ = value_index[anchor], value_index[near], value_index[far]
        if D[ia, in_] < D[ia, if_]:
            correct += 1
        evaluated += 1

    return {
        "accuracy":    float(correct / evaluated) if evaluated else float("nan"),
        "n_correct":   correct,
        "n_evaluated": evaluated,
        "n_skipped":   skipped,
    }


# ── diagnostic 3: lattice_r2 ──────────────────────────────────────────────────

def lattice_r2(
    E: np.ndarray,
    value_index: ValueIndex,
    component_vectors: dict[str, list[int]],
    component_names: list[str],
    alpha: float = 0.5,
) -> DiagResult:
    """Leave-one-out ridge-regression R² for Boolean-lattice component reconstruction.

    Applicable to features of type ``"boolean_lattice"``, where each value is
    a combination of binary components (e.g. WALS 7A Glottalized Consonants:
    ejectives=1 or 0, implosives=1 or 0, glottalized resonants=1 or 0).

    Fits a ridge regressor from the multi-hot component vector to the embedding
    vector using leave-one-out cross-validation.  An R² near 1 means the embedding
    geometry is fully explained by the component structure; R² < 0 means the
    embedding adds no information over the component-vector mean.

    With small n (< 6), negative R² is expected even for partially structured
    embeddings because LOO variance dominates.  Use as a *relative* measure
    (Learned vs T-CF) rather than as an absolute quality indicator.

    Parameters
    ----------
    E : np.ndarray, shape (n, d)
        Within-feature-centred value embeddings.
    value_index : ValueIndex
        ``{value_string: row_index}`` mapping into E.
    component_vectors : dict
        ``{value_string: [c1, c2, ...]}`` multi-hot vectors, from the annotation's
        ``"values"`` field (for boolean_lattice features).
    component_names : list[str]
        Names of the components, from the annotation's ``"components"`` field.
    alpha : float
        Ridge regularisation strength (default 0.5).

    Returns
    -------
    dict with keys:
        r2          : float   (LOO variance-weighted R²; NaN if n < 3)
        n_values    : int
        components  : list[str]
    """
    items = [(v, c) for v, c in component_vectors.items() if v in value_index]
    n = len(items)
    if n < 3:
        return {"r2": float("nan"), "n_values": n, "components": component_names}

    X = np.array([c for _, c in items], dtype=float)
    Y = E[[value_index[v] for v, _ in items]]

    loo   = LeaveOneOut()
    preds = np.zeros_like(Y)
    for tr, te in loo.split(X):
        preds[te] = Ridge(alpha=alpha).fit(X[tr], Y[tr]).predict(X[te])

    ss_res = float(((Y - preds) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0)) ** 2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    return {"r2": r2, "n_values": n, "components": component_names}


# ── diagnostic 4: cross_product_probe ─────────────────────────────────────────

def cross_product_probe(
    E: np.ndarray,
    value_index: ValueIndex,
    axis_assignments: dict[str, dict[str, Any]],
    axes: list[str],
) -> DiagResult:
    """Per-axis centroid-direction LOO accuracy for permutation/cross-product features.

    Applicable to features of type ``"permutation"``, where each value is tagged
    with one or more relational axes (e.g. WALS 81A: verb_pos, OV_VO, SO_OS).

    For each axis, groups values by their axis label and tests whether each value's
    normalised embedding vector is most similar to its own group's centroid (using
    LOO to exclude the query from its own centroid).  Reports accuracy per axis,
    plus chance-level baseline.

    Parameters
    ----------
    E : np.ndarray, shape (n, d)
        Within-feature-centred value embeddings.
    value_index : ValueIndex
        ``{value_string: row_index}`` mapping into E.
    axis_assignments : dict
        ``{value_string: {axis_name: label}}`` from the annotation's ``"values"``
        field (for permutation features).
    axes : list[str]
        Which axes to probe, from the annotation's ``"axes"`` field.

    Returns
    -------
    dict with keys:
        probes : dict[axis_name, {accuracy, chance, n}]
            Per-axis LOO centroid-direction accuracy and majority-class chance.
        overall_mean_accuracy : float
            Mean accuracy across all probed axes.
    """
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    probes: dict[str, dict] = {}

    for axis in axes:
        # Group row indices by this axis's label
        groups: dict[Any, list[int]] = {}
        for v, axdict in axis_assignments.items():
            if v in value_index and axis in axdict:
                groups.setdefault(axdict[axis], []).append(value_index[v])

        if len(groups) < 2:
            continue

        class_labels = list(groups.keys())
        # Precompute full centroids (will be adjusted per-LOO below)
        centroids = {
            c: En[idxs].mean(0)
            for c, idxs in groups.items()
        }
        centroids = {
            c: vec / (np.linalg.norm(vec) + 1e-12)
            for c, vec in centroids.items()
        }

        correct = 0
        total   = 0
        for cls, idxs in groups.items():
            for i in idxs:
                # LOO centroid: exclude query from its own group
                others = [ii for ii in groups[cls] if ii != i]
                self_c = (
                    En[others].mean(0) / (np.linalg.norm(En[others].mean(0)) + 1e-12)
                    if others else centroids[cls]
                )
                scores = {
                    c: float(En[i] @ (self_c if c == cls else centroids[c]))
                    for c in class_labels
                }
                if max(scores, key=scores.get) == cls:
                    correct += 1
                total += 1

        n_per_class = [len(v) for v in groups.values()]
        chance = max(n_per_class) / sum(n_per_class)
        probes[axis] = {
            "accuracy": float(correct / total) if total else float("nan"),
            "chance":   float(chance),
            "n":        total,
        }

    overall = float(
        np.nanmean([p["accuracy"] for p in probes.values()])
    ) if probes else float("nan")

    return {"probes": probes, "overall_mean_accuracy": overall}


# ── diagnostic 5: nn_purity ───────────────────────────────────────────────────

def nn_purity(
    E: np.ndarray,
    value_index: ValueIndex,
    family_labels: dict[str, str],
) -> DiagResult:
    """Nearest-neighbour family purity within the feature's value set.

    Applicable to features of type ``"strategy_family"``, where values are grouped
    into labelled families (e.g. WALS 117A Predicative Possession: 'have', 'locational',
    'genitive', 'topic', 'conjunctional').

    For each value v, finds its nearest neighbour among all other values of the same
    feature and checks whether they share the same family label.  Purity = proportion
    of values whose nearest neighbour belongs to the same family.

    Expected result: purity ≈ 0 for most WALS strategy-family features.  This is the
    paradigmatic-competitor effect: softmax training pushes same-feature values apart,
    so cross-feature typological co-occurrence geometry dominates within-feature geometry.
    A near-zero purity is therefore *consistent with correct architecture behaviour*,
    not a failure.  See cross_feature_nn() for where the positive structure lives.

    Parameters
    ----------
    E : np.ndarray, shape (n, d)
        Within-feature-centred value embeddings.
    value_index : ValueIndex
        ``{value_string: row_index}`` mapping into E.
    family_labels : dict
        ``{value_string: family_label}`` from the annotation's ``"values"`` field.

    Returns
    -------
    dict with keys:
        purity          : float   (NaN if < 3 values present)
        n_correct_nn    : int
        n_values        : int
        family_counts   : dict[str, int]   (number of values per family present)
        chance_purity   : float   (expected purity under random assignment)
    """
    items = [(v, f) for v, f in family_labels.items() if v in value_index]
    n = len(items)
    if n < 3:
        return {
            "purity": float("nan"),
            "n_correct_nn": 0,
            "n_values": n,
            "family_counts": {},
            "chance_purity": float("nan"),
        }

    D = cosine_distance_matrix(E)
    correct = 0
    for v, fam in items:
        i = value_index[v]
        # distances to all other values in this feature
        neighbors = [(D[i, value_index[v2]], f2) for v2, f2 in items if v2 != v]
        nn_fam = min(neighbors, key=lambda x: x[0])[1]
        if nn_fam == fam:
            correct += 1

    family_counts: dict[str, int] = {}
    for _, f in items:
        family_counts[f] = family_counts.get(f, 0) + 1
    chance = sum(c * (c - 1) for c in family_counts.values()) / (n * (n - 1)) if n > 1 else float("nan")

    return {
        "purity":        float(correct / n),
        "n_correct_nn":  correct,
        "n_values":      n,
        "family_counts": family_counts,
        "chance_purity": float(chance),
    }


# ── dispatcher: run all appropriate diagnostics for a feature ─────────────────

def run_feature(
    feat_id: str,
    arch: str = "learned",
    seed: int = 42,
    checkpoint_dir: str | Path = "checkpoints",
    annotation_path: str | Path = "data/substructure_annotation.json",
) -> DiagResult:
    """Run the appropriate diagnostic for a single feature and return results.

    Looks up the feature's type in the annotation file, loads embeddings from the
    checkpoint, and dispatches to the correct diagnostic function.

    Parameters
    ----------
    feat_id : str
        WALS feature identifier, e.g. ``"81A"``.
    arch : str
        ``"learned"`` or ``"tcf"``.
    seed : int
        Training seed (default 42).
    checkpoint_dir : path-like
        Root directory of checkpoint subdirs.
    annotation_path : path-like
        Path to ``substructure_annotation.json``.

    Returns
    -------
    dict with at least:
        feat_id : str
        arch    : str
        seed    : int
        type    : str
        name    : str
        n_found : int
        missing : list[str]
        ...     (diagnostic-specific fields)

    Returns ``None`` if the feature is not in the annotation file or fewer than
    2 values are present in the checkpoint vocabulary.
    """
    annotations = load_annotation(annotation_path)
    features    = annotations.get("features", annotations)  # handle nested or flat

    if feat_id not in features:
        warnings.warn(f"Feature {feat_id!r} not found in annotation file.")
        return None

    spec = features[feat_id]
    try:
        E, value_index = load_feature_embeddings(feat_id, arch, seed, checkpoint_dir)
    except (FileNotFoundError, KeyError) as e:
        warnings.warn(str(e))
        return None

    if len(value_index) < 2:
        return None

    all_values = list(spec["values"].keys())
    found   = [v for v in all_values if v in value_index]
    missing = [v for v in all_values if v not in value_index]

    base = {
        "feat_id": feat_id,
        "arch":    arch,
        "seed":    seed,
        "type":    spec["type"],
        "name":    spec["name"],
        "n_found": len(found),
        "missing": missing,
    }

    stype = spec["type"]

    if stype == "ordinal":
        result = ordinal_spearman(E, value_index, spec["values"])

    elif stype == "presence_subtype":
        result = triplet_accuracy(E, value_index, spec.get("triplets", []))

    elif stype == "boolean_lattice":
        result = lattice_r2(
            E, value_index, spec["values"], spec.get("components", [])
        )

    elif stype == "permutation":
        result = cross_product_probe(
            E, value_index, spec["values"], spec.get("axes", [])
        )
        if "triplets" in spec or "presence_triplets" in spec:
            trp_key = "triplets" if "triplets" in spec else "presence_triplets"
            result["presence_triplets"] = triplet_accuracy(
                E, value_index, spec[trp_key]
            )

    elif stype == "strategy_family":
        result = nn_purity(E, value_index, spec["values"])

    else:
        warnings.warn(f"Unknown structure type {stype!r} for {feat_id}.")
        return None

    return {**base, **result}


# ── cross-feature nearest-neighbour ───────────────────────────────────────────

def cross_feature_nn(
    feat_id: str,
    arch: str = "learned",
    seed: int = 42,
    checkpoint_dir: str | Path = "checkpoints",
    annotation_path: str | Path = "data/substructure_annotation.json",
    top_k: int = 10,
) -> dict[str, list[dict]]:
    """Top-k cross-feature nearest neighbours for every annotated value of feat_id.

    Same-feature values are excluded from the neighbour list.  This is where the
    positive typological structure lives for strategy-family features: e.g.
    ``81A=SOV``'s top neighbours are Greenbergian head-final correlates
    (``95A=OV and Postpositions``, ``83A=OV``, ``85A=Postpositions``).

    Parameters
    ----------
    feat_id : str
        WALS feature identifier.
    arch : str
        Architecture name.
    seed : int
        Training seed.
    checkpoint_dir : path-like
    annotation_path : path-like
    top_k : int
        Number of neighbours to return per value.

    Returns
    -------
    dict mapping ``"feat_id=value_string"`` to a list of
    ``{"key": "other_feat=value", "sim": float}`` dicts sorted by descending
    cosine similarity.
    """
    ckpt = Path(checkpoint_dir) / f"wals_{arch}_s{seed}"
    if arch == "learned":
        all_emb = np.load(ckpt / "featvalue_embeddings.npy")
        k2id    = json.load(open(ckpt / "featvalue2id.json"))
    else:
        all_emb = np.load(ckpt / "binarycol_embeddings.npy")
        k2id    = json.load(open(ckpt / "binarycol2id.json"))

    annotations = load_annotation(annotation_path)
    features    = annotations.get("features", annotations)
    spec        = features[feat_id]

    all_keys = list(k2id.keys())
    all_E    = all_emb[[k2id[k] for k in all_keys]]
    En       = all_E / (np.linalg.norm(all_E, axis=1, keepdims=True) + 1e-12)

    prefix  = feat_id + "="
    results = {}
    for v in spec["values"]:
        key = f"{feat_id}={v}"
        if key not in k2id:
            continue
        sims  = En @ En[k2id[key]]
        order = np.argsort(-sims)
        nns   = []
        for idx in order:
            nb = all_keys[idx]
            if nb == key or nb.startswith(prefix):
                continue
            nns.append({"key": nb, "sim": float(sims[idx])})
            if len(nns) == top_k:
                break
        results[key] = nns

    return results


# ── shuffled-label null baseline ──────────────────────────────────────────────

def shuffled_null(
    diag_fn,
    E: np.ndarray,
    value_index: ValueIndex,
    extra_args: tuple = (),
    n_shuffles: int = 500,
    rng: np.random.Generator | None = None,
    score_key: str | None = None,
) -> dict[str, float]:
    """Monte-Carlo null distribution for a scalar diagnostic.

    Permutes the value_index mapping (keeping embeddings fixed) to break the
    alignment between labels and geometry, then re-runs the diagnostic.

    Parameters
    ----------
    diag_fn : callable
        One of ordinal_spearman, triplet_accuracy, lattice_r2, nn_purity.
        (cross_product_probe returns per-axis dicts; use per-axis calls instead.)
    E : np.ndarray
    value_index : ValueIndex
    extra_args : tuple
        Additional positional arguments for diag_fn beyond (E, value_index).
    n_shuffles : int
        Number of permutations (default 500).
    rng : np.random.Generator, optional
        Seeded generator for reproducibility.
    score_key : str, optional
        Key to extract the primary scalar from the diagnostic result dict.
        If None, defaults to 'spearman_rho' / 'accuracy' / 'r2' / 'purity' by type.

    Returns
    -------
    dict with keys:
        null_mean : float
        null_std  : float
        z_score   : float   (observed score relative to null, requires running
                             the diagnostic separately and passing observed_score)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    keys     = list(value_index.keys())
    idx_vals = list(value_index.values())

    # auto-detect score key
    if score_key is None:
        test_result = diag_fn(E, value_index, *extra_args)
        for candidate in ("spearman_rho", "accuracy", "r2", "purity"):
            if candidate in test_result:
                score_key = candidate
                break
        if score_key is None:
            raise ValueError("Cannot auto-detect score_key; pass it explicitly.")

    scores = []
    for _ in range(n_shuffles):
        perm  = rng.permutation(idx_vals).tolist()
        shuf  = dict(zip(keys, perm))
        res   = diag_fn(E, shuf, *extra_args)
        s     = res.get(score_key, float("nan"))
        if not (isinstance(s, float) and np.isnan(s)):
            scores.append(float(s))

    if not scores:
        return {"null_mean": float("nan"), "null_std": float("nan")}

    return {
        "null_mean": float(np.mean(scores)),
        "null_std":  float(np.std(scores)),
    }


# ── convenience: run all features ─────────────────────────────────────────────

def run_all(
    arch: str = "learned",
    seed: int = 42,
    checkpoint_dir: str | Path = "checkpoints",
    annotation_path: str | Path = "data/substructure_annotation.json",
    verbose: bool = True,
) -> list[DiagResult]:
    """Run diagnostics for every annotated feature and return a list of results.

    Convenience wrapper around :func:`run_feature`.  Results that are ``None``
    (missing checkpoint or fewer than 2 values) are silently excluded.
    """
    annotations = load_annotation(annotation_path)
    features    = annotations.get("features", annotations)
    results     = []
    for feat_id in features:
        r = run_feature(feat_id, arch, seed, checkpoint_dir, annotation_path)
        if r is not None:
            results.append(r)
            if verbose:
                _summarise(r)
    return results


def _summarise(r: DiagResult) -> None:
    """Print a one-line summary of a diagnostic result."""
    fid   = r.get("feat_id", "?")
    name  = r.get("name", "")[:35]
    stype = r.get("type", "")
    if stype == "ordinal":
        print(f"  {fid:6s} {name:35s} ρ = {r.get('spearman_rho', float('nan')):+.3f}")
    elif stype == "presence_subtype":
        print(f"  {fid:6s} {name:35s} triplet acc = {r.get('accuracy', float('nan')):.3f}")
    elif stype == "boolean_lattice":
        print(f"  {fid:6s} {name:35s} R² = {r.get('r2', float('nan')):+.3f}")
    elif stype == "permutation":
        probes = r.get("probes", {})
        acc_str = "  ".join(f"{ax}={p['accuracy']:.3f}" for ax, p in probes.items())
        print(f"  {fid:6s} {name:35s} {acc_str}")
    elif stype == "strategy_family":
        print(f"  {fid:6s} {name:35s} nn_purity = {r.get('purity', float('nan')):.3f}")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run WALS substructure diagnostics.")
    parser.add_argument("--feat",  default=None, help="Single feature ID (e.g. 81A); omit to run all.")
    parser.add_argument("--arch",  default="learned", choices=["learned", "tcf"])
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--checkpoints", default="checkpoints", help="Checkpoint root dir.")
    parser.add_argument("--annotations", default="data/substructure_annotation.json")
    parser.add_argument("--null", action="store_true", help="Also compute shuffled-label null.")
    parser.add_argument("--nn",   action="store_true", help="Also print cross-feature nearest neighbours.")
    args = parser.parse_args()

    if args.feat:
        r = run_feature(args.feat, args.arch, args.seed, args.checkpoints, args.annotations)
        if r:
            import pprint
            pprint.pprint(r, sort_dicts=False)
            if args.null:
                ann = load_annotation(args.annotations)
                spec = ann.get("features", ann)[args.feat]
                E, vi = load_feature_embeddings(args.feat, args.arch, args.seed, args.checkpoints)
                stype = spec["type"]
                if stype == "ordinal":
                    fn, extra, sk = ordinal_spearman, (spec["values"],), "spearman_rho"
                elif stype == "presence_subtype":
                    fn, extra, sk = triplet_accuracy, (spec.get("triplets", []),), "accuracy"
                elif stype == "strategy_family":
                    fn, extra, sk = nn_purity, (spec["values"],), "purity"
                else:
                    fn = None
                if fn:
                    null = shuffled_null(fn, E, vi, extra, score_key=sk)
                    observed = r.get(sk, float("nan"))
                    z = (observed - null["null_mean"]) / (null["null_std"] + 1e-12)
                    print(f"\nNull: mean={null['null_mean']:.3f}  std={null['null_std']:.3f}  "
                          f"z={z:+.2f}")
            if args.nn:
                nns = cross_feature_nn(args.feat, args.arch, args.seed,
                                       args.checkpoints, args.annotations)
                print(f"\nCross-feature nearest neighbours (top 10):")
                for key, neighbours in nns.items():
                    print(f"  {key}:")
                    for nn in neighbours[:7]:
                        print(f"    {nn['key']:50s}  sim={nn['sim']:.4f}")
    else:
        print(f"Running all annotated features — arch={args.arch}, seed={args.seed}")
        run_all(args.arch, args.seed, args.checkpoints, args.annotations)
