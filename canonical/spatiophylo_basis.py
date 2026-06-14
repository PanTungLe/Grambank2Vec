"""
spatiophylo_basis.py — Spatiophylogenetic conditioning basis for RQ2.

Goal
----
Remove genealogical (phylogenetic) and areal (geographic) autocorrelation from
the *learned value-embedding geometry*, in the spirit of Verkerk et al. (2026,
Nat. Hum. Behav.), who test grammatical universals with Bayesian models that
include genealogical-descent and geographical-proximity terms and keep only the
fixed-effect signal whose 95% CI does not straddle zero.

Why a *basis* rather than a per-language residualisation
--------------------------------------------------------
A value embedding has no per-language index, so a phylogenetic/spatial random
effect (defined over languages) cannot attach to it directly. The control must
enter through the language side. A Gaussian-process / mixed random effect with
covariance C is, in its eigen/dual form, ridge-penalised regression on the
eigenvectors of C scaled by sqrt(eigenvalue). The canonical conditioning model
(canonical/conditioning_model.py) ALREADY applies a zero-initialised,
L2-penalised linear offset to the language embedding:

    conditioned_lang_emb = lang_emb + cond_proj( concat(geo, phylo) )

So if we feed a *spatiophylogenetic eigenvector basis* in place of the URIEL+
geo/phylo vectors, the existing L2 (cond_proj.weight, --l2_coef) becomes the
shrinkage that approximates a low-rank phylogenetic + spatial random effect —
with no change to the model or the objective. The value embeddings trained
alongside are then "net of" genealogy and area, and the existing substructure
suite (canonical/analyze_geometry.py) can be re-run on them unchanged.

Drop-in mapping into the existing conditioning model
----------------------------------------------------
    phylo_E, spatial_E, meta = build_spatiophylo_basis(
        lang2id, coords, phylo_cov, phylo_taxa)
    model = TypologicalMF_Conditioned(            # or *_TCF_Conditioned
        n_languages, n_total_values, feat_to_global_ids,
        geo_matrix   = spatial_E,   # areal control
        phylo_matrix = phylo_E,     # genealogical control
        cond_mode    = "both",      # 'phylo_only' = genealogy-only control,
    )                               # 'geo_only'  = area-only control

Inputs NOT shipped in the repo (supply on the lab server)
---------------------------------------------------------
  * coords    : {glottocode: (lat_deg, lon_deg)} — from WALS/Grambank CLDF
                languages.csv (Latitude/Longitude).
  * phylo_cov : a (taxon x taxon) Brownian covariance. For parity with Verkerk
                use the Grambank/Verkerk global *dated* tree
                (-> phylo_covariance_from_newick). Glottolog has no branch
                lengths; phylo_covariance_from_classification builds a
                unit-branch (topology) covariance as a fallback.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)
EARTH_R_KM = 6371.0


# ---------------------------------------------------------------- spatial side
def great_circle_km(coords: np.ndarray) -> np.ndarray:
    """coords: (n, 2) in DEGREES [lat, lon]. Returns (n, n) great-circle km."""
    lat = np.radians(coords[:, 0])[:, None]
    lon = np.radians(coords[:, 1])[:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    return EARTH_R_KM * 2.0 * np.arcsin(np.sqrt(a))


def spatial_kernel(coords: np.ndarray,
                   lengthscale_km: Optional[float] = None) -> np.ndarray:
    """
    Exponential spatial covariance K_ij = exp(-d_ij / lengthscale).
    lengthscale defaults to the median pairwise distance (a scale-free choice).
    This is the geographical-proximity analogue of Verkerk's spatial term.
    (Moran eigenvector maps are a recognised alternative for the spatial basis;
    the kernel-eigenvector form here keeps the GP/random-effect duality exact.)
    """
    D = great_circle_km(coords)
    if lengthscale_km is None:
        iu = np.triu_indices_from(D, k=1)
        lengthscale_km = float(np.median(D[iu])) if iu[0].size else 1.0
        lengthscale_km = max(lengthscale_km, 1.0)
    K = np.exp(-D / lengthscale_km)
    log.info("spatial kernel: lengthscale=%.0f km, n=%d", lengthscale_km, len(D))
    return K


# ------------------------------------------------------------ phylogenetic side
def phylo_covariance_from_classification(
    classification: Dict[str, Sequence[str]],
    taxa: List[str],
) -> np.ndarray:
    """
    Topology-only (unit-branch) Brownian covariance: C_ij = number of ancestral
    nodes shared by i and j on their root->tip classification path; C_ii = own
    path depth. No-branch-length fallback for when only a Glottolog
    classification is available (glottocode -> [ancestor node ids, ROOT FIRST]).

    For parity with Verkerk, prefer phylo_covariance_from_newick with a dated
    tree; unit branches over-weight shallow splits relative to a dated tree.
    """
    paths = [list(classification.get(t, [])) for t in taxa]
    n = len(taxa)
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        pi = paths[i]
        for j in range(i, n):
            pj = paths[j]
            s = 0
            for a, b in zip(pi, pj):           # shared root-first prefix
                if a == b:
                    s += 1
                else:
                    break
            C[i, j] = C[j, i] = s
        C[i, i] = max(len(pi), 1)
    return C


def phylo_covariance_from_newick(newick_path: str, taxa: List[str]) -> np.ndarray:
    """
    Brownian phylogenetic VCV from a tree WITH branch lengths:
    C_ij = shared root->MRCA(i, j) path length (the standard ape::vcv matrix),
    aligned to `taxa`. Requires dendropy. Use the Grambank/Verkerk global dated
    tree here for closest parity with the published spatiophylogenetic model.
    """
    try:
        import dendropy
    except ImportError as e:
        raise ImportError(
            "phylo_covariance_from_newick needs dendropy (`pip install "
            "dendropy`); or use phylo_covariance_from_classification for a "
            "topology fallback."
        ) from e

    tree = dendropy.Tree.get(path=newick_path, schema="newick")
    pdm = tree.phylogenetic_distance_matrix()
    leaves = {lf.taxon.label: lf for lf in tree.leaf_node_iter()}

    depth: Dict[str, float] = {}
    for lab, lf in leaves.items():
        d, nd = 0.0, lf
        while nd.parent_node is not None:
            d += (nd.edge.length or 0.0)
            nd = nd.parent_node
        depth[lab] = d

    n = len(taxa)
    C = np.zeros((n, n), dtype=np.float64)
    for i, ti in enumerate(taxa):
        for j in range(i, n):
            tj = taxa[j]
            if ti in leaves and tj in leaves:
                if i == j:
                    C[i, i] = depth.get(ti, 0.0)
                else:
                    pd = pdm.distance(leaves[ti].taxon, leaves[tj].taxon)
                    C[i, j] = C[j, i] = max(
                        (depth[ti] + depth[tj] - pd) / 2.0, 0.0)
    return C


# ----------------------------------------------------- shared eigen machinery
def kernel_eigenvectors(C: np.ndarray,
                        k: Optional[int] = None,
                        var_target: float = 0.95,
                        scale_by_eigval: bool = True
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Low-rank random-effect basis for covariance C.

    A Gaussian random effect with covariance C is, in eigen/dual form, ridge
    regression on U * diag(sqrt(eigval)). Returning eigenvectors scaled by
    sqrt(eigenvalue) lets the uniform L2 in cond_proj approximate the
    covariance's shrinkage profile (deep/broad structure shrunk less than fine
    structure). k overrides var_target if given; only positive-eigenvalue
    components are kept.
    """
    C = 0.5 * (C + C.T)
    vals, vecs = np.linalg.eigh(C)                 # ascending
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    pos = vals > 1e-9
    vals, vecs = vals[pos], vecs[:, pos]
    if vals.size == 0:
        return np.zeros((C.shape[0], 1), np.float32), np.zeros(1, np.float32)
    if k is None:
        csum = np.cumsum(vals) / vals.sum()
        k = int(np.searchsorted(csum, var_target) + 1)
    k = min(k, vecs.shape[1])
    U, lam = vecs[:, :k], vals[:k]
    if scale_by_eigval:
        U = U * np.sqrt(lam)[None, :]
    log.info("kernel basis: kept %d comps (%.1f%% var)",
             k, 100 * lam.sum() / vals.sum())
    return U.astype(np.float32), lam.astype(np.float32)


# --------------------------------------------------------------- main builder
def build_spatiophylo_basis(
    lang2id: Dict[str, int],
    coords: Dict[str, Tuple[float, float]],
    phylo_cov: np.ndarray,
    phylo_taxa: List[str],
    k_phylo: Optional[int] = None,
    k_spatial: Optional[int] = None,
    lengthscale_km: Optional[float] = None,
    var_target: float = 0.95,
    scale_by_eigval: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Per-language phylogenetic and spatial eigenvector matrices aligned to
    lang2id, drop-in for the URIEL+ phylo_matrix / geo_matrix in
    canonical/conditioning_model.py. Languages missing coords or tree membership
    get zero rows (no conditioning signal) — matching the URIEL+ loader.

    Returns
    -------
    phylo_E   : (n_langs, k_phylo)   float32   -> pass as phylo_matrix
    spatial_E : (n_langs, k_spatial) float32   -> pass as geo_matrix
    meta      : coverage / rank diagnostics
    """
    n_langs = max(lang2id.values()) + 1

    # phylogenetic eigenvectors over the tree taxa, scattered to lang2id rows
    taxon2row = {t: i for i, t in enumerate(phylo_taxa)}
    U_phy, _ = kernel_eigenvectors(
        phylo_cov, k=k_phylo, var_target=var_target,
        scale_by_eigval=scale_by_eigval)
    phylo_E = np.zeros((n_langs, U_phy.shape[1]), dtype=np.float32)
    phy_found = 0
    for gc, idx in lang2id.items():
        r = taxon2row.get(gc)
        if r is not None:
            phylo_E[idx] = U_phy[r]
            phy_found += 1

    # spatial eigenvectors over languages that have coordinates
    geo_gcs = [gc for gc in lang2id if gc in coords]
    if geo_gcs:
        coord_arr = np.array([coords[gc] for gc in geo_gcs], dtype=np.float64)
        K = spatial_kernel(coord_arr, lengthscale_km=lengthscale_km)
        U_sp, _ = kernel_eigenvectors(
            K, k=k_spatial, var_target=var_target,
            scale_by_eigval=scale_by_eigval)
        spatial_E = np.zeros((n_langs, U_sp.shape[1]), dtype=np.float32)
        for gc, row in zip(geo_gcs, U_sp):
            spatial_E[lang2id[gc]] = row
    else:
        spatial_E = np.zeros((n_langs, 1), dtype=np.float32)

    meta = {
        "n_langs": n_langs,
        "phylo_dim": int(phylo_E.shape[1]),
        "spatial_dim": int(spatial_E.shape[1]),
        "phylo_coverage": phy_found / n_langs if n_langs else 0.0,
        "spatial_coverage": len(geo_gcs) / n_langs if n_langs else 0.0,
        "lengthscale_km": lengthscale_km,
        "scale_by_eigval": scale_by_eigval,
    }
    log.info(
        "spatiophylo basis: phylo %d-dim (cov %.1f%%), spatial %d-dim (cov %.1f%%)",
        meta["phylo_dim"], 100 * meta["phylo_coverage"],
        meta["spatial_dim"], 100 * meta["spatial_coverage"])
    return phylo_E, spatial_E, meta


if __name__ == "__main__":
    # Self-contained smoke test (synthetic geography + classification).
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    gcs = [f"lang{i:03d}" for i in range(60)]
    lang2id = {gc: i for i, gc in enumerate(gcs)}
    centres = np.array([[10, 10], [-20, 120], [40, -100]], dtype=float)
    coords = {gc: tuple(centres[i % 3] + rng.normal(0, 5, size=2))
              for i, gc in enumerate(gcs)}
    classification = {
        gc: [f"fam{i % 3}", f"fam{i % 3}.sub{(i // 3) % 4}", gc]
        for i, gc in enumerate(gcs)
    }
    C = phylo_covariance_from_classification(classification, gcs)
    phylo_E, spatial_E, meta = build_spatiophylo_basis(
        lang2id, coords, C, gcs, var_target=0.9)
    assert phylo_E.shape[0] == 60 and spatial_E.shape[0] == 60
    print("OK:", meta)
    print("phylo_E", phylo_E.shape, "spatial_E", spatial_E.shape)
