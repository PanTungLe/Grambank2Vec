"""
downstream/distances.py
========================
Compute typological distance matrices between language pairs.

For each pair of languages (s, t), compute distance under each representation:

  - Repr A (direct):   d_A(s,t) = 1 - cosine(λ_s, λ_t)
  - Repr B (expected): d_B(s,t) = 1 - cosine(T_s, T_t)
  - Repr C (predicted binary): d_C(s,t) = Hamming(profile_s, profile_t)
                                         / n_features
  - B+WALS / B+Grambank: if both databases have embeddings,
    concatenate B_wals and B_grambank → single joint representation
  - URIEL geo/phylo:   from uriel_plus_vectors.parquet if available

Usage
-----
    python downstream/distances.py \\
        --repr_dirs downstream/repr_wals_s42,downstream/repr_grambank_s42 \\
        --glottocodes_file downstream/transfer/glottocodes.json \\
        --out_dir downstream/distances \\
        --uriel_parquet analysis/conditioning_uriel_plus/uriel_plus_vectors.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Distance computation utilities
# ---------------------------------------------------------------------------

def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distance matrix.

    Parameters
    ----------
    X : (N, D) float32

    Returns
    -------
    D_cos : (N, N) float32  in [0, 2]
    """
    # Normalise rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X_norm = X / norms
    # Cosine similarity
    sim = X_norm @ X_norm.T
    # Clip for numerical safety
    sim = np.clip(sim, -1.0, 1.0)
    return (1.0 - sim).astype(np.float32)


def hamming_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise normalised Hamming distance for binary vectors.

    Parameters
    ----------
    X : (N, D) binary float32

    Returns
    -------
    D_ham : (N, N) float32  in [0, 1]
    """
    # For binary vectors: Hamming = sum of XOR / D
    N, D = X.shape
    D_ham = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        diff = np.abs(X - X[i]).sum(axis=1)
        D_ham[i] = diff / D
    return D_ham


def euclidean_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix."""
    diff = X[:, None, :] - X[None, :, :]          # (N, N, D)
    return np.sqrt((diff ** 2).sum(-1)).astype(np.float32)


# ---------------------------------------------------------------------------
# Load representations
# ---------------------------------------------------------------------------

def load_repr(repr_dir: str, repr_name: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load a representation matrix and its language list.

    Parameters
    ----------
    repr_dir    directory containing repr_*.npy and repr_*_langs.json
    repr_name   e.g. "A", "B", "C", "B_masked30"

    Returns
    -------
    matrix  (N, D) float32
    langs   list of glottocodes
    """
    mat_path = os.path.join(repr_dir, f"repr_{repr_name}.npy")
    langs_path = os.path.join(repr_dir, f"repr_{repr_name}_langs.json")
    if not os.path.exists(mat_path):
        return None, None
    matrix = np.load(mat_path)
    with open(langs_path) as f:
        langs = json.load(f)
    return matrix, langs


def align_to_glottocodes(
    matrix: np.ndarray,
    langs: List[str],
    target_glottocodes: List[str],
) -> np.ndarray:
    """
    Re-order and filter rows of matrix to match target_glottocodes.
    Missing languages get zero vectors.

    Returns
    -------
    aligned  (len(target_glottocodes), D) float32
    """
    lang2row = {g: i for i, g in enumerate(langs)}
    D = matrix.shape[1]
    aligned = np.zeros((len(target_glottocodes), D), dtype=np.float32)
    for j, glot in enumerate(target_glottocodes):
        if glot in lang2row:
            aligned[j] = matrix[lang2row[glot]]
    return aligned


# ---------------------------------------------------------------------------
# URIEL+ loader
# ---------------------------------------------------------------------------

def load_uriel_plus(
    parquet_path: str,
    glottocodes: List[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load URIEL+ geo and phylo vectors for a set of glottocodes.

    Returns
    -------
    geo_matrix    (N, D_geo) float32  or None if parquet unavailable
    phylo_matrix  (N, D_phylo) float32 or None
    """
    if not os.path.exists(parquet_path):
        return None, None

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None, None

    df = pq.read_table(parquet_path).to_pandas()
    df = df.set_index("glottocode")

    n = len(glottocodes)
    geo_rows, phylo_rows = [], []
    for glot in glottocodes:
        if glot in df.index:
            row = df.loc[glot]
            geo_rows.append(np.array(row["geo_vec"], dtype=np.float32))
            phylo_rows.append(np.array(row["phylo_vec"], dtype=np.float32))
        else:
            geo_rows.append(None)
            phylo_rows.append(None)

    # Find dimensions (first non-None row)
    geo_dim   = next((r.shape[0] for r in geo_rows   if r is not None), 0)
    phylo_dim = next((r.shape[0] for r in phylo_rows if r is not None), 0)

    geo_mat   = np.zeros((n, geo_dim),   dtype=np.float32) if geo_dim   else None
    phylo_mat = np.zeros((n, phylo_dim), dtype=np.float32) if phylo_dim else None

    for i, (gr, pr) in enumerate(zip(geo_rows, phylo_rows)):
        if gr is not None and geo_mat is not None:
            geo_mat[i] = gr
        if pr is not None and phylo_mat is not None:
            phylo_mat[i] = pr

    return geo_mat, phylo_mat


# ---------------------------------------------------------------------------
# Main: compute all distance matrices
# ---------------------------------------------------------------------------

def compute_all_distances(
    glottocodes: List[str],
    repr_dirs: List[str],
    out_dir: str,
    uriel_parquet: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compute all pairwise distance matrices and save to out_dir.

    Returns
    -------
    distances : {repr_name -> pd.DataFrame(N×N)}
    """
    os.makedirs(out_dir, exist_ok=True)
    N = len(glottocodes)
    distances: Dict[str, pd.DataFrame] = {}

    def save_dist(name: str, D_mat: np.ndarray):
        df = pd.DataFrame(D_mat, index=glottocodes, columns=glottocodes)
        df.to_csv(os.path.join(out_dir, f"dist_{name}.csv"))
        distances[name] = df
        print(f"  Saved dist_{name}.csv  shape={D_mat.shape}  "
              f"mean={D_mat[~np.eye(N, dtype=bool)].mean():.4f}")

    # Load representations from each directory
    all_reprs: Dict[str, np.ndarray] = {}

    for repr_dir in repr_dirs:
        db_tag = Path(repr_dir).name  # e.g. "repr_wals_s42"
        for repr_name in ["A", "B", "C", "B_masked30", "B_masked50", "B_masked70"]:
            mat, langs = load_repr(repr_dir, repr_name)
            if mat is None:
                continue
            aligned = align_to_glottocodes(mat, langs, glottocodes)
            key = f"{db_tag}_{repr_name}"
            all_reprs[key] = aligned
            print(f"  Loaded {key}: {aligned.shape}")

    print(f"\nComputing distance matrices for {N} languages...")

    # Compute distances for each representation
    for key, X in all_reprs.items():
        if key.endswith("_C"):
            D_mat = hamming_distance_matrix(X)
            save_dist(key, D_mat)
        else:
            # For dense embeddings (A, B, B_masked*): cosine distance
            D_cosine = cosine_distance_matrix(X)
            save_dist(key, D_cosine)

    # Joint representation: concatenate WALS B + Grambank B
    wals_B_key = next((k for k in all_reprs if "wals" in k.lower()
                       and k.endswith("_B")), None)
    gram_B_key = next((k for k in all_reprs if "grambank" in k.lower()
                       and k.endswith("_B")), None)
    if wals_B_key and gram_B_key:
        X_joint = np.concatenate(
            [all_reprs[wals_B_key], all_reprs[gram_B_key]], axis=1)
        D_joint = cosine_distance_matrix(X_joint)
        save_dist("joint_B_wals_grambank", D_joint)
        print(f"  Joint B (WALS+Grambank): {X_joint.shape}")

    # URIEL+ geo and phylo
    if uriel_parquet:
        print(f"\nLoading URIEL+ vectors from {uriel_parquet}...")
        geo_mat, phylo_mat = load_uriel_plus(uriel_parquet, glottocodes)
        if geo_mat is not None:
            save_dist("uriel_geo", cosine_distance_matrix(geo_mat))
        if phylo_mat is not None:
            save_dist("uriel_phylo", cosine_distance_matrix(phylo_mat))
        if geo_mat is not None and phylo_mat is not None:
            X_uriel_joint = np.concatenate([geo_mat, phylo_mat], axis=1)
            save_dist("uriel_geo_phylo",
                      cosine_distance_matrix(X_uriel_joint))

    # Baseline: random distances (sanity check)
    rng = np.random.default_rng(42)
    X_random = rng.standard_normal((N, 64)).astype(np.float32)
    save_dist("random", cosine_distance_matrix(X_random))

    # Save glottocodes used
    with open(os.path.join(out_dir, "glottocodes.json"), "w") as f:
        json.dump(glottocodes, f, indent=2)

    return distances


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compute pairwise typological distance matrices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repr_dirs", required=True,
                   help="Comma-separated paths to representation directories "
                        "(output of representations.py)")
    p.add_argument("--glottocodes_file", required=True,
                   help="JSON file with list of glottocodes to use")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--uriel_parquet", default=None,
                   help="Path to URIEL+ parquet file (optional)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    with open(args.glottocodes_file) as f:
        glottocodes = json.load(f)
    print(f"Computing distances for {len(glottocodes)} glottocodes")

    repr_dirs = [d.strip() for d in args.repr_dirs.split(",")]
    distances = compute_all_distances(
        glottocodes=glottocodes,
        repr_dirs=repr_dirs,
        out_dir=args.out_dir,
        uriel_parquet=args.uriel_parquet,
    )
    print(f"\nAll distance matrices saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
