"""
downstream/representations.py
==============================
Extract three typological language representations from trained checkpoints
for use in source-language transfer prediction.

Representation A — Direct language embedding
---------------------------------------------
    λ_ℓ ∈ ℝ⁶⁴  from lang_embeddings.npy
    The model's learned holistic language vector.
    Available immediately from the checkpoint; no inference needed.

Representation B — Expected value embedding (theory-central)
-------------------------------------------------------------
    T_ℓ = (1/|F|) Σ_f  Σ_v  p(v | ℓ, f) · e_{f,v}
    where p(v|ℓ,f) = softmax(λ_ℓ · E_f)  and  E_f = value embs for feature f.
    This is the NOVEL representation: it projects each language into the
    learned value geometry via probability-weighted aggregation.
    Crucially it handles missing typological coverage gracefully — every
    language gets a vector regardless of database sparsity.

Representation C — Predicted categorical profile (binary-style baseline)
-------------------------------------------------------------------------
    For each feature f, take argmax_v p(v|ℓ,f) and one-hot encode.
    Concatenated across all features → sparse binary-style vector.
    Dimensionality = total number of feature-value pairs (640 for WALS,
    392 for Grambank).
    Serves as the "model-predicted binary baseline" — directly comparable
    to raw WALS/Grambank one-hot vectors.

Usage
-----
    python downstream/representations.py \\
        --checkpoint checkpoints/wals_learned_s42 \\
        --out_dir downstream/repr_wals_s42

    python downstream/representations.py \\
        --checkpoint checkpoints/grambank_learned_s42 \\
        --out_dir downstream/repr_grambank_s42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import softmax as scipy_softmax

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_dir: str) -> Dict:
    """
    Load all artifacts from a canonical checkpoint directory.

    Returns a dict with:
        lang_embs       (N_langs, D) float32
        fv_embs         (N_values, D) float32
        lang2id         {glottocode -> row_index}  (glottocodes)
        lang2id_full    {wals_code -> row_index}   (WALS only; else same as lang2id)
        featvalue2id    {"feat=value" -> global_embedding_row}
        feat2values     {feat_name -> [value_name, ...]}
        feat_to_global_ids  {feat_idx -> [global_row, ...]}
        config          dict
    """
    d = Path(ckpt_dir)

    lang_embs = np.load(d / "lang_embeddings.npy").astype(np.float32)
    fv_embs   = np.load(d / "featvalue_embeddings.npy").astype(np.float32)

    with open(d / "lang2id.json") as f:
        lang2id = json.load(f)
    
    # lang2id_full may not exist for Grambank
    lang2id_full_path = d / "lang2id_full.json"
    if lang2id_full_path.exists():
        with open(lang2id_full_path) as f:
            lang2id_full = json.load(f)
    else:
        lang2id_full = lang2id

    with open(d / "featvalue2id.json") as f:
        featvalue2id = json.load(f)

    with open(d / "feat2values.json") as f:
        feat2values = json.load(f)

    with open(d / "config.json") as f:
        config = json.load(f)

    # Build feat_idx -> [global_row] mapping (feat_idx = alphabetical order of feature names)
    feat_names_sorted = sorted(feat2values.keys())
    feat_to_global_ids: Dict[int, List[int]] = {}
    for fi, fname in enumerate(feat_names_sorted):
        vals = feat2values[fname]
        global_rows = [featvalue2id[f"{fname}={v}"] for v in vals]
        feat_to_global_ids[fi] = global_rows

    return {
        "lang_embs": lang_embs,
        "fv_embs": fv_embs,
        "lang2id": lang2id,
        "lang2id_full": lang2id_full,
        "featvalue2id": featvalue2id,
        "feat2values": feat2values,
        "feat_to_global_ids": feat_to_global_ids,
        "feat_names_sorted": feat_names_sorted,
        "config": config,
    }


def compute_repr_A(ckpt: Dict, glottocodes: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Representation A: direct language embedding λ_ℓ.

    Parameters
    ----------
    ckpt        output of load_checkpoint()
    glottocodes list of glottocodes to extract

    Returns
    -------
    reprs   (N_found, D) float32
    found   list of glottocodes that were in the checkpoint
    """
    lang2id = ckpt["lang2id"]
    lang_embs = ckpt["lang_embs"]

    reprs, found = [], []
    for glot in glottocodes:
        if glot in lang2id:
            idx = lang2id[glot]
            reprs.append(lang_embs[idx])
            found.append(glot)

    if not reprs:
        return np.empty((0, lang_embs.shape[1]), dtype=np.float32), []
    return np.stack(reprs, axis=0), found


def compute_repr_B(
    ckpt: Dict,
    glottocodes: List[str],
    feature_subset: List[int] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Representation B: expected value embedding.

        T_ℓ = (1/|F|) Σ_f  Σ_v  p(v|ℓ,f) · e_{f,v}

    where p(v|ℓ,f) = softmax(λ_ℓ · E_f^T).

    This is the NOVEL representation: language as a probability-weighted
    average over the learned value embedding space.

    Parameters
    ----------
    feature_subset  if given, only average over these feature indices (int list).
                    Useful for task-specific weighting (e.g. word-order features
                    only for dependency parsing).

    Returns
    -------
    reprs   (N_found, D) float32
    found   list of glottocodes present in checkpoint
    """
    lang2id = ckpt["lang2id"]
    lang_embs = ckpt["lang_embs"]
    fv_embs = ckpt["fv_embs"]
    feat_to_global_ids = ckpt["feat_to_global_ids"]

    feat_indices = feature_subset if feature_subset is not None \
                   else list(feat_to_global_ids.keys())

    reprs, found = [], []
    for glot in glottocodes:
        if glot not in lang2id:
            continue
        idx = lang2id[glot]
        lang_emb = lang_embs[idx]  # (D,)

        feat_expected = []
        for fi in feat_indices:
            gids = feat_to_global_ids[fi]
            val_embs_f = fv_embs[gids]           # (K_f, D)
            scores = val_embs_f @ lang_emb        # (K_f,)
            probs = scipy_softmax(scores)         # (K_f,)
            expected = (probs[:, None] * val_embs_f).sum(0)  # (D,)
            feat_expected.append(expected)

        # Mean over features → single language vector
        T_l = np.stack(feat_expected, axis=0).mean(0)  # (D,)
        reprs.append(T_l.astype(np.float32))
        found.append(glot)

    if not reprs:
        return np.empty((0, lang_embs.shape[1]), dtype=np.float32), []
    return np.stack(reprs, axis=0), found


def compute_repr_C(ckpt: Dict, glottocodes: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Representation C: one-hot predicted categorical profile.

    For each language, for each feature f, take argmax_v p(v|ℓ,f) and
    one-hot encode (local index within feature).  Concatenate across all
    features → sparse binary vector of dimension = total feature-value pairs.

    This is a "model-predicted binary baseline" — structurally similar to
    raw WALS/Grambank one-hot vectors, but filled in by the model for
    all languages regardless of database coverage.
    """
    lang2id = ckpt["lang2id"]
    lang_embs = ckpt["lang_embs"]
    fv_embs = ckpt["fv_embs"]
    feat_to_global_ids = ckpt["feat_to_global_ids"]
    featvalue2id = ckpt["featvalue2id"]

    n_total_values = fv_embs.shape[0]
    feat_indices = sorted(feat_to_global_ids.keys())

    reprs, found = [], []
    for glot in glottocodes:
        if glot not in lang2id:
            continue
        idx = lang2id[glot]
        lang_emb = lang_embs[idx]

        vec = np.zeros(n_total_values, dtype=np.float32)
        for fi in feat_indices:
            gids = feat_to_global_ids[fi]
            val_embs_f = fv_embs[gids]       # (K_f, D)
            scores = val_embs_f @ lang_emb   # (K_f,)
            pred_local = int(np.argmax(scores))
            pred_global = gids[pred_local]
            vec[pred_global] = 1.0

        reprs.append(vec)
        found.append(glot)

    if not reprs:
        return np.empty((0, n_total_values), dtype=np.float32), []
    return np.stack(reprs, axis=0), found


def compute_repr_B_masked(
    ckpt: Dict,
    glottocodes: List[str],
    mask_frac: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """
    Representation B under simulated missing-data conditions.

    For each language, randomly mask `mask_frac` of features (treat as if
    the typological database has no entry for those features).  The masked
    features are simply dropped from the mean — they contribute nothing.

    This directly tests the claim: "under sparse typological coverage,
    expected value embeddings degrade gracefully compared to binary zeros."

    Returns
    -------
    reprs_masked   (N_found, D) float32  — B under masking
    reprs_full     (N_found, D) float32  — B without masking (for comparison)
    found          list of glottocodes
    """
    lang2id = ckpt["lang2id"]
    lang_embs = ckpt["lang_embs"]
    fv_embs = ckpt["fv_embs"]
    feat_to_global_ids = ckpt["feat_to_global_ids"]

    feat_indices = sorted(feat_to_global_ids.keys())
    n_feats = len(feat_indices)
    rng = np.random.default_rng(seed)

    # Number of features to keep per language
    n_keep = max(1, int(np.ceil(n_feats * (1.0 - mask_frac))))

    masked_reprs, full_reprs, found = [], [], []
    for glot in glottocodes:
        if glot not in lang2id:
            continue
        idx = lang2id[glot]
        lang_emb = lang_embs[idx]

        all_expected = []
        for fi in feat_indices:
            gids = feat_to_global_ids[fi]
            val_embs_f = fv_embs[gids]
            scores = val_embs_f @ lang_emb
            probs = scipy_softmax(scores)
            expected = (probs[:, None] * val_embs_f).sum(0)
            all_expected.append(expected)

        all_expected = np.stack(all_expected, axis=0)  # (n_feats, D)
        T_full = all_expected.mean(0)

        # Mask: randomly drop features
        kept_idx = rng.choice(n_feats, size=n_keep, replace=False)
        T_masked = all_expected[kept_idx].mean(0)

        full_reprs.append(T_full.astype(np.float32))
        masked_reprs.append(T_masked.astype(np.float32))
        found.append(glot)

    if not found:
        D = lang_embs.shape[1]
        empty = np.empty((0, D), dtype=np.float32)
        return empty, empty, []

    return (np.stack(masked_reprs), np.stack(full_reprs), found)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Extract typological language representations from a "
                    "canonical checkpoint for downstream transfer prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to canonical checkpoint directory "
                        "(e.g. checkpoints/wals_learned_s42)")
    p.add_argument("--out_dir", required=True,
                   help="Output directory for representation .npy files")
    p.add_argument("--glottocodes", default=None,
                   help="Comma-separated list of glottocodes to extract "
                        "(default: all languages in checkpoint)")
    p.add_argument("--mask_fracs", default="0.0,0.3,0.5,0.7",
                   help="Comma-separated mask fractions for Repr B ablation")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)
    n_langs = ckpt["lang_embs"].shape[0]
    n_feats = len(ckpt["feat_to_global_ids"])
    n_vals  = ckpt["fv_embs"].shape[0]
    db = ckpt["config"].get("database", "unknown")
    arch = ckpt["config"].get("architecture", "unknown")
    print(f"  DB={db}, arch={arch}, "
          f"{n_langs} languages, {n_feats} features, {n_vals} total values")

    if args.glottocodes:
        glottocodes = [g.strip() for g in args.glottocodes.split(",")]
    else:
        glottocodes = list(ckpt["lang2id"].keys())
    print(f"  Extracting representations for {len(glottocodes)} glottocodes")

    # Representation A
    print("\n[Repr A] Direct language embeddings...")
    A, found_A = compute_repr_A(ckpt, glottocodes)
    print(f"  Found {len(found_A)}/{len(glottocodes)} languages in checkpoint")
    np.save(os.path.join(args.out_dir, "repr_A.npy"), A)
    with open(os.path.join(args.out_dir, "repr_A_langs.json"), "w") as f:
        json.dump(found_A, f, indent=2)
    print(f"  Shape: {A.shape},  mean norm: {np.linalg.norm(A, axis=1).mean():.4f}")

    # Representation B
    print("\n[Repr B] Expected value embeddings...")
    B, found_B = compute_repr_B(ckpt, glottocodes)
    np.save(os.path.join(args.out_dir, "repr_B.npy"), B)
    with open(os.path.join(args.out_dir, "repr_B_langs.json"), "w") as f:
        json.dump(found_B, f, indent=2)
    print(f"  Shape: {B.shape},  mean norm: {np.linalg.norm(B, axis=1).mean():.4f}")

    # Representation C
    print("\n[Repr C] Predicted one-hot profile...")
    C, found_C = compute_repr_C(ckpt, glottocodes)
    np.save(os.path.join(args.out_dir, "repr_C.npy"), C)
    with open(os.path.join(args.out_dir, "repr_C_langs.json"), "w") as f:
        json.dump(found_C, f, indent=2)
    print(f"  Shape: {C.shape},  sparsity: "
          f"{(C == 0).mean():.3f}")

    # Representation B under masking
    mask_fracs = [float(x) for x in args.mask_fracs.split(",")]
    for mf in mask_fracs:
        if mf <= 0.0:
            continue
        print(f"\n[Repr B masked @ {mf:.0%}]...")
        Bm, Bf, found_m = compute_repr_B_masked(ckpt, glottocodes,
                                                  mask_frac=mf, seed=args.seed)
        tag = f"repr_B_masked{int(mf*100):02d}"
        np.save(os.path.join(args.out_dir, f"{tag}.npy"), Bm)
        with open(os.path.join(args.out_dir, f"{tag}_langs.json"), "w") as f:
            json.dump(found_m, f, indent=2)
        print(f"  Shape: {Bm.shape}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Representations saved to: {args.out_dir}")
    print(f"  repr_A.npy           ({A.shape})  — direct lang embeddings")
    print(f"  repr_B.npy           ({B.shape})  — expected value embeddings")
    print(f"  repr_C.npy           ({C.shape})  — predicted binary profile")
    for mf in mask_fracs:
        if mf > 0:
            print(f"  repr_B_masked{int(mf*100):02d}.npy — B under {mf:.0%} feature masking")


if __name__ == "__main__":
    main()
