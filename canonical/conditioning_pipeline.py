"""
RQ4 conditioning pipeline — full factorial evaluation.

For each (database, architecture, imputation, seed) configuration:
  1. Load data, URIEL+ geo+phylo vectors, and pre-computed baseline F1 scores.
  2. For each qualifying branch × in_branch_frac × repeat × conditioning_variant:
       a. Look up baseline F1 from analysis/baselines/comparison_results_{db}.csv.
          Skip the cell (with a warning) if the tuple is absent.
       b. Build train/eval split.
       c. Train CONDITIONED model only (URIEL+ geo+phylo offset via cond_proj).
       d. Evaluate; record baseline and conditioned F1 rows.
  3. Save results to analysis/conditioning_results/<run_id>.csv.

Baseline reuse halves per-cell wall-clock vs training both models from scratch.

IMPORTANT — full factorial is ~12 000 model trainings; run on a lab server.
See canonical/LAB_SERVER_RUNBOOK.md
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_preparation import (
    load_wals_cldf,
    load_grambank_cldf,
    binarise_features,
)
from model_learned import (
    CategoricalTypDataset,
    prepare_categorical,
    split_by_branch_categorical,
    evaluate_learned,
)
from canonical.conditioning_model import (
    TypologicalMF_Conditioned,
    TypologicalMF_TCF_Conditioned,
    load_uriel_vectors_for_lang2id,
    train_conditioned,
)
from canonical.train_canonical import build_lang2id
from utils import (
    seed_everything,
    build_all_triples_binary,
    build_all_triples_categorical,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)

# Maps --conditioning_variants token → cond_mode string accepted by the models.
_VARIANT_TO_COND_MODE = {
    "geo":   "geo_only",
    "phylo": "phylo_only",
    "both":  "both",
}

# Maps pipeline architecture name → model column in comparison_results CSV.
_ARCH_TO_MODEL_NAME = {
    "tcf":     "T-CF",
    "learned": "Learned",
}

_DEFAULT_BASELINES_DIR = str(_REPO_ROOT / "analysis" / "baselines")


def load_baseline_lookup(
    baselines_dir: str,
    database: str,
    architecture: str,
) -> Dict[Tuple[str, float, int], float]:
    """
    Load pre-computed baseline F1 scores from comparison_results_{database}.csv.

    Returns a dict keyed on (branch, in_branch_frac, repeat) → f1_baseline.
    Only rows whose 'model' column matches the current architecture are kept.

    Raises FileNotFoundError if the CSV is absent.
    """
    csv_path = Path(baselines_dir) / f"comparison_results_{database}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Baseline CSV not found: {csv_path}\n"
            f"Expected at analysis/baselines/comparison_results_{{database}}.csv"
        )
    model_name = _ARCH_TO_MODEL_NAME[architecture]
    df_bl = pd.read_csv(csv_path)
    subset = df_bl[df_bl["model"] == model_name]
    lookup: Dict[Tuple[str, float, int], float] = {
        (row["branch"],
         round(float(row["in_branch_frac"]), 6),
         int(row["repeat"])): float(row["f1"])
        for _, row in subset.iterrows()
    }
    log.info("Loaded %d baseline entries (model=%s) from %s",
             len(lookup), model_name, csv_path.name)
    return lookup


# ──────────────────────────────────────────────────────────────────────────────
# Spatiophylogenetic helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_coords_from_cldf(
    data_path: str,
    database: str,
) -> Dict[str, Tuple[float, float]]:
    """
    Load {glottocode: (lat_deg, lon_deg)} from CLDF languages.csv.

    For WALS, uses the 'Glottocode' column paired with 'Latitude'/'Longitude'.
    For Grambank, uses the 'ID' column (which is already the Glottocode).
    Languages without coordinates are omitted (zero-row fallback in
    build_spatiophylo_basis matches the URIEL+ loader convention).
    """
    langs_csv = Path(data_path) / "cldf" / "languages.csv"
    if not langs_csv.exists():
        log.warning(
            "CLDF languages.csv not found at %s; spatial basis will be empty.",
            langs_csv)
        return {}

    df_l = pd.read_csv(langs_csv, low_memory=False)
    df_l.columns = [c.strip() for c in df_l.columns]

    lat_col = next((c for c in df_l.columns if c.lower() == "latitude"), None)
    lon_col = next((c for c in df_l.columns if c.lower() == "longitude"), None)
    if lat_col is None or lon_col is None:
        log.warning(
            "No Latitude/Longitude columns in %s; spatial basis will be empty.",
            langs_csv.name)
        return {}

    if database == "grambank":
        gc_col = next((c for c in df_l.columns if c.lower() == "id"), None)
    else:  # wals
        gc_col = next((c for c in df_l.columns if c.lower() == "glottocode"), None)

    if gc_col is None:
        log.warning(
            "No glottocode column found in %s; spatial basis will be empty.",
            langs_csv.name)
        return {}

    coords: Dict[str, Tuple[float, float]] = {}
    for _, row in df_l.iterrows():
        gc = str(row[gc_col]).strip() if pd.notna(row[gc_col]) else ""
        if not gc or gc.lower() in ("nan", "none", ""):
            continue
        lat, lon = row[lat_col], row[lon_col]
        if pd.notna(lat) and pd.notna(lon):
            coords[gc] = (float(lat), float(lon))

    log.info("Loaded coordinates for %d languages from %s", len(coords), langs_csv.name)
    return coords


def build_phylo_classification(
    df: pd.DataFrame,
    lang2id_glot: Dict[str, int],
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Build a {glottocode: [family, genus, glottocode]} classification dict
    and the aligned taxa list for phylo_covariance_from_classification.

    Constructs a root-first ancestral path from the df's 'family' and 'genus'
    columns (both present in WALS and Grambank loaders). Missing/empty nodes
    are omitted so the path is never shallower than [glottocode].
    """
    taxa = list(lang2id_glot.keys())
    has_family = "family" in df.columns
    has_genus = "genus" in df.columns

    classification: Dict[str, List[str]] = {}
    for gc, row_idx in lang2id_glot.items():
        row = df.iloc[row_idx]
        path: List[str] = []

        if has_family:
            fam = str(row["family"]).strip()
            if fam and fam.lower() not in ("nan", "none", ""):
                path.append(fam)
        if has_genus:
            gen = str(row["genus"]).strip()
            if gen and gen.lower() not in ("nan", "none", "") and (not path or gen != path[-1]):
                path.append(gen)
        path.append(gc)
        classification[gc] = path

    return classification, taxa


def _save_spatiophylo_checkpoint(
    architecture: str,
    df: pd.DataFrame,
    data_dict: dict,
    lang2id_glot: Dict[str, int],
    lang2id_full: Dict[str, int],
    geo_matrix: np.ndarray,
    phylo_matrix: np.ndarray,
    args: argparse.Namespace,
    out_dir: str,
    spatiophylo_meta: Optional[dict],
) -> None:
    """
    Train a full-data (all-observations) conditioned model with the
    spatiophylo basis (cond_mode='both') and save analyze_geometry.py-
    compatible artifacts to out_dir.

    Artifact layout mirrors canonical/train_canonical.py::dump_artifacts:
      lang_embeddings.npy, featvalue_embeddings.npy / binarycol_embeddings.npy,
      featvalue2id.json / binarycol2id.json, feat2values.json,
      lang2id.json, lang2id_full.json, config.json
    """
    log.info("Training full-data conditioned model for canonical checkpoint …")
    seed_everything(args.seed)

    if architecture == "learned":
        cat_matrix       = data_dict["cat_matrix"]
        feat_to_global_ids = data_dict["feat_to_global_ids"]
        feat_to_value_names = data_dict["feat_to_value_names"]
        kept_feature_names  = data_dict["kept_feature_names"]
        n_langs = cat_matrix.shape[0]
        n_total_values = max(
            max(gids) for gids in feat_to_global_ids.values()) + 1

        all_l, all_f, all_v = build_all_triples_categorical(cat_matrix)
        train_ds = CategoricalTypDataset(all_l, all_f, all_v)

        model = TypologicalMF_Conditioned(
            n_langs, n_total_values, feat_to_global_ids,
            geo_matrix, phylo_matrix, args.embed_dim, cond_mode="both")
    else:  # tcf
        binary_matrix    = data_dict["binary_matrix"]
        binary_col_names = data_dict["binary_col_names"]
        feature_value_names = data_dict["feature_value_names"]
        n_langs, n_binary = binary_matrix.shape

        all_l, all_f, all_v = build_all_triples_binary(binary_matrix)
        from torch.utils.data import TensorDataset
        train_ds = TensorDataset(
            torch.LongTensor(all_l),
            torch.LongTensor(all_f),
            torch.FloatTensor(all_v),
        )

        model = TypologicalMF_TCF_Conditioned(
            n_langs, n_binary, geo_matrix, phylo_matrix,
            args.embed_dim, cond_mode="both")

    train_conditioned(
        model, train_ds,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        lr=args.lr, l2_reg=args.l2_coef,
        device=args.device, patience=args.patience)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        lang_emb = model.lang_embeddings.weight.detach().cpu().numpy()
    np.save(str(out_path / "lang_embeddings.npy"), lang_emb)

    if architecture == "learned":
        val_emb = model.value_embeddings.weight.detach().cpu().numpy()
        np.save(str(out_path / "featvalue_embeddings.npy"), val_emb)

        featvalue2id: dict = {}
        for fi, feat_name in enumerate(kept_feature_names):
            bare = feat_name.removeprefix("feat_")
            for gid, vname in zip(feat_to_global_ids[fi], feat_to_value_names[fi]):
                featvalue2id[f"{bare}={vname}"] = int(gid)
        with open(out_path / "featvalue2id.json", "w") as fh:
            json.dump(featvalue2id, fh, indent=2)

        feat2values = {
            feat_name.removeprefix("feat_"): list(feat_to_value_names[fi])
            for fi, feat_name in enumerate(kept_feature_names)
        }
    else:
        bin_emb = model.feat_embeddings.weight.detach().cpu().numpy()
        np.save(str(out_path / "binarycol_embeddings.npy"), bin_emb)

        binarycol2id: dict = {
            name.removeprefix("feat_"): i
            for i, name in enumerate(binary_col_names)
        }
        with open(out_path / "binarycol2id.json", "w") as fh:
            json.dump(binarycol2id, fh, indent=2)

        feat2values = {
            feat_name.removeprefix("feat_"): list(val_names)
            for feat_name, val_names in feature_value_names.items()
        }

    with open(out_path / "feat2values.json", "w") as fh:
        json.dump(feat2values, fh, indent=2)
    with open(out_path / "lang2id.json", "w") as fh:
        json.dump(lang2id_glot, fh, indent=2)
    with open(out_path / "lang2id_full.json", "w") as fh:
        json.dump(lang2id_full, fh, indent=2)

    config = {
        "database": args.database,
        "architecture": architecture,
        "seed": args.seed,
        "embed_dim": args.embed_dim,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "l2_coef": args.l2_coef,
        "cond_mode": "both",
        "conditioning_source": "spatiophylo",
        "phylo_source": args.phylo_source,
        "newick_path": args.newick_path,
        "spatiophylo_meta": spatiophylo_meta,
    }
    with open(out_path / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)
    with open(out_path / "spatiophylo_meta_both.json", "w") as fh:
        json.dump(spatiophylo_meta or {}, fh, indent=2)

    log.info("Saved de-confounded canonical checkpoint → %s", out_dir)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RQ4 conditioning pipeline — full factorial evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--database", required=True, choices=["wals", "grambank"])
    p.add_argument("--architecture", required=True, choices=["tcf", "learned"])
    p.add_argument(
        "--data_path", required=True,
        help="Path to the cloned CLDF repo (WALS or Grambank).")
    p.add_argument(
        "--glottolog_repo", default=None,
        help="Path to cloned glottolog-cldf repo (Grambank genus only).")
    p.add_argument(
        "--uriel_vectors_path",
        default=str(_REPO_ROOT / "analysis" / "conditioning_uriel_plus"
                    / "uriel_plus_vectors.parquet"),
        help="Path to the URIEL+ vectors parquet produced by uriel_plus_loader.py.",
    )
    p.add_argument(
        "--baselines_dir",
        default=_DEFAULT_BASELINES_DIR,
        help="Directory containing comparison_results_{database}.csv files.",
    )
    p.add_argument(
        "--out_dir",
        default="analysis/conditioning_results",
        help="Output directory for results.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2_coef", type=float, default=0.1)
    p.add_argument("--min_branch_size", type=int, default=5)
    p.add_argument(
        "--in_branch_fracs",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.10, 0.20],
        help="In-branch training fractions to sweep.",
    )
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument(
        "--smoke_n_langs", type=int, default=100,
        help="[smoke only] Randomly subset to this many languages to speed up "
             "the sanity check. Set to 0 to use all languages.",
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke", action="store_true",
        help="Quick sanity check: 2 branches, 2 fracs, 1 repeat, "
             "subset to --smoke_n_langs languages.")
    mode.add_argument(
        "--full", action="store_true",
        help="Full factorial (all branches, fracs, repeats). Lab server only.")

    p.add_argument(
        "--resume", action="store_true",
        help="Skip already-completed (branch, frac, repeat, conditioning) rows.",
    )
    p.add_argument(
        "--conditioning_variants",
        nargs="+",
        choices=["geo", "phylo", "both"],
        default=["both"],
        metavar="VARIANT",
        help="Conditioning input modes to sweep per (branch, frac, repeat) combo. "
             "'geo'=geo_only, 'phylo'=phylo_only, 'both'=both (default). "
             "Each additional variant trains one extra conditioned model, "
             "doubling the result rows for that combo.",
    )

    sp = p.add_argument_group("spatiophylogenetic basis")
    sp.add_argument(
        "--phylo_source",
        choices=["uriel", "glottolog_topology", "newick"],
        default="uriel",
        help="Phylogenetic conditioning source.  "
             "'uriel' (default): existing URIEL+ parquet, no change.  "
             "'glottolog_topology': topology-only Brownian VCV from the family+genus "
             "columns already in the loaded dataframe (no dated tree required).  "
             "'newick': full Brownian VCV from a dated tree (--newick_path required, "
             "closest parity to Verkerk et al. 2026).")
    sp.add_argument(
        "--newick_path", default=None,
        help="[--phylo_source newick] Path to a dated newick tree for Brownian VCV.")
    sp.add_argument(
        "--k_phylo", type=int, default=None,
        help="[phylo_source != uriel] Fixed rank for the phylo eigenvector basis "
             "(default: auto-selected to capture 95%% variance).")
    sp.add_argument(
        "--k_spatial", type=int, default=None,
        help="[phylo_source != uriel] Fixed rank for the spatial eigenvector basis "
             "(default: auto-selected to capture 95%% variance).")
    sp.add_argument(
        "--lengthscale_km", type=float, default=None,
        help="[phylo_source != uriel] Spatial kernel lengthscale in km "
             "(default: median pairwise great-circle distance).")
    sp.add_argument(
        "--spatiophylo_out_dir", default=None,
        help="[phylo_source != uriel] After the factorial loop, train a full-data "
             "conditioned model and save analyze_geometry.py-compatible artifacts here "
             "(lang_embeddings.npy, featvalue_embeddings.npy, config.json, "
             "spatiophylo_meta_both.json …). Skipped in --smoke mode.")

    return p.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data(args: argparse.Namespace) -> Tuple[pd.DataFrame, Any, Any, Any]:
    """
    Load typological data and return (df, data_dict, lang2id_glot, lang2id_full).

    data_dict keys depend on architecture:
      tcf    : binary_matrix, binary_col_names, feature_groups, feature_value_names
      learned: cat_matrix, kept_feature_names, feat_to_global_ids, feat_to_value_names
    """
    log.info("Loading %s data from %s …", args.database, args.data_path)

    if args.database == "wals":
        df, feature_cols = load_wals_cldf(args.data_path)
    else:
        df, feature_cols = load_grambank_cldf(
            args.data_path,
            genus_source="glottolog" if args.glottolog_repo else "family",
            glottolog_repo_dir=args.glottolog_repo,
        )

    lang2id_glot, lang2id_full = build_lang2id(df, args.database, args.data_path)

    if args.architecture == "tcf":
        binary_matrix, binary_col_names, feature_groups, feature_value_names = \
            binarise_features(df, feature_cols)
        data_dict = {
            "binary_matrix": binary_matrix,
            "binary_col_names": binary_col_names,
            "feature_groups": feature_groups,
            "feature_value_names": feature_value_names,
        }
    else:
        cat_matrix, kept_feature_names, feat_to_global_ids, feat_to_value_names = \
            prepare_categorical(df, feature_cols)
        data_dict = {
            "cat_matrix": cat_matrix,
            "kept_feature_names": kept_feature_names,
            "feat_to_global_ids": feat_to_global_ids,
            "feat_to_value_names": feat_to_value_names,
        }

    log.info("Loaded %d languages, %d features.",
             len(df), len(feature_cols))
    return df, data_dict, lang2id_glot, lang2id_full


# ──────────────────────────────────────────────────────────────────────────────
# TCF branch split (binary version of split_by_branch_categorical)
# ──────────────────────────────────────────────────────────────────────────────

def split_by_branch_binary(
    df: pd.DataFrame,
    binary_matrix: np.ndarray,
    target_branch: str,
    in_branch_train_frac: float = 0.20,
    rng: Optional[np.random.Generator] = None,
):
    """
    Train/eval split on the BINARY (binarised) matrix.

    Same protocol as the categorical version:
    - Out-of-branch: all observed cells → train
    - In-branch: 80% per language for eval, remaining × in_branch_train_frac → train

    Returns
    -------
    train_l, train_f, train_v : arrays for training dataset
    eval_l,  eval_f,  eval_v  : arrays for evaluation
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_langs, n_feats = binary_matrix.shape
    in_branch_mask = (df["genus"] == target_branch).values

    out_l, out_f, out_v = [], [], []
    for i in range(n_langs):
        if in_branch_mask[i]:
            continue
        observed = np.where(binary_matrix[i] >= 0)[0]
        out_l.append(np.full(len(observed), i))
        out_f.append(observed)
        out_v.append(binary_matrix[i, observed])

    in_tr_l, in_tr_f, in_tr_v = [], [], []
    ev_l, ev_f, ev_v = [], [], []

    for i in np.where(in_branch_mask)[0]:
        observed = np.where(binary_matrix[i] >= 0)[0].tolist()
        rng.shuffle(observed)
        n_obs = len(observed)
        n_eval = int(np.ceil(n_obs * 0.80))
        eval_feats_i = observed[:n_eval]
        train_pool = observed[n_eval:]
        n_use = min(round(n_obs * in_branch_train_frac), len(train_pool))
        train_feats_i = train_pool[:n_use]

        for fi in eval_feats_i:
            ev_l.append(i); ev_f.append(fi); ev_v.append(binary_matrix[i, fi])
        for fi in train_feats_i:
            in_tr_l.append(i); in_tr_f.append(fi); in_tr_v.append(binary_matrix[i, fi])

    all_l = np.concatenate(out_l + [np.array(in_tr_l, dtype=int)])
    all_f = np.concatenate(out_f + [np.array(in_tr_f, dtype=int)])
    all_v = np.concatenate(out_v + [np.array(in_tr_v, dtype=int)])

    return (all_l, all_f, all_v.astype(np.float32),
            np.array(ev_l, dtype=int),
            np.array(ev_f, dtype=int),
            np.array(ev_v, dtype=np.float32))


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_tcf(
    model: torch.nn.Module,
    eval_l: np.ndarray,
    eval_f: np.ndarray,
    eval_v: np.ndarray,
    device: str = "cpu",
) -> float:
    """Compute micro-F1 for the binary TCF model."""
    model.eval()
    with torch.no_grad():
        all_probs = model.predict_all().cpu().numpy()  # (L, F)

    y_true = eval_v.astype(int)
    y_pred = (all_probs[eval_l, eval_f] >= 0.5).astype(int)

    if len(y_true) == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def _run_conditioned(
    architecture: str,
    df: pd.DataFrame,
    data_dict: dict,
    lang2id_glot: dict,
    geo_matrix: np.ndarray,
    phylo_matrix: np.ndarray,
    target_branch: str,
    in_branch_frac: float,
    seed: int,
    embed_dim: int,
    n_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    l2_coef: float,
    device: str,
    cond_mode: str = "both",
) -> float:
    """
    Train the conditioned model for one (branch, frac, cond_mode) split.

    Returns conditioned F1.  Baseline F1 is loaded externally from the
    pre-computed comparison CSV (see load_baseline_lookup).
    """
    rng = np.random.default_rng(seed)

    if architecture == "learned":
        cat_matrix = data_dict["cat_matrix"]
        feat_to_global_ids = data_dict["feat_to_global_ids"]
        feat_to_value_names = data_dict["feat_to_value_names"]
        n_langs = cat_matrix.shape[0]
        n_total_values = max(
            max(gids) for gids in feat_to_global_ids.values()) + 1

        train_ds, ev_l, ev_f, ev_v = split_by_branch_categorical(
            df, cat_matrix, target_branch, in_branch_frac, rng)

        if len(ev_v) == 0:
            return float("nan")

        cond = TypologicalMF_Conditioned(
            n_langs, n_total_values, feat_to_global_ids,
            geo_matrix, phylo_matrix, embed_dim,
            cond_mode=cond_mode)
        train_conditioned(
            cond, train_ds, n_epochs=n_epochs, batch_size=batch_size,
            lr=lr, l2_reg=l2_coef, device=device, patience=patience)
        return evaluate_learned(cond, ev_l, ev_f, ev_v, feat_to_value_names, device)

    else:  # tcf
        binary_matrix = data_dict["binary_matrix"]
        n_langs, n_binary = binary_matrix.shape

        tr_l, tr_f, tr_v, ev_l, ev_f, ev_v = split_by_branch_binary(
            df, binary_matrix, target_branch, in_branch_frac, rng)

        if len(ev_v) == 0:
            return float("nan")

        from torch.utils.data import TensorDataset
        train_ds = TensorDataset(
            torch.LongTensor(tr_l),
            torch.LongTensor(tr_f),
            torch.FloatTensor(tr_v),
        )

        cond_tcf = TypologicalMF_TCF_Conditioned(
            n_langs, n_binary, geo_matrix, phylo_matrix, embed_dim,
            cond_mode=cond_mode)
        train_conditioned(
            cond_tcf, train_ds, n_epochs=n_epochs, batch_size=batch_size,
            lr=lr, l2_reg=l2_coef, device=device, patience=patience)
        return evaluate_tcf(cond_tcf, ev_l, ev_f, ev_v, device)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    """Execute the full pipeline and return the results DataFrame."""
    t0 = time.time()
    seed_everything(args.seed)

    # ── 1. Load typological data ──
    df, data_dict, lang2id_glot, lang2id_full = load_data(args)

    # ── 2. Load conditioning vectors (URIEL+ or spatiophylogenetic basis) ──
    if args.phylo_source != "uriel":
        from canonical.spatiophylo_basis import (
            build_spatiophylo_basis,
            phylo_covariance_from_classification,
            phylo_covariance_from_newick,
        )
        coords = load_coords_from_cldf(args.data_path, args.database)
        if args.phylo_source == "newick":
            if not args.newick_path:
                raise ValueError(
                    "--phylo_source newick requires --newick_path to be set.")
            phylo_taxa = list(lang2id_glot.keys())
            phylo_cov = phylo_covariance_from_newick(args.newick_path, phylo_taxa)
        else:  # glottolog_topology
            classification, phylo_taxa = build_phylo_classification(df, lang2id_glot)
            phylo_cov = phylo_covariance_from_classification(classification, phylo_taxa)
        phylo_E, spatial_E, spatiophylo_meta = build_spatiophylo_basis(
            lang2id_glot, coords, phylo_cov, phylo_taxa,
            k_phylo=args.k_phylo,
            k_spatial=args.k_spatial,
            lengthscale_km=args.lengthscale_km,
        )
        # phylo_E → phylo_matrix, spatial_E → geo_matrix (drop-in convention)
        geo_matrix, phylo_matrix = spatial_E, phylo_E
        log.info(
            "Spatiophylo basis: phylo %d-dim (cov %.1f%%), spatial %d-dim (cov %.1f%%)",
            phylo_E.shape[1], 100.0 * spatiophylo_meta["phylo_coverage"],
            spatial_E.shape[1], 100.0 * spatiophylo_meta["spatial_coverage"],
        )
    else:
        spatiophylo_meta = None
        parquet_path = Path(args.uriel_vectors_path)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"URIEL+ parquet not found: {parquet_path}\n"
                f"Run: python canonical/uriel_plus_loader.py --full"
            )
        geo_matrix, phylo_matrix, uriel_meta = load_uriel_vectors_for_lang2id(
            str(parquet_path), lang2id_glot,
            geo_dim=299, phylo_dim=3718,
        )
        log.info("URIEL+ coverage for %s: %.1f%% (%d/%d)",
                 args.database, uriel_meta["pct_found"],
                 uriel_meta["found"], uriel_meta["n_langs"])

    # ── 2b. Load pre-computed baseline F1 scores ──
    baseline_lookup = load_baseline_lookup(
        args.baselines_dir, args.database, args.architecture)

    # ── 3. Determine qualifying branches ──
    branch_counts = df["genus"].value_counts()
    qualifying = branch_counts[
        branch_counts >= args.min_branch_size
    ].index.tolist()
    log.info("Qualifying branches (≥%d langs): %d",
             args.min_branch_size, len(qualifying))

    in_branch_fracs = args.in_branch_fracs
    n_repeats = args.n_repeats

    if args.smoke:
        qualifying = qualifying[:1]   # 1 branch is enough to verify wiring
        in_branch_fracs = [0.0]       # 1 frac level
        n_repeats = 1
        args.n_epochs = min(args.n_epochs, 1)  # cap at 1 epoch for speed
        log.info("[smoke] Restricted to %d branches, fracs=%s, repeats=%d, "
                 "n_epochs=%d",
                 len(qualifying), in_branch_fracs, n_repeats, args.n_epochs)
        # Subset languages for speed — keeps subset reproducible via seed
        if args.smoke_n_langs > 0 and len(df) > args.smoke_n_langs:
            rng_sub = np.random.default_rng(args.seed)
            # Ensure at least one language per qualifying branch is retained
            keep_idx = set()
            for branch in qualifying:
                branch_rows = df.index[df["genus"] == branch].tolist()
                keep_idx.update(branch_rows[:5])  # keep up to 5 per branch
            remaining = [i for i in range(len(df)) if i not in keep_idx]
            extra = rng_sub.choice(
                remaining,
                max(0, args.smoke_n_langs - len(keep_idx)),
                replace=False,
            ).tolist()
            keep_idx.update(extra)
            keep_sorted = sorted(keep_idx)
            df = df.iloc[keep_sorted].reset_index(drop=True)
            if args.architecture == "learned":
                data_dict["cat_matrix"] = data_dict["cat_matrix"][keep_sorted]
            else:
                data_dict["binary_matrix"] = data_dict["binary_matrix"][keep_sorted]
            geo_matrix = geo_matrix[keep_sorted]
            phylo_matrix = phylo_matrix[keep_sorted]
            log.info("[smoke] Subset to %d languages.", len(df))

    # ── 4. Resume: load existing results ──
    conditioning_variants: List[str] = getattr(
        args, "conditioning_variants", ["both"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.database}_{args.architecture}_s{args.seed}"
    if args.phylo_source != "uriel":
        run_id += f"_sp_{args.phylo_source}"
    results_path = out_dir / f"{run_id}.csv"

    done_keys: set = set()
    existing_rows: List[dict] = []
    if args.resume and results_path.exists():
        existing_df = pd.read_csv(results_path)
        existing_rows = existing_df.to_dict("records")
        for row in existing_rows:
            # Backwards-compatible default for rows written before this column existed.
            done_keys.add((
                row["branch"],
                row["in_branch_frac"],
                row["repeat"],
                row.get("conditioning", "both"),
            ))
        log.info("[resume] Loaded %d existing rows (%d done combos).",
                 len(existing_rows), len(done_keys))

    # ── 5. Full factorial loop ──
    rows: List[dict] = list(existing_rows)
    total_combos = (len(qualifying) * len(in_branch_fracs)
                    * n_repeats * len(conditioning_variants))
    combo_i = 0

    for branch in qualifying:
        macroarea = df.loc[df["genus"] == branch, "macroarea"].iloc[0]
        for frac in in_branch_fracs:
            for rep in range(n_repeats):
                for variant in conditioning_variants:
                    combo_i += 1
                    key = (branch, frac, rep, variant)

                    if key in done_keys:
                        log.info(
                            "[%d/%d] Skip (done): branch=%s frac=%.2f rep=%d variant=%s",
                            combo_i, total_combos, branch, frac, rep, variant)
                        continue

                    log.info("[%d/%d] branch=%-30s frac=%.2f rep=%d variant=%s",
                             combo_i, total_combos, branch, frac, rep, variant)

                    lookup_key = (branch, round(frac, 6), rep)
                    f1_base = baseline_lookup.get(lookup_key)
                    if f1_base is None:
                        log.warning(
                            "  No baseline F1 for branch=%s frac=%.2f rep=%d"
                            " — skipping cell",
                            branch, frac, rep)
                        continue

                    rep_seed = (args.seed * 10_000 + rep * 1_000
                                + hash(branch) % 1000)
                    cond_mode = _VARIANT_TO_COND_MODE[variant]
                    try:
                        f1_cond = _run_conditioned(
                            architecture=args.architecture,
                            df=df,
                            data_dict=data_dict,
                            lang2id_glot=lang2id_glot,
                            geo_matrix=geo_matrix,
                            phylo_matrix=phylo_matrix,
                            target_branch=branch,
                            in_branch_frac=frac,
                            seed=rep_seed,
                            embed_dim=args.embed_dim,
                            n_epochs=args.n_epochs,
                            patience=args.patience,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            l2_coef=args.l2_coef,
                            device=args.device,
                            cond_mode=cond_mode,
                        )
                    except Exception as exc:
                        log.warning(
                            "  ERROR branch=%s frac=%.2f rep=%d variant=%s: %s",
                            branch, frac, rep, variant, exc)
                        continue

                    base_row = dict(
                        database=args.database,
                        architecture=args.architecture,
                        seed=args.seed,
                        branch=branch,
                        macroarea=macroarea,
                        in_branch_frac=frac,
                        repeat=rep,
                        conditioning=variant,
                        model_type="baseline",
                        f1=f1_base,
                    )
                    cond_row = {**base_row,
                                "model_type": "conditioned",
                                "f1": f1_cond}

                    rows.append(base_row)
                    rows.append(cond_row)

                    log.info(
                        "  → [%s] baseline F1=%.4f  conditioned F1=%.4f  delta=%+.4f",
                        variant, f1_base, f1_cond,
                        (f1_cond - f1_base)
                        if np.isfinite(f1_base) and np.isfinite(f1_cond)
                        else float("nan"))

                    # Checkpoint after every combo
                    pd.DataFrame(rows).to_csv(results_path, index=False)

    results_df = pd.DataFrame(rows)
    elapsed = time.time() - t0

    # ── 6. Save final results and config ──
    results_df.to_csv(results_path, index=False)
    log.info("Saved %d result rows → %s", len(results_df), results_path)

    cfg = {
        "database": args.database,
        "architecture": args.architecture,
        "conditioning_source": "spatiophylo" if args.phylo_source != "uriel" else "uriel+",
        "phylo_source": args.phylo_source,
        "seed": args.seed,
        "embed_dim": args.embed_dim,
        "n_epochs": args.n_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "l2_coef": args.l2_coef,
        "n_branches": len(qualifying),
        "in_branch_fracs": in_branch_fracs,
        "n_repeats": n_repeats,
        "conditioning_variants": conditioning_variants,
        "min_branch_size": args.min_branch_size,
        "smoke": args.smoke,
        "n_result_rows": len(results_df),
        "elapsed_s": round(elapsed, 1),
    }
    if args.phylo_source != "uriel":
        cfg["spatiophylo_meta"] = spatiophylo_meta
        cfg["newick_path"] = args.newick_path
    else:
        cfg["uriel_vectors_path"] = str(parquet_path)
        cfg["uriel_coverage_pct"] = round(uriel_meta["pct_found"], 2)
    cfg_path = out_dir / f"config_{run_id}.json"
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    log.info("Saved config → %s", cfg_path)

    # Write per-variant spatiophylo meta sidecars into the results dir
    if args.phylo_source != "uriel" and spatiophylo_meta is not None:
        for variant in conditioning_variants:
            cond_mode = _VARIANT_TO_COND_MODE[variant]
            meta_sidecar = {**spatiophylo_meta,
                            "cond_mode": cond_mode,
                            "phylo_source": args.phylo_source}
            meta_path = out_dir / f"spatiophylo_meta_{cond_mode}.json"
            with open(meta_path, "w") as fh:
                json.dump(meta_sidecar, fh, indent=2)
        log.info("Saved spatiophylo meta sidecars → %s", out_dir)

    # ── 6b. Save de-confounded canonical checkpoint (spatiophylo mode only) ──
    if args.phylo_source != "uriel" and args.spatiophylo_out_dir:
        if args.smoke:
            log.warning(
                "[smoke] Skipping canonical checkpoint — data was subsetted. "
                "Re-run with --full to save de-confounded embeddings.")
        else:
            _save_spatiophylo_checkpoint(
                architecture=args.architecture,
                df=df,
                data_dict=data_dict,
                lang2id_glot=lang2id_glot,
                lang2id_full=lang2id_full,
                geo_matrix=geo_matrix,
                phylo_matrix=phylo_matrix,
                args=args,
                out_dir=args.spatiophylo_out_dir,
                spatiophylo_meta=spatiophylo_meta,
            )

    return results_df


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.full:
        log.warning(
            "═" * 60 + "\n"
            "  FULL MODE: this will train thousands of models.\n"
            "  Run on a lab server. See LAB_SERVER_RUNBOOK.md\n"
            + "═" * 60
        )

    results = run_pipeline(args)

    if len(results) == 0:
        log.warning("No results produced.")
        return

    # Summary statistics
    by_type = results.groupby("model_type")["f1"].agg(["mean", "std", "count"])
    print("\n" + "=" * 60)
    print(f"  Run: {args.database} / {args.architecture} / seed={args.seed}")
    print("=" * 60)
    print(by_type.to_string())
    if "baseline" in by_type.index and "conditioned" in by_type.index:
        delta = by_type.loc["conditioned", "mean"] - by_type.loc["baseline", "mean"]
        print(f"\n  Mean F1 delta (conditioned − baseline): {delta:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
