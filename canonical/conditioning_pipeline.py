"""
RQ4 conditioning pipeline — full factorial evaluation.

For each (database, architecture, imputation, seed) configuration:
  1. Load data and URIEL+ geo+phylo vectors.
  2. For each qualifying branch × in_branch_frac × repeat:
       a. Build train/eval split (branch-based protocol).
       b. Train BASELINE model (no conditioning).
       c. Train CONDITIONED model (URIEL+ geo+phylo offset via cond_proj).
       d. Evaluate both; record F1 rows.
  3. Save results to analysis/conditioning_results/<run_id>.parquet and .csv.

IMPORTANT — full factorial is ~24 000 model trainings; Claude runs smoke only.
Run the full factorial on a lab server using the runbook in
  canonical/LAB_SERVER_RUNBOOK.md

Usage (smoke — safe to run locally)
-------------------------------------
python canonical/conditioning_pipeline.py \\
    --database wals \\
    --architecture learned \\
    --data_path /path/to/wals \\
    --imputation familymean \\
    --seed 42 \\
    --smoke

Usage (full — lab server only)
-------------------------------
python canonical/conditioning_pipeline.py \\
    --database wals \\
    --architecture learned \\
    --data_path /path/to/wals \\
    --imputation familymean \\
    --seed 42 \\
    --full
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
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_preparation import (
    load_wals_cldf,
    load_grambank_cldf,
    binarise_features,
)
from model import TypologicalMF, WALSDataset
from model_learned import (
    TypologicalMF_Learned,
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
from utils import seed_everything, build_all_triples_binary, build_all_triples_categorical

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)


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
        "--imputation",
        required=True,
        choices=["familymean", "knn", "softimpute"],
        help="URIEL+ imputation method (determines which parquet to load).",
    )
    p.add_argument(
        "--uriel_dir",
        default="analysis/conditioning_uriel_plus",
        help="Directory containing uriel_plus_vectors_<method>.parquet files.",
    )
    p.add_argument(
        "--out_dir",
        default="analysis/conditioning_results",
        help="Output directory for results.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
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
        help="Skip already-completed (branch, frac, repeat, model_type) rows.",
    )
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


def _run_one(
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
) -> Tuple[float, float]:
    """
    Train baseline + conditioned models for one (branch, frac) split.

    Returns (f1_baseline, f1_conditioned).
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
            return float("nan"), float("nan")

        # Baseline
        baseline = TypologicalMF_Learned(
            n_langs, n_total_values, feat_to_global_ids, embed_dim)
        from model_learned import train_model_learned
        train_model_learned(
            baseline, train_ds, n_epochs=n_epochs, batch_size=batch_size,
            lr=lr, l2_reg=l2_coef, device=device)
        f1_base = evaluate_learned(
            baseline, ev_l, ev_f, ev_v, feat_to_value_names, device)

        # Conditioned
        cond = TypologicalMF_Conditioned(
            n_langs, n_total_values, feat_to_global_ids,
            geo_matrix, phylo_matrix, embed_dim)
        train_conditioned(
            cond, train_ds, n_epochs=n_epochs, batch_size=batch_size,
            lr=lr, l2_reg=l2_coef, device=device, patience=patience)
        f1_cond = evaluate_learned(
            cond, ev_l, ev_f, ev_v, feat_to_value_names, device)

        return f1_base, f1_cond

    else:  # tcf
        binary_matrix = data_dict["binary_matrix"]
        n_langs, n_binary = binary_matrix.shape

        tr_l, tr_f, tr_v, ev_l, ev_f, ev_v = split_by_branch_binary(
            df, binary_matrix, target_branch, in_branch_frac, rng)

        if len(ev_v) == 0:
            return float("nan"), float("nan")

        from torch.utils.data import TensorDataset
        train_ds = TensorDataset(
            torch.LongTensor(tr_l),
            torch.LongTensor(tr_f),
            torch.FloatTensor(tr_v),
        )

        # Baseline
        baseline_tcf = TypologicalMF(n_langs, n_binary, embed_dim)
        _train_tcf(baseline_tcf, train_ds, n_epochs, batch_size,
                   lr, l2_coef, device)
        f1_base = evaluate_tcf(baseline_tcf, ev_l, ev_f, ev_v, device)

        # Conditioned
        cond_tcf = TypologicalMF_TCF_Conditioned(
            n_langs, n_binary, geo_matrix, phylo_matrix, embed_dim)
        train_conditioned(
            cond_tcf, train_ds, n_epochs=n_epochs, batch_size=batch_size,
            lr=lr, l2_reg=l2_coef, device=device, patience=patience)
        f1_cond = evaluate_tcf(cond_tcf, ev_l, ev_f, ev_v, device)

        return f1_base, f1_cond


def _train_tcf(
    model: TypologicalMF,
    train_ds,
    n_epochs: int,
    batch_size: int,
    lr: float,
    l2_coef: float,
    device: str,
) -> None:
    """Thin training wrapper for the baseline TCF model."""
    import torch.optim as optim
    model = model.to(device)
    loader = DataLoader(train_ds, batch_size=batch_size,
                        shuffle=True, drop_last=False)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    for _ in range(n_epochs):
        model.train()
        for lang_idx, feat_idx, vals in loader:
            lang_idx = lang_idx.to(device)
            feat_idx = feat_idx.to(device)
            vals = vals.to(device)
            preds, batch_l2 = model(lang_idx, feat_idx)
            loss = F.binary_cross_entropy(preds, vals) + (l2_coef / 2.0) * batch_l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    """Execute the full pipeline and return the results DataFrame."""
    t0 = time.time()
    seed_everything(args.seed)

    # ── 1. Load typological data ──
    df, data_dict, lang2id_glot, lang2id_full = load_data(args)

    # ── 2. Load URIEL+ vectors ──
    parquet_path = (
        Path(args.uriel_dir)
        / f"uriel_plus_vectors_{args.imputation}.parquet"
    )
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"URIEL+ parquet not found: {parquet_path}\n"
            f"Run: python canonical/uriel_plus_loader.py "
            f"--imputation {args.imputation} --full"
        )

    n_langs = len(df)
    geo_matrix, phylo_matrix, uriel_meta = load_uriel_vectors_for_lang2id(
        str(parquet_path), lang2id_glot,
        geo_dim=299, phylo_dim=3718,
    )
    # geo_matrix / phylo_matrix are indexed by Glottocode lang2id, but the
    # model uses the full df index (0..n_langs-1).  We need to re-map.
    # lang2id_glot: glottocode → df-row index; geo_matrix[i] is the vector
    # for the language at df row i (because load_uriel_vectors_for_lang2id
    # uses lang2id value as the row index).
    # So geo_matrix is already aligned to df row indices for covered languages,
    # and zero elsewhere. The model uses df row indices directly.

    log.info("URIEL+ coverage for %s: %.1f%% (%d/%d)",
             args.database, uriel_meta["pct_found"],
             uriel_meta["found"], uriel_meta["n_langs"])

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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = (f"{args.database}_{args.architecture}"
              f"_{args.imputation}_s{args.seed}")
    results_path = out_dir / f"{run_id}.csv"

    done_keys: set = set()
    existing_rows: List[dict] = []
    if args.resume and results_path.exists():
        existing_df = pd.read_csv(results_path)
        existing_rows = existing_df.to_dict("records")
        for row in existing_rows:
            done_keys.add(
                (row["branch"], row["in_branch_frac"], row["repeat"]))
        log.info("[resume] Loaded %d existing rows (%d done combos).",
                 len(existing_rows), len(done_keys))

    # ── 5. Full factorial loop ──
    rows: List[dict] = list(existing_rows)
    total_combos = len(qualifying) * len(in_branch_fracs) * n_repeats
    combo_i = 0

    for branch in qualifying:
        macroarea = df.loc[df["genus"] == branch, "macroarea"].iloc[0]
        for frac in in_branch_fracs:
            for rep in range(n_repeats):
                combo_i += 1
                key = (branch, frac, rep)

                if key in done_keys:
                    log.info("[%d/%d] Skip (done): branch=%s frac=%.2f rep=%d",
                             combo_i, total_combos, branch, frac, rep)
                    continue

                log.info("[%d/%d] branch=%-30s frac=%.2f rep=%d",
                         combo_i, total_combos, branch, frac, rep)

                rep_seed = args.seed * 10_000 + rep * 1_000 + hash(branch) % 1000
                try:
                    f1_base, f1_cond = _run_one(
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
                    )
                except Exception as exc:
                    log.warning("  ERROR branch=%s frac=%.2f rep=%d: %s",
                                branch, frac, rep, exc)
                    continue

                base_row = dict(
                    database=args.database,
                    architecture=args.architecture,
                    imputation=args.imputation,
                    seed=args.seed,
                    branch=branch,
                    macroarea=macroarea,
                    in_branch_frac=frac,
                    repeat=rep,
                    model_type="baseline",
                    f1=f1_base,
                )
                cond_row = {**base_row,
                            "model_type": "conditioned",
                            "f1": f1_cond}

                rows.append(base_row)
                rows.append(cond_row)

                log.info("  → baseline F1=%.4f  conditioned F1=%.4f  "
                         "delta=%+.4f",
                         f1_base, f1_cond,
                         (f1_cond - f1_base) if np.isfinite(f1_base) and np.isfinite(f1_cond) else float("nan"))

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
        "imputation": args.imputation,
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
        "min_branch_size": args.min_branch_size,
        "smoke": args.smoke,
        "uriel_coverage_pct": round(uriel_meta["pct_found"], 2),
        "n_result_rows": len(results_df),
        "elapsed_s": round(elapsed, 1),
    }
    cfg_path = out_dir / f"config_{run_id}.json"
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    log.info("Saved config → %s", cfg_path)

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
    print(f"  Run: {args.database} / {args.architecture} / "
          f"{args.imputation} / seed={args.seed}")
    print("=" * 60)
    print(by_type.to_string())
    if "baseline" in by_type.index and "conditioned" in by_type.index:
        delta = by_type.loc["conditioned", "mean"] - by_type.loc["baseline", "mean"]
        print(f"\n  Mean F1 delta (conditioned − baseline): {delta:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
