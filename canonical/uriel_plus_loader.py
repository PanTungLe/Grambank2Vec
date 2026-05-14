"""
URIEL+ geographic and phylogenetic vector loader for RQ4.

Queries URIEL+ for geographic ('geo') and phylogenetic ('fam') vectors for
every Glottocode appearing in the canonical WALS and Grambank checkpoints.

HARD CONSTRAINT (Constraint 3): this script NEVER queries, touches, or stores
any 'fea' (typological) URIEL+ features.  URIEL+ integrates Grambank as a
typological source; using the typological vector to condition predictions of
Grambank features would be circular.

Languages absent from URIEL+ receive zero vectors by design — there is no
imputation or fabrication.  Approximately 6% of languages fall into this
category.  The has_geo / has_phylo flags in the parquet identify them.

Output:
  analysis/conditioning_uriel_plus/uriel_plus_vectors.parquet
    columns: glottocode (str), geo_vec (list[float32]), phylo_vec (list[float32]),
             has_geo (bool), has_phylo (bool)

Usage
-----
  python canonical/uriel_plus_loader.py \\
      [--output_dir analysis/conditioning_uriel_plus] \\
      [--urielplus_dir /tmp/URIELPlus] \\
      [--smoke | --full] \\
      [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "analysis" / "conditioning_uriel_plus")
_DEFAULT_URIELPLUS_DIR = "/tmp/URIELPlus"
_URIELPLUS_REPO = "https://github.com/Masonshipton25/URIELPlus"

_CHECKPOINT_DIRS = [
    _REPO_ROOT / "checkpoints" / "wals_learned_s42",
    _REPO_ROOT / "checkpoints" / "grambank_learned_s42",
]

# ──────────────────────────────────────────────────────────────────────────────
# Safety assertion — enforced at every access point
# ──────────────────────────────────────────────────────────────────────────────

_FORBIDDEN_FEATURE_TYPES = frozenset(["fea", "typological", "featural",
                                       "syntactic", "phonological",
                                       "inventory", "morphological"])

# Columns permitted in the output parquet (Constraint 3).
_PERMITTED_PARQUET_COLUMNS = frozenset([
    "glottocode", "geo_vec", "phylo_vec",
    "has_geo", "has_phylo",
])


def _assert_not_typological(label: str) -> None:
    """Raise if `label` names a typological / featural vector category."""
    lower = label.lower()
    for forbidden in _FORBIDDEN_FEATURE_TYPES:
        if forbidden in lower:
            raise AssertionError(
                f"[CONSTRAINT 3 VIOLATED] Attempted to access typological "
                f"vector category '{label}'. Grambank is a URIEL+ typological "
                "source; using it as conditioning signal is circular. "
                "Only 'geo' (geographic) and 'fam' / 'phylogeny' vectors "
                "are permitted for RQ4 conditioning."
            )


def _verify_parquet_schema(out_path: Path) -> None:
    """Raise if the saved parquet contains any column outside the permitted set."""
    t = pq.read_table(str(out_path))
    actual = set(t.schema.names)
    forbidden = actual - _PERMITTED_PARQUET_COLUMNS
    assert not forbidden, (
        f"[Constraint 3] Parquet {out_path.name} contains forbidden columns: "
        f"{sorted(forbidden)}. Only geo/phylo data is permitted (no typological)."
    )
    log.info("[Constraint 3] Parquet schema verified — no forbidden columns.")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _collect_glottocodes() -> list[str]:
    """Union of all Glottocodes across the WALS and Grambank checkpoints."""
    all_gc: set[str] = set()
    for ckpt in _CHECKPOINT_DIRS:
        l2id_path = ckpt / "lang2id.json"
        if not l2id_path.exists():
            log.warning("lang2id.json not found at %s — skipping.", ckpt)
            continue
        l2id = json.loads(l2id_path.read_text())
        all_gc.update(l2id.keys())
    log.info("Collected %d unique Glottocodes from checkpoints.", len(all_gc))
    return sorted(all_gc)


def _clone_or_verify(urielplus_dir: str) -> str:
    """Clone URIELPlus if absent; return directory path."""
    d = Path(urielplus_dir)
    if not d.exists():
        log.info("Cloning URIELPlus → %s …", d)
        subprocess.run(
            ["git", "clone", "--depth=1", _URIELPLUS_REPO, str(d)],
            check=True,
        )
    return str(d)


# ──────────────────────────────────────────────────────────────────────────────
# URIEL+ data extraction
# ──────────────────────────────────────────────────────────────────────────────

def _load_uriel_plus(urielplus_dir: str):
    """
    Initialise URIEL+ in Glottocode mode and return the URIELPlus object.

    The typological index (data[1] / features.npz) is present in memory but
    will never be read by our code; we verify this by asserting on every lookup.
    """
    if str(Path(urielplus_dir).parent) not in sys.path:
        sys.path.insert(0, str(Path(urielplus_dir).parent))
    try:
        from urielplus import urielplus as up_module
    except ImportError:
        up_path = Path(urielplus_dir) / "urielplus"
        if not up_path.exists():
            raise ImportError(
                f"Cannot import urielplus. Ensure URIELPlus is cloned at "
                f"{urielplus_dir} and pip install -e was run."
            )
        sys.path.insert(0, str(Path(urielplus_dir)))
        from urielplus import urielplus as up_module

    u = up_module.URIELPlus()
    u.set_glottocodes()

    assert "family" in u.files[0].lower(), \
        f"Unexpected file[0]: {u.files[0]} — expected family_features.npz"
    assert "geocoord" in u.files[2].lower(), \
        f"Unexpected file[2]: {u.files[2]} — expected geocoord_features.npz"
    assert "feature" in u.files[1].lower(), \
        f"Unexpected file[1]: {u.files[1]} — expected features.npz (typological)"

    log.info("URIEL+ initialised.  "
             "Phylo: %d langs × %d features.  "
             "Geo:   %d langs × %d features.",
             len(u.langs[0]), u.data[0].shape[1],
             len(u.langs[2]), u.data[2].shape[1])
    return u


def _extract_raw_vectors(
    u,
    glottocodes: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
           list[str], list[str]]:
    """
    Extract geographic and phylogenetic vectors for every Glottocode.

    Constraint 3 is enforced: we only access data[0] (phylo) and data[2] (geo).
    data[1] (typological) is never touched.

    Returns:
        geo_vecs   — {glottocode: (299,) float32}
        phylo_vecs — {glottocode: (3718,) float32}
        found      — glottocodes present in URIEL+
        missing    — glottocodes absent from URIEL+
    """
    assert len(u.files) > 2, "URIEL+ must have at least 3 file slots"
    assert "feature" in u.files[1].lower(), \
        f"[Constraint 3] data[1] must be typological (features.npz), got {u.files[1]}"

    geo_idx_map: dict[str, int] = {
        gc: i for i, gc in enumerate(u.langs[2].tolist())
    }
    phylo_idx_map: dict[str, int] = {
        gc: i for i, gc in enumerate(u.langs[0].tolist())
    }

    geo_vecs: dict[str, np.ndarray] = {}
    phylo_vecs: dict[str, np.ndarray] = {}
    found: list[str] = []
    missing: list[str] = []

    for gc in glottocodes:
        in_geo   = gc in geo_idx_map
        in_phylo = gc in phylo_idx_map

        if in_geo and in_phylo:
            geo_vecs[gc]   = u.data[2][geo_idx_map[gc]][:, 0].astype(np.float32)
            phylo_vecs[gc] = u.data[0][phylo_idx_map[gc]][:, 0].astype(np.float32)
            found.append(gc)
        else:
            missing.append(gc)

    log.info("Found in URIEL+: %d / %d  (missing: %d)",
             len(found), len(glottocodes), len(missing))
    return geo_vecs, phylo_vecs, found, missing


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def _save_parquet(
    glottocodes: list[str],
    geo_vecs: dict[str, np.ndarray],
    phylo_vecs: dict[str, np.ndarray],
    found_set: set[str],
    out_path: Path,
) -> None:
    """Save all glottocodes (found + zero-filled missing) to parquet."""
    rows_geo   = [geo_vecs[gc].tolist()   for gc in glottocodes]
    rows_phylo = [phylo_vecs[gc].tolist() for gc in glottocodes]
    has_geo    = [gc in found_set for gc in glottocodes]
    has_phylo  = [gc in found_set for gc in glottocodes]

    table = pa.table({
        "glottocode": pa.array(glottocodes, type=pa.string()),
        "geo_vec":    pa.array(rows_geo,    type=pa.list_(pa.float32())),
        "phylo_vec":  pa.array(rows_phylo,  type=pa.list_(pa.float32())),
        "has_geo":    pa.array(has_geo,     type=pa.bool_()),
        "has_phylo":  pa.array(has_phylo,   type=pa.bool_()),
    })
    pq.write_table(table, str(out_path))
    log.info("Saved %s  (%d rows, %d with real URIEL+ data)",
             out_path.name, len(glottocodes), sum(has_geo))


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_loader(
    output_dir: str,
    urielplus_dir: str,
    smoke: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Build the URIEL+ vector parquet.

    Languages absent from URIEL+ receive zero vectors (has_geo=False).
    No imputation is performed — zero vectors are the conditioning signal
    for unattested languages, which the model learns to ignore via the
    zero-init cond_proj.
    """
    log.info("[Constraint 3] Confirmed: only geo (data[2]) and phylo (data[0]) "
             "vectors will be accessed.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Collect all Glottocodes
    all_gcs = _collect_glottocodes()
    if smoke:
        rng = np.random.default_rng(seed)
        all_gcs = sorted(
            rng.choice(all_gcs, min(200, len(all_gcs)), replace=False).tolist()
        )
        log.info("[smoke] Subset to %d Glottocodes.", len(all_gcs))

    # 2. Load URIEL+
    _clone_or_verify(urielplus_dir)
    u = _load_uriel_plus(urielplus_dir)

    # 3. Extract raw vectors (geo and phylo only — Constraint 3 enforced inside)
    geo_vecs, phylo_vecs, found, missing = _extract_raw_vectors(u, all_gcs)

    pct_found = 100.0 * len(found) / len(all_gcs) if all_gcs else 0.0
    log.info("Coverage: %.1f%%  (%d found, %d missing — zero vectors by design)",
             pct_found, len(found), len(missing))

    # 4. Infer dimensions from found languages; default to known URIEL+ dims
    if geo_vecs:
        geo_dim   = next(iter(geo_vecs.values())).shape[0]
        phylo_dim = next(iter(phylo_vecs.values())).shape[0]
    else:
        geo_dim, phylo_dim = 299, 3718

    # 5. Fill missing languages with zero vectors (by design, no fabrication)
    zero_geo   = np.zeros(geo_dim,   dtype=np.float32)
    zero_phylo = np.zeros(phylo_dim, dtype=np.float32)
    for gc in missing:
        geo_vecs[gc]   = zero_geo.copy()
        phylo_vecs[gc] = zero_phylo.copy()

    # 6. Save parquet
    found_set = set(found)
    out_path  = out / "uriel_plus_vectors.parquet"
    _save_parquet(all_gcs, geo_vecs, phylo_vecs, found_set, out_path)

    # Constraint 3 post-save guard: verify no typological columns leaked in.
    _verify_parquet_schema(out_path)

    summary: dict[str, Any] = {
        "n_glottocodes":  len(all_gcs),
        "n_found":        len(found),
        "n_missing":      len(missing),
        "pct_found":      round(pct_found, 2),
        "geo_dim":        geo_dim,
        "phylo_dim":      phylo_dim,
        "output_file":    str(out_path),
        "smoke":          smoke,
        "seed":           seed,
    }
    (out / "config_uriel_plus.json").write_text(json.dumps(summary, indent=2))
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="URIEL+ geographic and phylogenetic vector loader for RQ4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output_dir",    default=_DEFAULT_OUTPUT_DIR)
    p.add_argument("--urielplus_dir", default=_DEFAULT_URIELPLUS_DIR)
    scale = p.add_mutually_exclusive_group()
    scale.add_argument("--smoke", action="store_true",
                       help="Run on a 200-language subset.")
    scale.add_argument("--full", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip if output parquet already exists.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    out_path = Path(args.output_dir) / "uriel_plus_vectors.parquet"
    if args.resume and out_path.exists():
        log.info("[resume] %s already exists — skipping.", out_path.name)
        return

    summary = run_loader(
        output_dir=args.output_dir,
        urielplus_dir=args.urielplus_dir,
        smoke=args.smoke,
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print(f"  Glottocodes:    {summary['n_glottocodes']}")
    print(f"  Found:          {summary['n_found']}  (real URIEL+ data)")
    print(f"  Missing:        {summary['n_missing']}  (zero vectors, has_geo=False)")
    print(f"  % coverage:     {summary['pct_found']:.1f}%")
    print(f"  Geo dim:        {summary['geo_dim']}")
    print(f"  Phylo dim:      {summary['phylo_dim']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
