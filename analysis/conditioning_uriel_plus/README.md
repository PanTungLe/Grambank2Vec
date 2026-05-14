# URIEL+ Conditioning Vectors — RQ4

## File

`uriel_plus_vectors.parquet`

| Column      | Type            | Description |
|-------------|-----------------|-------------|
| glottocode  | string          | Glottocode identifier |
| geo_vec     | list\<float32\> | Geographic vector from URIEL+ geocoord_features.npz (dim=299) |
| phylo_vec   | list\<float32\> | Phylogenetic vector from URIEL+ family_features.npz (dim=3718) |
| has_geo     | bool            | True if this language's geo vector comes from real URIEL+ data |
| has_phylo   | bool            | True if this language's phylo vector comes from real URIEL+ data |

## Coverage

Approximately **94% of languages** have real URIEL+ data (`has_geo=True`).
The remaining ~6% receive **zero vectors by design** (`has_geo=False`).

**No imputation is performed.** Languages absent from URIEL+ get zero vectors
and `has_geo=False` / `has_phylo=False`.  This is intentional:

- The conditioned model uses a zero-init `cond_proj` layer, so zero-vector
  languages start with the same embedding as the unconditioned baseline.
- The model learns to use URIEL+ signal only where it exists; zero inputs are
  effectively a "no-op" conditioning.
- There is no fabrication, no family-mean or KNN proxy, and no circular use of
  typological features (Constraint 3).

## Constraint 3

This parquet contains **only** geographic (`geo_vec`) and phylogenetic
(`phylo_vec`) information.  Typological / featural data from URIEL+ is never
queried, stored, or used.  URIEL+ integrates Grambank as a typological source;
using those features to condition Grambank predictions would be circular.

Schema verification is run automatically after every save:

```python
import pyarrow.parquet as pq
t = pq.read_table('analysis/conditioning_uriel_plus/uriel_plus_vectors.parquet')
print(t.schema)
# Must show only: glottocode, geo_vec, phylo_vec, has_geo, has_phylo
assert set(t.schema.names) == {"glottocode", "geo_vec", "phylo_vec",
                                "has_geo", "has_phylo"}
```

## Rebuilding

```bash
python canonical/uriel_plus_loader.py --full
# or for a quick sanity check:
python canonical/uriel_plus_loader.py --smoke
```
