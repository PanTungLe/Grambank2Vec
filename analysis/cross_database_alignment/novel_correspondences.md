# Novel WALS ↔ Grambank Correspondences — Two-Tier Discovery

Candidates are WALS feature-values **absent from the gold standard** that appear in the top-5 of one or more alignment approaches.

**Tier 1 (B∩C, higher confidence):** Both the language-profile method (Approach B) and the CCA-projection method (Approach C) independently place the same Grambank feature-value in their top-5 lists.  The two methods use different mathematical machinery, so agreement between them raises confidence.

**Tier 2 (pure-B, exploratory):** Only Approach B (headline method) lists the Grambank feature-value in its top-5.  Broader coverage than Tier 1; requires expert validation before use as research claims.

_All candidates require expert validation before use as research claims._

## Tier 1 — B∩C Agreement (Top 20)

### Complex Sentences

| # | WALS | Grambank | rank_B | rank_C | sim_B | sim_C |
|---|------|----------|--------|--------|-------|-------|
| 1 | `110A=Purposive but no sequential` | `GB025=1` | 1 | 1 | 0.974 | 0.358 |

### Lexicon

| # | WALS | Grambank | rank_B | rank_C | sim_B | sim_C |
|---|------|----------|--------|--------|-------|-------|
| 2 | `115A=Predicate negation also present` | `GB093=0` | 1 | 4 | 0.990 | 0.321 |
| 3 | `113A=Both` | `GB131=0` | 2 | 4 | 0.978 | 0.336 |
| 4 | `114A=A/Cat` | `GB409=0` | 5 | 1 | 0.961 | 0.586 |
| 5 | `128A=Deranked` | `GB025=1` | 4 | 2 | 0.945 | 0.241 |
| 6 | `144D=More than one construction` | `GB024=1` | 2 | 4 | 0.945 | 0.331 |

### Morphology

| # | WALS | Grambank | rank_B | rank_C | sim_B | sim_C |
|---|------|----------|--------|--------|-------|-------|
| 7 | `30A=Two` | `GB203=1` | 2 | 2 | 0.973 | 0.576 |
| 8 | `19A=Pharyngeals` | `GB025=1` | 3 | 2 | 0.951 | 0.422 |

### Nominal Categories

| # | WALS | Grambank | rank_B | rank_C | sim_B | sim_C |
|---|------|----------|--------|--------|-------|-------|
| 9 | `49A=10 or more cases` | `GB025=1` | 2 | 2 | 0.978 | 0.340 |
| 10 | `43A=Unrelated` | `GB082=0` | 3 | 3 | 0.973 | 0.541 |

### Nominal Syntax

| # | WALS | Grambank | rank_B | rank_C | sim_B | sim_C |
|---|------|----------|--------|--------|-------|-------|
| 11 | `59A=No possessive classification` | `GB203=1` | 5 | 1 | 0.970 | 0.625 |

### Tone

| # | WALS | Grambank | rank_B | rank_C | sim_B | sim_C |
|---|------|----------|--------|--------|-------|-------|
| 12 | `4A=In both plosives and fricatives` | `GB031=0` | 4 | 1 | 0.931 | 0.523 |
| 13 | `4A=In both plosives and fricatives` | `GB117=1` | 1 | 5 | 0.936 | 0.351 |
| 14 | `6A=Uvular stops and continuants` | `GB025=1` | 5 | 1 | 0.951 | 0.338 |

### Word Order

| # | WALS | Grambank | rank_B | rank_C | sim_B | sim_C |
|---|------|----------|--------|--------|-------|-------|
| 15 | `91A=Degree word-Adjective` | `GB025=1` | 1 | 1 | 0.951 | 0.596 |
| 16 | `92A=Final` | `GB203=2` | 1 | 1 | 0.910 | 0.506 |
| 17 | `81B=VSO or VOS` | `GB117=0` | 1 | 5 | 0.938 | 0.454 |
| 18 | `82A=SV` | `GB131=0` | 2 | 4 | 0.974 | 0.507 |
| 19 | `90D=Internally-headed or RelN` | `GB022=0` | 4 | 2 | 0.956 | 0.463 |
| 20 | `96A=VO and NRel` | `GB074=1` | 1 | 5 | 0.913 | 0.299 |

## Tier 2 — Pure-B Exploratory (Top 20)

### Complex Sentences

| # | WALS | Grambank | rank_B | sim_B |
|---|------|----------|--------|-------|
| 1 | `111A=Morphological but no compound` | `GB204=1` | 1 | 0.988 |

### Lexicon

| # | WALS | Grambank | rank_B | sim_B |
|---|------|----------|--------|-------|
| 2 | `136A=No M-T pronouns` | `GB204=0` | 1 | 0.993 |
| 3 | `130A=Different` | `GB204=1` | 1 | 0.992 |
| 4 | `115A=Predicate negation also present` | `GB093=0` | 1 | 0.990 |
| 5 | `144L=SV&OV&[V-Neg]` | `GB047=1` | 1 | 0.989 |
| 6 | `114A=A/Fin` | `GB257=0` | 1 | 0.989 |
| 7 | `137A=No N-M pronouns` | `GB140=1` | 1 | 0.988 |

### Morphology

| # | WALS | Grambank | rank_B | sim_B |
|---|------|----------|--------|-------|
| 8 | `21B=monoexponential TAM` | `GB020=0` | 1 | 0.994 |
| 9 | `25B=Non-zero marking` | `GB250=0` | 1 | 0.993 |
| 10 | `22A=6-7 categories per word` | `GB273=0` | 1 | 0.993 |
| 11 | `18A=All present` | `GB140=1` | 1 | 0.989 |

### Nominal Categories

| # | WALS | Grambank | rank_B | sim_B |
|---|------|----------|--------|-------|
| 12 | `41A=Two-way contrast` | `GB059=0` | 1 | 0.992 |
| 13 | `43A=Related to remote demonstratives` | `GB204=1` | 1 | 0.988 |

### Nominal Syntax

| # | WALS | Grambank | rank_B | sim_B |
|---|------|----------|--------|-------|
| 14 | `58A=Absent` | `GB140=1` | 1 | 0.992 |
| 15 | `65A=No grammatical marking` | `GB187=0` | 1 | 0.989 |
| 16 | `56A=Formally similar, without interrogative` | `GB266=0` | 1 | 0.989 |

### Tone

| # | WALS | Grambank | rank_B | sim_B |
|---|------|----------|--------|-------|
| 17 | `10A=Contrast absent` | `GB250=0` | 1 | 0.991 |

### Verbal Categories

| # | WALS | Grambank | rank_B | sim_B |
|---|------|----------|--------|-------|
| 18 | `79B=None (= no suppletive imperatives reported in the reference material)` | `GB304=0` | 1 | 0.993 |
| 19 | `73A=Inflectional optative absent` | `GB250=1` | 1 | 0.993 |

### Word Order

| # | WALS | Grambank | rank_B | sim_B |
|---|------|----------|--------|-------|
| 20 | `90C=Noun-Relative clause (NRel) dominant` | `GB123=0` | 1 | 0.990 |
