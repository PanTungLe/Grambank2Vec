# Replicating the URIEL+ categorical recast

Independent re-run of the experiment testing whether URIEL+'s independent-binary-flag
encoding of mutually exclusive typological values can be recast into categorical groups,
and whether that recast improves whole-feature accuracy and eliminates structurally
impossible profiles.

Nothing here was tuned toward the prior numbers. Where I diverge I say so and give a
diagnosis rather than adjusting.

**Claim tags.** `[V]` computed in this run · `[R]` read from repo files ·
`[P]` from a paper · `[prior]` supplied by the requester.

**Provenance.** Data: `/tmp/URIELPlus/urielplus/database/original_uriel/features.npz`,
cloned fresh with `git clone --depth=1 https://github.com/Masonshipton25/URIELPlus`
`[R]`. Keys `feats` (289, `<U72`), `langs` (7970, `<U3`), `sources` (10, `<U14`),
`data` (7970 × 289 × 10, `float32`, values exactly `{-1, 0, 1}`) `[V]`.
NumPy 2.4.6, pandas 3.0.5, SciPy 1.17.1, scikit-learn 1.9.0 `[V]`. `PYTHONHASHSEED=0`;
all randomness via `numpy.random.default_rng(seed)`, no global `np.random` calls `[V]`.

---

## 1. Gate-by-gate

| Gate | Criterion | Result | Verdict |
|---|---|---|---|
| 1a | union shape `(3357, 289)` | `(3357, 289)` `[V]` | **PASS** |
| 1a | observed density `0.3922` | `0.392198` → `0.3922` `[V]` | **PASS** |
| 1b | `basic_order` WALS pct_one 94.4 | 94.4 `[V]` | **PASS** |
| 1b | `basic_order` SSWL pct_multi 44.7 / max 6 | 44.7 / 6 `[V]` | **PASS** |
| 1b | `basic_order` union pct_multi 10.1 / max 6 | 10.1 / 6 `[V]` | **PASS** |
| 1b | `vowel_quality_inventory` union pct_multi 9.6 / max 5 | 9.6 / 5 `[V]` | **PASS** |
| 3 | group-protocol metrics bit-identical, naive vs constrained | identical to machine precision `[V]` | **PASS** |
| 4 | SoftImpute shrink → 0.5 | 0.5 `[V]` | **PASS** |
| 4 | KNN k → 50 | **25** `[V]` | **DIVERGED** |
| 4 | decode tau → 0.35 | **0.40** on `cell_F1`; **0.35** on `group_acc` `[V]` | **DIVERGED** (rule-dependent) |
| 4 | MF rank → 64 | 64 `[V]` | **PASS** |
| 4 | MF L2 → 0.02 | **0.08** `[V]` | **DIVERGED** |
| 6 | group protocol, SoftImpute + MF means | all within tolerance `[V]` | **PASS** |
| 6 | group protocol, KNN means | +0.003 … +0.007, outside ±0.002 `[V]` | **MARGINAL** |
| 6 | cell protocol, SoftImpute means | within 0.005 `[V]` | **PASS** |
| 6 | cell protocol, `knn_cat` means | +0.028 … +0.045 `[V]` | **DIVERGED** |
| 6 | all group-accuracy deltas ≥ 4/5 wins | 5/5 everywhere `[V]` | **PASS** |
| 6 | MF `cell_F1` must not favour categorical | favours binary, −0.0077 `[V]` | **PASS** |
| 6 | ordinal probe sign + win count | +0.18 … +0.23, 5/5 `[V]` | **PASS** |
| — | `mf_cat` under the cell protocol | degenerate as specified `[V]` | **NEW FAILURE** |

Phase 0 was not attempted: Phases 1–6 did not finish early, and Phase 0 is explicitly an
optional anchor that blocks nothing. The Bafna et al. non-replication (arXiv 2510.27183)
is carried as `[prior]`, not independently verified here. Phase 7 was not run — its
precondition is a clean Phase 6, and it needs a task dataset that is not staged.

**Configs.** Because three Phase-4 selections diverged, everything downstream was run
twice: **config A** = the prior hyperparameters (shrink 0.5, k 50, tau 0.35, rank 64,
L2 0.02) and **config B** = mine (shrink 0.5, k 25, tau 0.40, rank 64, L2 0.08). Config A
is the like-for-like comparison; config B is the sensitivity check. `_strict` marks the
corrected categorical gold rule described in §5.

---

## 2. Phase 1 — the coherence audit stands

39 groups over 98 of the 289 features, 38 `exclusive` + 1 `ordinal`; all 98 names resolve
against `feats`, no column claimed twice, 191 features left ungrouped `[V]`.

The audit reproduces exactly `[V]`. The headline is unchanged: SSWL alone puts **44.7%**
of its observed `basic_order` languages in a structurally impossible state, with up to
**all six** word orders flagged simultaneously; union aggregation reduces but does not
remove this (**10.1%**, max 6). Full table in `results/audit_coherence.csv`.

One measurement added, because the gold-label convention depends on it: **98.7% of
observed group instances are fully co-observed** (median 1.0000, min 0.8459 for
`subordinator`) `[V]`. Group members are essentially all-or-nothing in the source, so the
"observed at all" convention for counting active flags is safe.

---

## 3. Phase 3 — the scoring trap

Confirmed and passed `[V]`. Under the group protocol, constrained and naive decoding give
metrics identical to machine precision. The two completion matrices do differ — 1869
member cells — but **0 of those differences fall inside a masked row** `[V]`: they are all
languages the naive decoder overwrote despite their values still being observed. That is
exactly the failure mode the trap describes, and it is invisible to the group protocol and
material under the cell protocol.

---

## 4. Phase 6 — my numbers beside yours

### Group protocol, means over seeds 42–46 (config A)

| arm | cell F1 `[V]` / `[prior]` | group acc `[V]` / `[prior]` | exact `[V]` / `[prior]` | validity `[V]` / `[prior]` |
|---|---|---|---|---|
| softimpute_bin | **.8355** / .8373 | **.8156** / .8180 | **.7866** / .7887 | **.9609** / .9587 |
| softimpute_cat | **.8348** / .8322 | **.8575** / .8564 | **.8089** / .8069 | **1.0000** / 1.0000 |
| knn_bin | **.7811** / .7780 | **.7710** / .7659 | **.7350** / .7307 | **.9705** / .9699 |
| knn_cat | **.7809** / .7764 | **.7976** / .7905 | **.7522** / .7451 | **1.0000** / 1.0000 |
| mf_bin | **.8127** / .8107 | **.7764** / .7741 | **.7532** / .7504 | **.9175** / .9159 |
| mf_cat | **.8049** / .8034 | **.8352** / .8321 | **.7883** / .7842 | **1.0000** / 1.0000 |

SoftImpute agrees to ≤ .0026, MF to ≤ .0041 (tolerance ±0.010), KNN to ≤ .0071.

### Cell protocol, means over seeds 42–46 (config A)

| arm | cell F1 `[V]` / `[prior]` | group acc `[V]` / `[prior]` | exact `[V]` / `[prior]` | validity `[V]` / `[prior]` |
|---|---|---|---|---|
| softimpute_bin | **.9162** / .9130 | **.9619** / .9606 | **.9403** / .9380 | **.9629** / .9624 |
| softimpute_cat | **.8980** / .8935 | **.9730** / .9714 | **.9278** / .9250 | **.9918** / .9917 |
| knn_bin | **.8246** / .8189 | **.8984** / .8959 | **.8778** / .8752 | **.9490** / .9484 |
| knn_cat | **.8497** / .8154 | **.9403** / .8951 | **.8967** / .8691 | **.9918** / .9536 |

`knn_cat` is the one substantive divergence in the whole replication. See §6.

### Paired contrasts, group protocol (config A), Cat − Bin

| family | metric | delta `[V]` | wins `[V]` | t `[V]` | p `[V]` | `[prior]` |
|---|---|---|---|---|---|---|
| softimpute | group_acc | **+.0419** | 5/5 | 30.2 | 1e-05 | +.0384, 5/5, t=23.5 |
| softimpute | exact | **+.0224** | 5/5 | 18.8 | 5e-05 | +.0182, 5/5 |
| softimpute | cell_F1 | **−.0007** | 1/5 | −0.76 | .487 | −.0051, 0/5, p=.023 |
| knn | group_acc | **+.0265** | 5/5 | 18.9 | 5e-05 | +.0246, 5/5, p=.0001 |
| knn | cell_F1 | **−.0001** | 1/5 | −0.14 | .899 | −.0016, 1/5, p=.34 (ns) |
| mf | group_acc | **+.0588** | 5/5 | 38.3 | 4e-06 | +.0579, 5/5, t=60.1 |
| mf | exact | **+.0351** | 5/5 | 23.1 | 2e-05 | +.0338 |
| mf | validity | **+.0825** | 5/5 | 81.2 | <1e-06 | +.0841 |
| mf | cell_F1 | **−.0077** | 1/5 | −3.31 | .030 | −.0073, 0/5, p=.0002 |

Every sign and every win count matches, and every group-accuracy delta is 5/5. The two
`cell_F1` win counts land 1/5 where you had 0/5 — the deltas are small enough
(−.0007, −.0077) that one seed crossing zero is expected. The one magnitude gap worth
noting is SoftImpute `cell_F1`: I get −.0007 (ns) where you got −.0051 (p=.023). The sign
agrees; the cost of the categorical decode on the metric it is expected to lose is smaller
in my run and does not reach significance.

### Ordinal probe (MF arms, 28 pairs, Spearman rho)

| run | mf_cat `[V]` | mf_bin `[V]` | delta `[V]` | wins | t | p |
|---|---|---|---|---|---|---|
| A, group | +.1698 ± .0984 | −.0103 ± .0568 | **+.1801** | 5/5 | 5.28 | .0062 |
| A, cell | +.2359 ± .0765 | +.0126 ± .0392 | **+.2233** | 5/5 | 9.39 | .0007 |
| A_strict, cell | +.2418 ± .0815 | +.0126 ± .0392 | **+.2291** | 5/5 | 8.10 | .0013 |
| `[prior]` | +.2394 ± .0994 | +.0003 ± .0779 | +.2391 | 5/5 | 6.02 | .0038 |

Replicates: positive, 5/5 seeds, significant, binary arm at chance. The categorical
softmax recovers the vowel-inventory ordinal scale that the flat encoding discards; the
binary arm does not. Your protocol did not say which masking protocol the probe ran under
— my cell-protocol figures (+.2233/+.2291) sit closer to yours (+.2391) than my
group-protocol figure (+.1801), which suggests yours was the cell protocol.

---

## 5. The negative result under the cell protocol

**It replicates for SoftImpute, and it is real.** Reporting it plainly, unsoftened:

| metric | delta `[V]` | wins `[V]` | p `[V]` | `[prior]` |
|---|---|---|---|---|
| softimpute cell_F1 | **−.0182** | 0/5 | .00029 | −.0195, 0/5, p=.001 |
| softimpute exact | **−.0125** | 0/5 | .00018 | −.0130, 0/5, p=.001 |
| softimpute group_acc | **+.0111** | 5/5 | .00064 | +.0108, 5/5, p=.001 |

Under URIEL+'s own cell-level protocol the categorical recast **buys group accuracy and
pays for it in cell F1 and exact-profile match**, losing on both in 5 seeds out of 5. This
is not noise and it is not a wash: the recast is a net loss on two of four metrics under
the protocol URIEL+ actually uses. All three numbers land within .0013 of yours. The
partial failure is confirmed.

The mechanism is the leak the protocol description already identifies. When five of six
`basic_order` members remain observed, the binary arm gets those five for free and only
has to guess one cell; the categorical arm must commit to a single class for the whole
group, and when it commits wrongly it loses the whole profile rather than one cell. The
group protocol removes the leak and the sign flips.

**What does not replicate is the KNN half of it.** You report `knn` group_acc −.0008
(2/5, p=.36, no effect) and exact −.0061 (0/5). I get **+.0418 (5/5, p=1e-05)** and
**+.0189 (5/5)** — categorical wins clearly. See §6.

### A second, new failure: `mf_cat` is degenerate under the cell protocol

Implemented exactly as specified, `mf_cat` under the cell protocol collapses `[V]`:

| arm | cell F1 | group acc | exact | validity |
|---|---|---|---|---|
| mf_bin | .8901 | .9329 | .9182 | .9411 |
| mf_cat (as specified) | **.0344** | **.6539** | **.6239** | .9918 |
| mf_cat (`strict_gold`) | .8882 | .9664 | .9217 | .9918 |

This is not a tuning artifact, it is structural. The specified gold rule — "`len(cols)`
(NONE) if none is active" — cannot distinguish *"no member is active"* from *"the active
member is the cell that was masked out."* Under the cell protocol that mislabels
**6.84% of training instances** as NONE, versus **0.00%** under the group protocol, where
masked instances are dropped entirely because no member remains observed `[V]`. The
mislabelled instances are precisely the held-out evaluation positions, so the model is
trained to predict NONE exactly where it is about to be scored. The binary arm is immune:
a masked cell contributes nothing to the BCE rather than contributing a wrong label.

The fix is one line and it is principled rather than tuned: drop instances whose observed
members are all zero *while some member cell is missing*, on the same grounds the spec
already drops instances with two or more active flags — no single class represents the
state. That is `strict_gold` in `models.py`, and it restores the arm to .8882 F1 /
.9664 group acc, at which point the MF contrast behaves like the others: group_acc
**+.0334** (5/5, p<1e-05), exact **+.0035** (5/5, p=.068), cell_F1 −.0019 (2/5, p=.42),
validity **+.0507** (5/5) `[V]`. Note the cell-protocol pattern is milder here than for
SoftImpute — the categorical MF loses only .0019 of cell F1 rather than .0182.

I suspect this is why your Phase 6 cell-protocol table has no MF rows.

---

## 6. Divergences

### DIVERGED — `knn_cat`, cell protocol

Yours: F1 .8154 / acc .8951 / exact .8691 / validity .9536. Mine: **.8497 / .9403 /
.8967 / .9918** `[V]`. Categorical wins on every metric in my run; in yours it loses or
ties on three.

Diagnosis. The decoder is not the problem: `softimpute_cat` on the same protocol through
the same code path matches you to ≤ .0045 on all four metrics. What isolates it is
**validity**. Under constrained decoding on the cell protocol, validity is
*model-independent* — it is fixed by how often the gold's still-observed cells already
carry two active flags, not by anything the completion says. My three cat arms confirm
this: `softimpute_cat`, `knn_cat` and `mf_cat` return **identical validity, seed by seed**
(.9920, .9914, .9915, .9914, .9926) `[V]`. Your run has `softimpute_cat` at .9917 —
matching mine — but `knn_cat` at .9536. Two arms sharing one decoder cannot disagree on a
model-independent quantity. So your `knn_cat` cell-protocol arm did not go through the
same constrained decode.

I could not reproduce your exact values with any near-miss variant `[V]`: naive decoding
gives 1.0000 validity, dropping the "observed member already 1" short-circuit gives .9129,
and plain binary gives .9490 — none matches .9536. So I can localise the divergence to
that arm without pinning the precise line, and I have not adjusted anything to fit.

Consequence: the cell-protocol negative result holds for SoftImpute but not for KNN. The
`[prior]` claim that KNN shows "no effect" under the cell protocol does not survive.

### DIVERGED — MF L2 (0.08 vs 0.02), and an underdetermined spec

"L2 `l2` on `U` and `W` only" does not say whether the data loss is **summed** or
**averaged** over supervised units, and that decides everything. Tested both `[V]`:

| `loss_norm` | l2=0.005 | l2=0.02 | l2=0.08 |
|---|---|---|---|
| `mean` | .6105 | .6203 | .6203 |
| `sum` | .8109 | .8169 | .8215 |

Averaging crushes `U` to zero — the model degenerates to a per-column bias predictor, and
l2=0.02 and 0.08 give *bit-identical* output because both have fully collapsed. Summing
reproduces your ~.81 regime, so `sum` is the intended reading and is what I used
throughout. But within it the pinned grid {0.005, 0.02, 0.08} **rises monotonically and
never turns over**, so `cell_F1` selection takes the endpoint, 0.08. I ran the grid as
specified rather than extending it.

This one has a resolution you will want. L2 0.08 is better on the selection metric but
**materially worse on the ordinal probe**: rho delta +.0568 (4/5, p=.176, not significant)
at L2 0.08 versus **+.1801 (5/5, p=.0062)** at L2 0.02 `[V]`. The heavier penalty shrinks
`W` and flattens exactly the geometry the probe measures. Your 0.02 is the better choice;
the `cell_F1` selection rule simply cannot see the property that makes it better. Worth
stating explicitly in any write-up, since it is a case where the pre-registered selection
rule picks the wrong model.

Your single-validation-mask caveat can be retired: selection does **not** move between
seeds 901 and 902 — both pick rank 64, L2 0.08 `[V]`.

### DIVERGED — KNN k (25 vs 50)

k=25 beats k=50 on mean `cell_F1` (.7899 vs .7874) and on *both* validation seeds
individually `[V]`. Small but consistent. Related: my KNN runs sit systematically ~.003 to
~.006 above yours on every arm and protocol, which points to an implementation difference
in `knn_fill` rather than mask noise — candidates are top-k tie-breaking, the column-mean
fallback for cells no neighbour observes, or predicting all cells rather than only missing
ones. It changes no conclusion: every KNN group-protocol contrast replicates in sign and
win count under both k.

### DIVERGED — decode tau (0.40 vs 0.35), a selection-rule ambiguity

Your stated rule is "select on `cell_F1`", which gives **0.40** (.83780 vs .83714 at 0.35)
`[V]`. But the evidence you quote for 0.35 is a *group-accuracy* peak, and my group
accuracy peaks at **0.35** exactly as yours does (.85891 vs your .8607) `[V]`. The two
rules disagree by .0007 in cell F1 — a tie. Recommend fixing the rule for tau explicitly
in the write-up, since tau is the categorical decoder's own parameter and selecting it on
the metric the categorical arm loses is arguably the wrong instrument.

### On the ±0.002 tolerance

The deterministic arms are deterministic *given the mask*, but two independent
implementations do not draw the same mask from the same seed — RNG consumption order is
implementation-specific. Between-seed SD in my runs is .0034–.0043, so the standard error
of a 5-seed mean is ~.0015–.0019 `[V]`. A ±0.002 band is therefore about ±1 SE and is
tighter than independent reimplementation can deliver; ±0.004 (≈2 SE) is the honest
threshold. Under that band SoftImpute passes everywhere and only KNN is genuinely out.

---

## 7. Conclusion

The central claim replicates. Under the realistic group protocol, recasting URIEL+'s
independent binary flags into categorical groups improves whole-feature accuracy for
**all three** completion arms — SoftImpute **+.0419**, KNN **+.0265**, MF **+.0588**,
each 5/5 seeds, each p ≤ 5e-05 — improves exact-profile match, and drives structurally
impossible profiles to **exactly zero** (validity 1.0000 in every cat arm, against
.9175–.9705 binary) `[V]`. The cost is a negligible and mostly non-significant loss in
cell F1, the metric the binary baseline was tuned on. The MF embedding additionally
recovers the vowel-inventory ordinal scale that the flat encoding discards (rho
+.18 to +.23, 5/5 seeds) `[V]`. All of this survives the hyperparameter divergences: every
contrast holds in sign and win count under both configs.

Two things should not be smoothed over. Under URIEL+'s own cell-level protocol the recast
is a **net loss** on cell F1 and exact profile for SoftImpute, 0/5 seeds, confirmed within
.0013 of the prior numbers. And the categorical MF arm is **structurally incompatible**
with that protocol as specified, for a reason that has nothing to do with typology: the
gold rule cannot tell "no value" apart from "the value we hid."

### Caveats carried forward (all from the requester, all standing)

1. The 39-group mapping is one reading of URIEL feature names against WALS chapter titles.
   `alignment`, `object_marking_locus` and `tendency_affix` are the shakiest: WALS 98A/99A
   carry neutral and tripartite values URIEL cannot express, and locus-of-marking is
   arguably multi-label in the source. `basic_order`, `adposition`, `adjective`,
   `possessor` and the vowel group are safe.
2. **The MF arms sit below SoftImpute in absolute terms (.8127 vs .8355 cell F1** `[V]`,
   matching the prior's .8107 vs .8373). The MF has no row or column bias terms beyond a
   single output bias and one global learning rate. It is a controlled contrast, not a
   competitive imputer, and **must never be reported as beating SoftImpute.**
3. The MF-hyperparameter-from-one-mask caveat is now retired: seeds 901 and 902 select
   identically `[V]`.

### Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy scikit-learn pyarrow
git clone --depth=1 https://github.com/Masonshipton25/URIELPlus /tmp/URIELPlus
cd uriel_recast
PYTHONHASHSEED=0 python audit.py                       # Phase 1, gates 1a/1b
PYTHONHASHSEED=0 python run.py sanity --seeds 42       # Phase 3, the trap
PYTHONHASHSEED=0 python run.py tune  --seeds 901 902 --what all   # Phase 4
PYTHONHASHSEED=0 python run.py test  --tag A --seeds 42 43 44 45 46 \
    --shrink 0.5 --k 50 --tau 0.35 --rank 64 --l2 0.02            # Phase 4 test
PYTHONHASHSEED=0 python aggregate.py                   # Phase 5
```

Chunk the MF runs two or three seeds at a time (~80 s per seed for the pair). Outputs land
in `results/`: `audit_coherence.csv`, `tuning.csv`, `results_all.csv`,
`summary_means.csv`, `paired_contrasts.csv`, `ordinal_probe.csv`,
`results_all_Wvowel.npz`.
