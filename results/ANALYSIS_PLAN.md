# Analysis plan

Written before the models were fitted, from `roadmap.md`. Amendments are appended at
the bottom with the reason for each, so that anything decided after seeing results is
visible as such.

## Scope, frozen in `src/config.py`

Conventional SpCas9; *Homo sapiens*; hg38 for every coordinate-dependent step;
substitutions only in the primary analysis, with bulge-containing rows retained,
flagged and reported separately; processed and supplementary tables rather than raw
reads; prediction, not guide design.

## Primary outcome

Ranking of candidate sites by the probability of being called active by a **defined
cellular confirmatory assay** (targeted tag sequencing, 0.1% stated sensitivity).
Deliberately narrower than "probability of cleavage", and testable.

## Candidate space

All hg38 sites within 6 substitutions of the 23-nt pattern `<spacer>NGG`, counted over
the 22 informative positions (position 21 is a wildcard), both strands, no hard PAM
filter. This is the space CHANGE-seq itself searched, so the enumerated set is a strict
superset of every released detection; the coverage check in `src/scan.py` asserts this.
Performance claims apply to this space only, never to "the genome".

## Hypotheses

1. Relative to assuming "unreported = negative", using the unlabelled set improves
   recovery of independently confirmed sites.
2. Modelling technology and study differences transfers to a new study better than
   pooling the data.
3. Where activity and detection cannot be separated, a range or a ranking is more
   defensible than a claimed cleavage probability.

Hypothesis 3 can be supported by a negative result, and that is an acceptable outcome.

## Models

| level | formulation |
|---|---|
| L1 | supervised; every unreported candidate is a biological negative |
| L2 | nnPU on positives vs unlabelled, with a class-prior sensitivity sweep |
| L3 | assay-aware latent model: per-context activity heads, capacity-constrained propensity, false-positive term, hierarchical study shrinkage, inverse-selection-weighted confirmatory term |

Ablations: no observation branch; no confirmatory anchor; unconstrained propensity;
no false-positive term.

## Evaluation protocols

1. **Unseen sgRNA.** Single-linkage clusters of sgRNAs within 4 substitutions, then
   grouped folds, so near-identical spacers cannot straddle a split. Splits are made on
   the sequence key, not the locus, because a sequence model cannot distinguish two
   loci sharing a 23-mer.
2. **Unseen study.** A whole study held out of the external labelled set.
3. **Unseen technology.** A whole technology held out, reported only where the outcome
   and context remain comparable.
4. **Confirmatory subset.** Leave-one-sgRNA-out over the sgRNAs with targeted
   validation. Conclusions restricted to that subpopulation: sites were chosen for
   confirmation by a documented rule, not sampled at random.
5. **Semi-synthetic masking.** Positives hidden (i) completely at random and
   (ii) preferentially where the in-vitro signal is weak. A controlled check, not a
   substitute for real validation.

## Metrics

Average precision and AUROC; recall@K and precision@K computed **per sgRNA** and
macro-averaged; log loss and Brier score against the confirmatory label only;
calibration curves; risk–coverage. Confidence intervals from a bootstrap over sgRNAs
(and over studies where enough exist), never over rows.

Two prohibitions, enforced in `src/metrics.py` rather than merely stated: no precision
or false-positive rate is computed against unlabelled rows as though they were
confirmed negatives, and functions needing true negatives require an explicit
`labelled` mask.

## Identifiability

`P(report) = p·e + (1−p)·f` is what the data show. Each configuration is fitted from
several initialisations; we record the fit to the observed reports and the spread of
the recovered activity probabilities across initialisations. A configuration that fits
equally well but lands on different activity estimates is declared not identified, and
its point estimates are not reported as cleavage probabilities.

## Stopping rules

Taken from the roadmap and fixed in advance:

| pilot finding | decision |
|---|---|
| `FALSE` labels not separable from true negatives | drop from definitive evaluation; change the target definition |
| technology overlap too small | abandon the full-separation claim; focus on robust transfer |
| activity and detection give very different answers at similar fit | report non-identification; move to bounds or ranking |
| improvement disappears under leakage control | current hypothesis rejected; redesign |
| only within-study tests improve | cross-study generalisation claim unsupported |
| insufficient independent validation | restrict claims to the probability of being *reported*, not of cutting |

---

## Amendments

**A1.** The candidate protocol was changed from "≤6 mismatches with an NGG/NAG PAM" to
"≤6 mismatches over the 22 informative positions of `<spacer>NGG`, no PAM filter" after
inspecting the released CHANGE-seq site list, which reaches distance 6 and contains NGA
PAMs. The original rule would not have been a superset of the published detections.
Decided before any model was fitted.

**A2.** The activity call for the confirmatory assay was defined as a one-sided
Fisher's exact test on replicate-pooled counts against the matched control, BH-corrected,
**plus** a 0.1% effect floor matching the assay's published sensitivity. Because the
headline quantity turned out to depend strongly on where that floor is placed, the floor
is swept over 0–1% and reported as a curve rather than a single number
(`results/floor_sensitivity.csv`). The sweep was added after seeing that dependence; the
single-floor call was specified beforehand.

**A3.** hg19 rows from CRISPRoffT were retained rather than dropped, because the model
consumes sequence pairs and not coordinates. They carry their build and are excluded
from every coordinate-dependent step. This raised the number of usable external studies
from 26 to 45.

**A4.** Base- and prime-editor assays are excluded by assay name in addition to the
nuclease filter. Filtering on `Cas9_type == SpCas9` alone was not sufficient: 201
`ABE-Digenome-seq` rows in the release carry `SpCas9`, presumably for the nickase
component of the base editor, and 156 of them reach the external labelled set. An assay
whose read-out is an editor rather than a double-strand break does not belong in a study
scoped to conventional SpCas9 cleavage. Found while validating the cross-technology
split, before those protocols were run.

**A5.** The unseen-technology protocol is reported as largely a restatement of the
unseen-study protocol, not as independent evidence. In the external labelled set every
usable study contributes exactly one technology, and only Digenome-seq (5 studies) and
CIRCLE-seq (3 studies) appear in more than one study, so for the remaining units
"hold out a technology" and "hold out a study" are the same experiment. The plan listed
insufficient technology overlap as a reason to drop the full-separation claim; that
condition is met and the claim is dropped. Established by inspecting the label
structure before the transfer protocols were run.

**A6.** External rows flagged `in_pilot` are excluded from multi-study training and
from the external labelled test set. CRISPRoffT re-publishes the pilot study in full
under its own PMID, so 199,971 of the 226,118 external reported-positive rows (88.4%)
are duplicates of the pilot. Left in, the pilot would have entered training twice and an
"external" test could have scored rows the model had already trained on under a
different study id. After the filter, 26,147 external reported-positive rows remain
across 44 studies, and the external labelled set is 1,208 rows across 32 studies.
Found while verifying a figure caption, before the transfer protocols were run.

**A7.** sgRNA identity is resolved by target sequence rather than by sample name. Four
names in the GUIDE-seq sheet (`CTLA4site_2`, `LAG_site_3`, `CBLB_site_10`,
`CBLC_site_4`) appear nowhere in the CHANGE-seq sheet; each resolves to exactly one
known sgRNA by its target sequence. Matching on the name alone silently discarded 17
cellular detections, and made the number of sgRNAs with cellular profiling look like 60
when it is 57. Found while reconciling a row count between the source table and the
harmonised observations.

---

## Post-hoc additions

A1–A7 were decided before or during fitting and are amendments in the ordinary sense.
Everything below was added **after** the primary results were in hand, during an
adversarial review of the manuscript. They are separated here rather than folded into
the list above, because a reader entitled to rely on a pre-registered plan is entitled
to know which analyses the plan did not anticipate. None of them was chosen after
seeing whether it would favour our model; several did not.

**P1. The anchored control (`C1_supervised_anchor`).** The pre-registered levels were
L1 (supervised), L2 (nnPU) and L3 (assay-aware), with ablations removing components of
L3. That design cannot separate "the assay-aware formulation helps" from "training on
the target assay's labels helps", because only L3 receives the confirmatory labels. The
control — L1 plus the identical confirmatory term, no observation model — was added to
break that confound. It became the best model on the confirmatory protocol and is the
study's principal negative result. **Prompted by the confound, not by the outcome; the
outcome was against the pre-registered hypothesis.**

**P2. The unweighted-anchor ablation (`C2_anchor_unweighted`).** Once P1 showed the
anchor rather than the observation model carried the result, the inverse-selection
weighting inside the anchor became the open question. C2 holds everything fixed and sets
every confirmatory weight to 1. The weighting proved inert (+0.016 in favour of no
weighting, p=0.65).

**P3. External validation on TrueOT.** Not in the plan, which specified the pilot's own
confirmatory subset as the primary outcome. Six sgRNAs from one laboratory is too narrow
a base for the paper's central claim, and an external benchmark of the same estimand
exists. 1,073 rows, 192 positive, 25 evaluable sgRNAs, 9 studies, after excluding the
pilot study (which TrueOT contains) and seven sgRNAs whose spacers are in the pilot.
**It did not confirm the anchor advantage or the method-ordering reversal.** Added
knowing it might not, and reported as it came out.

**P4. Reimplemented published architectures (`R1`, `R2`).** The plan named CFD, a
mismatch count and a position-weighted score as baselines. Those are a 2016 lookup table
and two trivial scores, which is not a fair test of whether this work improves on what
the field uses. Two published architectures were reimplemented and fitted on the same
folds under the same labelling convention. Both proved strong (highest AUROC of any
model on the unseen-sgRNA protocol), which is what makes the comparison worth reading.

**P5. Multiplicity correction across model comparisons.** The plan fixed
Benjamini–Hochberg for the detection-floor analysis and said nothing about comparisons
between models, of which 23 are reported. BH is applied across that family and the
adjusted value is given wherever it changes the reading. Three nominally significant
contrasts do not survive, two of them claims the paper had made.

**P6. Depth-confound check on the detection-floor curve.** The curve's upper end counts
significant elevations, and significance buys more with more reads. Read depth across
these sites spans 403-fold, and the upper end does track it (17.1% to 31.4% across depth
quartiles, p=0.006) while the 0.1% end does not. The plan did not anticipate this, and
the headline model-free result is qualified accordingly.

**P7. One configuration deviation.** The external-benchmark models were trained at batch
1024 with activation checkpointing instead of the batch 8192 used everywhere else,
because a shared GPU left 1.9 GiB against the ~4 GiB that configuration needs. Every
model in that table was trained identically, so its internal comparisons hold; its
absolute values are not compared with the other protocols'. The 8-seed identifiability
rerun and the seed-variance study were **not** run at a reduced batch for the same
reason in reverse — their purpose is comparability with the reported configuration — and
are therefore outstanding rather than substituted.
