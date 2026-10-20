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
