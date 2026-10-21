# Provenance log

Built by `src/prep.py`. Every row records which assay saw a site, in which context, and what the report means. No step converts an absent report into a biological negative.

## CHANGE-seq supplementary tables

Source: `41587_2020_555_MOESM3_ESM.xlsx` (Lazzarotto et al. 2020, Nat Biotechnol 38:1317, Supplementary Tables 2-8).
- Table 3: 202043 CHANGE-seq detected sites (110 sgRNAs), cell-free genomic DNA from human CD4+/CD8+ T-cells, hg38.
  - 48 rows have an empty sequence cell (spans [22, 23, 24] nt, i.e. bulge sites); kept and flagged, excluded from the primary analysis.
  - reported mismatch distance ranges 0-6; the distance is counted over the 22 informative positions of the 23-nt pattern, so sites with a non-NGG PAM are present.
- sgRNA table: 110 sgRNAs over 13 loci; specificity ratio available for 109 (Table 4 omits TRBC1_site_2). Spacers agree with Table 4 wherever both exist.
- Table 6: 1702 GUIDE-seq detected sites (60 sgRNAs), in cell.
- Tables 7+8: 791 sites assayed by targeted tag sequencing over 6 sgRNAs; selection classes {'both': 232, 'A': 66, 'B': 175, 'C': 121, 'D': 197}.
- Table 5: on-target indel rates for 110 sites.

### Validation calls (FDR 0.05, effect floor 0.1%)

- active by tag integration: 124; by indel: 98; either: 130 of 791.
- sufficiently powered to see the floor: 791.

| selection class | sites | active | active rate |
|---|---|---|---|
| A | 66 | 2 | 3.0% |
| B | 175 | 2 | 1.1% |
| C | 121 | 1 | 0.8% |
| D | 197 | 0 | 0.0% |
| both | 232 | 125 | 53.9% |

### Sensitivity of the central quantity to the activity floor

`GUIDE-seq-negative` = CHANGE-seq-detected sites in selection classes A-D, which GUIDE-seq did not report.

| activity floor | GUIDE-seq-negative active | rate | detected-by-both active | rate |
|---|---|---|---|---|
| 0.00% | 126/559 | 22.5% | 228/232 | 98.3% |
| 0.01% | 57/559 | 10.2% | 214/232 | 92.2% |
| 0.02% | 27/559 | 4.8% | 197/232 | 84.9% |
| 0.05% | 7/559 | 1.3% | 159/232 | 68.5% |
| 0.10% | 5/559 | 0.9% | 125/232 | 53.9% |
| 0.20% | 1/559 | 0.2% | 87/232 | 37.5% |
| 0.50% | 1/559 | 0.2% | 56/232 | 24.1% |
| 1.00% | 0/559 | 0.0% | 50/232 | 21.6% |

- 4 sgRNA names in the GUIDE-seq sheet are absent from the detection sheet and were resolved by target sequence, each to exactly one known sgRNA: `CBLB_site_10` -> `CBLB_site_9`, `CBLC_site_4` -> `CBLB_site_4`, `CTLA4site_2` -> `CTLA4_site_2`, `LAG_site_3` -> `LAG3_site_3`. Matching on the name alone would silently discard these in-cell detections.

- GUIDE-seq was run for 57 of 110 sgRNAs; for the remaining sgRNAs a site is unobserved in cell, not negative.

- pairs: 202162; observations: 204536; validated pairs: 648.
- pairs with a bulge (excluded from the primary analysis): 10515 = 7468 with a gap character, 3045 with a window wider than 23 nt, 48 with an empty sequence.
- among sgRNAs with matched GUIDE-seq: 1342 pairs detected by both, 60867 by CHANGE-seq only, 119 by GUIDE-seq only.

## CRISPRoffT

Source: `allframe_update_addEpige.txt` (Zhang et al. 2025, Nucleic Acids Res 53:D914).
- 245848 rows; dropping 2 rows whose fields are shifted (non-numeric Year).
- scope filter (SpCas9, Homo sapiens, hg38 or hg19): 234838 of 245846 rows retained. hg19 rows are kept because the model consumes sequence pairs, not coordinates; they carry their build and are excluded from every coordinate-dependent step (candidate enumeration, epigenomic features, positional de-duplication).
- builds retained: {'hg19': 9452, 'hg38': 225386}
- substitution-only filter: dropping 4267 bulge rows.
- status mapping: {'reported_positive': 226228, 'uninterpretable': 3749, 'tested_negative': 594}
- evidence tier: {'discovery': 225321, 'confirmed': 907, 'rejected_or_failed': 4343}
- of 4343 rows marked `Validation == FALSE`, only 594 carry an indel measurement. The CRISPRoffT documentation states that `FALSE` can mean either a rejected off-target or a failed PCR, so the remainder are recorded as uninterpretable and never used as training negatives.
- technologies: 22; studies (PMID): 45; guides: 249.
- guide length before normalisation: {23: 229574, 22: 997}
- CRISPRoffT mapped to 215560 pairs and 230458 observations.

- 198082 pairs appear in both the CHANGE-seq pilot and CRISPRoffT (CRISPRoffT re-publishes the CHANGE-seq release); flagged as `in_pilot` so external tests can exclude them.

- 219640 genomic pairs collapse to 150697 distinct (guide, site) sequence pairs; 74536 pairs share a sequence with another locus (repeat families). Splits are made on `seq_key`, never on the locus.

## Written

- `pairs_v1.parquet`: 219640 pairs
- `observations_v1.parquet`: 434994 observations
- `validation_v1.parquet`: 791 validation rows
