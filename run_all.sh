#!/usr/bin/env bash
# Full pipeline. Expects ASSAYOT_DATA to point at the dataset folder described in
# datasets/MANIFEST.md (Part B), and a CUDA GPU for the candidate scan and training.
set -euo pipefail
PY=${PY:-.venv/bin/python}

$PY -m src.run prep        # harmonise sources        -> crispr_processed/ + PROVENANCE.md
$PY -m src.run scan        # candidate space (GPU)    -> candidates_v1.parquet  (~2.5 min)
$PY -m src.multiplicity    # Benjamini-Hochberg across all model comparisons
$PY -m src.audit_manuscript   # the paper's numbers vs the data
$PY -m src.selfcheck       # assert the invariants the claims rest on
$PY -m src.run guide       # unseen sgRNA
$PY -m src.run validated   # confirmatory assay, leave-one-sgRNA-out
$PY -m src.run prior       # PU class-prior sensitivity sweep
$PY -m src.run identify    # identifiability / sensitivity
$PY -m src.run semisynth   # masking under a known mechanism
$PY -m src.run study       # unseen study
$PY -m src.run tech        # unseen technology
$PY -m src.run_trueot      # external measured-activity benchmark (TrueOT): 25 sgRNAs,
                           # 9 laboratories, no row of it ever trained on
$PY -m src.run_reference all  # published architectures, reimplemented on these folds

$PY -m src.figures         # -> paper/figures/*.pdf
$PY -m src.tables          # -> paper/tables/*.tex

( cd paper      && latexmk -pdf -interaction=nonstopmode main.tex )   # journal-neutral
( cd paper-spj  && latexmk -pdf -interaction=nonstopmode spj.tex )   # AAAS SPJ (current
                                                                     # CSBJ publisher)
( cd paper-csbj && latexmk -pdf -interaction=nonstopmode main.tex )  # Elsevier, former
                                                                     # publisher: kept
                                                                     # only as a fallback
