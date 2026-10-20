#!/usr/bin/env bash
# Full pipeline. Expects ASSAYOT_DATA to point at the dataset folder described in
# datasets/MANIFEST.md (Part B), and a CUDA GPU for the candidate scan and training.
set -euo pipefail
PY=${PY:-.venv/bin/python}

$PY -m src.run prep        # harmonise sources        -> crispr_processed/ + PROVENANCE.md
$PY -m src.run scan        # candidate space (GPU)    -> candidates_v1.parquet  (~2.5 min)
$PY -m src.selfcheck       # assert the invariants the claims rest on
$PY -m src.run guide       # unseen sgRNA
$PY -m src.run validated   # confirmatory assay, leave-one-sgRNA-out
$PY -m src.run prior       # PU class-prior sensitivity sweep
$PY -m src.run identify    # identifiability / sensitivity
$PY -m src.run semisynth   # masking under a known mechanism
$PY -m src.run study       # unseen study
$PY -m src.run tech        # unseen technology

$PY -m src.figures         # -> paper/figures/*.pdf
$PY -m src.tables          # -> paper/tables/*.tex

( cd paper      && latexmk -pdf -interaction=nonstopmode main.tex )
( cd paper-csbj && latexmk -pdf -interaction=nonstopmode main.tex )
