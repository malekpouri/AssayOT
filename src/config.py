"""Paths and frozen scope decisions.

Scope is deliberately narrow (roadmap section 3): conventional SpCas9, human, hg38,
substitution-only mismatches in the primary analysis, processed/supplementary tables
rather than raw reads, prediction rather than guide design.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- paths -------------------------------------------------------------------

# Point ASSAYOT_DATA at the dataset folder described in datasets/MANIFEST.md. The
# default is a sibling `datasets/` directory beside this project; it is deliberately
# relative, because an absolute path here would publish one contributor's home
# directory layout.
_DEFAULT_DATA = Path(__file__).resolve().parents[2] / "datasets"
DATA = Path(os.environ.get("ASSAYOT_DATA", _DEFAULT_DATA))
PROJ = Path(__file__).resolve().parent.parent
RESULTS = PROJ / "results"
PAPER = PROJ / "paper"

CHANGESEQ_XLSX = DATA / "crispr_changeseq" / "41587_2020_555_MOESM3_ESM.xlsx"
OFFT_ALL = DATA / "crispr_offt" / "allframe_update_addEpige.txt"
OFFT_TECH = DATA / "crispr_offt" / "Each_tech_statistics_update.csv"
OFFT_GUIDE = DATA / "crispr_offt" / "Each_guide_statistics_update.csv"
HG38_FA = DATA / "reference_hg38" / "hg38.fa"

PROC = DATA / "crispr_processed"

PAIRS = PROC / "pairs_v1.parquet"
OBS = PROC / "observations_v1.parquet"
VALID = PROC / "validation_v1.parquet"
CANDIDATES = PROC / "candidates_v1.parquet"
FEATURES = PROC / "features_v1.npz"
PROVENANCE_LOG = PROC / "PROVENANCE.md"

for _d in (RESULTS, PROC):
    _d.mkdir(parents=True, exist_ok=True)

# --- frozen scope ------------------------------------------------------------

GENOME_BUILD = "hg38"
NUCLEASE = "SpCas9"
SPECIES = "Homo sapiens"

#: protospacer length, excluding PAM
SPACER_LEN = 20
#: 23 nt = 20 nt protospacer + 3 nt PAM; the layout used by every source table here
SITE_LEN = 23

#: Candidate enumeration protocol. Substitutions only. Mismatches are counted over the
#: 22 *informative* positions of the 23-nt pattern `<spacer>NGG`: the 20 spacer
#: positions plus the two G's, with PAM position 21 a wildcard. There is no hard PAM
#: filter, because this is exactly the search space CHANGE-seq itself reported over
#: (its released sites reach distance 6 and include NGA PAMs), so the enumerated
#: candidate set is a strict superset of every detected site.
MAX_MM_PRIMARY = 6
#: Tighter space for the candidate-space sensitivity analysis (roadmap section 5, rule 4).
MAX_MM_TIGHT = 4
#: Position 21 of the 23-nt site (0-based index 20) is the PAM wildcard.
PAM_WILDCARD_IDX = 20
#: PAM classes recorded as a feature rather than used as a filter.
PAM_CLASSES = ("NGG", "NAG", "NGA", "other")

#: Primary chromosomes plus the mitochondrial genome; no alt/random/patch contigs. The
#: released CHANGE-seq and GUIDE-seq site lists contain no non-primary contigs, but they
#: do contain chrM sites, so chrM is required for the candidate space to remain a strict
#: superset of the published detections.
CHROMS = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX", "chrY", "chrM")

#: Rows with a bulge are excluded from the primary analysis and reported separately.
EXCLUDE_BULGES = True

#: Assays that read out a base or prime editor rather than a Cas9 nuclease. The
#: nuclease field alone does not exclude these: 201 `ABE-Digenome-seq` rows in the
#: release carry `Cas9_type == "SpCas9"`, presumably for the nickase component, and
#: would otherwise enter a study restricted to conventional SpCas9 cleavage. Matched
#: case-insensitively against the technology name.
EXCLUDED_ASSAY_PATTERNS = (r"^ABE[-_ ]", r"^CBE[-_ ]", r"\bprime[-_ ]?edit",
                           r"^PE[-_ ]?\d", r"\bbase[-_ ]?edit")

# --- observation vocabulary --------------------------------------------------

#: How a site/assay row relates to the latent event. Kept explicit so that
#: "not reported" is never silently a biological negative (roadmap section 1).
STATUS_POSITIVE = "reported_positive"      # assay reported this site
STATUS_TESTED_NEG = "tested_negative"      # site interrogated, signal below LoD
STATUS_UNOBSERVED = "unobserved"           # not reported by this assay; unknown
STATUS_UNINTERPRETABLE = "uninterpretable"  # assay/PCR failure, QC fail

#: in vitro (cell-free genomic DNA) vs in cell. Never pooled as one latent event.
CTX_INVITRO = "in_vitro"
CTX_INCELL = "in_cell"

#: Validation-selection classes from Lazzarotto et al. 2020, Supplementary Tables 7-8.
#: 'both' = detected by CHANGE-seq and GUIDE-seq; A-D are CHANGE-seq-detected and
#: GUIDE-seq-negative, chosen by a documented rule, so the selection is modellable.
VALIDATION_CLASSES = {
    "both": "detected by both CHANGE-seq and GUIDE-seq",
    "A": "1-3 mismatches, medium-to-high CHANGE-seq reads; GUIDE-seq negative",
    "B": "<=4 mismatches, high CHANGE-seq reads; GUIDE-seq negative",
    "C": "1-3 mismatches, low CHANGE-seq reads; GUIDE-seq negative",
    "D": "remaining sites with 4-6 mismatches; GUIDE-seq negative",
}

# --- validation calling ------------------------------------------------------

#: A targeted-tag-seq site is called active when the treated indel (or dsODN tag
#: integration) fraction exceeds the matched control at this FDR, pooling the three
#: replicates. Chosen instead of a fixed percentage threshold so that the call
#: carries the read depth with it.
VALIDATION_FDR = 0.05
#: Effect-size floor applied on top of significance, in fraction units.
VALIDATION_MIN_DELTA = 1e-3

# --- splits ------------------------------------------------------------------

SEED = 20261020
#: Guides are clustered before splitting so near-identical spacers cannot leak
#: across folds (roadmap section 9, test 1).
GUIDE_CLUSTER_MAX_MM = 4
N_FOLDS = 5
