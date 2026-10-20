"""Enumerate the candidate space: every hg38 site within MAX_MM_PRIMARY substitutions
of each sgRNA pattern, on both strands.

Without this step the only rows available are detected sites, i.e. positives, and the
unlabelled set U would be undefined. The protocol is fixed in advance and reported, and
model performance is only ever claimed over this space, never over "the genome"
(roadmap section 5, rule 4).

Method: the genome is one-hot encoded in chunks and correlated against all sgRNA
patterns at once with a single grouped 1-D convolution on the GPU, which yields the
number of matching positions at every offset for every guide. This is a GPU
reimplementation of the standard seedless mismatch enumeration; it is exact, not
heuristic, so the output is a strict superset of any seed-based search.

Run:  python -m src.scan
"""
from __future__ import annotations

import gzip
import sys
import time

import numpy as np
import pandas as pd
import torch

try:
    from . import config as C
except ImportError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from src import config as C

BASES = "ACGT"
CODE = np.full(256, 4, dtype=np.uint8)
for i, b in enumerate(BASES):
    CODE[ord(b)] = i
    CODE[ord(b.lower())] = i
COMP = np.array([3, 2, 1, 0, 4], dtype=np.uint8)  # A<->T, C<->G, N->N

CHUNK = 1 << 21  # 2 Mbp per convolution pass


# ---------------------------------------------------------------------------
# genome
# ---------------------------------------------------------------------------

def read_genome(path=None) -> dict[str, np.ndarray]:
    """Load the primary chromosomes as uint8 code arrays (0-3 = ACGT, 4 = N)."""
    path = path or C.HG38_FA
    gz = path.with_suffix(path.suffix + ".gz") if not path.exists() else None
    src = gz if (gz and gz.exists()) else path
    opener = gzip.open if str(src).endswith(".gz") else open
    print(f"reading {src}")
    want = set(C.CHROMS)
    out: dict[str, np.ndarray] = {}
    name = None
    buf: list[bytes] = []

    def flush():
        if name in want and buf:
            raw = np.frombuffer(b"".join(buf), dtype=np.uint8)
            out[name] = CODE[raw]
            print(f"  {name}: {len(out[name]):,} bp")

    with opener(src, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                flush()
                name = line[1:].split()[0].decode()
                buf = []
            elif name in want:
                buf.append(line.strip())
    flush()
    missing = want - set(out)
    if missing:
        raise RuntimeError(f"missing chromosomes: {sorted(missing)}")
    return out


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

def build_weight(patterns: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
    """One-hot weights over informative positions, plus the count of informative
    positions per guide (so mismatches = informative - matches)."""
    g = len(patterns)
    w = torch.zeros(g, 4, C.SITE_LEN, dtype=torch.float16)
    n_inf = torch.zeros(g, dtype=torch.float16)
    for i, p in enumerate(patterns):
        for j, ch in enumerate(p.upper()):
            if j == C.PAM_WILDCARD_IDX or ch == "N":
                continue  # wildcard: contributes to neither matches nor the total
            b = BASES.find(ch)
            if b < 0:
                continue
            w[i, b, j] = 1.0
            n_inf[i] += 1
    return w.to(device), n_inf.to(device)


def pam_class(site: str) -> str:
    pam = site[C.SPACER_LEN:]
    if len(pam) < 3:
        return "other"
    if pam[1:3] == "GG":
        return "NGG"
    if pam[1:3] == "AG":
        return "NAG"
    if pam[1:3] == "GA":
        return "NGA"
    return "other"


def scan_guides(genome: dict[str, np.ndarray], patterns: list[str],
                names: list[str], max_mm: int, device="cuda") -> pd.DataFrame:
    w, n_inf = build_weight(patterns, device)
    g = len(patterns)
    hits: list[pd.DataFrame] = []
    t0 = time.time()

    for chrom, codes in genome.items():
        n = len(codes)
        for strand in ("+", "-"):
            seq = codes if strand == "+" else COMP[codes[::-1]].copy()
            for off in range(0, n - C.SITE_LEN + 1, CHUNK):
                end = min(off + CHUNK + C.SITE_LEN - 1, n)
                sub = torch.from_numpy(seq[off:end].astype(np.int64)).to(device)
                L = sub.numel()
                if L < C.SITE_LEN:
                    continue
                oh = torch.zeros(4, L, dtype=torch.float16, device=device)
                valid = sub < 4
                oh[sub.clamp(max=3), torch.arange(L, device=device)] = torch.where(
                    valid, torch.ones(L, dtype=torch.float16, device=device),
                    torch.zeros(L, dtype=torch.float16, device=device))
                matches = torch.nn.functional.conv1d(oh[None], w)[0]  # (g, L-22)
                mm = n_inf[:, None] - matches
                sel = (mm <= max_mm).nonzero(as_tuple=False)
                if sel.numel() == 0:
                    del oh, matches, mm
                    continue
                gi = sel[:, 0].cpu().numpy()
                pos = sel[:, 1].cpu().numpy()
                nmm = mm[sel[:, 0], sel[:, 1]].cpu().numpy().astype(np.int16)
                # local (chunk) coordinate -> genomic 1-based start on the + strand
                local = off + pos
                if strand == "+":
                    start1 = local + 1
                else:
                    start1 = n - (local + C.SITE_LEN) + 1
                site_codes = seq[local[:, None] + np.arange(C.SITE_LEN)[None, :]]
                site_seq = np.array(
                    ["".join(BASES[c] if c < 4 else "N" for c in row)
                     for row in site_codes])
                hits.append(pd.DataFrame(dict(
                    guide_sample=np.asarray(names)[gi],
                    chromosome=chrom, start=start1.astype(np.int64),
                    end=(start1 + C.SITE_LEN - 1).astype(np.int64),
                    strand=strand, site_sequence=site_seq, n_mm_total=nmm)))
                del oh, matches, mm, sel
        done = time.time() - t0
        tot = sum(len(h) for h in hits)
        print(f"  {chrom} done  cum_hits={tot:,}  {done:.0f}s", flush=True)
    out = pd.concat(hits, ignore_index=True)
    out["pam_class"] = [pam_class(s) for s in out.site_sequence]
    return out


def guide_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Every sgRNA for which a candidate space can be enumerated.

    The pilot sgRNAs carry a published 23-nt target pattern. External sgRNAs are taken
    from their 20-nt spacer with an NGG pattern appended, and only where the study
    reported hg38 coordinates -- an hg19-only sgRNA can still contribute labelled
    confirmatory rows, but it has no comparable candidate space.
    """
    pilot = pairs[pairs.source_set == "pilot"]
    g1 = (pilot[["guide_sample", "guide_sequence", "target_pattern"]]
          .drop_duplicates("guide_sample").dropna())
    g1["source_set"] = "pilot"

    ext = pairs[(pairs.source_set == "external")
                & pairs.genome_build.eq(C.GENOME_BUILD)
                & pairs.guide_sequence.str.len().eq(C.SPACER_LEN)]
    g2 = (ext[["guide_sequence"]].drop_duplicates()
          .assign(guide_sample=lambda d: d.guide_sequence,
                  target_pattern=lambda d: d.guide_sequence + "NGG",
                  source_set="external"))
    g2 = g2[~g2.guide_sequence.isin(set(g1.guide_sequence))]
    out = pd.concat([g1, g2], ignore_index=True)
    out = out.drop_duplicates("guide_sequence").reset_index(drop=True)
    print(f"sgRNA patterns: {len(g1)} pilot + {len(g2)} external = {len(out)}")
    return out


def main() -> None:
    pairs = pd.read_parquet(C.PAIRS)
    pilot = pairs[pairs.source_set == "pilot"]
    guides = guide_table(pairs)
    genome = read_genome()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cand = scan_guides(genome, guides.target_pattern.tolist(),
                       guides.guide_sample.tolist(), C.MAX_MM_PRIMARY, device)
    gmap = dict(zip(guides.guide_sample, guides.guide_sequence))
    cand["guide_sequence"] = cand.guide_sample.map(gmap)
    cand["pair_id"] = (cand.guide_sequence + "|" + cand.chromosome + ":"
                       + cand.start.astype(str) + "-" + cand.end.astype(str)
                       + ":" + cand.strand)
    cand = cand.drop_duplicates("pair_id")
    cand["genome_build"] = C.GENOME_BUILD
    cand = cand.merge(guides[["guide_sample", "source_set"]], on="guide_sample",
                      how="left")
    for col in ("guide_sample", "chromosome", "strand", "pam_class", "source_set"):
        cand[col] = cand[col].astype("category")
    cand.to_parquet(C.CANDIDATES, index=False)
    print(f"\n{len(cand):,} candidates -> {C.CANDIDATES}")
    print(cand.groupby("n_mm_total").size())
    print(cand.pam_class.value_counts())

    # Coverage check: the candidate space must contain every substitution-only detected
    # site. The flags are cast explicitly because `~` on an object-dtype Series is a
    # bitwise not, which would turn this assertion into a no-op.
    det_flag = pilot.detected_changeseq.fillna(False).astype(bool)
    bulge = pilot.has_bulge.fillna(False).astype(bool)
    det = pilot[det_flag & ~bulge]
    covered = det.pair_id.isin(set(cand.pair_id))
    print(f"\ncandidate space covers {covered.mean():.4%} of "
          f"{len(det):,} substitution-only CHANGE-seq detected sites")
    if not covered.all():
        miss = det[~covered]
        print(f"  {len(miss)} uncovered; mismatch counts: "
              f"{miss.n_mismatch.value_counts().sort_index().to_dict()}")
        miss[["pair_id", "site_sequence", "n_mismatch", "n_mismatch_pam", "span"]] \
            .to_csv(C.PROC / "candidate_coverage_misses.csv", index=False)


if __name__ == "__main__":
    main()
