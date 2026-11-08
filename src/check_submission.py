"""Check the separate submission files against the publisher's stated limits.

Highlights and graphical abstracts fail quietly: a bullet one character over the limit,
or an image a hundred pixels too narrow, is returned by a technical check days later
rather than refused at upload. This verifies them before submission.

    python -m src.check_submission

Limits encoded here are Elsevier's published ones. They are not read from the live Guide
for Authors, because ScienceDirect refuses automated requests; confirm them by hand once
and they rarely change.
"""
from __future__ import annotations

import re
import sys

try:
    from . import config as C
except ImportError:  # pragma: no cover
    from src import config as C

#: Elsevier: at most five highlights, each at most 85 characters including spaces.
MAX_HIGHLIGHTS = 5
MAX_HIGHLIGHT_CHARS = 85
#: Elsevier: graphical abstract at least 531 x 1328 pixels (height x width).
MIN_GA_WIDTH, MIN_GA_HEIGHT = 1328, 531

PKG = C.PROJ.parent / "02-Assay-Learning-Cross-Study-CRISPR–Cas9" / "paper-cbc"
if not PKG.exists():                                    # running from the project root
    PKG = C.PROJ / "paper-cbc"


def _plain(s: str) -> str:
    """Strip the LaTeX a reviewer never sees, so the count is of visible characters."""
    s = re.sub(r"\\%", "%", s)
    s = re.sub(r"``|''", '"', s)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    return re.sub(r"[{}]", "", s).strip()


def main() -> None:
    problems: list[str] = []

    hl = PKG / "highlights.tex"
    if hl.exists():
        items = [_plain(m) for m in re.findall(r"\\item\s+(.*)", hl.read_text())]
        print(f"highlights: {len(items)} bullets")
        for i, s in enumerate(items, 1):
            flag = "" if len(s) <= MAX_HIGHLIGHT_CHARS else "  <-- OVER LIMIT"
            print(f"  {len(s):3d}  {s}{flag}")
            if len(s) > MAX_HIGHLIGHT_CHARS:
                problems.append(f"highlight {i} is {len(s)} characters "
                                f"(limit {MAX_HIGHLIGHT_CHARS})")
        if len(items) > MAX_HIGHLIGHTS:
            problems.append(f"{len(items)} highlights (limit {MAX_HIGHLIGHTS})")
    else:
        problems.append(f"missing {hl}")

    ga = PKG / "graphical_abstract.png"
    if ga.exists():
        from PIL import Image
        w, h = Image.open(ga).size
        ok = w >= MIN_GA_WIDTH and h >= MIN_GA_HEIGHT
        print(f"\ngraphical abstract: {w} x {h} px "
              f"({'meets' if ok else 'BELOW'} the {MIN_GA_WIDTH} x {MIN_GA_HEIGHT} minimum)")
        if not ok:
            problems.append(f"graphical abstract is {w}x{h}, "
                            f"below {MIN_GA_WIDTH}x{MIN_GA_HEIGHT}")
    else:
        problems.append(f"missing {ga}")

    for name in ("title_page.pdf", "highlights.pdf", "graphical_abstract.pdf", "cbc.pdf"):
        p = PKG / name
        print(f"{'' if name == 'title_page.pdf' else ''}  {name}: "
              f"{'present' if p.exists() else 'MISSING'}")
        if not p.exists():
            problems.append(f"missing {p}")

    # the manuscript must not repeat what is uploaded separately
    man = PKG / "cbc.tex"
    if man.exists():
        t = man.read_text()
        for env in ("highlights", "graphicalabstract"):
            if f"\\begin{{{env}}}" in t:
                problems.append(f"cbc.tex still contains a {env} environment; it is "
                                "uploaded as a separate file and would be duplicated")

    print()
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print("all submission files present and within the stated limits")


if __name__ == "__main__":
    main()
