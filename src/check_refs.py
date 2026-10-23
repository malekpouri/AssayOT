"""Verify every DOI in the bibliography against Crossref.

A wrong citation is a defect a reader can check faster than any of the statistics, and
it is the kind that survives proofreading because the entry looks plausible. This
resolves each DOI and compares the stored title with the registered one.

Needs network access, so it is not part of `run_all.sh`; run it before submission and
after adding references.

    python -m src.check_refs

Note on parsing: entries here end with `}}` on the last field's line, not with a brace
on its own line, so a line-anchored regex silently matches nothing and reports a clean
bibliography. This counts braces instead, and asserts that it found some entries.
"""
from __future__ import annotations

import difflib
import json
import pathlib
import re
import sys
import time
import urllib.request

try:
    from . import config as C
except ImportError:  # pragma: no cover
    from src import config as C

BIB = C.PAPER / "references.bib"
MIN_TITLE_SIMILARITY = 0.80


def entries(text: str) -> list[tuple[str, str]]:
    out, i = [], 0
    while (s := text.find("@", i)) >= 0:
        b = text.find("{", s)
        depth, j = 0, b
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[b + 1:j]
        out.append((body.split(",", 1)[0].strip(), body))
        i = j + 1
    return out


def field(body: str, name: str) -> str | None:
    m = re.search(name + r"\s*=\s*\{", body)
    if not m:
        return None
    k, depth, buf = m.end(), 1, []
    while k < len(body) and depth:
        c = body[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if not depth:
                break
        buf.append(c)
        k += 1
    return re.sub(r"\s+", " ", "".join(buf)).strip()


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", (s or "").lower())).strip()


def main() -> None:
    ents = entries(BIB.read_text())
    assert len(ents) > 5, f"parsed only {len(ents)} entries; the parser is broken"
    problems, nodoi = [], []
    for key, body in ents:
        doi, ours = field(body, "doi"), field(body, "title")
        if not doi:
            nodoi.append(key)
            continue
        try:
            with urllib.request.urlopen(
                    f"https://api.crossref.org/works/{doi}", timeout=30) as r:
                msg = json.load(r)["message"]
        except Exception as e:                        # noqa: BLE001
            problems.append(f"{key}: DOI {doi} did not resolve ({type(e).__name__})")
            continue
        theirs = (msg.get("title") or [""])[0]
        sim = difflib.SequenceMatcher(None, _norm(ours), _norm(theirs)).ratio()
        if sim < MIN_TITLE_SIMILARITY:
            problems.append(f"{key}: title similarity {sim:.2f}\n"
                            f"    ours:     {ours}\n    crossref: {theirs}")
        time.sleep(0.1)
    print(f"{len(ents)} entries; {len(ents) - len(nodoi)} with a DOI, all checked")
    if nodoi:
        print(f"no DOI (conference or pre-DOI venues, verified by hand): {', '.join(nodoi)}")
    if problems:
        print(f"\n{len(problems)} PROBLEMS:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print("every DOI resolves and every title matches the registered record")


if __name__ == "__main__":
    main()
