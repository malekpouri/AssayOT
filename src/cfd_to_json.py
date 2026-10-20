"""Convert the published CFD lookup tables from pickle to JSON.

The Cutting Frequency Determination tables of Doench et al. 2016 circulate as a pair of
protocol-0 pickles. Unpickling executes whatever opcodes the file contains, so nothing
in this pipeline unpickles third-party data at run time: the tables are converted once,
here, behind a restricted unpickler that refuses every opcode capable of importing or
calling anything, and `src/baselines.py` reads only the resulting JSON.

    python -m src.cfd_to_json [<directory containing the pickles>]

Writes `cfd_mismatch_scores.json` and `cfd_pam_scores.json` beside the inputs.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

try:
    from . import config as C
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config as C


class RestrictedUnpickler(pickle.Unpickler):
    """Loads plain data and nothing else."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("blocked persistent id")


def load_table(path: Path) -> dict:
    with open(path, "rb") as fh:
        obj = RestrictedUnpickler(fh).load()
    if not isinstance(obj, dict):
        raise TypeError(f"{path.name}: expected a dict, got {type(obj).__name__}")
    bad = [k for k, v in obj.items()
           if not isinstance(k, str) or not isinstance(v, float)]
    if bad:
        raise TypeError(f"{path.name}: non str->float entries, e.g. {bad[:3]}")
    return obj


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else C.DATA / "cfd"
    mm = load_table(src / "mismatch_score.pkl")
    pam = load_table(src / "pam_scores.pkl")

    # Published anchors. If these drift, the tables are not the ones we think they are.
    assert len(mm) == 240, f"expected 240 mismatch entries, got {len(mm)}"
    assert len(pam) == 16, f"expected 16 PAM entries, got {len(pam)}"
    assert abs(pam["GG"] - 1.0) < 1e-9
    assert abs(pam["AG"] - 0.259259259) < 1e-6
    assert abs(pam["TT"]) < 1e-12

    (src / "cfd_mismatch_scores.json").write_text(json.dumps(mm, indent=0,
                                                             sort_keys=True))
    (src / "cfd_pam_scores.json").write_text(json.dumps(pam, indent=0,
                                                        sort_keys=True))
    print(f"wrote {len(mm)} mismatch and {len(pam)} PAM entries to {src}")


if __name__ == "__main__":
    main()
