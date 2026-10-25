#!/usr/bin/env python
"""Run a Tahoe method script with some cell lines removed from the data entirely.

Usage: EXCLUDE_LINES=CVCL_1571 python scripts/run_excluding_lines.py scripts/<runner>.py [args]

Wraps run_rhaister_o_compact.build_or_load_compact_cache so every downstream
runner sees a CompactData without the excluded lines (never held out in any
split, so only reference fitting changes). Used for the NCI-H596 (CVCL_1571)
label check: its Tahoe DMSO profile does not match DepMap NCI-H596 (rank 32/44)
while EmeraldBay's NCI-H596 does (rank 1).
"""
import os
import runpy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rhaister_o_compact as roc  # noqa: E402

EXCLUDE = set(filter(None, os.environ.get("EXCLUDE_LINES", "").split(",")))
_orig = roc.build_or_load_compact_cache


def filtered(*a, **k):
    c = _orig(*a, **k)
    keep = ~np.isin(c.cells, list(EXCLUDE))
    idx = np.flatnonzero(keep)
    print(f"[exclude] removing {sorted(EXCLUDE)}: {int((~keep).sum())} rows", flush=True)
    return roc.CompactData(cells=c.cells[idx], treatments=c.treatments[idx],
                           plates=[c.plates[i] for i in idx], delta=np.asarray(c.delta[idx]), genes=c.genes)


roc.build_or_load_compact_cache = filtered

if __name__ == "__main__":
    target = sys.argv[1]
    sys.argv = [target, *sys.argv[2:]]
    if Path(target).name == "run_rhaister_o_compact.py":
        roc.main()
    else:
        runpy.run_path(target, run_name="__main__")
