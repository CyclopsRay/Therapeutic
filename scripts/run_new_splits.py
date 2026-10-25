#!/usr/bin/env python
"""Run a Tahoe method script on five NEW splits built from never-held-out lines.

Pre-registered as C1 in report/10_02_record.md (2026-09-29). The Rhaister
benchmark's five splits hold out 25 of Tahoe's 50 lines; the other 25 have only
ever been reference lines. This wrapper defines five fresh splits from those 25
(seeded permutation, 5 lines each) and patches run_rhaister_o_compact.load_split
so every downstream runner uses them. Reference lines are the other 45,
including the benchmark's original holdouts -- they are plain reference data
here, so no outcome from the frozen benchmark leaks in.

Test conditions for a query line are every treatment observed in it that is
also present in all 45 reference lines.

Usage:
    python scripts/run_new_splits.py scripts/<runner>.py [args]
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rhaister_o_compact as roc  # noqa: E402

SEED = int(os.environ.get("NEW_SPLIT_SEED", "0"))
_orig_load_split = roc.load_split
_orig_cache = roc.build_or_load_compact_cache
_state: dict = {}


def _compact(*a, **k):
    compact = _orig_cache(*a, **k)
    _state["compact"] = compact
    return compact


def _assignment(compact) -> dict[int, list[str]]:
    if "splits" in _state:
        return _state["splits"]
    repo = roc.DEFAULT_PROJECT_ROOT / "external" / "Rhaister"
    ever_held: set[str] = set()
    for s in range(5, 10):
        holdout, _, _ = _orig_load_split(repo, s)
        ever_held |= set(holdout)
    never = sorted(set(compact.cells.tolist()) - ever_held)
    if len(never) != 25:
        raise RuntimeError(f"expected 25 never-held-out lines, found {len(never)}")
    order = np.random.default_rng(SEED).permutation(25)
    splits = {5 + i: sorted(never[j] for j in order[i * 5:(i + 1) * 5]) for i in range(5)}
    _state["splits"] = splits
    print(f"[new-splits] seed {SEED}: " + "; ".join(f"{s}={v}" for s, v in splits.items()), flush=True)
    return splits


def new_load_split(repo_root, split: int):
    compact = _state.get("compact")
    if compact is None:
        raise RuntimeError("load_split called before the compact cache was built")
    holdout = _assignment(compact)[split]
    reference = sorted(set(compact.cells.tolist()) - set(holdout))
    observed: dict[str, set[str]] = {}
    for cell, treatment in zip(compact.cells, compact.treatments):
        observed.setdefault(cell, set()).add(treatment)
    complete = set.intersection(*(observed[c] for c in reference))
    tests = {c: sorted(observed[c] & complete) for c in holdout}
    # A real path is still returned so runners that hash the split file work.
    _, _, path = _orig_load_split(repo_root, split)
    return holdout, {c: set(v) for c, v in tests.items()}, path


roc.build_or_load_compact_cache = _compact
roc.load_split = new_load_split


if __name__ == "__main__":
    target = sys.argv[1]
    sys.argv = [target, *sys.argv[2:]]
    print(f"[new-splits] running {target} {' '.join(sys.argv[1:])}", flush=True)
    if Path(target).name == "run_rhaister_o_compact.py":
        roc.main()
    else:
        runpy.run_path(target, run_name="__main__")
