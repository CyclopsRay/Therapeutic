#!/usr/bin/env python3
"""Mirror each figure's caption from the paper .tex into its dossier (paper/figures/<fig>.md); report drift.

The .tex is the single source of truth (GeneralAgent references/figures.md): this script only rewrites the fenced
block between `<!-- caption:begin -->` and `<!-- caption:end -->` in the dossier, never the .tex.
A dossier is matched to the figure environment whose \\includegraphics names `figures/<fig>` (any suffix).
Run: python3 scripts/figures/sync_captions.py [--check]   (--check: report drift, change nothing; exit 1 on drift)
"""
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2] / "paper"
BEGIN, END = "<!-- caption:begin -->", "<!-- caption:end -->"


def caption_of(env: str) -> str | None:
    i = env.find(r"\caption{")
    if i < 0:
        return None
    i += len(r"\caption{"); depth, j = 1, i
    while depth:
        depth += {"{": 1, "}": -1}.get(env[j], 0); j += 1
    return " ".join(env[i:j - 1].split())


def main() -> int:
    check = "--check" in sys.argv
    tex = "\n".join(p.read_text() for p in sorted((PAPER / "sections").glob("*.tex")))
    envs = re.findall(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", tex, flags=re.S)
    drift = 0
    for md in sorted((PAPER / "figures").glob("*.md")):
        name = md.stem
        env = next((e for e in envs if re.search(rf"\\includegraphics(\[[^]]*\])?\{{figures/{re.escape(name)}(_preview)?(\.pdf)?\}}", e)), None)
        text = md.read_text()
        if env is None:
            print(f"{name}: not included in any section"); continue
        cap = caption_of(env)
        block = f"{BEGIN}\n```latex\n{cap}\n```\n{END}"
        old = re.search(re.escape(BEGIN) + r".*?" + re.escape(END), text, flags=re.S)
        if old is None:
            print(f"{name}: dossier has no caption block"); drift += 1; continue
        if old.group(0) != block:
            drift += 1
            print(f"{name}: caption drift" + ("" if check else " -> refreshed"))
            if not check:
                md.write_text(text[:old.start()] + block + text[old.end():])
        else:
            print(f"{name}: in sync")
    return 1 if (check and drift) else 0


if __name__ == "__main__":
    sys.exit(main())
