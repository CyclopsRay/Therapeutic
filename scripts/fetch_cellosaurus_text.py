#!/usr/bin/env python3
"""Fetch Cellosaurus free-text comments for the Tahoe-observed cell lines.

Writes one JSON file holding, for each RRID, the raw Cellosaurus entry, the
extracted comment (CC) text, and a redacted variant with identity tokens and
drug-response statements removed.

Redaction matters for the strict zero-shot boundary.  Two separate risks are
handled:

* identity leakage -- the cell-line name acts as a retrieval key into a
  language model's pretraining memory, so names, synonyms and accessions are
  masked in the redacted text;
* drug-response leakage -- Cellosaurus comments can state sensitivity or
  resistance phenotypes, which are outcome information under our declared
  input boundary, so any sentence carrying those terms is dropped.

Run on the login node: 49 small API requests, no computation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_MAPPING = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/derived/depmap_24q4/tahoe_observed_Model.csv"
)
DEFAULT_OUTPUT = Path(
    "/nfs-stor/yuqi.lei/Therapeutic/derived/cellosaurus/cellosaurus_text.json"
)
API_TEMPLATE = "https://api.cellosaurus.org/cell-line/{rrid}?format=txt"

# Sentences containing these tokens are removed from the redacted text: they
# describe response to treatment, which is outcome information here.
DRUG_RESPONSE_TERMS = (
    "resistant", "resistance", "sensitiv", "susceptib", "ic50", "gi50",
    "ec50", "ld50", "chemoresist", "chemosensit", "drug", "inhibitor",
    "treated with", "treatment with", "selected in", "selected with",
    "cytotoxic", "apoptosis induced", "refractory", "responsive to",
    "mtx", "methotrexate", "doxorubicin", "cisplatin", "paclitaxel",
    "tamoxifen", "erlotinib", "gefitinib", "imatinib", "vemurafenib",
)

# Cellosaurus line codes whose content we keep as descriptive text.
KEEP_CODES = ("CC",)


def fetch_entry(rrid: str, retries: int = 3, pause: float = 0.5) -> str | None:
    url = API_TEMPLATE.format(rrid=rrid)
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Therapeutic-research/1.0"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
            if attempt == retries - 1:
                print(f"  FAILED {rrid}: {error}", file=sys.stderr, flush=True)
                return None
            time.sleep(pause * (attempt + 1))
    return None


def parse_fields(entry: str) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    for line in entry.splitlines():
        if len(line) < 5 or not line[:2].strip():
            continue
        code, _, value = line.partition("   ")
        code = code.strip()
        value = value.strip()
        if not code or not value:
            continue
        fields.setdefault(code, []).append(value)
    return fields


def identity_tokens(fields: dict[str, list[str]], extra: list[str]) -> list[str]:
    """Names, synonyms and accessions that identify this specific line.

    Database *names* from cross-reference lines are deliberately not masked:
    only the line-specific identifier after the separator is.  Masking generic
    resource names such as CCLE would corrupt ordinary prose without removing
    any identity signal.
    """
    tokens: list[str] = []
    for code in ("ID", "SY", "AC", "AS"):
        for value in fields.get(code, []):
            for piece in re.split(r"[;|]", value):
                piece = piece.strip()
                if len(piece) >= 3:
                    tokens.append(piece)
    for value in fields.get("DR", []):
        _, separator, identifier = value.partition(";")
        if separator:
            identifier = identifier.strip()
            if len(identifier) >= 3:
                tokens.append(identifier)
    tokens.extend(token for token in extra if token and len(str(token)) >= 3)
    # Longest first so that substrings do not pre-empt longer names.
    return sorted({str(token) for token in tokens}, key=len, reverse=True)


def redact(text: str, tokens: list[str]) -> tuple[str, int, int]:
    """Mask identity tokens and drop drug-response sentences."""
    masked = text
    n_masked = 0
    for token in tokens:
        # Word-boundary anchored so that a short name cannot be matched inside
        # an unrelated word (e.g. "CLO" inside "Encyclopedia").
        pattern = re.compile(
            rf"(?<![0-9A-Za-z]){re.escape(token)}(?![0-9A-Za-z])",
            flags=re.IGNORECASE,
        )
        masked, count = pattern.subn("[CELL_LINE]", masked)
        n_masked += count
    # Also mask bare Cellosaurus/DepMap-style accessions that survived.
    masked, extra = re.subn(r"\b(CVCL_[A-Z0-9]+|ACH-\d+|SIDM\d+)\b", "[ID]", masked)
    n_masked += extra

    sentences = re.split(r"(?<=[.;])\s+", masked)
    kept, dropped = [], 0
    for sentence in sentences:
        lowered = sentence.lower()
        if any(term in lowered for term in DRUG_RESPONSE_TERMS):
            dropped += 1
            continue
        kept.append(sentence)
    return " ".join(kept).strip(), n_masked, dropped


def main(args: argparse.Namespace) -> None:
    mapping = pd.read_csv(args.mapping).dropna(subset=["RRID"])
    records: dict[str, dict] = {}
    n_failed = 0
    for row in mapping.itertuples():
        rrid = str(row.RRID)
        entry = fetch_entry(rrid)
        if entry is None:
            n_failed += 1
            continue
        fields = parse_fields(entry)
        comments = fields.get("CC", [])
        raw_text = " ".join(comments)
        tokens = identity_tokens(
            fields,
            [
                getattr(row, "CellLineName", ""),
                getattr(row, "StrippedCellLineName", ""),
                getattr(row, "CCLEName", ""),
                getattr(row, "ModelID", ""),
                rrid,
            ],
        )
        redacted, n_masked, n_dropped = redact(raw_text, tokens)
        records[rrid] = {
            "rrid": rrid,
            "model_id": str(getattr(row, "ModelID", "")),
            "cell_line_name": str(getattr(row, "CellLineName", "")),
            "n_comment_lines": len(comments),
            "raw_text": raw_text,
            "raw_word_count": len(raw_text.split()),
            "redacted_text": redacted,
            "redacted_word_count": len(redacted.split()),
            "n_identity_tokens_masked": n_masked,
            "n_sentences_dropped_drug_response": n_dropped,
        }
        print(
            f"{rrid} {str(getattr(row, 'CellLineName', '')):20s} "
            f"raw={records[rrid]['raw_word_count']:4d}w "
            f"redacted={records[rrid]['redacted_word_count']:4d}w "
            f"masked={n_masked:3d} dropped_sentences={n_dropped}",
            flush=True,
        )
        time.sleep(args.pause)

    payload = {
        "source": "Cellosaurus REST API",
        "api_template": API_TEMPLATE,
        "n_requested": len(mapping),
        "n_fetched": len(records),
        "n_failed": n_failed,
        "kept_field_codes": list(KEEP_CODES),
        "drug_response_terms": list(DRUG_RESPONSE_TERMS),
        "redaction": (
            "Identity tokens (ID/SY/AC/AS/DR values, DepMap and Cellosaurus "
            "accessions, model names) replaced with placeholders; any sentence "
            "containing a drug-response term removed."
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, args.output)

    words = [record["redacted_word_count"] for record in records.values()]
    print(
        f"\nfetched {len(records)}/{len(mapping)} lines, {n_failed} failed\n"
        f"redacted word count: min={min(words)} median={sorted(words)[len(words)//2]} "
        f"max={max(words)} mean={sum(words)/len(words):.1f}\n"
        f"written to {args.output}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pause", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
