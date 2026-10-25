#!/usr/bin/env python3
"""Freeze language-model embeddings for two cell-line text representations.

``template`` text is generated deterministically from the same DepMap
``Model.csv`` fields used by the structured arm of the ablation.  It therefore
contains no fact the structured encoding lacks; comparing the two isolates the
effect of the language model's prior from the effect of new information.

``cellosaurus`` text is the redacted free-text comment block fetched by
``scripts/fetch_cellosaurus_text.py``, which may contain curated facts absent
from the structured fields.

Both are embedded with the same frozen encoder and pooling.  No fine-tuning:
45 reference lines per split cannot identify a large text encoder, and a frozen
comparison is the controlled test.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE_FIELDS = (
    ("OncotreeLineage", "tissue lineage"),
    ("OncotreePrimaryDisease", "primary disease"),
    ("OncotreeSubtype", "disease subtype"),
    ("Sex", "patient sex"),
    ("AgeCategory", "patient age category"),
    ("PatientRace", "patient reported ancestry"),
    ("PrimaryOrMetastasis", "tumour origin"),
    ("SampleCollectionSite", "sample collection site"),
    ("TissueOrigin", "tissue origin"),
    ("GrowthPattern", "culture growth pattern"),
    ("OnboardedMedia", "culture medium"),
    ("SerumFreeMedia", "serum-free medium"),
    ("PlateCoating", "plate coating"),
    ("EngineeredModel", "engineered model"),
)


def build_template_text(row: pd.Series) -> str:
    """Deterministic prose rendering of exactly the structured fields."""
    parts = ["This is a human cancer cell line."]
    for column, phrase in TEMPLATE_FIELDS:
        value = row.get(column)
        if pd.isna(value) or str(value).strip() in {"", "nan", "Unknown"}:
            continue
        parts.append(f"Its {phrase} is {str(value).strip()}.")
    age = row.get("Age")
    if pd.notna(age):
        try:
            parts.append(f"The donor was {int(float(age))} years old.")
        except (TypeError, ValueError):
            pass
    return " ".join(parts)


def embed_texts(
    texts: list[str],
    model_source: str,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, int]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModel.from_pretrained(model_source)
    model.eval()

    n_truncated = 0
    for text in texts:
        if len(tokenizer(text)["input_ids"]) > max_length:
            n_truncated += 1

    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            output = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0).astype(np.float32), n_truncated


def save(
    path: Path,
    rrids: list[str],
    embeddings: np.ndarray,
    model_name: str,
    text_kind: str,
    texts: list[str],
    n_truncated: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        rrid=np.asarray(rrids, dtype=str),
        embedding=embeddings,
        model_name=np.asarray([model_name], dtype=str),
        pooling=np.asarray(["attention_masked_mean"], dtype=str),
        text_kind=np.asarray([text_kind], dtype=str),
        word_count=np.asarray([len(text.split()) for text in texts], dtype=np.int32),
        n_truncated=np.asarray([n_truncated], dtype=np.int32),
    )
    os.replace(temporary, path)
    words = [len(text.split()) for text in texts]
    print(
        f"{text_kind:14s} -> {path.name}: {embeddings.shape[0]} lines x "
        f"{embeddings.shape[1]} dims, words min={min(words)} "
        f"median={sorted(words)[len(words)//2]} max={max(words)}, "
        f"truncated={n_truncated}",
        flush=True,
    )


def main(args: argparse.Namespace) -> None:
    mapping = pd.read_csv(args.mapping).dropna(subset=["RRID"])
    with args.cellosaurus_text.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["records"]

    rrids, template_texts, cellosaurus_texts = [], [], []
    for row in mapping.itertuples():
        rrid = str(row.RRID)
        if rrid not in records:
            print(f"  skipping {rrid}: no Cellosaurus text", flush=True)
            continue
        rrids.append(rrid)
        template_texts.append(build_template_text(mapping.loc[row.Index]))
        cellosaurus_texts.append(records[rrid]["redacted_text"])

    # A pre-staged local snapshot keeps the compute node offline; the recorded
    # model name stays the canonical hub id for provenance.
    model_source = str(args.model_path) if args.model_path else args.model_name
    print(f"embedding {len(rrids)} cell lines with {args.model_name}", flush=True)
    print(f"loading weights from {model_source}", flush=True)
    print(f"\nexample template text:\n  {template_texts[0]}\n", flush=True)
    print(f"example redacted Cellosaurus text:\n  {cellosaurus_texts[0][:400]}...\n", flush=True)

    template_embeddings, template_truncated = embed_texts(
        template_texts, model_source, args.batch_size, args.max_length
    )
    cellosaurus_embeddings, cellosaurus_truncated = embed_texts(
        cellosaurus_texts, model_source, args.batch_size, args.max_length
    )

    save(
        args.output_root / "embeddings_template.npz",
        rrids, template_embeddings, args.model_name, "template", template_texts, template_truncated,
    )
    save(
        args.output_root / "embeddings_cellosaurus.npz",
        rrids, cellosaurus_embeddings, args.model_name, "cellosaurus", cellosaurus_texts, cellosaurus_truncated,
    )

    texts_path = args.output_root / "embedded_texts.json"
    temporary = texts_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": args.model_name,
                "pooling": "attention_masked_mean",
                "max_length": args.max_length,
                "records": {
                    rrid: {"template": template, "cellosaurus": cellosaurus}
                    for rrid, template, cellosaurus in zip(rrids, template_texts, cellosaurus_texts)
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    os.replace(temporary, texts_path)
    print(f"texts written to {texts_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("/nfs-stor/yuqi.lei/Therapeutic/derived/depmap_24q4/tahoe_observed_Model.csv"),
    )
    parser.add_argument(
        "--cellosaurus-text",
        type=Path,
        default=Path("/nfs-stor/yuqi.lei/Therapeutic/derived/cellosaurus/cellosaurus_text.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/nfs-stor/yuqi.lei/Therapeutic/derived/cellosaurus"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local snapshot directory to load from; --model-name is still recorded.",
    )
    parser.add_argument(
        "--model-name", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
