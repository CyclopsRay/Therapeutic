#!/usr/bin/env python3
"""Frozen SMILES language-model embeddings (ChemBERTa-77M-MLM, MoLFormer-XL) per compound.

Unseen-drug branch, report/10_09_record.md. Input: unique_smiles.tsv from
build_drug_features.py (standardized parent SMILES keyed by InChIKey). Output:
smiles_lm_<model>.npz with `inchikey` and `embedding` (masked mean of the last
hidden layer). Frozen, no fine-tuning: the question is whether the pretrained
space puts transcriptionally similar compounds close together.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

MODELS = {
    "chemberta": ("DeepChem/ChemBERTa-77M-MLM", False),
    "molformer": ("ibm/MoLFormer-XL-both-10pct", True),
}


def embed(name: str, smiles: list[str], batch: int, max_length: int) -> np.ndarray:
    repo, remote = MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=remote)
    model = AutoModel.from_pretrained(repo, trust_remote_code=remote, deterministic_eval=True) \
        if name == "molformer" else AutoModel.from_pretrained(repo)
    model.eval()
    out = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(smiles), batch):
            tokens = tokenizer(smiles[start:start + batch], padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt")
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            out.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).float().numpy())
            if (start // batch) % 50 == 0:
                print(f"  {name}: {start + len(tokens['input_ids']):,}/{len(smiles):,} "
                      f"({time.time() - started:.0f}s)", flush=True)
    return np.concatenate(out).astype(np.float32)


def main(args: argparse.Namespace) -> None:
    print(f"host {platform.node()} python {sys.version.split()[0]} torch {torch.__version__} "
          f"transformers {transformers.__version__} threads {torch.get_num_threads()}", flush=True)
    table = pd.read_csv(args.smiles, sep="\t")
    if args.limit:
        table = table.head(args.limit)
    for name in args.models:
        values = embed(name, table["smiles_std"].astype(str).tolist(), args.batch, args.max_length)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: non-finite embeddings")
        target = args.out / f"smiles_lm_{name}{'_test' if args.limit else ''}.npz"
        np.savez(target, inchikey=np.asarray(table["inchikey"].astype(str).tolist(), dtype=str), embedding=values,
                 model=np.asarray([MODELS[name][0]]))
        print(f"{name}: wrote {values.shape} to {target}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/l1000_unseen_drug")
    p.add_argument("--smiles", type=Path, default=base / "unique_smiles.tsv")
    p.add_argument("--out", type=Path, default=base)
    p.add_argument("--models", nargs="+", default=["chemberta", "molformer"])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main(parse_args())
