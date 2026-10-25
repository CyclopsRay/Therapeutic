#!/usr/bin/env python3
"""Frozen text features for PrimeKG disease nodes (integrated model; user decision 2026-10-08, item 5).

name        SapBERT-from-PubMedBERT, [CLS] of the disease name (SapBERT's entity-linking convention)
definition  PubMedBERT (BiomedBERT abstract+fulltext), attention-masked mean of "name. definition";
            the name alone when PrimeKG has no Mondo/UMLS/Orphanet definition
Input derived/e2e_kg/v1/disease_text.csv (build_e2e_kg.py); output disease_text_emb.npz next to it.
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

SAPBERT = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
PUBMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


@torch.no_grad()
def embed(texts: list[str], name: str, pooling: str, max_length: int, batch: int) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(name); model = AutoModel.from_pretrained(name).eval()
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        h = model(**enc).last_hidden_state
        if pooling == "cls":
            out.append(h[:, 0].numpy())
        else:
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).numpy())
        if s % (batch * 50) == 0:
            print(f"  {name.split('/')[-1]} {s:,}/{len(texts):,}", flush=True)
    return np.concatenate(out).astype(np.float32)


def main(a: argparse.Namespace) -> None:
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    print(f"host {platform.node()} python {sys.version.split()[0]} torch {torch.__version__} "
          f"transformers {transformers.__version__}", flush=True)
    d = pd.read_csv(a.root / "disease_text.csv")
    names = d.name.fillna("").astype(str).tolist()
    defs = [f"{n}. {x}" if isinstance(x, str) and x else n for n, x in zip(names, d.definition)]
    e_name = embed(names, SAPBERT, "cls", 32, 256)
    e_def = embed(defs, PUBMEDBERT, "mean", a.max_length, 64)
    np.savez_compressed(a.root / "disease_text_emb.npz", kg_index=d.kg_index.to_numpy(), node=d.node.to_numpy(),
                        sapbert_name=e_name, pubmedbert_def=e_def, models=np.array([SAPBERT, PUBMEDBERT]),
                        max_length=np.array([32, a.max_length]))
    print(f"{len(d):,} diseases embedded ({sum(isinstance(x, str) for x in d.definition):,} with a definition) "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/nfs-stor/yuqi.lei/Therapeutic/derived/e2e_kg/v1"))
    p.add_argument("--max-length", type=int, default=256)
    main(p.parse_args())
