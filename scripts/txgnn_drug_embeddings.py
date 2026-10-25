#!/usr/bin/env python3
"""TxGNN drug-node embeddings for the L1000 unseen-drug benchmark (record 2026-10-07 13:40).

TxGNN (mims-harvard/TxGNN @ 0e0203c) feeds each node a frozen random input `inp` drawn in
model_initialize() without a seed; save_model() stores only the layer weights (the graph save is
commented out upstream). Two modes:

  ckpt   load a released checkpoint and test whether its drug embeddings survive a re-draw of the
         inputs: extract twice under two torch seeds and report the per-drug cosine between draws.
         Embeddings are written only with --accept-ckpt, after that check is read.
  train  the authors' recipe (reproduce/train.py: n_hid = n_inp = n_out = 100, proto, proto_num 3,
         all_nodes_profile, rarity; pretrain 1 epoch lr 1e-3 batch 1024; finetune 500 epochs
         lr 5e-4, validation every 20) on split='full_graph' with every RNG seeded. Saves the model,
         its inputs, and the best-validation model's embeddings.

Output npz: drug_idx, drugbank_id, inchikey ('' when unmapped), embedding (n_drugs x n_out).
Embeddings are the two-layer GNN output (forward(return_h=True)), before the disease-prototype step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    import dgl
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)


def drug_embedding(tx, model) -> np.ndarray:
    model.eval()
    G = tx.G.to(tx.device)
    with torch.no_grad():
        h = model(G, G, return_h=True)
    return h["drug"].detach().cpu().numpy().astype(np.float32)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def write_embeddings(tx, E: np.ndarray, map_path: Path, out: Path, manifest: dict) -> None:
    idx2id = tx.retrieve_id_mapping()["idx2id_drug"]
    idx = np.arange(E.shape[0])
    dbid = np.array([str(idx2id.get(float(i), idx2id.get(i, ""))) for i in idx])
    m = pd.read_csv(map_path, sep="\t", dtype=str).dropna(subset=["inchikey"]).drop_duplicates("drugbank_id")
    lookup = dict(zip(m.drugbank_id, m.inchikey))
    ik = np.array([lookup.get(d, "") for d in dbid])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, drug_idx=idx, drugbank_id=dbid, inchikey=ik, embedding=E)
    manifest.update({"n_drug_nodes": int(E.shape[0]), "dim": int(E.shape[1]),
                     "n_with_drugbank_id": int((dbid != "").sum()), "n_with_inchikey": int((ik != "").sum()),
                     "map_sha256": sha256(map_path), "out": str(out), "out_sha256": sha256(out)})
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def main(args: argparse.Namespace) -> None:
    import dgl
    from txgnn import TxData, TxGNN
    started = time.time()
    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"host {platform.node()} python {sys.version.split()[0]} torch {torch.__version__} "
          f"cuda {torch.version.cuda} dgl {dgl.__version__} device {device} mode {args.mode} "
          f"split {args.split} seed {args.seed}", flush=True)
    if args.require_gpu and device == "cpu":
        raise SystemExit("--require-gpu: torch.cuda.is_available() is false")
    for f in ("kg.csv", "node.csv", "edges.csv"):
        if not (args.data / f).exists():
            raise FileNotFoundError(f"{args.data / f} missing (TxData would download it; fetch into raw/ first)")
    manifest = {"mode": args.mode, "split": args.split, "seed": args.seed, "device": device,
                "torch": torch.__version__, "dgl": dgl.__version__, "data": str(args.data),
                "kg_sha256": sha256(args.data / "kg.csv"), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}

    seed_all(args.seed)
    data = TxData(data_folder_path=str(args.data))
    data.prepare_split(split=args.split, seed=args.seed)
    tx = TxGNN(data=data, weight_bias_track=False, proj_name="TxGNN", exp_name="therapeutic", device=device)

    if args.mode == "ckpt":
        draws = []
        for s in (args.seed, args.seed + 1):
            seed_all(s)
            tx.load_pretrained(str(args.ckpt))  # re-runs initialize_node_embedding under seed s
            draws.append(drug_embedding(tx, tx.model))
        cos = cosine_rows(draws[0], draws[1])
        q = np.quantile(cos, [0.05, 0.5, 0.95])
        print(f"ckpt input re-draw check: per-drug cosine between two input draws "
              f"median {q[1]:.4f} (5% {q[0]:.4f}, 95% {q[2]:.4f}); ~1 means the inputs do not matter", flush=True)
        manifest.update({"ckpt": str(args.ckpt), "ckpt_model_sha256": sha256(args.ckpt / "model.pt"),
                         "redraw_cosine_q05_q50_q95": [float(x) for x in q]})
        if not args.accept_ckpt:
            (args.out.parent).mkdir(parents=True, exist_ok=True)
            args.out.with_suffix(".ckpt_check.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print("not writing embeddings (pass --accept-ckpt after reading the check)", flush=True)
            return
        write_embeddings(tx, draws[0], args.map, args.out, manifest)
        return

    seed_all(args.seed)
    tx.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3, attention=False,
                        sim_measure="all_nodes_profile", agg_measure="rarity")
    print(f"initialised ({time.time() - started:.0f}s)", flush=True)
    tx.pretrain(n_epoch=args.pretrain_epochs, learning_rate=1e-3, batch_size=1024, train_print_per_n=200)
    print(f"pretrained ({time.time() - started:.0f}s)", flush=True)
    tx.finetune(n_epoch=args.finetune_epochs, learning_rate=5e-4, train_print_per_n=20, valid_per_n=20)
    print(f"finetuned ({time.time() - started:.0f}s)", flush=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    tx.save_model(str(args.model_dir))
    G = tx.G.to("cpu")
    torch.save({nt: G.nodes[nt].data["inp"].detach().cpu() for nt in G.ntypes}, args.model_dir / "inputs.pt")
    manifest.update({"pretrain_epochs": args.pretrain_epochs, "finetune_epochs": args.finetune_epochs,
                     "model_dir": str(args.model_dir), "model_sha256": sha256(args.model_dir / "model.pt"),
                     "inputs_sha256": sha256(args.model_dir / "inputs.pt"),
                     "seconds_training": round(time.time() - started, 1)})
    write_embeddings(tx, drug_embedding(tx, tx.best_model), args.map, args.out, manifest)


def parse_args() -> argparse.Namespace:
    base = Path("/nfs-stor/yuqi.lei/Therapeutic")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["ckpt", "train"], required=True)
    p.add_argument("--data", type=Path, default=base / "derived/txgnn/data",
                   help="TxData folder (symlinks to raw/txgnn/primekg; TxGNN writes splits here)")
    p.add_argument("--split", default="full_graph")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", type=Path, default=base / "raw/txgnn/ckpt")
    p.add_argument("--accept-ckpt", action="store_true")
    p.add_argument("--pretrain-epochs", type=int, default=1)
    p.add_argument("--finetune-epochs", type=int, default=500)
    p.add_argument("--model-dir", type=Path, default=base / "derived/txgnn/model_full_graph_seed42")
    p.add_argument("--map", type=Path, default=base / "derived/l1000_unseen_drug/primekg_drugbank_pubchem_inchikey.tsv")
    p.add_argument("--out", type=Path, default=base / "derived/l1000_unseen_drug/txgnn_drug_embedding.npz")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--require-gpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
