#!/usr/bin/env python3
"""Drug-free MLP backbone + multi-view drug kernel on the Bai et al. 2026 drug-blind L1000 benchmark (exploratory).

Data: `prep_xpert_sdst.py` arrays (XPert L1000 SDST, 78,453 control/treated pairs, 8,276 compounds, 164 lines, 10 uM
24 h). Fold f: test = drug group f, valid = group (f+1) % 5, train = the other three (the benchmark's own split).
Target: delta = treated - control over 978 landmarks; metric PCC_DEG = mean over samples of the Pearson correlation
of true and predicted delta across genes (the benchmark's pcc_deg_mean).

Arms, all fitted on train rows only; every choice made on valid PCC_DEG; test scored once per arm:
  mlp       the benchmark's drug-free MLP (978 -> 3 x 2048 -> 978, loss 0.5 MSE + 0.5 (1 - PCC_DEG), Adam 1e-4,
            batch 512, early stop on valid PCC_DEG, patience 50), our own run, seed 1729.
  krr_<v>   mlp + a * g(d): g = KRR over train drugs of the drug's mean MLP residual (across all its lines), with drug
            kernel v (chem: Morgan-2 Tanimoto; multi: KTA-weighted chem + targets + MOA + Cell Painting; _norm:
            divided by sqrt(K_ii K_jj) so compounds missing views keep unit self-similarity). (lambda, a) on valid.
  pool_<v>  krr_<v> + b * h_L(d): h_L = per-line KRR of the line's residuals with the same kernel. (a, b) on valid.
  nnk_<v>   mlp + MLP(phi(d)), phi = Nystrom features of the drug kernel over train drugs (neural drug head, trained on
            train residuals, early stop on valid).
  pool_plate (--plate-control) per-line KRR with a plate-overlap drug kernel: the batch-effect null for pool_<v>.
  pool_plate_drug (--plate-drug) per-line kernel = plate overlap + gamma * multi_norm drug kernel; minus pool_plate
            = drug similarity beyond plate structure.
  --viewsets  which drug kernels to run (default chem, multi, multi_norm); view ablations, all normalised:
            only_<v> (one view), minus_<v> (multi without v), perm_<v> (multi with v's compound labels permuted among
            the compounds that have v, so coverage and kernel marginals are kept; --perm-seed).
  --input-mlp same information as features instead of a kernel (Morgan bits, target and MOA multi-hot, z-scored Cell
            Painting, three availability flags): mlp_input = the benchmark MLP on [control, features], trained end to
            end; nnf_multi = the nnk head architecture on the features, on top of the fixed mlp. Small grids on valid.
  --drug-frac learning curve: keep this fraction of the fold's train compounds (all their samples, every arm including
            the backbone; --frac-seed + fold); valid and test unchanged.
  --backbone-cache TAG  take the backbone B for all rows from results/sdst_hybrid/TAG/fold_<f>/cache.npz
            (`cache_undetect.py`) instead of training it, so every arm sits on that run's backbone.
  --input-arms  which --input-mlp arms to fit (default both).
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
D = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/xpert_sdst/v1")
CP = Path("/nfs-stor/yuqi.lei/Therapeutic/derived/l1000_unseen_drug/cell_painting_features.npz")
OUT = Path("/nfs-stor/yuqi.lei/Therapeutic/results/sdst_hybrid")
LAMS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
GAMMAS = (0.25, 1.0, 4.0)  # weight of the drug kernel added to the plate kernel (--plate-drug)
VIEWS = ["chem", "tgt", "moa", "morph"]
VIEWSETS = {"chem": (["chem"], False), "multi": (VIEWS, False), "multi_norm": (VIEWS, True)}
INPUT_GRID = {"mlp_input": [dict(drop=0.1), dict(drop=0.3)],
              "nnf_multi": [dict(hidden=512, drop=0.3), dict(hidden=2048, drop=0.3), dict(hidden=512, drop=0.5)]}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pcc_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / torch.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def tanimoto(F: np.ndarray) -> np.ndarray:
    F = torch.from_numpy(F.astype(np.float32)).to(DEV); inter = F @ F.T; n = F.sum(1)
    union = n[:, None] + n[None, :] - inter
    return torch.where(union > 0, inter / union.clamp_min(1e-9), torch.zeros_like(inter)).double().cpu().numpy()


def viewset(name: str) -> tuple[list, bool]:
    if name in VIEWSETS:
        return VIEWSETS[name]
    kind, v = name.split("_", 1)
    assert v in VIEWS, name
    return {"only": [v], "minus": [x for x in VIEWS if x != v],
            "perm": [x + "_perm" if x == v else x for x in VIEWS]}[kind], True


def drug_kernels(table: pd.DataFrame) -> tuple[dict, dict]:
    """Per-view kernels (zero rows/cols where a view is missing) and the raw per-view features + availability."""
    n = len(table); gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    F = np.zeros((n, 2048), np.uint8); ok = np.zeros(n, bool)
    for i, s in enumerate(table.canonical_smiles):
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None:
            DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), F[i]); ok[i] = True
    K = {"chem": tanimoto(F)}; has = {"chem": ok}; feats = {"chem": F}
    for v, col in (("tgt", "target"), ("moa", "moa")):
        sets = [set(x.split(";")) - {""} for x in table[col]]
        vocab = {x: j for j, x in enumerate(sorted(set().union(*sets)))}
        M = np.zeros((n, len(vocab)), np.uint8)
        for i, s in enumerate(sets):
            M[i, [vocab[x] for x in s]] = 1
        K[v] = tanimoto(M); has[v] = M.any(1); feats[v] = M
    cp = np.load(CP, allow_pickle=False)
    pos = {k[:14]: i for i, k in enumerate(cp["inchikey"].astype(str))}
    idx = np.array([pos.get(k[:14], -1) if k else -1 for k in table.inchi_key])
    has["morph"] = idx >= 0
    Fm = np.zeros((n, cp["morph10"].shape[1])); Fm[has["morph"]] = cp["morph10"][idx[has["morph"]]]
    Fm[has["morph"]] = (Fm[has["morph"]] - Fm[has["morph"]].mean(0)) / np.maximum(Fm[has["morph"]].std(0), 1e-9)
    feats["morph"] = Fm.copy()
    Fm /= np.maximum(np.linalg.norm(Fm, axis=1, keepdims=True), 1e-12)
    K["morph"] = Fm @ Fm.T
    for v in K:
        K[v][~has[v]] = 0; K[v][:, ~has[v]] = 0
    print("view coverage (drugs):", {v: int(h.sum()) for v, h in has.items()},
          "feature dims:", {v: x.shape[1] for v, x in feats.items()}, flush=True)
    feats["has"] = has
    return K, feats


def permute_view(K: dict, has: dict, v: str, seed: int) -> None:
    """K[v + '_perm']: v's kernel with compound labels permuted among the compounds that have v."""
    idx = np.flatnonzero(has[v]); src = np.random.default_rng(seed).permutation(idx)
    Kp = np.zeros_like(K[v]); Kp[np.ix_(idx, idx)] = K[v][np.ix_(src, src)]; K[v + "_perm"] = Kp


def alignment(K: np.ndarray, R: np.ndarray) -> float:
    Kc = K - K.mean(0)[None] - K.mean(1)[:, None] + K.mean(); Rc = R - R.mean(0)
    den = np.linalg.norm(Kc) * np.linalg.norm(Rc.T @ Rc)
    return float(np.einsum("ij,ij->", Rc.T @ Kc, Rc.T) / den) if den > 0 else 0.0


def train_mlp(Xtr, Ytr, Xva, Yva, seed: int, max_epochs: int, patience: int, hidden=2048, layers=3, drop=0.1,
              base_tr=None, base_va=None, lr=1e-4):
    """Benchmark MLP recipe. With base_* given, it learns a residual on top of a fixed prediction (drug head)."""
    torch.manual_seed(seed); np.random.seed(seed)
    dims = [Xtr.shape[1]] + [hidden] * layers
    net = torch.nn.Sequential(*[m for i in range(layers) for m in (torch.nn.Linear(dims[i], dims[i + 1]),
                                                                    torch.nn.ReLU(), torch.nn.Dropout(drop))],
                              torch.nn.Linear(hidden, Ytr.shape[1])).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xtr, Ytr, Xva, Yva = (torch.from_numpy(a).float().to(DEV) for a in (Xtr, Ytr, Xva, Yva))
    btr = torch.zeros_like(Ytr) if base_tr is None else torch.from_numpy(base_tr).float().to(DEV)
    bva = torch.zeros_like(Yva) if base_va is None else torch.from_numpy(base_va).float().to(DEV)
    best, best_ep, best_state, wait = -2.0, 0, None, 0
    for ep in range(1, max_epochs + 1):
        net.train()
        for b in torch.randperm(len(Ytr), device=DEV).split(512):
            p = btr[b] + net(Xtr[b])
            loss = 0.5 * ((p - Ytr[b]) ** 2).mean() + 0.5 * (1 - pcc_rows(p, Ytr[b]).mean())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(torch.cat([pcc_rows(bva[b] + net(Xva[b]), Yva[b]) for b in torch.arange(len(Yva), device=DEV).split(4096)]).mean())
        if v > best:
            best, best_ep, wait = v, ep, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state); net.eval()

    def predict(X):
        X = torch.from_numpy(X).float().to(DEV)
        with torch.no_grad():
            return torch.cat([net(X[b]) for b in torch.arange(len(X), device=DEV).split(4096)]).cpu().numpy()
    return predict, best, best_ep, ep


def score(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    return pcc_rows(torch.from_numpy(Y).to(DEV), torch.from_numpy(P).to(DEV)).cpu().numpy()


def run_fold(f: int, A: dict, K: dict, feats: dict, args) -> dict:
    t0 = time.time()
    g = A["group"]; te = g == f; va = g == (f + 1) % 5; tr = ~te & ~va
    ctl, Y, di, cell = A["ctl"], A["delta"], A["drow"], A["cell_idx"]
    if args.drug_frac < 1:
        pool = np.unique(di[tr]); rng = np.random.default_rng(args.frac_seed + f)
        tr &= np.isin(di, rng.choice(pool, int(round(args.drug_frac * len(pool))), replace=False))
        print(f"fold {f}: --drug-frac {args.drug_frac}: {len(np.unique(di[tr]))} of {len(pool)} train compounds, "
              f"{int(tr.sum())} rows", flush=True)
    out = OUT / args.tag / f"fold_{f}"; out.mkdir(parents=True, exist_ok=True)
    if args.backbone_cache:
        B = np.load(OUT / args.backbone_cache / f"fold_{f}" / "cache.npz")["B"]
        vbest, vep, last = float(score(Y[va], B[va]).mean()), -1, -1
    else:
        pred_mlp, vbest, vep, last = train_mlp(ctl[tr], Y[tr], ctl[va], Y[va], args.seed, args.epochs, args.patience)
        B = pred_mlp(ctl)
    print(f"fold {f}: mlp valid {vbest:.4f} at epoch {vep} (ran {last}); {time.time() - t0:.0f}s", flush=True)
    R = Y - B
    tdr = np.unique(di[tr]); tpos = {d: i for i, d in enumerate(tdr)}
    C = np.zeros((len(tdr), Y.shape[1])); np.add.at(C, np.array([tpos[d] for d in di[tr]]), R[tr])
    C /= np.bincount([tpos[d] for d in di[tr]], minlength=len(tdr))[:, None]
    align = {v: alignment(K[v][np.ix_(tdr, tdr)], C) for v in K}
    print(f"  alignment {({v: round(a, 4) for v, a in align.items()})}", flush=True)

    def kernel(vs):
        views, norm = viewset(vs)
        w = np.array([max(align[v], 0) for v in views]); w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
        Kf = sum(wi * K[v] for wi, v in zip(w, views))
        if norm:
            dg = np.sqrt(np.clip(np.diag(Kf), 1e-12, None)); Kf = Kf / dg[:, None] / dg[None, :]
        return Kf

    res = {"fold": f, "mlp_valid_bestepoch": vep, "alignment": align, "arms": {}}
    rows = {"valid": va, "test": te}
    per = {}; preds = {}

    def keep(name, P):  # per-sample test PCC; with --save-pred also the float16 test prediction
        if args.save_pred:
            preds[name] = P.astype(np.float16)
        return score(Y[te], P)
    per["mlp"] = keep("mlp", B[te])
    res["arms"]["mlp"] = {"valid": float(score(Y[va], B[va]).mean()), "test": float(per["mlp"].mean())}
    for vs in args.viewsets:
        Kf = kernel(vs); Ktt = Kf[np.ix_(tdr, tdr)]
        vals, vecs = np.linalg.eigh(Ktt); vals = np.clip(vals, 0, None); proj = vecs.T @ C
        udr = {s: np.unique(di[m], return_inverse=True) for s, m in rows.items()}  # predict per drug, index per row
        Kq = {s: Kf[np.ix_(u, tdr)] for s, (u, _) in udr.items()}
        grid = []
        for lam in LAMS:
            alpha = vecs @ (proj / (vals + lam)[:, None])
            G = (Kq["valid"] @ alpha)[udr["valid"][1]]
            for a in SCALES:
                grid.append((float(score(Y[va], B[va] + a * G).mean()), lam, a))
        vbest, lam, a = max(grid)
        alpha = vecs @ (proj / (vals + lam)[:, None])
        Gall = {s: (Kq[s] @ alpha)[udr[s][1]] for s in rows}
        per[f"krr_{vs}"] = keep(f"krr_{vs}", B[te] + a * Gall["test"])
        res["arms"][f"krr_{vs}"] = {"lam": lam, "a": a, "valid": vbest, "test": float(per[f"krr_{vs}"].mean())}
        # per-line term with the same kernel and lambda; residual rows averaged per (line, drug)
        H = {s: np.zeros((m.sum(), Y.shape[1])) for s, m in rows.items()}
        ridx = {s: np.flatnonzero(m) for s, m in rows.items()}
        for c in np.unique(cell[tr]):
            rc = tr & (cell == c); dl = np.unique(di[rc]); lp = {d: i for i, d in enumerate(dl)}
            Cl = np.zeros((len(dl), Y.shape[1])); np.add.at(Cl, np.array([lp[d] for d in di[rc]]), R[rc])
            Cl /= np.bincount([lp[d] for d in di[rc]], minlength=len(dl))[:, None]
            al = np.linalg.solve(Kf[np.ix_(dl, dl)] + lam * np.eye(len(dl)), Cl)
            for s in rows:
                sel = cell[ridx[s]] == c
                if sel.any():
                    H[s][sel] = Kf[np.ix_(di[ridx[s][sel]], dl)] @ al
        grid = [(float(score(Y[va], B[va] + a2 * Gall["valid"] + b2 * H["valid"]).mean()), a2, b2)
                for a2, b2 in product(SCALES, SCALES)]
        vb2, a2, b2 = max(grid)
        per[f"pool_{vs}"] = keep(f"pool_{vs}", B[te] + a2 * Gall["test"] + b2 * H["test"])
        res["arms"][f"pool_{vs}"] = {"lam": lam, "a": a2, "b": b2, "valid": vb2, "test": float(per[f"pool_{vs}"].mean())}
        print(f"  {vs}: krr lam {lam} a {a} valid {vbest:.4f} test {res['arms'][f'krr_{vs}']['test']:.4f} | "
              f"pool a {a2} b {b2} valid {vb2:.4f} test {res['arms'][f'pool_{vs}']['test']:.4f}; {time.time() - t0:.0f}s", flush=True)
        if vs in args.nn_views:
            # neural drug head on Nystrom features of the kernel over train drugs
            pos = vals > 1e-6 * vals.max(); top = np.argsort(vals[pos])[::-1][:args.nn_rank]
            V = vecs[:, pos][:, top] / np.sqrt(vals[pos][top])
            phi = Kf[:, tdr] @ V  # every drug's coordinates in the train-drug kernel space
            pnn, vnn, epn, _ = train_mlp(phi[di[tr]], Y[tr], phi[di[va]], Y[va], args.seed, args.epochs, args.patience,
                                         hidden=512, layers=2, drop=0.3, base_tr=B[tr], base_va=B[va])
            per[f"nnk_{vs}"] = keep(f"nnk_{vs}", B[te] + pnn(phi[di[te]]))
            res["arms"][f"nnk_{vs}"] = {"rank": int(V.shape[1]), "valid": vnn, "epoch": epn, "test": float(per[f"nnk_{vs}"].mean())}
            print(f"  nnk_{vs}: valid {vnn:.4f} at epoch {epn} test {res['arms'][f'nnk_{vs}']['test']:.4f}; {time.time() - t0:.0f}s", flush=True)
    if args.input_mlp:
        has = feats["has"]
        Fd = np.hstack([feats[v] for v in VIEWS] + [np.stack([has[v] for v in VIEWS[1:]], 1)]).astype(np.float32)
        for arm, cfgs in INPUT_GRID.items():
            if arm not in args.input_arms:
                continue
            fits = []
            for cfg in cfgs:
                if arm == "mlp_input":
                    fit = train_mlp(np.hstack([ctl[tr], Fd[di[tr]]]), Y[tr], np.hstack([ctl[va], Fd[di[va]]]), Y[va],
                                    args.seed, args.epochs, args.patience, **cfg)
                else:
                    fit = train_mlp(Fd[di[tr]], Y[tr], Fd[di[va]], Y[va], args.seed, args.epochs, args.patience,
                                    layers=2, base_tr=B[tr], base_va=B[va], **cfg)
                fits.append((fit[1], cfg, fit))
                print(f"  {arm} {cfg}: valid {fit[1]:.4f} at epoch {fit[2]}; {time.time() - t0:.0f}s", flush=True)
            vbest, cfg, (pred, _, ep, _) = max(fits, key=lambda x: x[0])
            Pt = pred(np.hstack([ctl[te], Fd[di[te]]])) if arm == "mlp_input" else B[te] + pred(Fd[di[te]])
            per[arm] = keep(arm, Pt)
            res["arms"][arm] = {**cfg, "epoch": ep, "valid": vbest, "test": float(per[arm].mean()),
                                "grid": [{**c, "valid": v} for v, c, _ in fits]}
            print(f"  {arm}: chose {cfg} valid {vbest:.4f} test {res['arms'][arm]['test']:.4f}", flush=True)
    if "S" in A:
        # plate-confound control: per-line KRR whose drug kernel is the Jaccard overlap of the two drugs' detection
        # plates in that line (metadata only; no structure or pharmacology). If it matches pool_<v>, the drug term
        # is a batch effect.
        Hp = {lam: {s: np.zeros((m.sum(), Y.shape[1])) for s, m in rows.items()} for lam in LAMS}
        # --plate-drug: per-line kernel J + gamma * K_drug(multi_norm); does drug similarity add beyond plates?
        Kpd = kernel("multi_norm") if args.plate_drug else None
        Hpd = {(gm, lam): {s: np.zeros((m.sum(), Y.shape[1])) for s, m in rows.items()}
               for gm in GAMMAS for lam in LAMS} if args.plate_drug else {}
        for c in np.unique(cell[tr]):
            allc = np.flatnonzero(cell == c); dall, inv = np.unique(di[allc], return_inverse=True)
            P = sp.csr_matrix((np.ones(len(allc)), (inv, np.arange(len(allc)))), shape=(len(dall), len(allc))) @ A["S"][allc]
            P = (P > 0).astype(np.float64); inter = (P @ P.T).toarray(); cnt = np.asarray(P.sum(1)).ravel()
            J = inter / np.maximum(cnt[:, None] + cnt[None, :] - inter, 1e-9)
            loc = {d: i for i, d in enumerate(dall)}
            rc = tr & (cell == c); dl = np.unique(di[rc]); lp = {d: i for i, d in enumerate(dl)}
            Cl = np.zeros((len(dl), Y.shape[1])); np.add.at(Cl, np.array([lp[d] for d in di[rc]]), R[rc])
            Cl /= np.bincount([lp[d] for d in di[rc]], minlength=len(dl))[:, None]
            li = np.array([loc[d] for d in dl]); w, U = np.linalg.eigh(J[np.ix_(li, li)]); w = np.clip(w, 0, None); pr = U.T @ Cl
            for s in rows:
                q = np.flatnonzero(cell[ridx[s]] == c)
                if len(q):
                    Jq = J[np.ix_([loc[d] for d in di[ridx[s][q]]], li)]
                    for lam in LAMS:
                        Hp[lam][s][q] = Jq @ (U @ (pr / (w + lam)[:, None]))
            if Kpd is not None:
                gl = dall[li]
                for gm in GAMMAS:
                    w2, U2 = np.linalg.eigh(J[np.ix_(li, li)] + gm * Kpd[np.ix_(gl, gl)]); w2 = np.clip(w2, 0, None); pr2 = U2.T @ Cl
                    for s in rows:
                        q = np.flatnonzero(cell[ridx[s]] == c)
                        if len(q):
                            qd = di[ridx[s][q]]
                            Kq2 = J[np.ix_([loc[d] for d in qd], li)] + gm * Kpd[np.ix_(qd, gl)]
                            for lam in LAMS:
                                Hpd[(gm, lam)][s][q] = Kq2 @ (U2 @ (pr2 / (w2 + lam)[:, None]))
        grid = [(float(score(Y[va], B[va] + b2 * Hp[lam]["valid"]).mean()), lam, b2) for lam in LAMS for b2 in SCALES]
        vb2, lam, b2 = max(grid)
        per["pool_plate"] = keep("pool_plate", B[te] + b2 * Hp[lam]["test"])
        res["arms"]["pool_plate"] = {"lam": lam, "b": b2, "valid": vb2, "test": float(per["pool_plate"].mean())}
        print(f"  pool_plate: lam {lam} b {b2} valid {vb2:.4f} test {res['arms']['pool_plate']['test']:.4f}; {time.time() - t0:.0f}s", flush=True)
        if Hpd:
            grid = [(float(score(Y[va], B[va] + b2 * Hpd[k]["valid"]).mean()), k, b2) for k in Hpd for b2 in SCALES]
            vb3, (gm, lam3), b3 = max(grid)
            per["pool_plate_drug"] = keep("pool_plate_drug", B[te] + b3 * Hpd[(gm, lam3)]["test"])
            res["arms"]["pool_plate_drug"] = {"gamma": gm, "lam": lam3, "b": b3, "valid": vb3,
                                              "test": float(per["pool_plate_drug"].mean())}
            print(f"  pool_plate_drug: gamma {gm} lam {lam3} b {b3} valid {vb3:.4f} "
                  f"test {res['arms']['pool_plate_drug']['test']:.4f}; {time.time() - t0:.0f}s", flush=True)
    np.savez(out / "per_sample_test.npz", pert_idx=A["pert_idx"][te], cell_idx=cell[te], **per)
    if args.save_pred:
        np.savez(out / "pred_test.npz", y=Y[te].astype(np.float16), ctl=ctl[te].astype(np.float16), **preds)
    (out / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--nn-views", nargs="*", default=["multi_norm"])
    p.add_argument("--nn-rank", type=int, default=512)
    p.add_argument("--plate-control", action="store_true")
    p.add_argument("--save-pred", action="store_true", help="also write float16 test predictions per arm")
    p.add_argument("--plate-drug", action="store_true", help="with --plate-control: add the pool_plate_drug arm")
    p.add_argument("--viewsets", nargs="+", default=list(VIEWSETS))
    p.add_argument("--perm-seed", type=int, default=0)
    p.add_argument("--input-mlp", action="store_true", help="add the same-input mlp_input and nnf_multi arms")
    p.add_argument("--drug-frac", type=float, default=1.0)
    p.add_argument("--frac-seed", type=int, default=0)
    p.add_argument("--backbone-cache", default=None)
    p.add_argument("--input-arms", nargs="+", default=list(INPUT_GRID), choices=list(INPUT_GRID))
    p.add_argument("--tag", default="v1")
    args = p.parse_args()
    print(vars(args), "device", DEV, flush=True)
    z = np.load(D / "sdst.npz", allow_pickle=False)
    A = {k: z[k] for k in ("ctl", "delta", "pert_idx", "cell_idx", "group")}
    table = pd.read_csv(D / "drugs.tsv", sep="\t", keep_default_na=False)
    rowof = {d: i for i, d in enumerate(table.pert_idx)}
    A["drow"] = np.array([rowof[d] for d in A["pert_idx"]])
    if args.plate_control:
        pz = np.load(D / "plates.npz", allow_pickle=False)
        A["S"] = sp.csr_matrix((np.ones(len(pz["plate_row"])), (pz["plate_row"], pz["plate_col"])),
                               shape=(len(A["pert_idx"]), int(pz["n_plates"])))
    K, feats = drug_kernels(table)
    for v in sorted({vs.split("_", 1)[1] for vs in args.viewsets if vs.startswith("perm_")}):
        permute_view(K, feats["has"], v, args.perm_seed)
    allres = [run_fold(f, A, K, feats, args) for f in args.folds]
    df = pd.DataFrame([{"fold": r["fold"], "arm": a, **v} for r in allres for a, v in r["arms"].items()])
    df.to_csv(OUT / args.tag / f"folds_{'-'.join(map(str, args.folds))}.csv", index=False)
    print(df.pivot(index="arm", columns="fold", values="test").assign(mean=lambda x: x.mean(1)).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
