#!/usr/bin/env python3
"""UnDetect extensions on the Bai et al. 2026 drug-blind benchmark: learned view weights, neighbour-line transfer and
line sensitivity (exploratory; design and reading rules `report/11_06_record.md` 2026-10-31 02:40).

Same data, folds, backbone (seed 1729) and selection protocol as `run_sdst_hybrid.py`: every choice on valid PCC_DEG,
test scored once. All arms use the diagonally normalised fused kernel K~ (paper Proposition 1).
  pool_kta / nnk_kta  current UnDetect (closed form / neural readout); view weights from centred alignment.
  pool_ml / nnk_ml    view weights w (softmax), amplitude and two noise terms (constant + 1/n_d) maximise the
                      multi-output Gaussian-process marginal likelihood of the training compounds' mean backbone
                      residuals (genes z-scored); no validation data enter w.
  pool_val            w from a simplex grid (step 1/SIMPLEX_STEP) chosen on the valid score of the shared term:
                      does the weighting matter at all?
  pool3_resp/_ctl     pool_kta + c * h_nbr; h_nbr(d, l) = sum_{l' != l} W_ll' h_l'(d) / sum W_ll', W = exp(-dist/(h s)),
                      dist = 1 - line similarity of the training compounds' residuals (resp; train rows only, shrunk by
                      KAPPA_LINE shared compounds) or the course control distance (ctl: 1 - Pearson of line-centred
                      mean control rows); s = median off-diagonal distance; (h, a, b, c) on valid.
  pool_sens           pool_kta with the shared term scaled per line, a * t_l * g(d); t_l = the line's least-squares
                      slope of training residuals on leave-one-compound-out g, relative to the pooled slope, shrunk
                      towards 1 by kappa pseudo-rows (kappa on valid; kappa = inf gives t = 1).
  pool_all            ML weights + resp neighbour term + line sensitivity.
Writes results/sdst_hybrid/<tag>/fold_<f>/{summary.json,per_sample_test.npz[,pred_test.npz]}; score PCC_plate with
`analyze_sdst_centred.py --tag <tag> --ref pool_kta`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sdst_hybrid import D, DEV, LAMS, OUT, SCALES, VIEWS, alignment, drug_kernels, pcc_rows, train_mlp  # noqa: E402

HS_NBR = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
KAPPA_LINE = 5.0
SENS_KAPPAS = (1.0, 10.0, 100.0, float("inf"))  # inf: t = 1, so the arm nests pool_kta (user rule 2026-10-31)
SIMPLEX_STEP = 6
F64 = torch.float64


def T(a, dtype=F64) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=DEV)


def fuse(Kv: list, w) -> torch.Tensor:
    ws = w if torch.is_tensor(w) else [float(x) for x in w]  # keep the autograd graph for tensor weights
    Kf = sum(ws[i] * Kv[i] for i in range(len(Kv)))
    dg = torch.sqrt(torch.diagonal(Kf).clamp_min(1e-12))
    return Kf / dg[:, None] / dg[None, :]


EIGH_FALLBACKS = {"jitter": 0, "cpu": 0}


def eigh(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU eigh with a jitter / LAPACK retry. cusolver syevd fails on some simplex-grid kernels (jobs 356099-356101,
    356153-356155, 356629) and leaves the CUDA context unusable ("illegal memory access"), so the retry cannot recover
    there; the simplex grid therefore calls `shared(..., cpu=True)`. The KTA/ML kernels never failed."""
    try:
        return torch.linalg.eigh(M)
    except torch.linalg.LinAlgError:
        fin = bool(torch.isfinite(M).all())
        print(f"    eigh fallback: n {len(M)}, finite {fin}", flush=True)
        if not fin:
            M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        out = torch.linalg.eigh(M + 1e-8 * torch.eye(len(M), dtype=M.dtype, device=M.device)); EIGH_FALLBACKS["jitter"] += 1
        return out
    except torch.linalg.LinAlgError:
        w, V = np.linalg.eigh(M.cpu().numpy()); EIGH_FALLBACKS["cpu"] += 1
        return T(w), T(V)


def score(Y: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    return torch.cat([pcc_rows(P[b], Y[b]) for b in torch.arange(len(Y), device=DEV).split(8192)])


def ml_weights(Kv: list, tdr: np.ndarray, C: np.ndarray, nd: np.ndarray, steps: int) -> dict:
    """Maximise the GP marginal likelihood of z-scored mean residuals over view logits, amplitude and noise."""
    idx = T(tdr, torch.long); Kt = [K[idx][:, idx] for K in Kv]
    Cz = T(C); Cz = (Cz - Cz.mean(0)) / Cz.std(0).clamp_min(1e-9); G = Cz.shape[1]
    inv_nd = 1.0 / T(nd)
    th = torch.zeros(len(Kv) + 3, dtype=F64, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([th], lr=0.05); trace = []
    for it in range(steps):
        w = torch.softmax(th[:len(Kv)], 0)
        Kn = fuse(Kt, w)
        S = torch.exp(th[-3]) * Kn + torch.diag(torch.exp(th[-2]) + torch.exp(th[-1]) * inv_nd)
        L = torch.linalg.cholesky(S)
        nll = 0.5 * (Cz * torch.cholesky_solve(Cz, L)).sum() / G + torch.log(torch.diagonal(L)).sum()
        opt.zero_grad(); nll.backward(); opt.step()
        if it % 50 == 0 or it == steps - 1:
            trace.append((it, float(nll)))
    th = th.detach()
    return {"w": torch.softmax(th[:len(Kv)], 0).cpu().numpy(), "amp": float(torch.exp(th[-3])),
            "noise_const": float(torch.exp(th[-2])), "noise_per_sample": float(torch.exp(th[-1])), "nll_trace": trace}


def line_distances(ctl: np.ndarray, cell: np.ndarray, nl: int) -> np.ndarray:
    """Course control distance: 1 - Pearson of line-centred mean distinct control rows (inputs only)."""
    key = np.unique(np.array([hashlib.md5(r.tobytes()).digest() for r in ctl]), return_inverse=True)[1]
    first = np.unique(key, return_index=True)[1]; kl = cell[first]; cnt = np.bincount(kl, minlength=nl)
    Lc = np.zeros((nl, ctl.shape[1])); np.add.at(Lc, kl, ctl[first].astype(np.float64)); Lc /= np.maximum(cnt, 1)[:, None]
    Lc -= Lc[cnt > 0].mean(0); Lc -= Lc.mean(1, keepdims=True)
    Lc /= np.maximum(np.linalg.norm(Lc, axis=1, keepdims=True), 1e-12)
    return 1.0 - Lc @ Lc.T


def resp_distances(R: np.ndarray, tr: np.ndarray, di: np.ndarray, cell: np.ndarray, nl: int) -> tuple[np.ndarray, dict]:
    """1 - shrunk mean Pearson of two lines' training residuals for the same compound (train rows only)."""
    rows = np.flatnonzero(tr); Z = R[rows] - R[rows].mean(1, keepdims=True)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    d, l = di[rows], cell[rows]; num = np.zeros((nl, nl)); den = np.zeros((nl, nl))
    order = np.argsort(d, kind="stable"); b = np.flatnonzero(np.r_[True, d[order][1:] != d[order][:-1], True])
    for a0, a1 in zip(b[:-1], b[1:]):
        idx = order[a0:a1]
        if len(idx) > 1:
            li = l[idx]; np.add.at(num, (li[:, None], li[None, :]), Z[idx] @ Z[idx].T); np.add.at(den, (li[:, None], li[None, :]), 1.0)
    S = num / (den + KAPPA_LINE); np.fill_diagonal(S, 1.0)
    iu = np.triu_indices(nl, 1); m = den[iu] >= 10
    return 1.0 - S, {"pairs_ge10": int(m.sum()), "mean_sim_ge10": float(S[iu][m].mean()) if m.any() else None}


def neighbour_weights(Dl: np.ndarray, h: float) -> np.ndarray:
    """exp(-dist / (h s)) without the own line; shifted per row (normalised later), so tiny h cannot underflow."""
    iu = np.triu_indices(len(Dl), 1); s = float(np.median(Dl[iu]))
    E = Dl.copy(); np.fill_diagonal(E, np.inf)
    W = np.exp(-(E - E.min(1, keepdims=True)) / (h * s)); np.fill_diagonal(W, 0.0)
    return W


def run_fold(f: int, A: dict, Kv: list, Knp: dict, Dctl: np.ndarray, nl: int, args) -> dict:
    t0 = time.time()
    g = A["group"]; te = g == f; va = g == (f + 1) % 5; tr = ~te & ~va
    ctl, Y, di, cell = A["ctl"], A["delta"], A["drow"], A["cell_idx"]
    out = OUT / args.tag / f"fold_{f}"; out.mkdir(parents=True, exist_ok=True)
    pred_mlp, vbest, vep, last = train_mlp(ctl[tr], Y[tr], ctl[va], Y[va], args.seed, args.epochs, args.patience)
    B = pred_mlp(ctl).astype(np.float64); R = Y - B
    print(f"fold {f}: mlp valid {vbest:.4f} at epoch {vep} (ran {last}); {time.time() - t0:.0f}s", flush=True)
    tdr, tinv = np.unique(di[tr], return_inverse=True); nd = np.bincount(tinv).astype(float)
    C = np.zeros((len(tdr), Y.shape[1])); np.add.at(C, tinv, R[tr]); C /= nd[:, None]
    Ct = T(C); tdr_t = T(tdr, torch.long)
    splits = {"valid": np.flatnonzero(va), "test": np.flatnonzero(te)}
    Ys = {s: T(Y[r]) for s, r in splits.items()}; Bs = {s: T(B[r]) for s, r in splits.items()}
    res = {"fold": f, "mlp_valid_bestepoch": int(vep), "arms": {}, "weights": {}}
    per, preds = {}, {}

    def keep(name, P, info):
        per[name] = score(Ys["test"], P).cpu().numpy()
        if args.save_pred:
            preds[name] = P.cpu().numpy().astype(np.float16)
        res["arms"][name] = {**info, "test": float(per[name].mean())}
        print(f"  {name}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["arms"][name].items()})
              + f"; {time.time() - t0:.0f}s", flush=True)
    keep("mlp", Bs["test"], {"valid": float(score(Ys["valid"], Bs["valid"]).mean())})

    qidx = {s: tuple(T(x, torch.long) for x in np.unique(di[r], return_inverse=True)) for s, r in splits.items()}

    def shared(Kf, lams=LAMS, cpu=False):
        """KRR of mean residuals on train compounds; per lambda the valid/test shared terms, and the eigensystem.
        cpu=True: LAPACK eigh on the host (simplex-grid kernels, see `eigh`)."""
        M = Kf[tdr_t][:, tdr_t]
        vals, vecs = tuple(x.to(DEV) for x in torch.linalg.eigh(M.cpu())) if cpu else eigh(M)
        vals = vals.clamp_min(0); proj = vecs.T @ Ct
        Kq = {s: Kf[u][:, tdr_t] for s, (u, _) in qidx.items()}; Gs = {}
        for lam in lams:
            alpha = vecs @ (proj / (vals + lam)[:, None])
            Gs[lam] = {s: (Kq[s] @ alpha)[inv] for s, (_, inv) in qidx.items()}
        return Gs, vals, vecs

    def pick_krr(Gs):
        grid = [(float(score(Ys["valid"], Bs["valid"] + a * Gs[lam]["valid"]).mean()), lam, a) for lam in Gs for a in SCALES]
        return max(grid)

    def line_models(Kf, lam):
        models = {}
        for c in np.unique(cell[tr]):
            rc = tr & (cell == c); dl, inv = np.unique(di[rc], return_inverse=True)
            Cl = np.zeros((len(dl), Y.shape[1])); np.add.at(Cl, inv, R[rc]); Cl /= np.bincount(inv)[:, None]
            dl_t = T(dl, torch.long)
            al = torch.linalg.solve(Kf[dl_t][:, dl_t] + lam * torch.eye(len(dl), dtype=F64, device=DEV), T(Cl))
            models[int(c)] = (dl_t, al)
        return models

    def own_line(Kf, models):
        H = {}
        for s, r in splits.items():
            Hs = torch.zeros((len(r), Y.shape[1]), dtype=F64, device=DEV); cr = cell[r]
            for c, (dl_t, al) in models.items():
                sel = np.flatnonzero(cr == c)
                if len(sel):
                    Hs[T(sel, torch.long)] = Kf[T(di[r[sel]], torch.long)][:, dl_t] @ al
            H[s] = Hs
        return H

    def nbr_line(Kf, models, Wd: dict):
        """For every weighting k: sum over other lines of W[l_i, l'] h_l'(d_i), normalised over lines with a model."""
        Hn = {}
        for s, r in splits.items():
            qd, qinv = np.unique(di[r], return_inverse=True); Kq = Kf[T(qd, torch.long)]; qinv_t = T(qinv, torch.long)
            qc = T(cell[r], torch.long)
            acc = {k: torch.zeros((len(r), Y.shape[1]), dtype=torch.float32, device=DEV) for k in Wd}
            den = {k: torch.zeros(len(r), dtype=F64, device=DEV) for k in Wd}
            Wt = {k: T(W) for k, W in Wd.items()}
            for c, (dl_t, al) in models.items():
                Pr = (Kq[:, dl_t] @ al)[qinv_t]
                for k in Wd:
                    wr = Wt[k][qc, c]; acc[k] += (wr[:, None] * Pr).float(); den[k] += wr
            Hn[s] = {k: acc[k].double() / den[k].clamp_min(1e-300)[:, None] for k in Wd}
        return Hn

    def sensitivity(vals, vecs, lam, kappas=SENS_KAPPAS):
        """Per-line scale of the shared term from leave-one-compound-out KRR predictions on training rows."""
        filt = vals / (vals + lam); hd = (vecs ** 2) @ filt
        Ghat = vecs @ (filt[:, None] * (vecs.T @ Ct)); Gloo = ((Ghat - hd[:, None] * Ct) / (1 - hd)[:, None]).cpu().numpy()
        lines = np.unique(cell[tr]); num = np.zeros(nl); den = np.zeros(nl); n = np.zeros(nl)
        Gi = Gloo[tinv]; Ri = R[tr]; ct = cell[tr]
        np.add.at(num, ct, (Gi * Ri).sum(1)); np.add.at(den, ct, (Gi * Gi).sum(1)); np.add.at(n, ct, 1)
        s0 = num.sum() / den.sum(); mbar = den.sum() / n.sum(); out_ = {}
        for kp in kappas:
            t = np.ones(nl)
            if np.isinf(kp):
                out_[kp] = t; continue
            t[lines] = (num[lines] / s0 + kp * mbar) / (den[lines] + kp * mbar)
            t[lines] /= (n[lines] * t[lines]).sum() / n[lines].sum()
            out_[kp] = t
        return out_

    def scaled(Gl, t):
        return {s: Gl[s] * T(t[cell[r]])[:, None] for s, r in splits.items()}

    def best_combo(parts: dict, grids: dict):
        """Grid over named scales of named valid components; returns (valid, choice)."""
        names = list(parts); best = (-2.0, None)
        for combo in product(*[grids[n] for n in names]):
            P = Bs["valid"].clone()
            for nme, sc in zip(names, combo):
                if sc:
                    P += sc * parts[nme]
            v = float(score(Ys["valid"], P).mean())
            if v > best[0]:
                best = (v, dict(zip(names, combo)))
        return best

    def assemble(parts_test: dict, choice: dict):
        P = Bs["test"].clone()
        for nme, sc in choice.items():
            if sc:
                P += sc * parts_test[nme]
        return P

    # ---------- current UnDetect: KTA weights
    align = {v: alignment(Knp[v][np.ix_(tdr, tdr)], C) for v in VIEWS}
    w_kta = np.array([max(align[v], 0) for v in VIEWS]); w_kta = w_kta / w_kta.sum()
    res["weights"]["kta"] = dict(zip(VIEWS, map(float, w_kta)))
    kern = {"kta": fuse(Kv, w_kta)}
    # ---------- ML weights (training compounds only)
    ml = ml_weights(Kv, tdr, C, nd, args.ml_steps)
    res["weights"]["ml"] = dict(zip(VIEWS, map(float, ml["w"]))); res["ml_fit"] = {k: v for k, v in ml.items() if k != "w"}
    print(f"  weights kta {res['weights']['kta']}\n  weights ml  {res['weights']['ml']} {res['ml_fit']}; {time.time() - t0:.0f}s", flush=True)
    kern["ml"] = fuse(Kv, ml["w"])

    comp = {}
    for kname, Kf in kern.items():
        Gs, vals, vecs = shared(Kf); vk, lam, a = pick_krr(Gs)
        keep(f"krr_{kname}", Bs["test"] + a * Gs[lam]["test"], {"lam": lam, "a": a, "valid": vk})
        models = line_models(Kf, lam); H = own_line(Kf, models)
        vb, ch = best_combo({"g": Gs[lam]["valid"], "h": H["valid"]}, {"g": SCALES, "h": SCALES})
        keep(f"pool_{kname}", assemble({"g": Gs[lam]["test"], "h": H["test"]}, ch), {"lam": lam, **ch, "valid": vb})
        comp[kname] = {"G": Gs[lam], "H": H, "models": models, "lam": lam, "vals": vals, "vecs": vecs}
        # neural readout on Nystrom features of this kernel (as nnk_multi_norm)
        pos = vals > 1e-6 * vals.max(); top = torch.argsort(vals[pos], descending=True)[:args.nn_rank]
        V = vecs[:, pos][:, top] / torch.sqrt(vals[pos][top])
        phi = (Kf[:, tdr_t] @ V).float().cpu().numpy()
        pnn, vnn, epn, _ = train_mlp(phi[di[tr]], Y[tr], phi[di[va]], Y[va], args.seed, args.epochs, args.patience,
                                     hidden=512, layers=2, drop=0.3, base_tr=B[tr], base_va=B[va])
        keep(f"nnk_{kname}", Bs["test"] + T(pnn(phi[di[te]])), {"rank": int(V.shape[1]), "valid": vnn, "epoch": int(epn)})

    # ---------- validation-chosen weights (simplex grid on the shared term)
    if not args.skip_val:
        st = args.simplex_step
        grid = [np.array(c) / st for c in product(range(st + 1), repeat=len(VIEWS)) if sum(c) == st]
        best = (-2.0, None, None, None)
        for w in grid:
            Gs, _, _ = shared(fuse(Kv, w), cpu=True); vk, lam, a = pick_krr(Gs)
            if vk > best[0]:
                best = (vk, w, lam, a)
        vk, w_val, lam, a = best; res["weights"]["val"] = dict(zip(VIEWS, map(float, w_val)))
        Kf = fuse(Kv, w_val); Gs, _, _ = shared(Kf, (lam,), cpu=True)
        H = own_line(Kf, line_models(Kf, lam))
        vb, ch = best_combo({"g": Gs[lam]["valid"], "h": H["valid"]}, {"g": SCALES, "h": SCALES})
        keep("pool_val", assemble({"g": Gs[lam]["test"], "h": H["test"]}, ch),
             {"w": res["weights"]["val"], "grid_points": len(grid), "lam": lam, **ch, "valid": vb, "krr_valid": vk})

    # ---------- O2: neighbour-line transfer (kta kernel)
    Dresp, rinfo = resp_distances(R, tr, di, cell, nl); res["line_similarity"] = rinfo
    Wd = {(src, h): neighbour_weights(Dm, h) for src, Dm in (("resp", Dresp), ("ctl", Dctl)) for h in HS_NBR}
    K0 = comp["kta"]; Hn = nbr_line(kern["kta"], K0["models"], Wd)
    print(f"  neighbour terms done; {time.time() - t0:.0f}s", flush=True)
    for src in ("resp", "ctl"):
        best = (-2.0, None, None)
        for h in HS_NBR:
            vb, ch = best_combo({"g": K0["G"]["valid"], "h": K0["H"]["valid"], "n": Hn["valid"][(src, h)]},
                                {"g": SCALES, "h": SCALES, "n": SCALES})
            if vb > best[0]:
                best = (vb, h, ch)
        vb, h, ch = best
        keep(f"pool3_{src}", assemble({"g": K0["G"]["test"], "h": K0["H"]["test"], "n": Hn["test"][(src, h)]}, ch),
             {"h_bw": h, **ch, "valid": vb})
    # ---------- O3b: line sensitivity (kta kernel)
    tk = sensitivity(K0["vals"], K0["vecs"], K0["lam"]); best = (-2.0, None, None)
    for kp, t in tk.items():
        Gsc = scaled(K0["G"], t)
        vb, ch = best_combo({"g": Gsc["valid"], "h": K0["H"]["valid"]}, {"g": SCALES, "h": SCALES})
        if vb > best[0]:
            best = (vb, kp, ch)
    vb, kp, ch = best; t = tk[kp]
    res["sensitivity"] = {"kappa": kp, "t_quantiles": np.quantile(t[np.unique(cell[tr])], [0, 0.1, 0.5, 0.9, 1]).tolist()}
    keep("pool_sens", assemble({"g": scaled(K0["G"], t)["test"], "h": K0["H"]["test"]}, ch), {"kappa": kp, **ch, "valid": vb})
    # ---------- all together (ML kernel)
    K1 = comp["ml"]; Hn1 = nbr_line(kern["ml"], K1["models"], {k: v for k, v in Wd.items() if k[0] == "resp"})
    tk1 = sensitivity(K1["vals"], K1["vecs"], K1["lam"]); best = (-2.0, None)
    for kp, t in tk1.items():
        Gsc = scaled(K1["G"], t)
        for h in HS_NBR:
            vb, ch = best_combo({"g": Gsc["valid"], "h": K1["H"]["valid"], "n": Hn1["valid"][("resp", h)]},
                                {"g": SCALES, "h": SCALES, "n": SCALES})
            if vb > best[0]:
                best = (vb, {"kappa": kp, "h_bw": h, **ch})
    vb, ch = best; t = tk1[ch["kappa"]]
    keep("pool_all", assemble({"g": scaled(K1["G"], t)["test"], "h": K1["H"]["test"], "n": Hn1["test"][("resp", ch["h_bw"])]},
                              {k: ch[k] for k in ("g", "h", "n")}), {**ch, "valid": vb})

    np.savez(out / "per_sample_test.npz", pert_idx=A["pert_idx"][te], cell_idx=cell[te], **per)
    if args.save_pred:
        np.savez(out / "pred_test.npz", y=Y[te].astype(np.float16), ctl=ctl[te].astype(np.float16), **preds)
    res["eigh_fallbacks"] = dict(EIGH_FALLBACKS)
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--nn-rank", type=int, default=512)
    p.add_argument("--ml-steps", type=int, default=250)
    p.add_argument("--skip-val", action="store_true", help="skip the simplex-grid weight search")
    p.add_argument("--simplex-step", type=int, default=SIMPLEX_STEP, help="grid step 1/n (6 -> 84 points)")
    p.add_argument("--save-pred", action="store_true")
    p.add_argument("--tag", default="v7_ext")
    args = p.parse_args()
    print(vars(args), "device", DEV, flush=True)
    z = np.load(D / "sdst.npz", allow_pickle=False)
    A = {k: z[k] for k in ("ctl", "delta", "pert_idx", "cell_idx", "group")}
    A["delta"] = A["delta"].astype(np.float64)
    table = pd.read_csv(D / "drugs.tsv", sep="\t", keep_default_na=False)
    rowof = {d: i for i, d in enumerate(table.pert_idx)}
    A["drow"] = np.array([rowof[d] for d in A["pert_idx"]])
    Knp, _ = drug_kernels(table)
    Kv = [T(Knp[v]) for v in VIEWS]
    nl = int(A["cell_idx"].max()) + 1
    Dctl = line_distances(A["ctl"], A["cell_idx"], nl)
    allres = [run_fold(f, A, Kv, Knp, Dctl, nl, args) for f in args.folds]
    df = pd.DataFrame([{"fold": r["fold"], "arm": a, **{k: v for k, v in v.items() if k in ("valid", "test")}}
                       for r in allres for a, v in r["arms"].items()])
    df.to_csv(OUT / args.tag / f"folds_{'-'.join(map(str, args.folds))}.csv", index=False)
    print(df.pivot(index="arm", columns="fold", values="test").assign(mean=lambda x: x.mean(1)).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
