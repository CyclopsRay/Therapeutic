#!/usr/bin/env python3
"""Calibration, power and the E9 proposition for the control ladder (PLAN.md Phase 6, E9; RECOMB 2027).

Design pre-declared in report/10_30_record.md (2026-10-26). Subcommands:
  pool       build null graphs for one network: degree-preserving rewires and Erdos-Renyi graphs (netnull.null)
  e9         configuration-model check: mean one-/two-hop mass over the rewire pool vs d_g D_T / 2m (and x kappa)
  simulate   synthetic responses s = [lam A_real + (1 - lam) A_G0] 1_T, G0 a rewire outside the ladder pool;
             ladder test (real vs R rewires) and naive test (real vs R ER graphs) per replicate and lam;
             --ladder rewired2k: 2K (joint-degree-preserving) ladder, paired with the 1K ladder on the same data
             (design pre-declared in report/10_30_record.md 2026-10-26 10:20)
  summarize  rejection rates with Wilson intervals

Propagation is downstream: M[g, t] = 1 when t -> g (undirected: t - g), so one-hop mass is M @ 1_T.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import netnull

OUT = Path(os.environ.get("LADDER_OUT", "/nfs-stor/yuqi.lei/Therapeutic/results/ladder_calibration"))  # smoke: _smoke dir
R_LADDER = int(os.environ.get("LADDER_R", 39))   # smoke only; production uses 39
LAMS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
# ladder rewires = seeds 0..38; G0 / surrogate draws come from seeds 40..99
G0_SEEDS = range(int(os.environ.get("LADDER_G0_START", 40)), int(os.environ.get("LADDER_G0_STOP", 100)))
N_DRUGS, N_READ, SIGMA, TRAIN = 300, 978, 1.7, 0.7


def banner(args) -> None:
    import scipy
    print(json.dumps({"host": platform.node(), "python": sys.version.split()[0], "numpy": np.__version__,
                      "scipy": scipy.__version__, "pandas": pd.__version__, "argv": sys.argv[1:],
                      "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}), flush=True)


def pool_path(net: str, kind: str, seed: int) -> Path:
    return OUT / "pool" / f"{net}_{kind}{seed}.npy"


def load_real(name: str) -> netnull.Network:
    cache = OUT / "pool" / f"{name}_real.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return netnull.Network(name, z["genes"], z["edges"], bool(z["directed"]))
    net = netnull.load(name)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, genes=net.genes, edges=net.edges, directed=net.directed)
    return net


def downstream(E: np.ndarray, n: int, directed: bool) -> sp.csr_matrix:
    src, dst = E[:, 0], E[:, 1]
    if not directed:
        src, dst = np.r_[src, dst], np.r_[dst, src]
    M = sp.csr_matrix((np.ones(len(src), np.float32), (dst, src)), shape=(n, n))
    M.data[:] = 1.0
    return M


def load_M(net: netnull.Network, kind: str, seed: int) -> sp.csr_matrix:
    E = net.edges if kind == "real" else np.load(pool_path(net.name, kind, seed))
    return downstream(E, net.n, net.directed)


# --------------------------------------------------------------------------- pool

def cmd_pool(args) -> None:
    net = load_real(args.network)
    print(netnull.summary(net), flush=True)
    real = set(map(tuple, net.edges.tolist()))
    for seed in range(args.start, args.stop):
        for kind in args.kinds:
            p = pool_path(net.name, kind, seed)
            if p.exists():
                continue
            t = time.time()
            E = netnull.null(net, kind, seed).edges.astype(np.int32)
            kept = sum(tuple(e) in real for e in E.tolist()) / len(E)
            np.save(p, E)
            print(f"{kind}{seed}: {len(E)} edges, real edges kept {kept:.4f}, {time.time() - t:.0f}s", flush=True)


# --------------------------------------------------------------------------- shared simulation pieces

def draw_targets(rng, deg_src: np.ndarray, n_drugs: int) -> sp.csc_matrix:
    cand = np.flatnonzero(deg_src > 0)
    p = (deg_src[cand] + 1.0) ** 0.5
    p /= p.sum()
    rows, cols = [], []
    for d in range(n_drugs):
        k = rng.integers(1, 5)
        t = rng.choice(cand, size=k, replace=False, p=p)
        rows += list(t); cols += [d] * k
    return sp.csc_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(deg_src), n_drugs))


def masses(M: sp.csr_matrix, T: sp.csc_matrix, read: np.ndarray):
    m1 = M @ T                                   # n x D, one-hop
    m2 = M[read] @ m1                            # read x D, two-hop
    return np.asarray(m1[read].todense()), np.asarray(m2.todense() if sp.issparse(m2) else m2)


def features(M, T, read) -> np.ndarray:
    m1, m2 = masses(M, T, read)
    return np.stack([np.log1p(m1), np.log1p(m2)], -1)    # read x D x 2


def score(X: np.ndarray, y: np.ndarray, train: np.ndarray) -> float:
    """OLS on train drugs, flattened Pearson on test drugs. X read x D x k, y read x D."""
    Xtr = X[:, train].reshape(-1, X.shape[-1]); Xte = X[:, ~train].reshape(-1, X.shape[-1])
    Xtr = np.c_[np.ones(len(Xtr)), Xtr]; Xte = np.c_[np.ones(len(Xte)), Xte]
    beta = np.linalg.lstsq(Xtr, y[:, train].ravel(), rcond=None)[0]
    pred = Xte @ beta; obs = y[:, ~train].ravel()
    if pred.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, obs)[0, 1])


# --------------------------------------------------------------------------- e9

def cmd_e9(args) -> None:
    net = load_real(args.network)
    n, directed = net.n, net.directed
    M_real = load_M(net, "real", 0)
    d_in = np.asarray(M_real.sum(1)).ravel(); d_out = np.asarray(M_real.sum(0)).ravel()
    m = d_in.sum()                                # number of (directed) adjacency entries = 2m if undirected
    rng = np.random.default_rng(0)
    T = draw_targets(rng, d_out, N_DRUGS)
    read = rng.choice(n, N_READ, replace=False)
    D_T = np.asarray(T.T @ d_out).ravel()                        # total out-degree of each drug's targets
    kappa = float((d_in * d_out).sum() / m)                       # sum_j d_in(j) d_out(j) / m
    f1 = np.outer(d_in[read], D_T) / m
    f2 = f1 * kappa
    seeds = range(args.n_rewires); kind = args.kind
    acc1 = np.zeros_like(f1); acc2 = np.zeros_like(f1); tot_err = 0.0
    for s in seeds:
        M = load_M(net, kind, s)
        tot_err = max(tot_err, float(np.abs(np.asarray((M @ T).sum(0)).ravel() - D_T).max()))
        a1, a2 = masses(M, T, read); acc1 += a1; acc2 += a2
    mean1, mean2 = acc1 / len(seeds), acc2 / len(seeds)
    r1, r2 = masses(M_real, T, read)
    tot_real = float(np.abs(np.asarray((M_real @ T).sum(0)).ravel() - D_T).max())

    def r2_of(obs, pred):
        return float(1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())

    bins = pd.qcut(np.repeat(d_in[read], N_DRUGS), 10, duplicates="drop")
    by_bin = pd.DataFrame({"bin": bins, "formula": f1.ravel(), "rewire_mean": mean1.ravel(), "real": r1.ravel()}) \
        .groupby("bin", observed=True).mean()
    by_bin["rel_err_rewire"] = by_bin.rewire_mean / by_bin.formula - 1
    by_bin["rel_err_real"] = by_bin.real / by_bin.formula - 1
    # mixing: real-edge overlap of a rewire vs rewire-rewire overlap vs the configuration-model expectation
    key = lambda E: set(map(tuple, E.tolist()))
    E0, E1 = np.load(pool_path(net.name, kind, 0)), np.load(pool_path(net.name, kind, 1))
    real_set, s0 = key(net.edges), key(E0)
    u, v = net.edges[:, 0], net.edges[:, 1]
    p_edge = np.minimum(1.0, d_out[u] * d_in[v] / m) if directed else np.minimum(1.0, d_in[u] * d_in[v] / m)
    mixing = {"real_edges_kept_in_rewired0": len(real_set & s0) / len(real_set),
              "rewired0_edges_in_rewired1": len(s0 & key(E1)) / len(s0),
              "real_edges_kept_expected_config": float(p_edge.mean())}
    res = {"network": net.name, "kind": kind, "directed": directed, "genes": n, "entries": int(m), "kappa": kappa, **mixing,
           "n_rewires": len(seeds), "max_abs_total_mass_error_rewired": tot_err,
           "max_abs_total_mass_error_real": tot_real,
           "R2_onehop_rewiremean_vs_formula": r2_of(mean1, f1), "R2_twohop_rewiremean_vs_formula": r2_of(mean2, f2),
           "R2_onehop_real_vs_formula": r2_of(r1, f1), "R2_twohop_real_vs_formula": r2_of(r2, f2),
           "R2_onehop_real_vs_rewiremean": r2_of(r1, mean1), "R2_twohop_real_vs_rewiremean": r2_of(r2, mean2)}
    d = OUT / net.name; d.mkdir(parents=True, exist_ok=True)
    sfx = "" if kind == "rewired" else f"_{kind}"
    (d / f"e9{sfx}.json").write_text(json.dumps(res, indent=1)); by_bin.to_csv(d / f"e9{sfx}_by_degree_bin.csv")
    print(json.dumps(res, indent=1)); print(by_bin.to_string(), flush=True)


# --------------------------------------------------------------------------- simulate

def cmd_simulate(args) -> None:
    net = load_real(args.network)
    n, two_k = net.n, args.ladder == "rewired2k"
    M_real = load_M(net, "real", 0)
    d_out = np.asarray(M_real.sum(0)).ravel()
    M_rw = [load_M(net, args.ladder, s) for s in range(R_LADDER)]
    # 1K: naive arm against ER graphs; 2K: paired 1K-ladder p on the same data instead (ER already measured)
    M_cmp = [load_M(net, "rewired" if two_k else "er", s) for s in range(R_LADDER)]
    rows = []
    for rep in range(args.rep_start, args.rep_stop):
        t0 = time.time()
        rng = np.random.default_rng(10_000 + rep)
        g0, sur = rng.choice(list(G0_SEEDS), 2, replace=False)
        M_g0, M_sur = load_M(net, args.ladder, int(g0)), load_M(net, args.ladder, int(sur))
        T = draw_targets(rng, d_out, N_DRUGS)
        read = rng.choice(n, N_READ, replace=False)
        train = np.zeros(N_DRUGS, bool); train[rng.permutation(N_DRUGS)[:int(TRAIN * N_DRUGS)]] = True
        noise = rng.standard_normal((N_READ, N_DRUGS))
        s0 = masses(M_g0, T, read)[0]
        X = {"real": features(M_real, T, read), "sur": features(M_sur, T, read), "g0": features(M_g0, T, read)}
        X_rw = [features(M, T, read) for M in M_rw]
        X_cmp = [features(M, T, read) for M in M_cmp]
        s_real, s_sur = masses(M_real, T, read)[0], masses(M_sur, T, read)[0]
        worlds = [("real", s_real, X["real"], s0), ("exchangeable", s_sur, X["sur"], s0)]
        if two_k:   # degree-only truth (G0 a 1K rewire) tested with the 2K ladder
            g0_1k = int(rng.choice(list(G0_SEEDS)))
            worlds.append(("real_truth1k", s_real, X["real"], masses(load_M(net, "rewired", g0_1k), T, read)[0]))
        for world, s_w, Xw, s_null in worlds:
            for lam in LAMS:
                s = lam * s_w + (1 - lam) * s_null
                y = (s - s.mean()) / (s.std() + 1e-12) + SIGMA * noise
                r = score(Xw, y, train)
                r_rw = np.array([score(Xi, y, train) for Xi in X_rw])
                r_cmp = np.array([score(Xi, y, train) for Xi in X_cmp])
                row = {"rep": rep, "world": world, "lam": lam, "r_real": r, "r_g0": score(X["g0"], y, train),
                       "r_rw_mean": r_rw.mean(), "r_rw_sd": r_rw.std(ddof=1),
                       "p_ladder": (1 + (r_rw >= r).sum()) / (R_LADDER + 1), "g0": int(g0), "sur": int(sur)}
                p_cmp = (1 + (r_cmp >= r).sum()) / (R_LADDER + 1)
                if two_k:
                    row.update({"r_rw1k_mean": r_cmp.mean(), "p_ladder1k": p_cmp})
                else:
                    row.update({"r_er_mean": r_cmp.mean(), "p_naive": p_cmp})
                rows.append(row)
        print(f"rep {rep}: {time.time() - t0:.1f}s", flush=True)
    d = OUT / net.name; d.mkdir(parents=True, exist_ok=True)
    pre = "sim2k" if two_k else "sim"
    pd.DataFrame(rows).to_csv(d / f"{pre}_{args.rep_start:03d}_{args.rep_stop:03d}.csv", index=False)


# --------------------------------------------------------------------------- summarize

def wilson(k: int, n: int, z: float = 1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def cmd_summarize(args) -> None:
    for name in args.networks:
        two_k = args.ladder == "rewired2k"
        files = sorted((OUT / name).glob("sim2k_*.csv" if two_k else "sim_*.csv"))
        df = pd.concat([pd.read_csv(f) for f in files])
        cmp, r_cmp = ("p_ladder1k", "r_rw1k_mean") if two_k else ("p_naive", "r_er_mean")
        lab = "ladder1k" if two_k else "naive"
        out = []
        for (world, lam), g in df.groupby(["world", "lam"]):
            k1, k2, nn = int((g.p_ladder <= 0.05).sum()), int((g[cmp] <= 0.05).sum()), len(g)
            out.append({"network": name, "world": world, "lam": lam, "reps": nn,
                        "reject_ladder": k1 / nn, "ladder_CI": "[%.3f, %.3f]" % wilson(k1, nn),
                        f"reject_{lab}": k2 / nn, f"{lab}_CI": "[%.3f, %.3f]" % wilson(k2, nn),
                        "r_real": g.r_real.mean(), "r_rewired": g.r_rw_mean.mean(), f"r_{lab}": g[r_cmp].mean(),
                        "r_g0": g.r_g0.mean(), "real_minus_rewired": (g.r_real - g.r_rw_mean).mean(),
                        f"rewired_minus_{lab}": (g.r_rw_mean - g[r_cmp]).mean()})
        t = pd.DataFrame(out)
        t.to_csv(OUT / name / ("summary2k.csv" if two_k else "summary.csv"), index=False)
        print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool"); p.add_argument("--network", required=True)
    p.add_argument("--start", type=int, default=0); p.add_argument("--stop", type=int, default=100)
    p.add_argument("--kinds", nargs="+", default=["rewired"])
    p = sub.add_parser("e9"); p.add_argument("--network", required=True)
    p.add_argument("--n-rewires", type=int, default=100)
    p.add_argument("--kind", default="rewired", choices=["rewired", "rewired2k"])
    p = sub.add_parser("simulate"); p.add_argument("--network", required=True)
    p.add_argument("--rep-start", type=int, default=0); p.add_argument("--rep-stop", type=int, default=200)
    p.add_argument("--ladder", default="rewired", choices=["rewired", "rewired2k"])
    p = sub.add_parser("summarize"); p.add_argument("--networks", nargs="+", required=True)
    p.add_argument("--ladder", default="rewired", choices=["rewired", "rewired2k"])
    args = ap.parse_args()
    banner(args)
    {"pool": cmd_pool, "e9": cmd_e9, "simulate": cmd_simulate, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
