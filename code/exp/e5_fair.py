"""E5: is SA a fair sampler of degenerate ground states?  (a TN-only yardstick)
TN: count the ground states N0 exactly, then enumerate them by exact Boltzmann
sampling at large beta until N0 distinct ground states are seen (guaranteed
complete because N0 is known).  SA (native and quadratized, M = A+1): histogram
of which ground state each successful replica lands in.  Fairness measured by
max/min frequency ratio, the normalised entropy H/log N0, and a chi-square test
against the uniform distribution."""
import json, sys, time
sys.path.insert(0, "..")
import numpy as np
from scipy.stats import chisquare
import instances as I
from tn import TNHOBO
from sa import SA

def ground_set(h, N0, beta, seed=0):
    tn = TNHOBO(h, method="greedy", max_repeats=16, target_size=2 ** 20, basis="natural")
    E0 = tn.ground_energy()
    seen = {}
    tries = 0
    while len(seen) < N0 and tries < 50:
        X = tn.sample(beta, 20000, seed=seed + tries)
        E = h.energy(X.astype(float))
        for x in X[np.abs(E - E0) < 1e-6]:
            seen[x.tobytes()] = x
        tries += 1
    return E0, seen

out = []
cands = []
for n in (24, 32, 40):
    for s in range(40):
        h = I.random_kspin(n, 2 * n, 3, s)
        tn = TNHOBO(h, method="greedy", max_repeats=16, basis="natural")
        E0, N0 = tn.ground_count()
        if 4 <= N0 <= 64:
            cands.append((n, s, h, E0, int(N0)))
        if sum(1 for c in cands if c[0] == n) >= 4:
            break
for n, s, h, E0, N0 in cands:
    E0t, gs = ground_set(h, N0, beta=6.0, seed=s)
    assert len(gs) == N0, (len(gs), N0)
    keys = list(gs)
    rec = {"n": n, "seed": s, "E0": E0, "N0": N0}
    for tag, hv in (("native", h), ("quad", h.quadratize(1.0, 1.0)[0])):
        counts = {k: 0 for k in keys}
        hits = 0
        for rep in range(8):
            r = SA(hv).run(2048, 1000 if tag == "native" else 3000, seed=100 * rep + s)
            X = r["best_X"][:, : h.n]
            E = h.energy(X.astype(float))
            for x in X[np.abs(E - E0) < 1e-6]:
                counts[x.astype(np.int8).tobytes()] += 1
                hits += 1
        f = np.array([counts[k] for k in keys], dtype=float)
        p = f / max(f.sum(), 1)
        H = -np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(N0)
        chi = chisquare(f).pvalue if f.sum() > 5 * N0 else float("nan")
        rec[tag] = {"hits": hits, "max_min_ratio": float(f.max() / max(f.min(), 1)), "never_found": int(np.sum(f == 0)),
                    "H_norm": float(H), "chi2_p": float(chi), "freq": f.tolist()}
    out.append(rec)
    nt, qt = rec["native"], rec["quad"]
    print(f"3spin n={n} s{s} N0={N0:3d} | native hits {nt['hits']:6d} max/min {nt['max_min_ratio']:7.1f} unseen {nt['never_found']:2d} "
          f"H/logN0 {nt['H_norm']:.3f} p={nt['chi2_p']:.1e} | quad hits {qt['hits']:6d} max/min {qt['max_min_ratio']:7.1f} "
          f"unseen {qt['never_found']:2d} H/logN0 {qt['H_norm']:.3f} p={qt['chi2_p']:.1e}", flush=True)
    json.dump(out, open("../../results/e5_fair.json", "w"), indent=1)
