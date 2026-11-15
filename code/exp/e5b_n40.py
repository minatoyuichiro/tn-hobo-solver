"""E5b: fairness follow-up for the unresolved n=40 instances.  Same four instances as E5
(seeds 2, 5, 7, 10), quadratized SA (M=A+1) with 10x the effort (32 x 4096 replicas, 5000
sweeps) so that the hit count is comparable to the native run; native rerun with the same
seeds for a like-for-like matched comparison."""
import json, sys
sys.path.insert(0, "..")
import numpy as np
from scipy.stats import chisquare
import instances as I
from tn import TNHOBO
from sa import SA
prev = {(r["n"], r["seed"]): r for r in json.load(open("../../results/e5_fair.json"))}
out = []
for s in (2, 5, 7, 10):
    h = I.random_kspin(40, 80, 3, s)
    tn = TNHOBO(h, method="greedy", max_repeats=16, target_size=2 ** 20, basis="natural")
    E0, N0 = tn.ground_count(); N0 = int(N0)
    seen = {}
    t = 0
    while len(seen) < N0 and t < 60:
        X = tn.sample(6.0, 20000, seed=s + t); E = h.energy(X.astype(float))
        for x in X[np.abs(E - E0) < 1e-6]:
            seen[x.tobytes()] = x
        t += 1
    assert len(seen) == N0, (len(seen), N0)
    keys = list(seen)
    rec = {"n": 40, "seed": s, "E0": E0, "N0": N0}
    for tag, hv, reps, sw in (("native", h, 8, 1000), ("quad", h.quadratize(1.0, 1.0)[0], 32, 5000)):
        counts = {k: 0 for k in keys}; hits = 0
        for rep in range(reps):
            r = SA(hv, ).run(4096 if tag == "quad" else 2048, sw, seed=1000 + 100 * rep + s)
            X = r["best_X"][:, : h.n]; E = h.energy(X.astype(float))
            for x in X[np.abs(E - E0) < 1e-6]:
                counts[x.astype(np.int8).tobytes()] += 1; hits += 1
        f = np.array([counts[k] for k in keys], float)
        p = f / max(f.sum(), 1); H = -np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(N0)
        rec[tag] = {"hits": hits, "freq": f.tolist(), "H_norm": float(H), "max_min_ratio": float(f.max() / max(f.min(), 1)),
                    "chi2_p": float(chisquare(f).pvalue) if f.min() >= 5 else float("nan"), "replicas": reps * (4096 if tag == "quad" else 2048), "sweeps": sw}
    # matched-hit comparison (native subsampled to quad hits, 1000 draws)
    rng = np.random.default_rng(0)
    fn = np.array(rec["native"]["freq"], int); k = int(sum(rec["quad"]["freq"]))
    pool = np.repeat(np.arange(N0), fn); Hs = []
    for _ in range(1000):
        c = np.bincount(rng.choice(pool, min(k, len(pool)), replace=False), minlength=N0); q = c / c.sum()
        Hs.append(-np.sum(q[q > 0] * np.log(q[q > 0])) / np.log(N0))
    rec["p_native_as_biased"] = float(np.mean(np.array(Hs) <= rec["quad"]["H_norm"]))
    rec["H_native_matched_mean"] = float(np.mean(Hs))
    out.append(rec)
    print(f"n=40 s{s} N0={N0} | native hits {rec['native']['hits']} H {rec['native']['H_norm']:.3f} | quad hits {rec['quad']['hits']} "
          f"H {rec['quad']['H_norm']:.3f} max/min {rec['quad']['max_min_ratio']:.1f} chi2 p {rec['quad']['chi2_p']:.1e} | p(native<=quad) {rec['p_native_as_biased']:.3f}", flush=True)
    json.dump(out, open("../../results/e5b_n40.json", "w"), indent=1)
