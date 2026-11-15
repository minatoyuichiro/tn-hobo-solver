"""E5 post-hoc: compare native vs quadratized SA ground-state bias at EQUAL hit counts.
Native hits are subsampled without replacement to the quadratized hit count
(1000 draws); the quadratized H/logN0 is compared with the native distribution."""
import json
import numpy as np
rng = np.random.default_rng(0)
data = json.load(open("../../results/e5_fair.json"))
rows = []
for r in data:
    fn = np.array(r["native"]["freq"], int); fq = np.array(r["quad"]["freq"], int)
    k = int(fq.sum()); N0 = len(fn)
    pool = np.repeat(np.arange(N0), fn)
    Hs = []
    for _ in range(1000):
        c = np.bincount(rng.choice(pool, k, replace=False), minlength=N0)
        p = c / k
        Hs.append(-np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(N0))
    Hs = np.array(Hs)
    pq = fq / k
    Hq = -np.sum(pq[pq > 0] * np.log(pq[pq > 0])) / np.log(N0)
    frac = float(np.mean(Hs <= Hq))  # how often native at equal hits looks as biased as quad
    rows.append({"n": r["n"], "seed": r["seed"], "N0": N0, "hits_quad": k, "H_quad": Hq,
                 "H_native_matched_mean": float(Hs.mean()), "H_native_matched_p05": float(np.quantile(Hs, 0.05)),
                 "p_native_as_biased": frac})
    print(f"n={r['n']} s{r['seed']:2d} N0={N0:3d} quad hits {k:5d} | H/logN0 quad {Hq:.3f} | native@same hits "
          f"{Hs.mean():.3f} (5% {np.quantile(Hs, 0.05):.3f}) | P(native <= quad) {frac:.3f}")
json.dump(rows, open("../../results/e5_matched.json", "w"), indent=1)
ps = np.array([x["p_native_as_biased"] for x in rows])
print(f"instances where quad is more biased than native at the 5% level: {int(np.sum(ps < 0.05))}/{len(ps)}")
