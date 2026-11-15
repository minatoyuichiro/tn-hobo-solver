"""E3: what quadratization does to the landscape, measured exactly with the TN.
Native HOBO vs Rosenberg-quadratized (M = A+1, tightest exact integer; and M = 1.5A):
contraction width, low-energy spectrum (counts at E0, E0+1, ...), exact equilibrium
ground-state probability P0(beta) = N0 exp(-beta E0) / Z(beta) at the schedule's cold
beta, and SA success at equal sweeps."""
import json, sys, time
sys.path.insert(0, "..")
import numpy as np
import instances as I
from tn import TNHOBO, factor_tables, optimise_tree, tree_stats
from sa import SA

K = 12
FAM = {"3spin_m2n": (lambda n, s: I.random_kspin(n, 2 * n, 3, s), [16, 24, 32, 40]),
       "3SAT_4.26": (lambda n, s: I.random_ksat(n, int(round(4.26 * n)), 3, s), [16, 24, 32, 40]),
       "LABS": (lambda n, s: I.labs(n), [10, 13, 16, 19])}
out = []
for fam, (gen, sizes) in [(f, FAM[f]) for f in (sys.argv[1:] or list(FAM))]:
    for n in sizes:
        for s in ([0] if fam == "LABS" else [0, 1, 2]):
            h = gen(n, s)
            rec = {"family": fam, "size": n, "seed": s}
            variants = {"native": h}
            for tag, (sc, ad) in {"quad_A+1": (1.0, 1.0), "quad_1.5A": (1.5, 0.0)}.items():
                q, aux = h.quadratize(sc, ad)
                variants[tag] = q
                rec[tag + "_aux"] = len(aux)
            for tag, hv in variants.items():
                tn = TNHOBO(hv, method="greedy", max_repeats=16, target_size=2 ** 22)
                st = tree_stats(tn.tree)
                unit = 0.5 if tag == "quad_1.5A" else 1.0
                E0, C = tn.spectrum(K, unit)
                sa = SA(hv)
                bc = sa.beta_cold
                lz = tn.log_partition(bc)
                P0 = float(np.exp(np.log(C[0]) - bc * E0 - lz))
                r = {"n_bits": hv.n, "terms": len(hv), "width": st["width"], "log2_cost": st["log2_cost"],
                     "E0": E0, "unit": unit, "spectrum": C.tolist(), "beta_hot": sa.beta_hot, "beta_cold": bc,
                     "P0_eq_beta_cold": P0}
                for sw in (100, 1000):
                    res = sa.run(1024, sw, seed=s + 7)
                    Eo = h.energy(res["best_X"][:, : h.n].astype(float))
                    r[f"sa{sw}_P0"] = float(np.mean(np.abs(Eo - E0) < 1e-6)) if tag == "native" else \
                        float(np.mean(np.abs(Eo - rec["native"]["E0"]) < 1e-6))
                    r[f"sa{sw}_sec"] = res["seconds"]
                rec[tag] = r
            out.append(rec)
            nat, qa, qb = rec["native"], rec["quad_A+1"], rec["quad_1.5A"]
            cum = lambda r, d: float(np.sum(np.array(r["spectrum"])[: int(round(d / r["unit"])) + 1]))
            print(f"{fam:10s} n={n:3d} s{s} aux={rec['quad_A+1_aux']:3d} | width {nat['width']:.0f}/{qa['width']:.0f}/{qb['width']:.0f}"
                  f" | N0 {nat['spectrum'][0]:.0f}/{qa['spectrum'][0]:.0f}/{qb['spectrum'][0]:.0f}"
                  f" | N(<=E0+2) {cum(nat, 2):.0f}/{cum(qa, 2):.0f}/{cum(qb, 2):.0f}"
                  f" | P0eq {nat['P0_eq_beta_cold']:.3f}/{qa['P0_eq_beta_cold']:.3f}/{qb['P0_eq_beta_cold']:.3f}"
                  f" | SA1000 {nat['sa1000_P0']:.3f}/{qa['sa1000_P0']:.3f}/{qb['sa1000_P0']:.3f}", flush=True)
            json.dump(out, open(f"../../results/e3_quad_{'_'.join(sys.argv[1:]) or 'all'}.json", "w"), indent=1)
