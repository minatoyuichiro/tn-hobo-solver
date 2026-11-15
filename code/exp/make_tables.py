"""LaTeX tables for the v2 manuscript, generated from results/*.json (paper/tables/*.tex)."""
import glob, json, os
import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "tables")
os.makedirs(OUT, exist_ok=True)


def sci(x):
    if x < 1e5:
        return f"{x:.0f}"
    e = int(np.floor(np.log10(x)))
    return f"{x / 10 ** e:.2f}\\times 10^{{{e}}}"


def w(name, s):
    open(os.path.join(OUT, name), "w").write(s)
    print("wrote", name)


# ---- Table: formulation vs bound (E0)
d = json.load(open(os.path.join(R, "e0_basis.json")))
rows = []
for r in d:
    rows.append(f"{r['instance']} & ${r['E0']:+.0f}$ & " + " & ".join(
        f"${r[b]['trivial']:+.1f}$ / ${r[b]['i4']:+.1f}$ / ${r[b]['i8']:+.1f}$ / ${r[b]['i12']:+.1f}$" for b in ("bit", "spin", "natural")) + r" \\")
w("tab_basis.tex", r"""\begin{tabular}{lc ccc}
\toprule
instance & $E_0$ & coefficient tensor (bit monomials) & spin monomials & natural factors \\
& & \multicolumn{3}{c}{lower bound: trivial / $i{=}4$ / $i{=}8$ / $i{=}12$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")

# ---- Table: reach (E1)
rows = {}
for f in glob.glob(os.path.join(R, "e1_width_*.json")):
    for r in json.load(open(f)):
        rows.setdefault((r["family"], r["size"]), []).append(r)
LAB = {"3R3X": "planted 3R3X", "3spin_m2n": r"random 3-spin ($m{=}2n$)", "3SAT_4.26": r"random 3-SAT ($m{=}4.26n$)",
       "LABS": "LABS", "plaq2d": "2D plaquette ($L{\\times}L$)", "tri2d": "2D triangle ($L{\\times}L$)"}
lines = []
for fam in ("3R3X", "3spin_m2n", "3SAT_4.26", "LABS", "plaq2d", "tri2d"):
    first = True
    for (f, size), rs in sorted(k for k in rows.items() if k[0][0] == fam):
        mf = [r["minfill"]["width"] for r in rs]
        ct = [min(r["greedy"]["width"], r["hyper"]["width"]) for r in rs]
        cb = [min(r["minfill"]["log2_cost"], r["greedy"]["log2_cost"], r["hyper"]["log2_cost"]) for r in rs]
        run = [r["run"]["t"] for r in rs if "run" in r]
        cnt = [r["run"]["count"] for r in rs if "run" in r]
        rng = lambda v, fmt: (fmt.format(min(v)) if min(v) == max(v) else fmt.format(min(v)) + "--" + fmt.format(max(v)))
        lines.append((LAB[fam] if first else "") + f" & {size} & {rs[0]['n']} & {rs[0]['terms']} & {rng(mf, '{:.0f}')} & {rng(ct, '{:.0f}')} & {rng(cb, '{:.1f}')} & "
                     + (f"{rng(run, '{:.2f}')} & " + ("$" + sci(min(cnt)) + "$" if min(cnt) == max(cnt) else "$" + sci(min(cnt)) + "$--$" + sci(max(cnt)) + "$") if run else "-- & --") + r" \\")
        first = False
    lines.append(r"\addlinespace[2pt]")
w("tab_reach.tex", r"""\begin{tabular}{l rrr rr r rr}
\toprule
family & size & $n$ & terms & \multicolumn{2}{c}{width} & $\log_2$ cost & time [s] & $N_0$ \\
& & & & min-fill & cotengra & (best) & (CPU) & \\
\midrule
""" + "\n".join(lines[:-1]) + r"""
\bottomrule
\end{tabular}""")

# ---- Table: 3R3X (E4 + E8)
d = json.load(open(os.path.join(R, "e4_xorsat_60_80_100.json"))) + json.load(open(os.path.join(R, "e8_gpu_xorsat_s0.json")))
lines = []
for r in d:
    gpu = "family" in r
    n = r["n"] if not gpu else r["size"]
    tn_t = r["tn_sec"]
    cnt = r["tn_count"] if not gpu else r["count"]
    sa3 = r["sa1000"]; sa4 = r["sa10000"]
    p3 = sa3["p_ground"] if "p_ground" in sa3 else sa3["p"]; p4 = sa4["p_ground"] if "p_ground" in sa4 else sa4["p"]
    lines.append(f"{n} & {r['seed']} & {'GPU' if gpu else 'CPU'} & {r['width']:.0f} & {r['log2_cost']:.1f} & {r['slices']} & {tn_t:.1f} & {cnt:.0f} & {r['deg_gf2']:.0f} & "
                 f"${sa3['best']:+.0f}$ & {p3:.3f} & ${sa4['best']:+.0f}$ & {p4:.3f} \\\\")
w("tab_xorsat.tex", r"""\begin{tabular}{rrl rrr r rr rr rr}
\toprule
$n$ & seed & device & width$^\dagger$ & $\log_2$ cost & slices & TN time [s] & $N_0$ (TN) & $N_0$ (GF(2)) & \multicolumn{2}{c}{SA, $10^3$ sweeps} & \multicolumn{2}{c}{SA, $10^4$ sweeps} \\
& & & & & & & & & best & $P(E_0)$ & best & $P(E_0)$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

# ---- Table: lattice exact (E6)
d = json.load(open(os.path.join(R, "e6_lattice_exact.json")))
def sci(x):
    if x < 1e5:
        return f"{x:.0f}"
    e = int(np.floor(np.log10(x)))
    return f"{x / 10 ** e:.2f}\\times 10^{{{e}}}"
lines = [f"{LAB[r['family']].split(' (')[0]} & {r['L']} & {r['L'] ** 2} & {r['width']:.0f} & {r['log2_cost']:.1f} & {r['slices']} & {r['tn_sec']:.0f} & ${r['E0']:+.0f}$ & ${sci(r['count'])}$ & ${r['sa_best']:+.0f}$ & {r['sa_p']:.3f} \\\\" for r in d]
w("tab_lattice.tex", r"""\begin{tabular}{lrr rrr r rr rr}
\toprule
family & $L$ & $n$ & width & $\log_2$ cost & slices & TN time [s] & $E_0$ & $N_0$ & SA best & SA $P(E_0)$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

# ---- Table: quadratization anatomy (E3 + common beta)
cbf = os.path.join(R, "e3_common_beta.json")
if os.path.exists(cbf):
    cb = {(r["family"], r["size"], r["seed"]): r for r in json.load(open(cbf))}
    sa = {(r["family"], r["size"], r["seed"]): r for r in json.load(open(os.path.join(R, "e3_quad_3spin_3sat.json"))) + json.load(open(os.path.join(R, "e3_quad_LABS.json")))}
    keys = sorted((k for k in sa if k in cb), key=lambda k: (["3spin_m2n", "3SAT_4.26", "LABS"].index(k[0]), k[1], k[2]))
    lines = []
    for k in keys:
        s, c = sa[k], cb[k]
        lev = lambda t: int(round(sum(s[t]["spectrum"][: int(round(2 / s[t]["unit"])) + 1])))
        lines.append(f"{LAB[k[0]].split(' (')[0]} & {k[1]} & {k[2]} & {s['quad_A+1_aux']} & "
                     f"{s['native']['width']:.0f} / {s['quad_A+1']['width']:.0f} / {s['quad_1.5A']['width']:.0f} & "
                     f"{s['native']['spectrum'][0]:.0f} & {lev('native')} / {lev('quad_A+1')} / {lev('quad_1.5A')} & "
                     f"{c['native']['P0_beta']:.3f} / {c['quad_A+1']['P0_beta']:.3f} / {c['quad_1.5A']['P0_beta']:.3f} & "
                     f"{c['native']['P0_2beta']:.3f} / {c['quad_A+1']['P0_2beta']:.3f} / {c['quad_1.5A']['P0_2beta']:.3f} & "
                     f"{s['native']['sa1000_P0']:.3f} / {s['quad_A+1']['sa1000_P0']:.3f} / {s['quad_1.5A']['sa1000_P0']:.3f} \\\\")
    w("tab_quad.tex", r"""\begin{tabular}{lrr r c r c c c c}
\toprule
family & $n$ & seed & aux. & width & $N_0$ & $N(E\le E_0{+}2)$ & $P_0(\beta_{\rm cold})$ & $P_0(2\beta_{\rm cold})$ & SA reaches $E_0$ \\
& & & bits & \multicolumn{6}{c}{each cell: native / $M{=}A{+}1$ / $M{=}1.5A$} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

# ---- Table: fairness (E5 matched)
m = json.load(open(os.path.join(R, "e5_matched.json")))
d = {(r["n"], r["seed"]): r for r in json.load(open(os.path.join(R, "e5_fair.json")))}
lines = []
for r in m:
    f = d[(r["n"], r["seed"])]
    pn = f['native']['chi2_p']; pns = "$<10^{-3}$" if pn < 1e-3 else f"{pn:.3f}"
    lines.append(f"{r['n']} & {r['seed']} & {r['N0']} & {f['native']['hits']} & {f['native']['max_min_ratio']:.1f} & {f['native']['H_norm']:.3f} & {pns} & "
                 f"{r['hits_quad']} & {f['quad']['max_min_ratio']:.1f} & {r['H_quad']:.3f} & {r['H_native_matched_mean']:.3f} & {r['p_native_as_biased']:.3f} \\\\")
# E5b: n=40 rerun with 10x quadratized effort (32 x 4096 replicas, 5000 sweeps)
e5b = os.path.join(R, "e5b_n40.json")
if os.path.exists(e5b):
    lines.append(r"\addlinespace[2pt]")
    for r in json.load(open(e5b)):
        pn = r['native']['chi2_p']; pns = "$<10^{-3}$" if pn < 1e-3 else f"{pn:.3f}"
        lines.append(f"{r['n']}$^\\dagger$ & {r['seed']} & {r['N0']} & {r['native']['hits']} & {r['native']['max_min_ratio']:.1f} & {r['native']['H_norm']:.3f} & {pns} & "
                     f"{r['quad']['hits']} & {r['quad']['max_min_ratio']:.1f} & {r['quad']['H_norm']:.3f} & {r['H_native_matched_mean']:.3f} & {r['p_native_as_biased']:.3f} \\\\")
w("tab_fair.tex", r"""\begin{tabular}{rrr rrrr rrr rr}
\toprule
$n$ & seed & $N_0$ & \multicolumn{4}{c}{native SA} & \multicolumn{3}{c}{quadratized SA} & $H/\log N_0$ native & $p$ \\
& & & hits & max/min & $H/\log N_0$ & $\chi^2$ $p$ & hits & max/min & $H/\log N_0$ & at matched hits & \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

# ---- Table: certified gaps (E2 natural)
rows = []
for f in glob.glob(os.path.join(R, "e2natural_gap_*.json")):
    rows += json.load(open(f))
bit = {}
for f in glob.glob(os.path.join(R, "e2bit_gap_*.json")):
    for r in json.load(open(f)):
        bit[(r["family"], r["size"], r["seed"])] = r
rows.sort(key=lambda r: (["3R3X", "3spin_m2n", "3SAT_4.26", "plaq2d", "tri2d"].index(r["family"]), r["size"], r["seed"]))
lines = []
for r in rows:
    b = bit.get((r["family"], r["size"], r["seed"]))
    lines.append(f"{LAB[r['family']].split(' (')[0]} & {r['size']} & {r['seed']} & {r['n']} & ${r['ub']:+.0f}$ & ${r['lb_dual']:+.1f}$ & ${r['lb']:+.1f}$ & {r['gap']:.0f}{'$^*$' if r['certified'] else ''} & "
                 + (f"${b['lb']:+.1f}$ & {b['gap']:.0f}{'$^*$' if b['certified'] else ''}" if b else "-- & --") + r" \\")
w("tab_gap.tex", r"""\begin{tabular}{lrrr r rrr rr}
\toprule
family & size & seed & $n$ & UB (SA) & \multicolumn{3}{c}{natural factors} & \multicolumn{2}{c}{coefficient tensor} \\
& & & & & dual & LB & gap & LB & gap \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")
