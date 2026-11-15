"""Figures for the v2 manuscript (PDF into paper/media/).  Reads results/*.json only."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "media")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "figure.dpi": 150})
LAB = {"3R3X": "planted 3R3X", "3spin_m2n": "random 3-spin, $m=2n$", "3SAT_4.26": "random 3-SAT, $m=4.26n$",
       "LABS": "LABS", "plaq2d": "2D plaquette", "tri2d": "2D triangle"}


# ---------------------------------------------------------------- fig 1: schematic
def fig_schematic():
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    vars_ = {1: (1.5, 4), 2: (4, 4), 3: (6.5, 4), 4: (9, 4)}
    facs = {"A": (2.75, 1.2, (1, 2, 3)), "B": (5.25, 1.2, (2, 3)), "C": (7.75, 1.2, (3, 4))}
    for name, (x, y, sc) in facs.items():
        for v in sc:
            vx, vy = vars_[v]
            ax.plot([x, vx], [y + 0.45, vy - 0.35], color="0.3", lw=1, zorder=1)
        ax.add_patch(plt.Rectangle((x - 0.55, y - 0.45), 1.1, 0.9, fc="#d9e6f2", ec="k", lw=1, zorder=2))
        ax.text(x, y, f"$T_{name}$", ha="center", va="center", zorder=3)
    for v, (x, y) in vars_.items():
        ax.add_patch(plt.Circle((x, y), 0.35, fc="white", ec="k", lw=1, zorder=2))
        ax.text(x, y, f"$x_{v}$", ha="center", va="center", zorder=3)
    ax.text(5, 0.15, r"$E(x)=T_A(x_1,x_2,x_3)+T_B(x_2,x_3)+T_C(x_3,x_4)$; circles = COPY tensors",
            ha="center", va="bottom", fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_tn.pdf")); plt.close(fig)


# ---------------------------------------------------------------- fig 2: reach
def fig_reach():
    rows = {}
    for f in glob.glob(os.path.join(R, "e1_width_*.json")):
        for r in json.load(open(f)):
            rows.setdefault(r["family"], {}).setdefault(r["n"], []).append(r)
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    markers = dict(zip(LAB, "osD^v<"))
    for fam, by_n in rows.items():
        ns = sorted(by_n)
        mf = [np.mean([r["minfill"]["log2_cost"] for r in by_n[n]]) for n in ns]
        best = [np.mean([min(r["minfill"]["log2_cost"], r["greedy"]["log2_cost"], r["hyper"]["log2_cost"]) for r in by_n[n]]) for n in ns]
        axs[0].plot(ns, best, marker=markers[fam], ms=3.5, lw=1, label=LAB[fam])
        wmf = [np.mean([r["minfill"]["width"] for r in by_n[n]]) for n in ns]
        whp = [np.mean([min(r["hyper"]["width"], r["greedy"]["width"]) for r in by_n[n]]) for n in ns]
        axs[1].plot(ns, np.array(wmf) - np.array(whp), marker=markers[fam], ms=3.5, lw=1, label=LAB[fam])
    axs[0].axhline(33, color="0.5", ls="--", lw=0.8); axs[0].text(255, 33.6, "CPU, minutes", fontsize=6.5, ha="right", color="0.3")
    axs[0].axhline(47, color="0.5", ls=":", lw=0.8); axs[0].text(255, 47.6, "1 GPU, hours", fontsize=6.5, ha="right", color="0.3")
    axs[0].set_xlabel("$n$ (bits)"); axs[0].set_ylabel(r"$\log_2$ contraction cost (best tree)")
    axs[1].set_xlabel("$n$ (bits)"); axs[1].set_ylabel("width: min-fill $-$ cotengra")
    axs[1].axhline(0, color="0.5", lw=0.8)
    axs[0].legend(ncol=1, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_reach.pdf")); plt.close(fig)


# ---------------------------------------------------------------- fig 3: quadratization anatomy
def fig_quad():
    f = os.path.join(R, "e3_common_beta.json")
    if not os.path.exists(f):
        print("e3_common_beta.json missing; skip fig_quad"); return
    cb = {(r["family"], r["size"], r["seed"]): r for r in json.load(open(f))}
    sa = {(r["family"], r["size"], r["seed"]): r for r in
          json.load(open(os.path.join(R, "e3_quad_3spin_3sat.json"))) + json.load(open(os.path.join(R, "e3_quad_LABS.json")))}
    keys = [k for k in sa if k in cb]
    keys.sort(key=lambda k: (["3spin_m2n", "3SAT_4.26", "LABS"].index(k[0]), k[1], k[2]))
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    x = np.arange(len(keys)); w = 0.27
    for ax, tag, title in ((axs[0], "P0_beta", r"equilibrium $P_0(\beta_{\rm cold})$ (exact, TN)"),
                           (axs[1], "sa", "SA: fraction of replicas reaching $E_0$")):
        for j, (var, col, lab) in enumerate((("native", "#1f77b4", "native HOBO"), ("quad_A+1", "#ff7f0e", "quadratized, $M=A+1$"),
                                             ("quad_1.5A", "#2ca02c", "quadratized, $M=1.5A$"))):
            vals = [cb[k][var]["P0_beta"] if tag == "P0_beta" else sa[k][var]["sa1000_P0"] for k in keys]
            ax.bar(x + (j - 1) * w, vals, w, color=col, label=lab)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([f"{k[1]}" for k in keys], fontsize=6)
        ax.set_xlabel("instance ($n$; 3-spin | 3-SAT | LABS)")
        b1 = sum(1 for k in keys if k[0] == "3spin_m2n"); b2 = b1 + sum(1 for k in keys if k[0] == "3SAT_4.26")
        for b in (b1, b2):
            ax.axvline(b - 0.5, color="0.6", lw=0.6)
    axs[0].set_ylim(0, 1.05); axs[0].legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_quad.pdf")); plt.close(fig)


# ---------------------------------------------------------------- fig 4: fairness
def fig_fair():
    d = {(r["n"], r["seed"]): r for r in json.load(open(os.path.join(R, "e5_fair.json")))}
    m = {(r["n"], r["seed"]): r for r in json.load(open(os.path.join(R, "e5_matched.json")))}
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.3))
    for ax, key in zip(axs[:2], [(24, 0), (32, 8)]):
        r = d[key]
        fn = np.array(r["native"]["freq"]); fq = np.array(r["quad"]["freq"])
        order = np.argsort(-fn)
        xx = np.arange(len(fn))
        ax.bar(xx - 0.2, fn[order] / fn.sum(), 0.4, label="native", color="#1f77b4")
        ax.bar(xx + 0.2, fq[order] / fq.sum(), 0.4, label="quadratized", color="#ff7f0e")
        ax.axhline(1 / len(fn), color="k", ls="--", lw=0.8)
        ax.set_title(f"3-spin $n={key[0]}$, $N_0={len(fn)}$", fontsize=8)
        ax.set_xlabel("ground state (sorted)"); ax.set_ylabel("SA landing frequency")
    axs[0].legend(frameon=False)
    ax = axs[2]
    pts = [(r["H_native_matched_mean"], r["H_quad"], r["H_native_matched_mean"] - r["H_native_matched_p05"], r["p_native_as_biased"], str(r["n"]), "o")
           for r in m.values() if r["n"] < 40]
    e5b = os.path.join(R, "e5b_n40.json")
    if os.path.exists(e5b):
        pts += [(r["H_native_matched_mean"], r["quad"]["H_norm"], 0.0, r["p_native_as_biased"], "40", "s") for r in json.load(open(e5b))]
    else:
        pts += [(r["H_native_matched_mean"], r["H_quad"], r["H_native_matched_mean"] - r["H_native_matched_p05"], r["p_native_as_biased"], "40", "o")
                for r in m.values() if r["n"] == 40]
    for x0, y0, xe, pv, lab, mk in pts:
        col = "#d62728" if pv < 0.05 else "0.5"
        ax.errorbar(x0, y0, xerr=[[xe], [0]], fmt=mk, ms=3.5, color=col, elinewidth=0.8, capsize=1.5)
        ax.annotate(lab, (x0, y0), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax.plot([0.75, 1], [0.75, 1], color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$H/\log N_0$, native (matched hits)"); ax.set_ylabel(r"$H/\log N_0$, quadratized")
    ax.set_xlim(0.75, 1.01); ax.set_ylim(0.75, 1.01)
    ax.set_title("entropy of landing distribution", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_fair.pdf")); plt.close(fig)


# ---------------------------------------------------------------- fig 5: certified gaps
def fig_gap():
    def load(prefix):
        out = {}
        for f in glob.glob(os.path.join(R, f"{prefix}_gap_*.json")):
            for r in json.load(open(f)):
                out[(r["family"], r["size"], r["seed"])] = r
        return out
    bit, nat = load("e2bit"), load("e2natural")
    keys = sorted(set(bit) & set(nat), key=lambda k: (list(LAB).index(k[0]), k[1], k[2]))
    fig, ax = plt.subplots(figsize=(6.8, 2.3))
    x = np.arange(len(keys)); w = 0.38
    ax.bar(x - w / 2, [bit[k]["gap"] for k in keys], w, label="coefficient-tensor tables (bit monomials)", color="#9467bd")
    ax.bar(x + w / 2, [nat[k]["gap"] for k in keys], w, label="natural factor tables", color="#17becf")
    ax.set_xticks(x); ax.set_xticklabels([str(k[1]) for k in keys], fontsize=6)
    # family separators and names
    fams = [k[0] for k in keys]
    start = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or fams[i] != fams[start]:
            ax.text((start + i - 1) / 2, ax.get_ylim()[1] * 0.55 if False else 130, LAB[fams[start]].split(",")[0].replace("random ", "").replace("planted ", ""),
                    ha="center", va="bottom", fontsize=6.5)
            if i < len(keys):
                ax.axvline(i - 0.5, color="0.7", lw=0.6)
            start = i
    ax.set_xlabel("instance size ($n$ or $L$)")
    ax.set_ylabel("certified gap  UB $-$ LB"); ax.set_yscale("symlog", linthresh=1)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_gap.pdf")); plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_schematic, fig_reach, fig_quad, fig_fair, fig_gap):
        fn(); print("made", fn.__name__)
