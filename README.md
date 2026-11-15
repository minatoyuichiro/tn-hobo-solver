# Tensor Network based HOBO Solver

Code, raw results and manuscript source for version 2 of

> Y. Minato, *Tensor Network based HOBO Solver*, [arXiv:2407.16106](https://arxiv.org/abs/2407.16106).

A higher-order binary optimisation (HOBO) cost `E(x) = c0 + sum_S c_S prod_{i in S} x_i` is written as
a factor network (one small table per term or clause, one COPY tensor per variable) and contracted
along a [cotengra](https://github.com/jcmgray/cotengra) tree. Changing the semiring of the
contraction changes what comes out:

| semiring | result |
|---|---|
| tropical (min, +) | exact minimum |
| counting tropical | minimum and ground-state degeneracy |
| truncated polynomial | number of states at each of the K lowest levels |
| (logsumexp, +) | log Z(beta) |
| top-down pass | a minimiser, or exact Boltzmann samples |

The repository also contains certified lower bounds (dual ascent + mini-bucket elimination), a
batched simulated-annealing baseline, Rosenberg quadratization, and the experiment scripts of the
paper.

**What this is not.** It is not a faster optimiser than simulated annealing. Exact contraction is
exponential in the contraction width, so it works for structured or sparse instances (about 100-150
variables in the paper) and is useless for dense random ones. The paper reports these negative
results as well.

## Layout

```
code/        solver modules and self-tests        (see code/README.md)
code/exp/    experiment scripts, table and figure generators
results/     raw JSON records and console logs of the runs used in the paper
paper/       LaTeX source, generated tables and figures, PDF
```

## Quick start

```bash
pip install -r requirements.txt          # CPU torch is enough
cd code
python3 tn.py                             # self-test against brute-force enumeration (a few minutes)
```

```python
import instances as I
from tn import TNHOBO

h = I.plaquette_2d(6, seed=0)             # 36 spins, 4-body terms on a torus
tn = TNHOBO(h, basis="natural")           # builds the network, finds a contraction tree
E0, N0 = tn.ground_count()                # exact minimum and number of ground states
E, x   = tn.ground_state()                # one minimiser
logZ   = tn.log_partition(beta=1.0)
X      = tn.sample(beta=1.0, n_samples=1000)   # exact Boltzmann samples
```

Your own problem: build a `hobo.HOBO(n)` and `add(indices, coefficient)` for each monomial (use
`HOBO.from_ising` for spin models), then `TNHOBO(h)`. If you have the problem's own terms as tables,
attach them as `h.factors` and use `basis="natural"`; relaxations are much tighter that way
(paper, Sec. 7.2).

## Reproducing the paper

`code/README.md` lists the six self-tests and the command for every experiment with approximate
run times. Every number in the paper's tables is generated from `results/*.json` by
`code/exp/make_tables.py`; figures by `code/exp/make_figs.py`. Seeds are recorded in the JSON.

Not rerun for this release, and stated as such in the paper: the n = 150 XORSAT contraction
(7.3 h on one V100 GPU), and two attempts that were abandoned (L = 16 lattices; quadratized 3-SAT
n = 40 / LABS N >= 16 spectra).

Tested with Python 3.10, numpy 1.26, scipy 1.15, torch 2.x (CPU), cotengra 0.8.2, quimb 1.11
(only for `exp/e10_compressed.py`).

## Citation

```bibtex
@misc{minato2024hobo,
  title         = {Tensor Network based {HOBO} Solver},
  author        = {Minato, Yuichiro},
  year          = {2024},
  eprint        = {2407.16106},
  archivePrefix = {arXiv}
}
```

## License

MIT (see `LICENSE`).
