# Code for "Tensor Network based HOBO Solver" v2

Python 3.10, numpy, scipy, torch (CPU is enough; CUDA optional), cotengra 0.8.x (+ kahypar, optuna),
matplotlib (figures). `pip install numpy scipy torch cotengra kahypar optuna cmaes matplotlib`.

| file | what |
|---|---|
| `hobo.py` | HOBO as a sparse coefficient tensor: build, evaluate (batched; dense v1 `einsum` form), spin/bit conversion, brute force, min-fill tropical elimination, `log_partition`, Rosenberg `quadratize(scale, add)` |
| `instances.py` | generators: random k-spin, planted 3R3X (+ GF(2) rank), LABS, 2D plaquette, 2D triangle, random k-SAT; spin/SAT models carry their natural factor tables |
| `tn.py` | factor-graph tensor network; cotengra tree (+slicing); semirings Tropical / CountingTropical / TruncatedPoly / LogSumExp; `ground_energy`, `ground_count`, `spectrum`, `log_partition`, `ground_state`, `sample`; table bases `bit` / `spin` / `natural` |
| `tn_torch.py` | the same tree executed as chunked min-plus (and counting) products in torch, CPU or GPU |
| `bounds.py` | lower bounds: dual ascent (min-sum diffusion) + mini-bucket with moment matching; `best_bound` |
| `bnb.py` | slice-level branch and bound (negative result, Sec. 8) |
| `sa.py` | colour-parallel batched simulated annealing, neal-style schedule; `solve_quadratized` |
| `exp/e1_width.py` | E1: contraction width / cost per family and size (Table 2, Fig. 2) |
| `exp/e2_gap.py` | E2: certified gaps, `BASIS=natural` or `bit` (Table 7, Fig. 5) |
| `exp/e3_quad.py`, `exp/e3_common_beta.py` | E3: quadratization anatomy (Table 5, Fig. 3) |
| `exp/e4_xorsat.py`, `exp/e8_gpu.py` | E4/E8: 3R3X exact vs SA, CPU / GPU (Table 4) |
| `exp/e5_fair.py`, `exp/e5_matched.py`, `exp/e5b_n40.py` | E5: fairness of SA over degenerate ground states (Table 6, Fig. 4) |
| `exp/e6_lattice_exact.py` | E6: L=12 lattices exact (Table 3) |
| `exp/e10_compressed.py` | E10: quimb `contract_compressed` vs exact log Z (Sec. 7.8) |
| `exp/e9_moves.py` | E9: mixed single-bit / coordinated moves on the quadratized model (not used in the paper; not yet run at scale) |
| `exp/make_tables.py`, `exp/make_figs.py` | regenerate `paper/tables/*.tex` and `paper/media/*.pdf` from `results/*.json` |

## Self-tests (each prints `ok` lines; all compare against brute-force enumeration)
```
cd code
python3 hobo.py        # energy == einsum, spin<->bit, tropical == brute force, log Z, n=25 ring
python3 instances.py   # planted 3R3X degeneracy == 2^(n-rank), LABS known optima
python3 tn.py          # E0, N0, log Z, argmin, sampler moments; with and without slicing; spectrum
python3 tn_torch.py    # torch executor == numpy (counting, sliced)
python3 bounds.py      # LB <= E0 for all i-bounds, dual ascent + MBE
python3 sa.py          # incremental dE bookkeeping, native vs quadratized on small 3-spin
```

## Reproducing the experiments (CPU, 32-core server; times approximate)
```
cd code/exp
python3 e1_width.py 3R3X 3spin_m2n ; python3 e1_width.py 3SAT_4.26 LABS ; python3 e1_width.py plaq2d tri2d   # ~1 h each, 2 threads
BASIS=natural python3 e2_gap.py plaq2d tri2d ; BASIS=natural python3 e2_gap.py 3spin_m2n 3R3X ; BASIS=natural python3 e2_gap.py 3SAT_4.26
BASIS=bit     python3 e2_gap.py ...   (same three)                                                    # minutes each
python3 e3_quad.py 3spin_m2n 3SAT_4.26 ; python3 e3_quad.py LABS ; python3 e3_common_beta.py           # ~2 h + 1.5 h
python3 e4_xorsat.py 60 80 100                                                                        # ~20 min
python3 e5_fair.py ; python3 e5_matched.py                                                            # ~40 min
python3 e6_lattice_exact.py 12                                                                        # ~15 min, 8 threads
python3 e8_gpu.py xorsat 0     # GPU only; 7 h on a V100 for n=150
python3 e10_compressed.py      # ~10 min: SVD-compressed contraction vs exact log Z (Sec. 7.8)
python3 make_tables.py ; python3 make_figs.py
```
`e3_quad.py` for 3-SAT n=40 and LABS N>=16 was aborted (quadratized width too large for the numpy
spectrum); `e8_gpu.py lattice` (L=16) was aborted (tree cost 2^57 after slicing).  Both are reported
as such in the paper.

Randomness: every generator and SA run takes an explicit seed; the JSON in `results/` records them.
`results/*.log` are the console logs of the runs used in the paper.
