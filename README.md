# The Lyapunov exponent of a randomly switched pair of stable planar systems is unimodal in the switching rate

Yitzchak Shmalo

Paper: [paper/main.pdf](paper/main.pdf) (source [paper/main.tex](paper/main.tex)).

Let $X_t$ solve $\dot X_t = A_{I_t} X_t$ in the plane, with

$$A_0=\begin{pmatrix}-\alpha&1\\0&-\alpha\end{pmatrix},\qquad A_1=\begin{pmatrix}-\alpha&0\\-1&-\alpha\end{pmatrix},$$

and $I_t$ switching between 0 and 1 at rate $r$. The top Lyapunov exponent is $G(r)-\alpha$. Malrieu
(Ann. Fac. Sci. Toulouse Math. (6) 24 (2015), Conjecture 4.3) conjectured that $G$ increases and then decreases,
with a single maximum, and tends to zero at both ends. The paper proves this: $G'>0$ on $(0,r_*)$ and $G'<0$ on
$(r_*,\infty)$ for a single $r_*\in(0.226,0.235)$, with $G(r_*)\approx 0.1985$. So for $\alpha<G(r_*)$ the
rates at which the system is unstable form one open interval.

The rates $r\le 0.01$ and $r\ge 5$ are handled by hand. The range $[0.01,100]$ is covered by interval
arithmetic (Arb, through python-flint), in three panels:

| panel | statement | intervals | output |
|---|---|---|---|
| $[0.01, 0.215]$ | $G'>0$ | 47 | `code/outputs/gp_grid_001_0215.jsonl` |
| $[0.215, 0.25]$ | $H'<0$, where $H=N'D-ND'$ has the sign of $G'$ | 35 | `code/outputs/hp_grid_0215_025.jsonl` |
| $[0.25, 100]$ | $G'<0$ | 83 | `code/outputs/gp_grid_025_10.jsonl`, `code/outputs/gp_grid_10_100.jsonl` |

Each output line is one interval of $r$ with the enclosure computed on it; lines marked `undecided` were
bisected and their halves appear further down.

## Code

Python 3.14 with python-flint 0.9.0, SciPy 1.17 and NumPy 2.4.

- `check_cover.py`: reads the outputs and checks that the decided intervals tile each panel with no gap and carry the
  stated sign. Runs in a second.
- `gp_grid.py A B q out.jsonl [minrel]`: certifies the sign of $G'$ on $[A,B]$, starting from intervals of ratio $q$
  and bisecting undecided ones. The panels above used $q=1.08$: `python gp_grid.py 0.01 0.215 1.08 out.jsonl`,
  and likewise for `0.25 10` and `10 100`.
- `hp_grid.py A B w out.jsonl [minw]`: certifies $H'<0$ on intervals of width $w$; the panel used
  `python hp_grid.py 0.215 0.25 0.001 out.jsonl`.
- `gp_ball.py`, `assemble.py`, `NDprime.py`, `NDpp.py`, `near0.py`, `tails.py`, `drd*.py`: the ball-arithmetic
  enclosures of $N$, $D$ and their first two derivatives in $r$, with the strip near $x=0$ and the tails in $x$.
- `paper_bounds_check.py`, `small_r_eps.py`, `e3_split.py`: floating-point checks of the small-rate bounds of
  Section 3. `large_r.py`: the large-rate bound of Section 4.
- `g_stable.py`, `J_form.py`, `sandwich.py`, `small_r_ND.py`: floating-point evaluation of $G$ and the kernel.
- `mc_check.py`: a Monte Carlo estimate of the exponent of the switched system at $\alpha=0$, compared with $G$.

The certification runs are slow: a single interval of $r$ takes from a few minutes to over an hour, and a whole
panel takes days on one core. The runs are resumable, since intervals already in the output file are skipped.

## Funding

The author was supported by the European Research Council under the Horizon Europe programme (grant 101041711),
the Simons Foundation (Collaboration on the Mathematical and Scientific Foundations of Deep Learning), Heights Labs,
Convex Nexus Capital, and the Israel Science Foundation (grants 2258/19 and 4101/25). Computations used Google Cloud
education credits (credit 530550103).
