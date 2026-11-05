"""small r: the four quantities of the N/D route, rD - 2, N - 2 log(1/r) + 2 gamma, r^2 D' + 2, r N' + 2,
with N', D' by central differences (h = 1e-3 r). Usage: python small_r_eps.py r [r ...]"""
import math, sys
from small_r_ND import ND

g = 0.5772156649015329
for r in [float(a) for a in sys.argv[1:]] or [0.01]:
    h = 1e-3 * r
    N, D = ND(r)
    Np_, Dp_ = ND(r + h)
    Nm, Dm = ND(r - h)
    Np, Dp = (Np_ - Nm) / (2 * h), (Dp_ - Dm) / (2 * h)
    H = Np * D - N * Dp
    print("r=%-7g e1=rD-2=%.5f e2=N-2L+2g=%.5f e3=r^2D'+2=%.5f e4=rN'+2=%.5f  r^2(N'D-ND')/4=%.5f  L-g-1=%.5f" % (
        r, r * D - 2, N - 2 * math.log(1 / r) + 2 * g, r * r * Dp + 2, r * Np + 2, r * r * H / 4,
        math.log(1 / r) - g - 1), flush=True)
