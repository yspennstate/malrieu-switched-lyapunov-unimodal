"""small r: sizes of N and D (g_stable quadrature) to fix the scaling used by a small-r H-route proof.
Prediction: d -> x^2-type behaviour for x >> 1 up to x ~ 1/r, so D ~ c/r and N ~ log(1/r); then
N'D - ND' ~ (c/r^2)(log(1/r) - const) > 0. Prints r, N, D, r*D, N - log(1/r), G/r - log(1/r)."""
import math, sys
import numpy as np
from scipy.integrate import quad
from g_stable import d_of


def ND(r):
    def parts(x):
        if abs(x) < 1e-9:
            x = 1e-9 if x >= 0 else -1e-9
        d = d_of(x, r)
        x2 = x * x
        return d / (x * (1 + x2)), 2 / (1 + x2) + (x2 - 1) * d / (x2 * (1 + x2))
    X = 1e3 / r
    pts = sorted(set([-X, -50, -10, -3, -1, -0.3, 0, 0.3, 1, 3, 10, 50] + [c / r for c in (0.03, 0.1, 0.3, 1, 3, 10, 30, 100)] + [X]))
    N = sum(quad(lambda x: parts(x)[0], a, b, limit=400, epsabs=1e-11)[0] for a, b in zip(pts[:-1], pts[1:]))
    D = sum(quad(lambda x: parts(x)[1], a, b, limit=400, epsabs=1e-10)[0] for a, b in zip(pts[:-1], pts[1:]))
    return N, D


if __name__ == "__main__":
    for r in [float(a) for a in sys.argv[1:]] or [0.03, 0.01, 0.003]:
        N, D = ND(r)
        L = math.log(1 / r)
        print("r=%-7g N=%.6f D=%.6f rD=%.6f N-log(1/r)=%.6f G/r-log(1/r)=%.6f" % (r, N, D, r * D, N - L, N / D / r - L), flush=True)
