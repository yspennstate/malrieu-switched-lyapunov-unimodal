"""G(r) computed without the large-r cancellation.
d(x) := (1+x^2) f0(x) - 1 = (1+x^2) int_0^U [e^{-u a(u)} - e^{-u(1+x^2)}] du - e^{-U(1+x^2)},
a(u) = 1 + x^2 r/(r+u x), U = inf (x>=0), r/|x| (x<0)  [u = r w in the g_xform.py integral].
Then f0 - f1 = d/x^2 and f0 + f1 = 2/(1+x^2) + (x^2-1) d/(x^2 (1+x^2))."""
import sys, math
import numpy as np
from scipy.integrate import quad


def d_of(x, r):
    b = 1.0 + x * x
    def h(u):
        arg = u * u * x ** 3 / (r + u * x)
        if arg < 1.0:
            return math.exp(-u * b) * math.expm1(arg)
        return math.exp(-u * b + arg) - math.exp(-u * b)
    if x >= 0:
        s = 1.0 / b
        pts = [0, 2 * s, 10 * s, 40 * s]
        tot = sum(quad(h, a, c, limit=200, epsabs=1e-15, epsrel=1e-11)[0] for a, c in zip(pts[:-1], pts[1:]))
        tot += quad(h, pts[-1], np.inf, limit=200, epsabs=1e-15)[0]
        return b * tot
    U = r / (-x)
    s = min(1.0 / b, U)
    pts = sorted(set([0.0] + [p for p in (2 * s, 10 * s, 40 * s) if p < U] + [U]))
    tot = sum(quad(h, a, c, limit=200, epsabs=1e-15, epsrel=1e-11)[0] for a, c in zip(pts[:-1], pts[1:]))
    return b * tot - math.exp(-U * b)


def G(r, X=np.inf):
    def parts(x):
        if abs(x) < 1e-9:
            x = 1e-9 if x >= 0 else -1e-9
        d = d_of(x, r)
        x2 = x * x
        return d / x2, 2 / (1 + x2) + (x2 - 1) * d / (x2 * (1 + x2))
    num = lambda x: parts(x)[0] * x / (1 + x * x)
    den = lambda x: parts(x)[1]
    pts = [-X, -50, -10, -3, -1, -0.3, 0, 0.3, 1, 3, 10, 50, X]
    N = sum(quad(num, a, b, limit=300, epsabs=1e-13)[0] for a, b in zip(pts[:-1], pts[1:]))
    D = sum(quad(den, a, b, limit=300, epsabs=1e-12)[0] for a, b in zip(pts[:-1], pts[1:]))
    return N / D


if __name__ == "__main__":
    for r in [float(a) for a in sys.argv[1:]] or [0.01, 0.23, 4.6]:
        g = G(r)
        print("r = %-8g G = %.8f   8rG = %.6f   G/r - log(1/r) = %.6f" % (r, g, 8 * r * g, g / r - math.log(1 / r)), flush=True)
