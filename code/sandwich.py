"""x > 0: sandwich for B and J (the small-rate section of the paper).
B is the smaller root of f(B) = xB^2 - (rb+tx)B + trx^2, b = 1+x^2. f(t) = -rt < 0 and f(rx) = -r^2 x < 0, so
L(t) := trx^2/(rb+tx) < B(t) < min(t, rx). Hence, with c = rx,
  e^{-c} < 1 - J < e^{-c} + Delta,  Delta = int_0^inf e^{-t} (min(t,c) - L(t)) dt,
and d = b(1-J) - 1 lies in (b e^{-c} - 1, b e^{-c} - 1 + b Delta). Checks the bounds against J_form.J and prints Delta."""
import math
from scipy.integrate import quad
from J_form import J


def Bexact(t, x, r):
    b = 1 + x * x
    return 2 * t * r * x * x / ((r * b + t * x) + math.sqrt((t * x - r * b) ** 2 + 4 * r * t * x))


def Delta(x, r):
    b, c = 1 + x * x, r * x
    g = lambda t: math.exp(-t) * (min(t, c) - t * r * x * x / (r * b + t * x))
    pts = sorted(set([0, c, r * b / x, 10 * c, 40]))
    pts = [p for p in pts if p <= 40]
    return sum(quad(g, a, e, limit=200, epsabs=1e-16, epsrel=1e-12)[0] for a, e in zip(pts[:-1], pts[1:])) + \
        quad(g, pts[-1], math.inf)[0]


if __name__ == "__main__":
    bad = 0
    for r in (0.001, 0.01, 0.1, 0.22, 3.0):
        for x in (1e-3, 0.05, 0.7, 2.0, 40.0, 0.3 / r, 3 / r, 30 / r):
            for t in (1e-6, 1e-3, 0.1, 1.0, 5.0, 30.0):
                b = 1 + x * x
                B = Bexact(t, x, r)
                L = t * r * x * x / (r * b + t * x)
                if not (L <= B * (1 + 1e-12) and B <= min(t, r * x) * (1 + 1e-12)):
                    bad += 1
            oneJ = 1 - J(x, r)
            lo, D = math.exp(-r * x), Delta(x, r)
            ok = lo * (1 - 1e-9) <= oneJ <= (lo + D) * (1 + 1e-9) + 1e-14
            bad += not ok
            print("r=%-6g x=%-9.4g 1-J=%.10g e^-rx=%.10g Delta=%.3e (1-J-e^-rx)/Delta=%.3f %s" % (
                r, x, oneJ, lo, D, (oneJ - lo) / D if D > 0 else float('nan'), "ok" if ok else "FAIL"), flush=True)
    print("violations", bad)
