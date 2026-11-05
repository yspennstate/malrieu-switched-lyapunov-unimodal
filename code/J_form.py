"""x > 0: d(x) = x^2 - (1+x^2) J(x), J = int_0^inf e^{-t} B(t) dt, B(t) = 2trx^2 / ((rb+tx) + sqrt((tx-rb)^2 + 4rtx)),
b = 1+x^2 (B = t - A, A the inverse of phi_+). Checked against g_stable.d_of."""
import math, sys
from scipy.integrate import quad
from g_stable import d_of


def J(x, r):
    b = 1 + x * x
    def B(t):
        return 2 * t * r * x * x / ((r * b + t * x) + math.sqrt((t * x - r * b) ** 2 + 4 * r * t * x))
    s = r * b / x
    pts = [0, s, 10 * s, 40]
    pts = sorted(set(p for p in pts if p <= 40))
    v = sum(quad(lambda t: math.exp(-t) * B(t), a, c, limit=200, epsabs=1e-15, epsrel=1e-12)[0] for a, c in zip(pts[:-1], pts[1:]))
    return v + quad(lambda t: math.exp(-t) * B(t), pts[-1], math.inf, limit=200)[0]


if __name__ == "__main__":
    worst = 0
    for r in (0.001, 0.01, 0.22, 3.0):
        for x in (1e-3, 0.05, 0.7, 2.0, 40.0, 0.3 / r, 3 / r, 30 / r):
            dJ = x * x - (1 + x * x) * J(x, r)
            dq = d_of(x, r)
            rel = abs(dJ - dq) / max(1.0, abs(dq))
            worst = max(worst, rel)
            print("r=%-6g x=%-9.4g d_J=%.12g d_of=%.12g rel=%.1e  J=%.8g  1-e^{-rx}=%.8g" % (
                r, x, dJ, dq, rel, J(x, r), 1 - math.exp(-r * x)), flush=True)
    print("worst rel", worst)
