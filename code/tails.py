"""x-tail bounds for |x| >= X.
x > 0:  d = (1+x^2) int_0^inf e^{-u} (e^{-g} - e^{-u x^2}) du,  g = r u x^2/(r + u x) <= u x^2, so d >= 0, and
        splitting at u0 = r/x (g >= u x^2/2 below, g >= r x/2 above; u x^2 - g = u^2 x^3/(r+ux) <= u^2 x^3/r):
        0 <= d <= (1+x^2) (16/(r x^3) + e^{-r x/2}).
x < 0:  s = U b t in the t-form (U = r/|x|, b = 1+x^2, Ub >= r|x|), 1 - e^{-z} <= min(1, z), split at t = 1/2:
        |d| <= 4/(r|x|) + e^{-Ub/2} + e^{-Ub} <= 4/(r|x|) + 2 e^{-r|x|/2}.
Tail integrals (y = 1/x): N tail = int_0^{1/X} d y/(1+y^2) dy, D tail = 2 atan(1/X) + int_0^{1/X} d (1-y^2)/(1+y^2) dy."""
import math
from flint import arb, ctx
ctx.prec = 64


def dbound_pos(x, r):
    return (1 + x * x) * (16 / (r * x ** 3) + math.exp(-r * x / 2))


def dbound_neg(x, r):
    return 4 / (r * abs(x)) + 2 * math.exp(-r * abs(x) / 2)


def tails(X, r):
    """Enclosures of the N and D contributions of x >= X and x <= -X (both sides together).
    The bounds are decreasing in x for r X >= 8, which is asserted, so sup over the tail is the value at X."""
    assert r * X >= 8
    Bp, Bn = arb(dbound_pos(X, r)) * arb("1.000001"), arb(dbound_neg(X, r)) * arb("1.000001")
    h = arb(1) / X
    n_rad = (Bp + Bn) * (h * h / 2)          # int_0^h y/(1+y^2) dy <= h^2/2
    d_rad = (Bp + Bn) * h                    # int_0^h (1-y^2)/(1+y^2) dy <= h
    N = arb(0, n_rad.upper())
    D = 4 * (1 / arb(X)).atan() + arb(0, d_rad.upper())
    return N, D


if __name__ == "__main__":
    from g_stable import d_of
    worst = 0.0
    for r in [0.01, 0.05, 0.23, 1.0, 5.0, 100.0]:
        for x in [10.0, 50.0, 200.0, 1000.0, 5000.0]:
            dp, dn = d_of(x, r), d_of(-x, r)
            bp, bn = dbound_pos(x, r), dbound_neg(-x, r)
            ok = (-1e-12 <= dp <= bp) and abs(dn) <= bn
            worst = max(worst, dp / bp, abs(dn) / bn)
            if not ok:
                print("FAIL", r, x, dp, bp, dn, bn)
        print("r=%g x=1000: d+=%.3e (bound %.3e, x*r*d %.3f)  d-=%.3e (bound %.3e, x*r*d %.3f)" % (
            r, d_of(1000.0, r), dbound_pos(1000.0, r), 1000 * r * d_of(1000.0, r),
            d_of(-1000.0, r), dbound_neg(-1000.0, r), 1000 * r * d_of(-1000.0, r)), flush=True)
    print("worst ratio actual/bound %.3f" % worst)
    for r in [0.01, 1.0]:
        print("tails X=1e4 r=%g:" % r, tails(1e4, r))
