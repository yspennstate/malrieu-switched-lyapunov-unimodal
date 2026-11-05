"""pointwise bounds for d_rr d and the strip/tail pieces of N''(r), D''(r) (same layout as drd_bounds.py).

d_rr d = b int e^{-u} e^{-g} K du, K = u^4x^6/s^4 + 2u^2x^3/s^3, s = r+ux, g = r u x^2/s (drd2.py).
x > 0, split at u0 = r/x.
  u <= u0: r <= s <= 2r, g >= u x^2/2, K <= u^4x^6/r^4 + 2u^2x^3/r^3;
           int u^4 e^{-ux^2/2} = 768/x^10, int u^2 e^{-ux^2/2} = 16/x^6  ->  768/(r^4x^4) + 32/(r^3x^3).
  u >= u0: g >= rx/2, ux/s <= 1, s >= 2r, so K <= x^2 + x/r; int e^{-u} <= 1.
  => |d_rr d| <= b (768/(r^4x^4) + 32/(r^3x^3) + (x^2 + x/r) e^{-rx/2})        (tail)
  K <= u^4x^6/r^4 + 2u^2x^3/r^3 everywhere, int u^4 e^{-u} = 24, int u^2 e^{-u} = 2:
  => |d_rr d| <= b (24 x^6/r^4 + 4 x^3/r^3)                                     (strip)
x < 0, t-form (drd2.py), U = r/|x|, Ub = r b/|x|, bU = r b/|x|, a = r|x|/4:
  [0, 1/2]: t^4/(1-t)^4 <= 16 t^4, t^2/(1-t)^3 <= 8 t^2, |e^y| <= 1:  384|x|^6/(r^4 b^4) + 32|x|^3/(r^3 b^2);
  [1/2, 1]: e^{-Utb} <= e^{-Ub/2}, int |e^y| K_t dt <= x^2 J2 + (2|x|/r) J1 with s0 = 2:
            J2 = e^{-2a}(4/a + 4/a^2 + 2/a^3), J1 = e^{-2a}(2/a + 1/a^2).
  => |d_rr d| <= 384|x|^6/(r^4b^4) + 32|x|^3/(r^3b^2) + (r b/|x|) e^{-Ub/2} (x^2 J2 + 2|x| J1/r)
"""
import math


def bpos_tail(x, r):
    return (1 + x * x) * (768 / (r ** 4 * x ** 4) + 32 / (r ** 3 * x ** 3) + (x * x + x / r) * math.exp(-r * x / 2))


def bpos_strip(x, r):
    return (1 + x * x) * (24 * x ** 6 / r ** 4 + 4 * x ** 3 / r ** 3)


def bneg(x, r):
    ax = abs(x); b = 1 + x * x; Ub = r * b / ax; a = r * ax / 4
    e = math.exp(-2 * a)
    J2 = e * (4 / a + 4 / a ** 2 + 2 / a ** 3); J1 = e * (2 / a + 1 / a ** 2)
    return 384 * ax ** 6 / (r ** 4 * b ** 4) + 32 * ax ** 3 / (r ** 3 * b * b) + \
        (r * b / ax) * math.exp(-Ub / 2) * (x * x * J2 + 2 * ax * J1 / r)


def pieces(r, eps, X):
    """Bounds on |contribution| to (N'', D'') from x >= X, x <= -X, 0 < x <= eps, -eps <= x < 0.
    Needs X >= 1, r X >= 4 (so a >= 1 on the tail) and r eps <= 4 (a <= 1 on the strip)."""
    assert X >= 1 and r * X >= 4 and r * eps <= 4 and eps < 1
    E = math.exp(-r * X / 2)
    tN_pos = 384 / (r ** 4 * X ** 4) + 64 / (3 * r ** 3 * X ** 3) + 2 * E * (2 * X / r + 6 / r ** 2)
    tD_pos = 512 / (r ** 4 * X ** 3) + 32 / (r ** 3 * X ** 2) + 2 * E * (2 * X * X / r + 10 * X / r ** 2 + 20 / r ** 3)
    tN_neg = 96 / (r ** 4 * X ** 4) + 32 / (3 * r ** 3 * X ** 3) + (2 / r) * E * (80 / X + 48 / (r * X * X))
    tD_neg = 128 / (r ** 4 * X ** 3) + 16 / (r ** 3 * X ** 2) + (2 / r) * E * (80 + 48 / (r * X))
    sN_pos = 4 * eps ** 6 / r ** 4 + 4 * eps ** 3 / (3 * r ** 3)
    sD_pos = 24 * eps ** 5 / (5 * r ** 4) + 2 * eps ** 2 / r ** 3
    Es = math.exp(-r / (2 * eps))
    sN_neg = 64 * eps ** 6 / r ** 4 + 32 * eps ** 3 / (3 * r ** 3) + 736 / (r * r * eps * eps) * Es
    sD_neg = 384 * eps ** 5 / (5 * r ** 4) + 16 * eps ** 2 / r ** 3 + 736 / (r * r * eps ** 3) * Es
    return dict(tN=tN_pos + tN_neg, tD=tD_pos + tD_neg, sN=sN_pos + sN_neg, sD=sD_pos + sD_neg)


if __name__ == "__main__":
    from flint import acb
    from drd2 import drr_pos, drr_neg
    worst = 0.0
    for r in (0.01, 0.1, 0.22, 1.0, 10.0):
        for k in range(-60, 61):
            for s in (1, -1):
                x = s * 10 ** (k / 10)
                v = abs(float((drr_pos if s > 0 else drr_neg)(acb(x), acb(r)).real.mid()))
                bd = min(bpos_strip(x, r), bpos_tail(x, r)) if s > 0 else bneg(x, r)
                ratio = v / bd if bd > 0 else float("inf")
                worst = max(worst, ratio)
                if ratio > 1:
                    print("VIOLATION r=%g x=%g |drr|=%g bound=%g" % (r, x, v, bd))
        eps, X = 1e-5 * min(r, 1.0), 1e5 / min(r, 1.0)
        print("r=%g eps=%g X=%g" % (r, eps, X), {k: "%.2e" % v for k, v in pieces(r, eps, X).items()}, flush=True)
    print("worst ratio %.3f" % worst)
