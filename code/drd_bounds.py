"""pointwise bounds for d_r d and the strip/tail pieces of N'(r), D'(r).

x > 0, split the u-integral at u0 = r/x.
  u <= u0: r <= r+ux <= 2r, so g >= u x^2/2 and the kernel u^2x^3/(r+ux)^2 <= u^2x^3/r^2;
           int_0^inf u^2 x^3/r^2 e^{-u x^2/2} du = 16/(r^2 x^3).
  u >= u0: g >= r x/2 and the kernel <= x; int e^{-u} <= 1.
  => |d_r d| <= b (16/(r^2 x^3) + x e^{-rx/2})          (used for x >= X)
  Also kernel <= u^2x^3/r^2 everywhere and int u^2 e^{-u} = 2:
  => |d_r d| <= 2 b x^3/r^2                            (used for 0 < x <= eps)
x < 0, t-form d_r d = -bU int_0^1 e^{-Utb} e^{y} x t^2/(1-t)^2 dt, y = -c t^2/(1-t), c = r|x|, bU|x| = r b:
  [0, 1/2]: t^2/(1-t)^2 <= 4t^2, |e^y| <= 1: <= 8/(Ub)^3;
  [1/2, 1]: e^{-Utb} <= e^{-Ub/2} and int |e^y| t^2/(1-t)^2 dt <= 4 e^{-c/2}/c;
  => |d_r d| <= 8|x|^3/(r^2 b^2) + 4 b e^{-(Ub+c)/2}/|x|,  Ub = r b/|x|  (all x < 0)
"""
import math, sys


def bpos_tail(x, r):
    return (1 + x * x) * (16 / (r * r * x ** 3) + x * math.exp(-r * x / 2))


def bpos_strip(x, r):
    return 2 * (1 + x * x) * x ** 3 / (r * r)


def bneg(x, r):
    a = abs(x); b = 1 + x * x; Ub = r * b / a
    return 8 * a ** 3 / (r * r * b * b) + 4 * b * math.exp(-(Ub + r * a) / 2) / a


def pieces(r, eps, X):
    """Bounds on |contribution| to (N', D') from x >= X, x <= -X, 0 < x <= eps, -eps <= x < 0."""
    tN_pos = 16 / (3 * r * r * X ** 3) + 2 * math.exp(-r * X / 2) / r
    tD_pos = 16 / (r * r * X ** 2) + 2 * math.exp(-r * X / 2) * (2 * X / r + 4 / (r * r))
    tN_neg = 8 / (3 * r * r * X ** 3) + 4 * math.exp(-r * X) / (r * X * X)
    tD_neg = 4 / (r * r * X ** 2) + 8 * math.exp(-r * X) / (r * X)
    sN_pos = 2 * eps ** 3 / (3 * r * r)
    sD_pos = eps ** 2 / (r * r)
    a = r / 2
    sN_neg = 8 * eps ** 3 / (3 * r * r) + 4 * (2 / r) * math.exp(-r / (2 * eps))
    sD_neg = 4 * eps ** 2 / (r * r) + 4 * (1 + eps * eps) * math.exp(-a / eps) * (1 / (a * eps) + 1 / (a * a))
    return dict(tN=tN_pos + tN_neg, tD=tD_pos + tD_neg, sN=sN_pos + sN_neg, sD=sD_pos + sD_neg)


if __name__ == "__main__":
    from flint import acb
    from drd import drd_pos, drd_neg
    worst = 0.0
    for r in (0.01, 0.1, 1.0, 10.0, 100.0):
        for k in range(-60, 61):
            for s in (1, -1):
                x = s * 10 ** (k / 10)
                v = abs(float((drd_pos if s > 0 else drd_neg)(acb(x), acb(r)).real.mid()))
                if s > 0:
                    bs = [bpos_strip(x, r)] + ([bpos_tail(x, r)] if r * x >= 8 else [])
                    bd = min(bs)
                else:
                    bd = bneg(x, r)
                ratio = v / bd if bd > 0 else float("inf")
                worst = max(worst, ratio)
                if ratio > 1:
                    print("VIOLATION r=%g x=%g |drd|=%g bound=%g" % (r, x, v, bd))
        eps, X = 1e-5 * min(r, 1.0), 1e4 / min(r, 1.0)
        print("r=%g eps=%g X=%g" % (r, eps, X), {k: "%.2e" % v for k, v in pieces(r, eps, X).items()}, flush=True)
    print("worst ratio %.3f" % worst)
