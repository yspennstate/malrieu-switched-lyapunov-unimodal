"""the strip |x| <= eps. d is C^infinity but not analytic at 0 (the x<0 branch carries e^{-r(1+x^2)/|x|}),
so no ellipse panel may contain 0; the strip is bounded directly and graded panels start at eps.
x > 0: 0 <= e^{-g} - e^{-ux^2} <= u x^2 - g = u^2 x^3/(r+ux) <= u^2 x^3/r, so 0 <= d <= (1+x^2) 2 x^3/r.
x < 0 (u-form, u in [0, U], U = r/|x|, y = u^2 x^3/(r+ux) <= 0, |e^y - 1| <= min(1, |y|); r - u|x| >= r/2 for u <= U/2):
       |d| <= (1+x^2)(4|x|^3/r + e^{-r/(2|x|)}) + e^{-r/|x|}.
Strip contributions (eps <= r/2 so e^{-r/(2x)}/x^k is increasing on (0, eps]):
  N: int_{-eps}^{eps} |d|/(|x|(1+x^2)) <= (1+eps^2)(2/3 + 4/3) eps^3/r + 2 eps e^{-r/(2 eps)}/eps (1+eps^2)
  D: 4 atan(eps) + radius int |d| (1-x^2)/(x^2(1+x^2)) <= (1+eps^2)(1 + 2) eps^2/r + 2 eps e^{-r/(2eps)}/eps^2 (1+eps^2)."""
import math
from flint import arb


def dnear_pos(x, r):
    return (1 + x * x) * 2 * x ** 3 / r


def dnear_neg(x, r):
    a = abs(x)
    return (1 + x * x) * (4 * a ** 3 / r + math.exp(-r / (2 * a))) + math.exp(-r / a)


def strip(eps, r):
    assert eps <= r / 2
    e1 = 1 + eps * eps
    ex = 2 * e1 * math.exp(-r / (2 * eps))
    n_rad = e1 * 2 * eps ** 3 / r + ex
    d_rad = e1 * 3 * eps ** 2 / r + ex / eps
    return arb(0, n_rad * 1.000001), 4 * arb(eps).atan() + arb(0, d_rad * 1.000001)


if __name__ == "__main__":
    from g_stable import d_of
    worst = 0.0
    for r in [0.01, 0.05, 0.23, 1.0, 5.0, 100.0]:
        for x in [1e-4, 1e-3, 3e-3, 1e-2, 0.05, 0.2]:
            if x > r / 2:
                continue
            dp, dn = d_of(x, r), d_of(-x, r)
            bp, bn = dnear_pos(x, r), dnear_neg(-x, r)
            if not (-1e-14 <= dp <= bp and abs(dn) <= bn):
                print("FAIL", r, x, dp, bp, dn, bn)
            worst = max(worst, abs(dp) / bp, abs(dn) / bn)
        x = min(1e-3, r / 4)
        print("r=%g x=%g: r d/x^3 = %+.4f (x>0), %+.4f (x<0)" % (r, x, r * d_of(x, r) / x ** 3, r * d_of(-x, r) / x ** 3))
    print("worst ratio actual/bound %.3f" % worst)
    for r in [0.01, 1.0]:
        print("strip eps=1e-5 r=%g:" % r, strip(1e-5, r))
