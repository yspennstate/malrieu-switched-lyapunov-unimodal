"""large r: -H+ = int J_r [D+/x + N+/x^2 - N+] dx > 0 from crude bounds (H = 4 H+, sign G' = sign H).
Bounds used (proved in Section 4 of the paper):
  0 <= d <= 4x^3/(r b^2) + x^2 e^{-r b/(2x)}      =>  0 <= N+ <= pi/(4r) + K1(r) =: Nbar
  D+ >= pi/2
  J_r <= 16/(r^2 x^3) + x e^{-r x/2}               =>  int_X^inf J_r <= 8/(r^2 X^2) + e^{-rX/2}(2X/r + 4/r^2)
  int_{1/2}^1 J_r >= 0.016 (1 - Q)/r^2, Q = e^{-a}(1 + a + a^2/2), a = 5r/2
  -H+ >= (pi/2 - Nbar) * 0.016(1-Q)/r^2 - Nbar * tail(X), X = (pi/2)/Nbar.
Checks: (1) the bound LB(r) on a log grid, smallest r where LB > 0; (2) the bounds on d, J_r against quadrature;
(3) 8 r G -> 1 (leading term G ~ 1/(8r))."""
import math, sys
from scipy.integrate import quad
from scipy.special import k1
from g_stable import d_of, G

pi = math.pi


def LB(r):
    Nb = pi / (4 * r) + k1(r)
    X = (pi / 2) / Nb
    a = 2.5 * r
    Q = math.exp(-a) * (1 + a + a * a / 2)
    pos = (pi / 2 - Nb) * 0.016 * (1 - Q) / r ** 2
    neg = Nb * (8 / (r * r * X * X) + math.exp(-r * X / 2) * (2 * X / r + 4 / r ** 2))
    return pos - neg, pos, neg


def Jr_t(x, r):
    f = lambda t: r * t * t / (1 + t) ** 2 * math.exp(-r * (x * t / (1 + t) + t / x))
    sc = min(1 / (r * x), x / r, 1.0)
    pts = [0, sc, 10 * sc, 100 * sc]
    return sum(quad(f, a, b, limit=200, epsabs=0, epsrel=1e-10)[0] for a, b in zip(pts[:-1], pts[1:])) + \
        quad(f, pts[-1], math.inf, limit=200, epsabs=0, epsrel=1e-10)[0]


if __name__ == "__main__":
    print("(1) LB(r) = bound on -H+ (pos part, neg part):")
    first = None
    r = 0.5
    while r <= 1e6:
        lb, p, n = LB(r)
        if lb > 0 and first is None:
            first = r
        if lb <= 0:
            first = None
        r *= 1.05
    for r in (1, 2, 5, 10, 100, 1e3, 1e4, 1e6):
        lb, p, n = LB(r)
        print("  r=%-8g LB=%.4e  r^2 LB=%.5f  pos=%.3e neg=%.3e" % (r, lb, r * r * lb, p, n))
    print("  LB > 0 on the whole grid from r = %.4g to 1e6 (step 1.05)" % first)
    print("  asymptotics: r^2 LB -> 0.016*pi/2 = %.5f, neg = O(r^-5)" % (0.016 * pi / 2))
    print("(2) bounds vs quadrature:")
    for r in (2.0, 10.0, 100.0):
        for x in (0.05, 0.5, 1.0, 3.0, 30.0, 300.0):
            b = 1 + x * x
            d = d_of(x, r)
            dub = 4 * x ** 3 / (r * b * b) + x * x * math.exp(-r * b / (2 * x))
            j = Jr_t(x, r)
            jub = 16 / (r * r * x ** 3) + x * math.exp(-r * x / 2)
            flag = "ok" if (-1e-14 <= d <= dub and 0 <= j <= jub) else "FAIL"
            print("  r=%-5g x=%-6g d=%.4e <= %.4e   J_r=%.4e <= %.4e  %s" % (r, x, d, dub, j, jub, flag))
        lo = quad(lambda x: Jr_t(x, r), 0.5, 1.0, epsrel=1e-8)[0]
        print("  r=%-5g int_{1/2}^1 J_r = %.5e >= %.5e : %s" % (r, lo, 0.016 / r ** 2 * (1 - math.exp(-2.5 * r) * (1 + 2.5 * r + 3.125 * r * r)),
                                                          lo >= 0.016 / r ** 2 * (1 - math.exp(-2.5 * r) * (1 + 2.5 * r + 3.125 * r * r))))
    print("(3) 8rG:")
    for r in (10.0, 30.0, 100.0):
        print("  r=%-5g 8rG=%.6f" % (r, 8 * r * G(r)), flush=True)
