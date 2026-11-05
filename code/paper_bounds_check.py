"""checks of the simplified small-r bounds used in the paper.
(i)  D+ = 4r int_0^inf u (1+w^2)/(1-w^2) K0(2ru) du, w = w(u) in (0,1] with u^2 w = (1-w)^2, against small_r_ND.
(ii) e1 = rD - 2 = 16 r^2 int_0^inf u w^2/(1-w^2) K0(2ru) du in (0, 8 r^2 (1 + 2 K0(2r))].
(iii) N <= 2 ln(1/r) + 4 K0(2 - 2 r^2) and N >= 2 ln(1/r) - 2 gamma.
(iv) the assembly lower bound on r^2 H/4 with these bounds, on (0, 0.01]."""
import math
from scipy.integrate import quad
from scipy.special import k0, k1
from small_r_ND import ND

g = 0.5772156649015329


def w_of(u):
    # positive root of u^2 w = (1-w)^2 in (0,1]: w = ((2+u^2) - u sqrt(u^2+4))/2
    return ((2 + u * u) - u * math.sqrt(u * u + 4)) / 2


def Dplus(r):
    f = lambda u: u * (1 + w_of(u) ** 2) / (1 - w_of(u) ** 2) * k0(2 * r * u) if u > 0 else 0.5 * k0(1e-300)
    P = [0, 1e-6, 1, 10, 1 / r, 10 / r, 100 / r]
    return 4 * r * sum(quad(f, a, b, limit=400)[0] for a, b in zip(P[:-1], P[1:]))


def e1_exact(r):
    def f(u):
        w = w_of(u)
        return u * w * w / ((1 - w) * (1 + w)) * k0(2 * r * u)
    P = [1e-12, 1e-6, 1, 10, 1 / r, 10 / r, 100 / r]
    return 16 * r * r * sum(quad(f, a, b, limit=400)[0] for a, b in zip(P[:-1], P[1:]))


for r in (0.001, 0.01, 0.03, 0.1, 0.5):
    N, D = ND(r)
    L = math.log(1 / r)
    print("r=%-6g D/2=%.6f D+bessel=%.6f | e1=%.6g e1_bessel=%.6g bound=%.6g | N=%.5f lo=%.5f hi=%.5f" % (
        r, D / 2, Dplus(r), r * D - 2, e1_exact(r), 8 * r * r * (1 + 2 * k0(2 * r)),
        N, 2 * L - 2 * g, 2 * L + 4 * k0(2 - 2 * r * r)), flush=True)

# (iv) assembly on (0, 0.01]: r^2 H/4 >= (L - g - 1) - e1/2 - e3 N/4 with
# e1 <= 8r^2(1 + 2K0(2r)), e3 <= 4r^2(1 + 2K0(2 sqrt2 r)), N <= 2L + 4K0(2-2r^2)
worst = 1e9
for k in range(0, 1001):
    r = 10 ** (-12 + 10 * k / 1000)
    L = math.log(1 / r)
    e1 = 8 * r * r * (1 + 2 * k0(2 * r))
    e3 = 4 * r * r * (1 + 2 * k0(2 * math.sqrt(2) * r))
    Nb = 2 * L + 4 * k0(2 - 2 * r * r)
    lb = (L - g - 1) - e1 / 2 - e3 * Nb / 4
    el = math.exp(-2 * L) * (2 * L + 0.456) * (2 * L + 4)  # the elementary form in the paper
    assert e1 / 2 + e3 * Nb / 4 <= el + 1e-15, (r, e1 / 2 + e3 * Nb / 4, el)
    worst = min(worst, lb)
    if k == 1000:
        print("r=0.01: lower bound on r^2H/4 = %.5f, loss = %.5f, elementary loss = %.5f, L-g-1 = %.5f" % (
            lb, e1 / 2 + e3 * Nb / 4, el, L - g - 1))
print("min lower bound over [1e-12, 0.01] = %.5f; elementary form dominates at all 1001 points" % worst)
print("K0(1) = %.6f, K0(2) = %.6f, K1(5) = %.6g" % (k0(1), k0(2), k1(5)))
