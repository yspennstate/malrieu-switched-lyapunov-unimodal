"""e3 = r^2 D' + 2 = 2r^2 [ I1 - I2 ], I1 = int J_r/x^2 dx, I2 = int (J_r - x e^{-rx}) dx, J_r = int e^{-t} B_r dt.
Cancellation-free B_r: with y = t x^2/b and f(y) = -t^2 x^3/b^2, t x^2 - bB = t^2 x^3/(b (rb + tx - x(y+B))), and
sqrt(disc) = sqrt((tx - rb)^2 + 4 r t x), so B_r = t^2 x^3 / (b (rb + tx - x(y+B)) sqrt((tx-rb)^2 + 4rtx)).
Prints I1, I2, 2r^2(I1 - I2) against the measured e3 = 0.0028 (small_r_eps.py, r = 0.01)."""
import math, sys
from scipy.integrate import quad
from sandwich import Bexact


def Br(t, x, r):
    b = 1 + x * x
    B, y = Bexact(t, x, r), t * x * x / b
    return t * t * x ** 3 / (b * (r * b + t * x - x * (y + B)) * math.sqrt((t * x - r * b) ** 2 + 4 * r * t * x))


def Jr(x, r):
    c = r * x
    pts = sorted(set([0, min(c, 40), min(r * (1 + x * x) / x, 40), 40]))
    return sum(quad(lambda t: math.exp(-t) * Br(t, x, r), a, e, limit=200, epsabs=1e-18, epsrel=1e-11)[0]
               for a, e in zip(pts[:-1], pts[1:]) if e > a)


for r in ([float(a) for a in sys.argv[1:]] or [0.01]) if __name__ == "__main__" else []:
    P = sorted(set([0, r / 2, r, 2 * r, 0.1, 1, 10] + [k / r for k in (0.1, 1, 3, 10, 30, 100)] + [3e3 / r]))
    S = lambda f: sum(quad(f, a, b, limit=300, epsrel=1e-9, epsabs=1e-16)[0] for a, b in zip(P[:-1], P[1:]))
    I1 = S(lambda x: Jr(x, r) / (x * x) if x > 0 else 0.0)
    I2 = S(lambda x: Jr(x, r) - x * math.exp(-r * x))
    print("r=%g I1=%.6g I2=%.6g e3=2r^2(I1-I2)=%.6g" % (r, I1, I2, 2 * r * r * (I1 - I2)), flush=True)
