"""certified N(r), D(r): strip |x| <= eps (near0.py) + graded ellipse panels [a, 2a] on eps..X each side
(d_acb for x > 0, d_acb_neg for x < 0) + tails |x| >= X (tails.py). G = N/D enclosed."""
import math, sys, time
from flint import acb, arb, ctx


def tol(x, r):
    """abs_tol for acb.integral, scaled to the size of d (~|x|^3/r near 0) and d_r d (~|x|^3/r^2); the default
    (~2^-prec) is coarser than both for |x| small. It sets tightness only; the enclosure is valid for any value."""
    a = float(abs(x).upper()); rr = float(abs(r).upper())
    return arb(2) ** -66 * min(1.0, a ** 3) / max(1.0, rr * rr)


def d_acb(x, r, L=50):
    """x > 0 side, cancellation-free: d = b int_0^inf e^{-u}(e^{-g} - e^{-u x^2}) du, g = r u x^2/(r+ux).
    With |arg x| < pi/4: arg g in [arg x, 2 arg x] so Re g >= 0, and Re x^2 > 0; tail beyond L <= 2 e^{-L}."""
    assert x.real.lower() > abs(x.imag).upper()
    b = 1 + x * x
    x2 = x * x
    # e^{-g} - e^{-ux^2} = -e^{-g} expm1(-w), w = u^2 x^3/(r+ux) (no cancellation for small x)
    h = lambda u, an: -(-u).exp() * (-(r * u * x2) / (r + u * x)).exp() * (-(u * u * x2 * x) / (r + u * x)).expm1()
    t = 2 * arb(-L).exp()
    return b * (acb.integral(h, 0, L, abs_tol=tol(x, r)) + acb(arb(0, t.upper()), arb(0, t.upper())))


def d_acb_neg(x, r, delta=arb("1e-15"), S=60):
    """x < 0 side, t-form (cert_neg.py) with the t-range cut at T = S/Re(Ub) when that is below 1 - delta:
    on [T, 1], |e^{-Utb}| <= e^{-S} and |e^{y} - 1| <= 2 (Re y <= 0), so that piece is <= 2 e^{-S}."""
    b = 1 + x * x
    assert x.real.upper() < 0 and b.real.lower() > 0
    U = -r / x
    Ub = U * b
    reub = Ub.real.lower()
    h = lambda t, an: (-U * t * b).exp() * (r * x * t * t / (1 - t)).expm1()
    if reub > S / (1 - float(delta)):
        v = acb.integral(h, 0, arb(S) / reub, abs_tol=tol(x, r))
        tail = 2 * arb(-S).exp()
    else:
        v = acb.integral(h, 0, 1 - delta, abs_tol=tol(x, r))
        m = max(arb(0), -reub)
        tail = 2 * delta * abs(U).upper() * m.exp()
    return b * U * (v + acb(arb(0, tail.upper()), arb(0, tail.upper()))) - (-Ub).exp()
from near0 import strip
from tails import tails
ctx.prec = 64


def fN(d, x):
    return d / (x * (1 + x * x))


def fD(d, x):
    return 2 / (1 + x * x) + (x * x - 1) * d / (x * x * (1 + x * x))


def panel2(a, b, r, n=24, rho=1.6, m=48):
    """Enclose (int_a^b fN, int_a^b fD) for a panel on one side of 0 (a < b, same sign)."""
    neg = b < 0
    dfun = d_acb_neg if neg else d_acb
    A, B, R = arb(a), arb(b), acb(r)
    c, hw = (A + B) / 2, (B - A) / 2
    sN = sD = acb(0)
    for k in range(n):
        t, w = arb.legendre_p_root(n, k, weight=True)
        x = acb(c + hw * t)
        d = dfun(x, R)
        sN += w * fN(d, x)
        sD += w * fD(d, x)
    MN = MD = arb(0)
    for j in range(m):
        th0, th1 = 2 * math.pi * j / m, 2 * math.pi * (j + 1) / m
        z = lambda th: acb(c) + acb(hw) * (acb(rho * math.cos(th), rho * math.sin(th)) + acb(math.cos(th) / rho, -math.sin(th) / rho)) / 2
        zc = z((th0 + th1) / 2)
        rad = 1.05 * max(abs(z(th0) - zc).upper(), abs(z(th1) - zc).upper())
        box = acb(arb(zc.real.mid(), rad), arb(zc.imag.mid(), rad))
        assert (box.real.upper() < 0) if neg else (box.real.lower() > 0)
        d = dfun(box, R)
        MN = max(MN, abs(fN(d, box)).upper())
        MD = max(MD, abs(fD(d, box)).upper())
    k = hw * 64 / (15 * (arb(rho) ** 2 - 1) * arb(rho) ** (2 * n))
    return sN.real * hw + arb(0, (k * MN).upper()), sD.real * hw + arb(0, (k * MD).upper())


def ND(r, eps=1e-5, X=1e4, log=None):
    N, D = strip(eps, r)
    tN, tD = tails(X, r)
    N, D = N + tN, D + tD
    edges = [eps]
    while edges[-1] < X:
        edges.append(min(2 * edges[-1], X))
    for s in (1, -1):
        for a, b in zip(edges[:-1], edges[1:]):
            t0 = time.time()
            lo, hi = (a, b) if s > 0 else (-b, -a)
            pN, pD = panel2(lo, hi, r)
            N, D = N + pN, D + pD
            if log:
                print("  [%g,%g] N %s D %s %.1fs" % (lo, hi, pN, pD, time.time() - t0), file=log, flush=True)
    return N, D


if __name__ == "__main__":
    r = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    if len(sys.argv) > 2 and sys.argv[2] == "probe":
        for a, b in [(5000.0, 10000.0), (1e-5, 2e-5), (-10000.0, -5000.0), (-2e-5, -1e-5), (0.5, 1.0)]:
            t0 = time.time(); print((a, b), panel2(a, b, r), "%.1fs" % (time.time() - t0), flush=True)
        sys.exit()
    t0 = time.time()
    N, D = ND(r, log=sys.stdout)
    print("r=%g N=%s D=%s G=%s  %.0fs" % (r, N, D, N / D, time.time() - t0))
