"""certified N'(r), D'(r), G'(r): the panels of assemble.py with d replaced by d_r d (drd.py), the strip |x| <= eps and
tails |x| >= X bounded by drd_bounds.pieces. N' = int d_r d/(x(1+x^2)), D' = int (x^2-1) d_r d/(x^2(1+x^2)) (the 2/(1+x^2)
term of D does not depend on r). G' = (N'D - ND')/D^2 with N, D from assemble.ND."""
import math, sys, time
from flint import acb, arb, ctx
from drd import drd_pos, drd_neg
from drd_bounds import pieces
from assemble import ND
ctx.prec = 64


def gN(v, x):
    return v / (x * (1 + x * x))


def gD(v, x):
    return (x * x - 1) * v / (x * x * (1 + x * x))


def panelP(a, b, r, n=24, rho=1.6, m=48):
    neg = b < 0
    f = drd_neg if neg else drd_pos
    A, B, R = arb(a), arb(b), acb(r)
    c, hw = (A + B) / 2, (B - A) / 2
    sN = sD = acb(0)
    for k in range(n):
        t, w = arb.legendre_p_root(n, k, weight=True)
        x = acb(c + hw * t)
        v = f(x, R)
        sN += w * gN(v, x)
        sD += w * gD(v, x)
    MN = MD = arb(0)
    for j in range(m):
        th0, th1 = 2 * math.pi * j / m, 2 * math.pi * (j + 1) / m
        z = lambda th: acb(c) + acb(hw) * (acb(rho * math.cos(th), rho * math.sin(th)) + acb(math.cos(th) / rho, -math.sin(th) / rho)) / 2
        zc = z((th0 + th1) / 2)
        rad = 1.05 * max(abs(z(th0) - zc).upper(), abs(z(th1) - zc).upper())
        box = acb(arb(zc.real.mid(), rad), arb(zc.imag.mid(), rad))
        assert (box.real.upper() < 0) if neg else (box.real.lower() > abs(box.imag).upper())
        v = f(box, R)
        MN = max(MN, abs(gN(v, box)).upper())
        MD = max(MD, abs(gD(v, box)).upper())
    k = hw * 64 / (15 * (arb(rho) ** 2 - 1) * arb(rho) ** (2 * n))
    return sN.real * hw + arb(0, (k * MN).upper()), sD.real * hw + arb(0, (k * MD).upper())


def NDp(r, eps, X, log=None):
    p = pieces(r, eps, X)
    Np = arb(0, p["tN"] + p["sN"])
    Dp = arb(0, p["tD"] + p["sD"])
    edges = [eps]
    while edges[-1] < X:
        edges.append(min(2 * edges[-1], X))
    for s in (1, -1):
        for a, b in zip(edges[:-1], edges[1:]):
            t0 = time.time()
            lo, hi = (a, b) if s > 0 else (-b, -a)
            pN, pD = panelP(lo, hi, r)
            Np, Dp = Np + pN, Dp + pD
            if log:
                print("  [%g,%g] N' %s D' %s %.1fs" % (lo, hi, pN, pD, time.time() - t0), file=log, flush=True)
    return Np, Dp


if __name__ == "__main__":
    r = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    eps, X = 1e-5 * min(r, 1.0), 1e5 / min(r, 1.0)
    t0 = time.time()
    Np, Dp = NDp(r, eps, X, log=sys.stdout)
    print("r=%g N'=%s D'=%s  %.0fs" % (r, Np, Dp, time.time() - t0), flush=True)
    N, D = ND(r, eps=1e-5 * min(r, 1.0), X=1e4 / min(r, 1.0))
    Gp = (Np * D - N * Dp) / (D * D)
    print("r=%g N=%s D=%s G=%s G'=%s  %.0fs" % (r, N, D, N / D, Gp, time.time() - t0))
