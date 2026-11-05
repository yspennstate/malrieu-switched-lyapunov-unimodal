"""certified N''(r), D''(r) and H'(r) = N''D - ND'' (H = N'D - ND' has the sign of G'; the N'D' terms cancel in H').
Panels of NDprime.panelP with d_r d replaced by d_rr d (drd2.py); strip and tails from drd2_bounds.pieces.
r may be a ball: the panels then hold for every r inside, and the pieces (decreasing in r term by term) are taken at r_lo.
Usage: python NDpp.py r_lo [r_hi]"""
import math, sys, time
from flint import acb, arb, ctx
from drd2 import drr_pos, drr_neg
from drd2_bounds import pieces
from NDprime import gN, gD
ctx.prec = 64


def panelPP(a, b, r, n=24, rho=1.6, m=48):
    neg = b < 0
    f = drr_neg if neg else drr_pos
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


def NDpp(R, r_lo, eps, X, log=None):
    p = pieces(r_lo, eps, X)
    Npp, Dpp = arb(0, p["tN"] + p["sN"]), arb(0, p["tD"] + p["sD"])
    edges = [eps]
    while edges[-1] < X:
        edges.append(min(2 * edges[-1], X))
    for s in (1, -1):
        for a, b in zip(edges[:-1], edges[1:]):
            t0 = time.time()
            lo, hi = (a, b) if s > 0 else (-b, -a)
            pN, pD = panelPP(lo, hi, R)
            Npp, Dpp = Npp + pN, Dpp + pD
            if log:
                print("  [%g,%g] N'' %s D'' %s %.1fs" % (lo, hi, pN, pD, time.time() - t0), file=log, flush=True)
    return Npp, Dpp


if __name__ == "__main__":
    from gp_ball import gp_interval
    r_lo = float(sys.argv[1]); r_hi = float(sys.argv[2]) if len(sys.argv) > 2 else r_lo
    R = acb(arb((r_lo + r_hi) / 2, (r_hi - r_lo) / 2))
    m = min(r_lo, 1.0)
    t0 = time.time()
    Npp, Dpp = NDpp(R, r_lo, 1e-5 * m, 1e5 / m)
    print("r in [%g,%g]: N''=%s D''=%s  %.0fs" % (r_lo, r_hi, Npp, Dpp, time.time() - t0), flush=True)
    N, D, Np, Dp, G, Gp = gp_interval(r_lo, r_hi) if r_hi > r_lo else gp_interval(r_lo, r_lo)
    Hp = Npp * D - N * Dpp
    print("r in [%g,%g]: N=%s D=%s N'=%s D'=%s G'=%s H'=%s sign=%s  %.0fs" % (r_lo, r_hi, N, D, Np, Dp, Gp, Hp,
          "neg" if Hp.upper() < 0 else ("pos" if Hp.lower() > 0 else "UNDECIDED"), time.time() - t0), flush=True)
