"""enclosure of G'(r) for ALL r in [r_lo, r_hi] at once: the panels take r as a ball, so their enclosures hold for
every r inside; the strip and tail bounds (near0.strip, tails.tails, drd_bounds.pieces) decrease in r in every term
(powers of 1/r and e^{-r ...}), so evaluated at r_lo they bound the whole interval. eps = 1e-5 min(r_lo,1),
X = 1e4/min(r_lo,1) for N, D and 1e5/min(r_lo,1) for N', D'."""
import sys, time
from flint import acb, arb, ctx
from assemble import panel2
from NDprime import panelP
from near0 import strip
from tails import tails
from drd_bounds import pieces
ctx.prec = 64


def edges(eps, X):
    e = [eps]
    while e[-1] < X:
        e.append(min(2 * e[-1], X))
    return e


def gp_interval(r_lo, r_hi, log=None):
    R = acb(arb((r_lo + r_hi) / 2, (r_hi - r_lo) / 2))
    m = min(r_lo, 1.0)
    eps, X, Xp = 1e-5 * m, 1e4 / m, 1e5 / m
    N, D = strip(eps, r_lo)
    tN, tD = tails(X, r_lo)
    N, D = N + tN, D + tD
    p = pieces(r_lo, eps, Xp)
    Np, Dp = arb(0, p["tN"] + p["sN"]), arb(0, p["tD"] + p["sD"])
    for s in (1, -1):
        for a, b in zip(edges(eps, X)[:-1], edges(eps, X)[1:]):
            lo, hi = (a, b) if s > 0 else (-b, -a)
            pN, pD = panel2(lo, hi, R)
            N, D = N + pN, D + pD
        for a, b in zip(edges(eps, Xp)[:-1], edges(eps, Xp)[1:]):
            lo, hi = (a, b) if s > 0 else (-b, -a)
            pN, pD = panelP(lo, hi, R)
            Np, Dp = Np + pN, Dp + pD
    G = N / D
    Gp = (Np * D - N * Dp) / (D * D)
    return N, D, Np, Dp, G, Gp


if __name__ == "__main__":
    r_lo, r_hi = float(sys.argv[1]), float(sys.argv[2])
    t0 = time.time()
    N, D, Np, Dp, G, Gp = gp_interval(r_lo, r_hi)
    print("r in [%g,%g]: G=%s G'=%s  sign=%s  %.0fs" % (r_lo, r_hi, G, Gp,
          "neg" if Gp.upper() < 0 else ("pos" if Gp.lower() > 0 else "UNDECIDED"), time.time() - t0), flush=True)
