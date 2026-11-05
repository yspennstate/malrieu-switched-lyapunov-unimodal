"""ball-arithmetic enclosures of d_r d(x), the r-derivative of d, for the certified N'(r), D'(r).

d_r d(x) = -(1+x^2) int_0^U exp(-u(1+x^2) + u^2x^3/(r+ux)) u^2x^3/(r+ux)^2 du  (gprime.py).
The exponent is -u - g with g = r u x^2/(r+ux), the same g as in d_acb.

x > 0 (|arg x| < pi/4): Re g >= 0 and |r+ux| >= u Re x, so the integrand is at most e^{-u} |x|^3/(Re x)^2;
    the tail beyond L is at most |x|^3/(Re x)^2 e^{-L}.
x < 0: u = U t, U = -r/x, b = 1+x^2:
    d_r d = -b U int_0^1 e^{-U t b} e^{y} x t^2/(1-t)^2 dt,  y = r x t^2/(1-t).
    With c = -Re(r x) > 0 and s = 1/(1-t): on [0, 1/2] the integrand of t is at most 1 in modulus (Re y <= 0);
    on [1/2, 1], Re y <= -c s/4 and t^2/(1-t)^2 dt <= ds, so int_a^1 |e^y| t^2/(1-t)^2 dt <= 4 e^{-c/(4(1-a))}/c
    for a >= 1/2, and int_0^1 ... <= 1/2 + 4 e^{-c/2}/c.
    Cut at T = S/Re(Ub) when Re(Ub) is large (|e^{-Utb}| <= e^{-S} beyond), else at 1 - delta.
"""
from flint import acb, arb, ctx
ctx.prec = 64


def tol(x, r):
    """abs_tol for acb.integral, scaled to the size of d (~|x|^3/r near 0) and d_r d (~|x|^3/r^2); the default
    (~2^-prec) is coarser than both for |x| small. It sets tightness only; the enclosure is valid for any value."""
    a = float(abs(x).upper()); rr = float(abs(r).upper())
    return arb(2) ** -66 * min(1.0, a ** 3) / max(1.0, rr * rr)


def drd_pos(x, r, L=50):
    assert x.real.lower() > abs(x.imag).upper()
    b = 1 + x * x
    x2, x3 = x * x, x * x * x

    def h(u, an):
        s = r + u * x
        return (-u).exp() * (-(r * u * x2) / s).exp() * u * u * x3 / (s * s)
    rex = x.real.lower()
    t = (abs(x) ** 3 / (arb(rex) ** 2) * arb(-L).exp()).upper()
    return -b * (acb.integral(h, 0, L, abs_tol=tol(x, r)) + acb(arb(0, t), arb(0, t)))


def drd_neg(x, r, delta=arb("1e-15"), S=60):
    b = 1 + x * x
    assert x.real.upper() < 0 and b.real.lower() > 0
    U = -r / x
    Ub = U * b
    reub = Ub.real.lower()
    c = (-(r * x).real).lower()
    assert c > 0

    def h(t, an):
        return (-U * t * b).exp() * (r * x * t * t / (1 - t)).exp() * x * t * t / ((1 - t) * (1 - t))
    if reub > S / (1 - float(delta)):
        T = arb(S) / reub
        v = acb.integral(h, 0, T, abs_tol=tol(x, r))
        tail = abs(x) * arb(-S).exp() * (arb(1) / 2 + 4 * (-arb(c) / 2).exp() / c)
    else:
        v = acb.integral(h, 0, 1 - delta, abs_tol=tol(x, r))
        m = max(arb(0), -reub)
        tail = abs(x) * m.exp() * 4 * (-arb(c) / (4 * delta)).exp() / c
    tu = tail.upper()
    return -b * U * (v + acb(arb(0, tu), arb(0, tu)))


if __name__ == "__main__":
    import sys, time
    from assemble import d_acb, d_acb_neg
    r = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    hstep = 1e-6
    for xv in (2e-5, 0.01, 0.7, 3.0, 60.0, 8000.0, -2e-5, -0.01, -0.7, -3.0, -60.0, -8000.0):
        x = acb(xv)
        f = d_acb if xv > 0 else d_acb_neg
        g = drd_pos if xv > 0 else drd_neg
        t0 = time.time()
        v = g(x, acb(r))
        fd = (f(x, acb(r + hstep)) - f(x, acb(r - hstep))) / (2 * hstep)
        print("x=%g drd=%s  cdiff=%.10e  rel=%.1e  %.2fs" % (xv, v.real, float(fd.real.mid()),
              abs(float(v.real.mid()) - float(fd.real.mid())) / max(1e-300, abs(float(fd.real.mid()))), time.time() - t0), flush=True)
    # complex boxes like the ellipse ones
    for zc, rad in ((acb(0.75, 0.2), 0.05), (acb(-0.75, 0.2), 0.05), (acb(7500, 2000), 300), (acb(-7500, 2000), 300),
                    (acb(1.5e-5, 4e-6), 1e-6), (acb(-1.5e-5, 4e-6), 1e-6)):
        box = acb(arb(zc.real.mid(), rad), arb(zc.imag.mid(), rad))
        g = drd_pos if zc.real.mid() > 0 else drd_neg
        t0 = time.time()
        print("box", zc, "->", abs(g(box, acb(r))).upper(), "%.2fs" % (time.time() - t0), flush=True)
