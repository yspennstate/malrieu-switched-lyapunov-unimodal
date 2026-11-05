"""ball enclosures of d_rr d(x), the second r-derivative of d, for N'', D'' (the peak certificate:
H = N'D - ND' has H' = N''D - ND'', the N'D' terms cancel, so H' < 0 on the peak interval makes the sign change of G'
unique there).

u-form (from drd.py): d_r d = -b int e^E u^2x^3/s^2 du, E = -u b + u^2x^3/s, s = r + ux, b = 1+x^2. Differentiating in r
(dE/dr = -u^2x^3/s^2):  d_rr d = b int_0^inf e^E [u^4x^6/s^4 + 2u^2x^3/s^3] du.

x > 0 (|arg x| < pi/4): |e^E| <= e^{-u}, |s| >= u Re x, so the integrand is at most
    e^{-u} (|x|^6/(Re x)^4 + 2|x|^3/(u (Re x)^3)); the tail beyond L is at most e^{-L}(|x|^6/(Re x)^4 + 2|x|^3/(L (Re x)^3)).
x < 0: u = U t, U = -r/x: s = r(1-t), u^4x^6/s^4 = x^2 t^4/(1-t)^4, 2u^2x^3/s^3 = 2x t^2/(r(1-t)^3), E = -Utb + y,
    y = r x t^2/(1-t):  d_rr d = b U int_0^1 e^{-Utb} e^y [x^2 t^4/(1-t)^4 + 2x t^2/(r(1-t)^3)] dt.
    With c = -Re(rx) > 0, s = 1/(1-t), a = c/4: on [1/2, 1], Re y <= -a s and dt = ds/s^2, so
    int_{t0}^1 |e^y||bracket| dt <= |x|^2 J2(s0) + (2|x|/r) J1(s0), s0 = 1/(1-t0) >= 2, with
    J2 = e^{-a s0}(s0^2/a + 2 s0/a^2 + 2/a^3), J1 = e^{-a s0}(s0/a + 1/a^2). On [0, 1/2] the bracket is at most
    |x|^2 + 4|x|/r and |e^y| <= 1. Cut at T = S/Re(Ub) when Re(Ub) is large, else at 1 - delta.
"""
from flint import acb, arb, ctx
from drd import tol
ctx.prec = 64


def drr_pos(x, r, L=50):
    assert x.real.lower() > abs(x.imag).upper()
    b = 1 + x * x
    x2, x3 = x * x, x * x * x

    def h(u, an):
        s = r + u * x
        q = u * u * x3 / s
        return (-u).exp() * (-(r * u * x2) / s).exp() * (q * q / (s * s) + 2 * q / (s * s))
    rex = arb(x.real.lower())
    ax = abs(x)
    t = ((-arb(L)).exp() * (ax ** 6 / rex ** 4 + 2 * ax ** 3 / (L * rex ** 3))).upper()
    return b * (acb.integral(h, 0, L, abs_tol=tol(x, r)) + acb(arb(0, t), arb(0, t)))


def _J(a, s0):
    e = (-a * s0).exp()
    return e * (s0 * s0 / a + 2 * s0 / (a * a) + 2 / (a ** 3)), e * (s0 / a + 1 / (a * a))


def drr_neg(x, r, delta=arb("1e-15"), S=60):
    b = 1 + x * x
    assert x.real.upper() < 0 and b.real.lower() > 0
    U = -r / x
    Ub = U * b
    reub = Ub.real.lower()
    c = (-(r * x).real).lower()
    assert c > 0
    a = arb(c) / 4
    ax = abs(x)
    rr = arb(abs(r).lower())

    def h(t, an):
        w = 1 - t
        return (-U * t * b).exp() * (r * x * t * t / w).exp() * (x * x * t ** 4 / w ** 4 + 2 * x * t * t / (r * w ** 3))
    if reub > S / (1 - float(delta)):
        T = arb(S) / reub
        v = acb.integral(h, 0, T, abs_tol=tol(x, r))
        J2, J1 = _J(a, arb(2))
        tail = arb(-S).exp() * ((ax * ax + 4 * ax / rr) / 2 + ax * ax * J2 + 2 * ax / rr * J1)
    else:
        v = acb.integral(h, 0, 1 - delta, abs_tol=tol(x, r))
        J2, J1 = _J(a, 1 / delta)
        m = max(arb(0), -reub)
        tail = m.exp() * (ax * ax * J2 + 2 * ax / rr * J1)
    tu = tail.upper()
    return b * U * (v + acb(arb(0, tu), arb(0, tu)))


if __name__ == "__main__":
    import sys, time
    from drd import drd_pos, drd_neg
    r = float(sys.argv[1]) if len(sys.argv) > 1 else 0.22
    hs = 1e-6 * r
    for xv in (2e-5, 0.01, 0.7, 3.0, 60.0, 8000.0, -2e-5, -0.01, -0.7, -3.0, -60.0, -8000.0):
        x = acb(xv)
        f, g = (drd_pos, drr_pos) if xv > 0 else (drd_neg, drr_neg)
        t0 = time.time()
        v = g(x, acb(r))
        fd = (f(x, acb(r + hs)) - f(x, acb(r - hs))) / (2 * hs)
        print("x=%g drr=%s  cdiff=%.10e  rel=%.1e  %.2fs" % (xv, v.real, float(fd.real.mid()),
              abs(float(v.real.mid()) - float(fd.real.mid())) / max(1e-300, abs(float(fd.real.mid()))), time.time() - t0), flush=True)
