"""Monte Carlo check of the top Lyapunov exponent of dX/dt = A_I X, A_i = N_i (alpha = 0),
I switching at rate r, using the exact flows e^{tN0} = [[1,t],[0,1]], e^{tN1} = [[1,0],[-t,1]] over Exp(r) holding times.
G(r) = lim log|X_T| / T. Compared with g_stable.G and with Figure 1 of Malrieu (2015)."""
import math, sys
import numpy as np
from g_stable import G
rng = np.random.default_rng(20261105)
for r in (0.23, 1.0, 4.6, 20.0):
    n = 2_000_000
    T = rng.exponential(1 / r, size=n)
    x, y, logn, tot = 1.0, 0.0, 0.0, 0.0
    for k in range(n):
        t = T[k]
        if k % 2 == 0:
            x = x + t * y
        else:
            y = y - t * x
        tot += t
        if k % 16 == 15:
            s = math.hypot(x, y); logn += math.log(s); x /= s; y /= s
    logn += math.log(math.hypot(x, y))
    print("r=%-5g MC log|X_T|/T = %.4f  (T=%.0f)   g_stable G = %.4f" % (r, logn / tot, tot, G(r)), flush=True)
