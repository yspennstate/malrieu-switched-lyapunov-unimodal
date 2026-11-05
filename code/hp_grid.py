"""certification of H' = N''D - ND'' < 0 on [A, B] (H = N'D - ND' has the sign of G'): intervals of width w,
H' enclosed on each r-ball by NDpp + gp_interval, an undecided interval bisected down to width minw, every result
appended to a jsonl as it lands (resumable). Usage: python hp_grid.py A B w out.jsonl [minw]"""
import json, os, sys, time
from flint import acb, arb
from NDpp import NDpp
from gp_ball import gp_interval


def hp(lo, hi):
    R = acb(arb((lo + hi) / 2, (hi - lo) / 2))
    m = min(lo, 1.0)
    Npp, Dpp = NDpp(R, lo, 1e-5 * m, 1e5 / m)
    N, D, Np, Dp, G, Gp = gp_interval(lo, hi)
    return Npp * D - N * Dpp, Gp


def main():
    A, B, w, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    minw = float(sys.argv[5]) if len(sys.argv) > 5 else w / 8
    done = {}
    if os.path.exists(out):
        for l in open(out):
            if l.strip():
                j = json.loads(l); done[(j["lo"], j["hi"])] = j["sign"]
    n = max(1, round((B - A) / w))
    stack = [(round(A + k * (B - A) / n, 12), round(A + (k + 1) * (B - A) / n, 12)) for k in range(n)][::-1]
    while stack:
        lo, hi = stack.pop()
        s = done.get((lo, hi))
        if s is None:
            t0 = time.time()
            H, Gp = hp(lo, hi)
            s = "neg" if H.upper() < 0 else ("pos" if H.lower() > 0 else "undecided")
            rec = dict(lo=lo, hi=hi, sign=s, Hp_lo=float(H.lower()), Hp_hi=float(H.upper()),
                       Gp_lo=float(Gp.lower()), Gp_hi=float(Gp.upper()), sec=round(time.time() - t0, 1))
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if s == "undecided" and hi - lo > minw * 1.0001:
            m = round((lo + hi) / 2, 12)
            stack.append((m, hi)); stack.append((lo, m))


if __name__ == "__main__":
    main()
