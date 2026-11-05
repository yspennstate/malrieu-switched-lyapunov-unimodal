"""adaptive certification of the sign of G' on [A, B]: start from intervals of ratio q, enclose G' on each with
gp_ball.gp_interval, bisect an undecided one down to relative width minrel, append every result to a jsonl as it lands
(resumable: intervals already in the file are skipped). Usage: python gp_grid.py A B q out.jsonl [minrel]"""
import json, os, sys, time
from gp_ball import gp_interval


def main():
    A, B, q, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    minrel = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
    done = {}
    if os.path.exists(out):
        for l in open(out):
            if l.strip():
                j = json.loads(l); done[(j["lo"], j["hi"])] = j["sign"]
    stack, r = [], A
    while r < B:
        stack.append((r, min(r * q, B))); r = min(r * q, B)
    stack.reverse()
    while stack:
        lo, hi = stack.pop()
        if (lo, hi) in done:
            if done[(lo, hi)] == "undecided" and (hi - lo) / lo > minrel:
                m = (lo + hi) / 2
                stack.append((m, hi)); stack.append((lo, m))
            continue
        t0 = time.time()
        N, D, Np, Dp, G, Gp = gp_interval(lo, hi)
        s = "neg" if Gp.upper() < 0 else ("pos" if Gp.lower() > 0 else "undecided")
        rec = dict(lo=lo, hi=hi, sign=s, Gp_lo=float(Gp.lower()), Gp_hi=float(Gp.upper()),
                   G_lo=float(G.lower()), G_hi=float(G.upper()), sec=round(time.time() - t0, 1))
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if s == "undecided" and (hi - lo) / lo > minrel:
            m = (lo + hi) / 2
            stack.append((m, hi)); stack.append((lo, m))


if __name__ == "__main__":
    main()
