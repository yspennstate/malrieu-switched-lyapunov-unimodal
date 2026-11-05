"""Check the certificates: for each panel, the decided intervals in the jsonl files tile the panel exactly
(no gap, no overlap) and all carry the stated sign. Usage: python check_cover.py [outputs_dir]"""
import json, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
PANELS = [("G' > 0", 0.01, 0.215, "pos", ["gp_grid_001_0215.jsonl"]),
          ("H' < 0", 0.215, 0.25, "neg", ["hp_grid_0215_025.jsonl"]),
          ("G' < 0", 0.25, 100.0, "neg", ["gp_grid_025_10.jsonl", "gp_grid_10_100.jsonl"])]
ok = True
for name, A, B, sign, files in PANELS:
    iv = []
    for f in files:
        for l in open(os.path.join(D, f)):
            if l.strip():
                j = json.loads(l)
                if j["sign"] != "undecided":
                    iv.append((j["lo"], j["hi"], j["sign"]))
    iv.sort()
    good = all(s == sign for _, _, s in iv) and abs(iv[0][0] - A) < 1e-12 and abs(iv[-1][1] - B) < 1e-9 * B
    for (a, b, _), (c, d, _) in zip(iv[:-1], iv[1:]):
        if abs(b - c) > 1e-12 * max(1, b):
            good = False
            print("  gap/overlap between [%g, %g] and [%g, %g]" % (a, b, c, d))
    ok &= good
    print("%s on [%g, %g]: %d decided intervals, all %s, tiling %s" % (name, A, B, len(iv), sign, "ok" if good else "FAILED"))
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
