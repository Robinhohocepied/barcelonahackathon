"""v3 leave-one-Metrica-game-out. python research/ext_v3eval.py <held-out game 1|2|3>"""
import glob, itertools, os, pickle, sys, numpy as np, multiprocessing as mp
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, "pipeline")); sys.path.insert(0, HERE)
import core, v3
from metrics import spell_f1, macro_f1
ROWS = [r for f in sorted(glob.glob(os.path.join(HERE, "ext", "feats3_*.pkl"))) for r in pickle.load(open(f, "rb"))]
for r in ROWS: r["yd"], r["yh"], r["yb"] = v3.labels5(r)
GRID = [dict(Ls=(a, a, d), a0=a0, b=b, lam_d=1.0) for a, d, a0, b in itertools.product((3, 5, 8, 12), (5, 10), (0.5, 1.0, 2.0), (0.0, 0.5, 1.0, 2.0))]
def s1(P, T): s = spell_f1(P, T); return s[0] if isinstance(s, tuple) else s
def _sc(c): return s1([v3.decode(*r["o"], r["n"], c) for r in TR], [r["label"] for r in TR])
if __name__ == "__main__":
    g = sys.argv[1]
    met = [r for r in ROWS if r["src"] == "metrica"]; sk = [r for r in ROWS if r["src"] == "sk"]
    tr_m = [r for r in met if r["game"] != g]; te = [r for r in met if r["game"] == g]
    m = v3.fit(tr_m + sk, tr_m)
    for r in tr_m + te: r["o"] = v3.apply(m, r)
    TR = tr_m
    with mp.Pool(4) as pool: sc = pool.map(_sc, GRID)
    c = GRID[int(np.argmax(sc))]
    P = [v3.decode(*r["o"], r["n"], c) for r in te]; T = [r["label"] for r in te]
    npos = sum(int(np.isin(q[np.r_[0, np.where(np.diff(q) != 0)[0] + 1]], [0, 1]).sum()) for q in P)
    ntrue = sum(int(np.isin(q[np.r_[0, np.where(np.diff(q) != 0)[0] + 1]], [0, 1]).sum()) for q in T)
    line = "v3 held-out game %s spellF1 %.3f macroF1 %.3f poss %d/%d D %.0f%% | cfg %s | train %.3f" % (g, s1(P, T), macro_f1(P, T), npos, ntrue, 100 * np.mean(np.concatenate(P) == 2), c, max(sc))
    print(line); open(os.path.join(HERE, "ext", "v3eval.log"), "a").write(line + "\n")
