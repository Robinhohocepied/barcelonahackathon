"""Robust selection. For each held-out Metrica game: predictions on its native AND stressed (extra ball loss) versions.
Ensemble weight + decoder chosen on the OTHER games, maximising the mean of native and stressed spellF1.
python research/ext_v5eval.py <name> <tcn tag pattern with {g}, or 'none'> <stack: v2|v5|none>"""
import itertools, os, sys, pickle, numpy as np, multiprocessing as mp
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, "pipeline")); sys.path.insert(0, HERE)
import core, stack, stack5, tcn
from metrics import spell_f1, macro_f1
def s1(P, T): s = spell_f1(P, T); return s[0] if isinstance(s, tuple) else s
def npos(q): return int(np.isin(q[np.r_[0, np.where(np.diff(q) != 0)[0] + 1]], [0, 1]).sum())
def load(v, g): return pickle.load(open(os.path.join(HERE, "ext", "p5_%s_metrica_%s.pkl" % (v, g)), "rb"))
def lab5(r):
    l = r["label"][::core.STRIDE][:len(r["ph"])]; r["yd"] = (l == 2).astype(int); r["yh"] = (l == 0).astype(int); return r
def p5(P):
    T5 = -(-len(P) // 5); return np.vstack([P, np.repeat(P[-1:], T5 * 5 - len(P), 0)]).reshape(T5, 5, 3).mean(1)
GRID = [dict(Ls=(a, a, d), lam_ha=lh, lam_d=ld, w=w) for a, d, lh, ld, w in itertools.product((5, 8, 12, 16), (5,), (0.0, 1.0), (1.0, 3.0), (0.0, 0.5, 0.7, 1.0))]
def dec(o, n, c):
    Pt, Pv = o
    P = Pv if Pt is None else Pt if Pv is None else c["w"] * Pt + (1 - c["w"]) * Pv
    s = core.viterbi(np.log(np.clip(P, 1e-4, 1)), c["Ls"], c["lam_ha"], c["lam_d"]); return s[np.minimum(np.arange(n) // 5, len(s) - 1)]
def sc(c, items): return np.mean([s1([dec(o, n, c) for o, n, y in it], [y for o, n, y in it]) for it in items])
if __name__ == "__main__":
    name, tpat, stk = sys.argv[1:4]
    OUT = {}
    for g in "123":
        tr = [lab5(r) for x in "123" if x != g for v in (("native", "aug") if stk == "v5" else ("native",)) for r in load(v, x)]
        st = stack5.fit(tr) if stk == "v5" else stack.fit(tr) if stk == "v2" else None
        prm = None if tpat == "none" else dict(np.load(os.path.join(HERE, "ext", tpat.format(g=g))))
        for v in ("native", "stress"):
            items = []
            for r in load(v, g):
                Pv = None
                if st is not None:
                    qh, qd = (stack5 if stk == "v5" else stack).apply(st, r["X"], r["ph"], r["pd"], r["E"])
                    Pv = np.column_stack([(1 - qd) * qh, (1 - qd) * (1 - qh), qd])
                Pt = None if prm is None else p5(tcn.np_forward(prm, r["F"]))
                if Pt is not None and Pv is not None: k = min(len(Pt), len(Pv)); Pt, Pv = Pt[:k], Pv[:k]
                items.append(((Pt, Pv), r["n"], r["label"]))
            OUT[(g, v)] = items
    grid = [c for c in GRID if (tpat != "none" and stk != "none") or c["w"] == (1.0 if stk == "none" else 0.0)]
    res = []
    for g in "123":
        other = [OUT[(x, v)] for x in "123" if x != g for v in ("native", "stress")]
        with mp.Pool(2) as pool: scores = pool.starmap(sc, [(c, other) for c in grid])
        c = grid[int(np.argmax(scores))]; row = [g]
        for v in ("native", "stress"):
            P = [dec(o, n, c) for o, n, y in OUT[(g, v)]]; T = [y for o, n, y in OUT[(g, v)]]
            row += [s1(P, T), macro_f1(P, T), sum(npos(q) for q in P)]
        res.append(row); print("%s game %s native %.3f (mF1 %.3f, poss %d) | stress %.3f (mF1 %.3f, poss %d) | %s" % (name, *row, c), flush=True)
    a = np.array([[r[1], r[4]] for r in res])
    line = "%s MEAN native %.3f stress %.3f | worst game-variant %.3f" % (name, a[:, 0].mean(), a[:, 1].mean(), a.min())
    print(line); open(os.path.join(HERE, "ext", "v5eval.log"), "a").write(line + "\n")
