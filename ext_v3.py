import itertools, numpy as np, multiprocessing as mp
import ext_v2 as V
from ext_v2 import R, fit_stack, apply, run
from metrics import spell_f1, macro_f1
GRID = [dict(Ls=(a, a, d), lam_ha=lh, lam_d=ld) for a, d, lh, ld in itertools.product((8, 12, 16), (5, 10), (0.0, 1.0, 2.0), (0.3, 1.0))]
def em(pd, pi_tr, it=50):
    pi = pi_tr
    for _ in range(it):
        a = pi / pi_tr * pd; b = (1 - pi) / (1 - pi_tr) * (1 - pd); q = a / (a + b); pi = q.mean()
    return q, pi
def s1(P, T): s = spell_f1(P, T); return s[0] if isinstance(s, tuple) else s
def _sc(c): p = run("stack", c, _TRI); return s1([p[i] for i in _TRI], [R[i]["label"] for i in _TRI])
if __name__ == "__main__":
    for useem in (False, True):
        tot = []
        for g in (1, 2, 3):
            tr = [r for r in R if r["game"] != g]; st = fit_stack(tr); pi_tr = np.mean(np.concatenate([r["yd"] for r in tr]))
            for i, r in enumerate(R):
                r["_ph"], r["_pd"] = apply(r, st)
                if useem: r["_pd"], r["pi"] = em(r["_pd"], pi_tr)
            V._V = "stack"; tri = [i for i, r in enumerate(R) if r["game"] != g]; te = [i for i, r in enumerate(R) if r["game"] == g]
            globals()["_TRI"] = tri
            with mp.Pool(4) as pool: sc = pool.map(_sc, GRID)
            c = GRID[int(np.argmax(sc))]; p = run("stack", c, te); P = [p[i] for i in te]; T = [R[i]["label"] for i in te]
            npos = sum(int(np.isin(q[np.r_[0, np.where(np.diff(q) != 0)[0] + 1]], [0, 1]).sum()) for q in P)
            tot.append(s1(P, T)); print("EM" if useem else "noEM", "game", g, "spellF1 %.3f macroF1 %.3f poss %d D %.0f%%" % (tot[-1], macro_f1(P, T), npos, 100*np.mean(np.concatenate(P)==2)),
                  ("pi_hat %.2f true %.2f" % (np.mean([R[i]["pi"] for i in te]), np.mean(np.concatenate([R[i]["yd"] for i in te])))) if useem else "", c, flush=True)
        print("EM" if useem else "noEM", "LOGO mean %.3f" % np.mean(tot), flush=True)
