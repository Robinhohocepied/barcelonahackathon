"""v4 ensemble: TCN (v4) + v2 (stack) probabilities, weight and decoder chosen on the OTHER held-out games."""
import itertools, os, sys, pickle, numpy as np, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "pipeline")); sys.path.insert(0, HERE)
import core, stack
from metrics import spell_f1, macro_f1
from ext_v4eval import H, s1, npos
R = {g: pickle.load(open(os.path.join(HERE, "ext", "feats3_metrica_%s.pkl" % g), "rb")) for g in "123"}
for g in "123":
    for r in R[g]:
        lab5 = r["label"][::core.STRIDE][:len(r["ph"])]; r["yd"] = (lab5 == 2).astype(int); r["yh"] = (lab5 == 0).astype(int)
V2 = {}
for g in "123":                                                     # v2 stack fitted without game g
    st = stack.fit([r for x in "123" if x != g for r in R[x]])
    V2[g] = []
    for r in R[g]:
        qh, qd = stack.apply(st, r["X"], r["ph"], r["pd"], r["E"])
        V2[g].append(np.column_stack([(1 - qd) * qh, (1 - qd) * (1 - qh), qd]))
def p5(P):
    T5 = -(-len(P) // 5); return np.vstack([P, np.repeat(P[-1:], T5 * 5 - len(P), 0)]).reshape(T5, 5, 3).mean(1)
GRID = [dict(Ls=(a, a, d), lam_ha=lh, lam_d=ld, w=w) for a, d, lh, ld, w in itertools.product((2, 5, 8, 12), (5,), (0.0, 1.0, 2.0), (1.0, 3.0), (0.0, 0.3, 0.5, 0.7, 1.0))]
def dec(g, k, c):
    Pt = p5(H[g][k][0]); Pv = V2[g][k]; m = min(len(Pt), len(Pv)); P = c["w"] * Pt[:m] + (1 - c["w"]) * Pv[:m]
    s = core.viterbi(np.log(np.clip(P, 1e-4, 1)), c["Ls"], c["lam_ha"], c["lam_d"]); n = len(H[g][k][1])
    return s[np.minimum(np.arange(n) // 5, len(s) - 1)]
def sc(c, games): return s1([dec(g, k, c) for g in games for k in range(2)], [H[g][k][1] for g in games for k in range(2)])
if __name__ == "__main__":
    tot = []
    for g in "123":
        other = [x for x in "123" if x != g]
        with mp.Pool(2) as pool: scores = pool.starmap(sc, [(c, other) for c in GRID])
        c = GRID[int(np.argmax(scores))]; P = [dec(g, k, c) for k in range(2)]; T = [H[g][k][1] for k in range(2)]
        tot.append(s1(P, T)); print("ens game %s spellF1 %.3f macroF1 %.3f poss %d/%d D %.0f%% | %s" % (g, tot[-1], macro_f1(P, T), sum(npos(q) for q in P), sum(npos(q) for q in T), 100*np.mean(np.concatenate(P)==2), c), flush=True)
    print("ensemble LOGO mean spellF1 %.3f   (v2: 0.528, v4: 0.561)" % np.mean(tot))
