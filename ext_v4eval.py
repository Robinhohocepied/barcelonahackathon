"""v4 decoder selection without peeking: for held-out game g, the decoder settings are chosen on the OTHER games'
held-out TCN outputs (each predicted by a model that never saw it), then applied to g."""
import itertools, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "pipeline")); sys.path.insert(0, HERE)
import core
from metrics import spell_f1, macro_f1
H = {}
for g in ("1", "2", "3"):
    z = np.load(os.path.join(HERE, "ext", "heldout_%s.npz" % g)); H[g] = [(z["P%d" % k], z["y%d" % k]) for k in range(2)]
GRID = [dict(Ls=(a, a, d), lam_ha=lh, lam_d=ld, tau=t) for a, d, lh, ld, t in itertools.product((1, 2, 3, 5, 8), (5, 10), (0.0, 0.5, 1.0, 2.0), (0.3, 1.0, 3.0), (0.5, 1.0))]
def dec(P, n, c):
    T5 = -(-len(P) // 5); Pp = np.vstack([P, np.repeat(P[-1:], T5 * 5 - len(P), 0)]).reshape(T5, 5, 3).mean(1)
    s = core.viterbi(c["tau"] * np.log(np.clip(Pp, 1e-4, 1)), c["Ls"], c["lam_ha"], c["lam_d"])
    return s[np.minimum(np.arange(n) // 5, len(s) - 1)]
def s1(P, T): s = spell_f1(P, T); return s[0] if isinstance(s, tuple) else s
def npos(q): return int(np.isin(q[np.r_[0, np.where(np.diff(q) != 0)[0] + 1]], [0, 1]).sum())
if __name__ == "__main__":
    import multiprocessing as mp
    def sc(c, games): return s1([dec(P, len(y), c) for g in games for P, y in H[g]], [y for g in games for P, y in H[g]])
    tot = []
    for g in ("1", "2", "3"):
        other = [x for x in ("1", "2", "3") if x != g]
        with mp.Pool(2) as pool: scores = pool.starmap(sc, [(c, other) for c in GRID])
        c = GRID[int(np.argmax(scores))]
        P = [dec(P, len(y), c) for P, y in H[g]]; T = [y for P, y in H[g]]
        raw = [P_.argmax(1).astype(np.int8) for P_, y in H[g]]
        tot.append(s1(P, T))
        print("v4 game %s spellF1 %.3f macroF1 %.3f poss %d/%d D %.0f%% | raw argmax spellF1 %.3f | %s" % (
            g, tot[-1], macro_f1(P, T), sum(npos(q) for q in P), sum(npos(q) for q in T), 100 * np.mean(np.concatenate(P) == 2), s1(raw, T), c), flush=True)
    print("v4 LOGO mean spellF1 %.3f   (v2: 0.528)" % np.mean(tot))
