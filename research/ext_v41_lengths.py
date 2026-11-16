"""Backs the claim 'about 3 in 4 longer possessions found': v4.1 on held-out Metrica games (leave-one-game-out),
share of true possessions matched at IoU >= 0.5, by length. Needs the ext/p5_* files (research/ext_prep5.py) and ext/tcn_<g>.npz (research/train_v4.py)."""
import sys, numpy as np
sys.argv = ["x", "x", "tcn_{g}.npz", "v5"]
import ext_v5eval as E, stack5
cfg = {'Ls': (12, 12, 5), 'lam_ha': 0.0, 'lam_d': 3.0, 'w': 0.7}
def runs(a):
    ch = np.r_[0, np.where(np.diff(a) != 0)[0] + 1, len(a)]; return [(ch[i], ch[i + 1], a[ch[i]]) for i in range(len(ch) - 1) if a[ch[i]] in (0, 1)]
hit, ln = [], []
for g in "123":
    st = stack5.fit([E.lab5(r) for x in "123" if x != g for v in ("native", "aug") for r in E.load(v, x)]); prm = dict(np.load("ext/tcn_%s.npz" % g))
    for r in E.load("native", g):
        qh, qd = stack5.apply(st, r["X"], r["ph"], r["pd"], r["E"]); Pv = np.column_stack([(1 - qd) * qh, (1 - qd) * (1 - qh), qd])
        Pt = E.p5(E.tcn.np_forward(prm, r["F"])); k = min(len(Pt), len(Pv)); pp = runs(E.dec((Pt[:k], Pv[:k]), r["n"], cfg))
        for (s, e, c) in runs(r["label"]):
            best = max([0] + [(min(b, e) - max(a, s)) / ((b - a) + (e - s) - (min(b, e) - max(a, s))) for (a, b, d) in pp if d == c and b > s and a < e])
            hit.append(best >= 0.5); ln.append((e - s) / 25)
hit, ln = np.array(hit), np.array(ln)
for lo, hi in ((0, 1), (1, 3), (3, 8), (8, 20), (20, 999)):
    m = (ln >= lo) & (ln < hi); print("%g-%gs: %.0f%% of possessions, found %.0f%%" % (lo, hi, 100 * m.mean(), 100 * hit[m].mean()))
