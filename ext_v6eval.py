"""Two-source robustness validation. Folds: Metrica game 1, 2, 3 and IDSSE groups. Each fold is predicted by models that never
saw it (sequence model and second stage), in two conditions: native ball and extra ball loss. Ensemble weight + decoder chosen
on the OTHER folds (mean over sources and conditions). The worst fold-condition decides.
python ext_v6eval.py <name> <tcn: v4|v6> <stack: s5|s6>"""
import itertools, os, sys, pickle, numpy as np, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "pipeline")); sys.path.insert(0, HERE)
import core, stack, stack5, stack6, tcn
from metrics import spell_f1, macro_f1
from ext_v5eval import s1, npos, p5, GRID, dec
IDS = sorted(f[len("p5_native_idsse_"):-4] for f in os.listdir(os.path.join(HERE, "ext")) if f.startswith("p5_native_idsse_"))
IG = {"I1": IDS[: (len(IDS) + 1) // 2], "I2": IDS[(len(IDS) + 1) // 2:]} if len(IDS) > 1 else {"I1": IDS}
FOLDS = [("metrica", ["1"]), ("metrica", ["2"]), ("metrica", ["3"])] + [("idsse", v) for v in IG.values() if v]
def load(v, src, g):
    rows = pickle.load(open(os.path.join(HERE, "ext", "p5_%s_%s_%s.pkl" % (v, src, g)), "rb"))
    for r in rows:
        l = r["label"][::core.STRIDE][:len(r["ph"])]; r["yd"] = (l == 2).astype(int); r["yh"] = (l == 0).astype(int)
        r["D"] = stack6.dead_evidence(r["B5"])
    return rows
def tcn_file(kind, src, gs):
    if kind == "v4": return "tcn_%s.npz" % gs[0] if src == "metrica" else "tcn_all.npz"
    return "tcn6_%s_%s.npz" % (src, "-".join(gs))
if __name__ == "__main__":
    name, kind, sk = sys.argv[1:4]
    OUT = {}
    for fi, (src, gs) in enumerate(FOLDS):
        trv = ("native", "aug")
        tr = [r for (s2, g2) in FOLDS if (s2, g2) != (src, gs) for g in g2 for v in trv for r in load(v, s2, g)
              if not (sk == "s5" and s2 == "idsse")]
        st = stack6.fit(tr) if sk == "s6" else stack5.fit(tr)
        prm = dict(np.load(os.path.join(HERE, "ext", tcn_file(kind, src, gs))))
        for v in ("native", "stress"):
            items = []
            for g in gs:
                for r in load(v, src, g):
                    qh, qd = stack6.apply(st, r["X"], r["ph"], r["pd"], r["E"], r["D"]) if sk == "s6" else stack5.apply(st, r["X"], r["ph"], r["pd"], r["E"])
                    Pv = np.column_stack([(1 - qd) * qh, (1 - qd) * (1 - qh), qd]); Pt = p5(tcn.np_forward(prm, r["F"]))
                    k = min(len(Pt), len(Pv)); items.append(((Pt[:k], Pv[:k]), r["n"], r["label"]))
            OUT[(fi, v)] = items
    keys_all = list(OUT.keys())
    def score_all(c): return [s1([dec(o, n, c) for o, n, y in OUT[k]], [y for o, n, y in OUT[k]]) for k in keys_all]
    with mp.Pool(2) as pool: S = np.array(pool.map(score_all, GRID))          # (configs, fold-conditions): each decode once
    gc = GRID[int(np.argmax(S.mean(1)))]; open(os.path.join(HERE, 'ext', 'final_cfg_%s.txt' % name), 'w').write(repr(gc)); print('GLOBAL cfg', gc, '%.3f' % S.mean(1).max(), flush=True)
    res = []
    for fi, (src, gs) in enumerate(FOLDS):
        other = [j for j, k in enumerate(keys_all) if k[0] != fi]
        c_i = int(np.argmax(S[:, other].mean(1))); c = GRID[c_i]
        row = []
        for v in ("native", "stress"):
            P = [dec(o, n, c) for o, n, y in OUT[(fi, v)]]; T = [y for o, n, y in OUT[(fi, v)]]
            row.append((S[c_i, keys_all.index((fi, v))], macro_f1(P, T)))
        res.append((src, "+".join(gs), row)); print("%s %s %s | native %.3f (mF1 %.3f) | stress %.3f (mF1 %.3f) | %s" % (name, src, "+".join(gs), *row[0], *row[1], c), flush=True)
    line = name
    for s in ("metrica", "idsse"):
        a = np.array([[r[2][0][0], r[2][1][0]] for r in res if r[0] == s])
        if len(a): line += " | %s native %.3f stress %.3f" % (s, a[:, 0].mean(), a[:, 1].mean())
    line += " | WORST %.3f" % min(x[0] for r in res for x in r[2])
    print(line); open(os.path.join(HERE, "ext", "v6eval.log"), "a").write(line + "\n")
