"""v4.1 = v4 sequence model + positive-only second stage (stack5, fitted on all 3 Metrica games, native + extra-ball-loss copies).
Ensemble weight + decoder chosen on all 3 games' held-out outputs, mean of native and stressed spellF1."""
import os, sys, pickle, numpy as np, multiprocessing as mp
sys.argv = ["x", "x", "tcn_{g}.npz", "v5"]
import ext_v5eval as E
import stack5
if __name__ == "__main__":
    OUT = {}
    for g in "123":
        tr = [E.lab5(r) for x in "123" if x != g for v in ("native", "aug") for r in E.load(v, x)]
        st = stack5.fit(tr); prm = dict(np.load("ext/tcn_%s.npz" % g))
        for v in ("native", "stress"):
            items = []
            for r in E.load(v, g):
                qh, qd = stack5.apply(st, r["X"], r["ph"], r["pd"], r["E"]); Pv = np.column_stack([(1 - qd) * qh, (1 - qd) * (1 - qh), qd])
                Pt = E.p5(E.tcn.np_forward(prm, r["F"])); k = min(len(Pt), len(Pv)); items.append(((Pt[:k], Pv[:k]), r["n"], r["label"]))
            OUT[(g, v)] = items
    allitems = [OUT[k] for k in OUT]
    with mp.Pool(2) as pool: scores = pool.starmap(E.sc, [(c, allitems) for c in E.GRID])
    cfg = E.GRID[int(np.argmax(scores))]; print("final cfg", cfg, "mean held-out spellF1 (native+stress) %.3f" % max(scores))
    st_all = stack5.fit([E.lab5(r) for x in "123" for v in ("native", "aug") for r in E.load(v, x)])
    v4 = pickle.load(open("pipeline/model_v4.pkl", "rb"))
    m = dict(v4); m.pop("stack"); m["stack5"] = st_all; m["decode"] = cfg
    m["stack5_trained_on"] = ["metrica_sample_1", "metrica_sample_2", "metrica_sample_3"]
    pickle.dump(m, open("pipeline/model_v41.pkl", "wb")); print("saved pipeline/model_v41.pkl")
