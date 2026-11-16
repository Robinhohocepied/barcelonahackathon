"""Fit the v2 second stage + decoder on ALL 3 Metrica open games (external), save pipeline/model_v2.pkl.
Needs: ./fetch_metrica.sh, then python ext_eval.py prep. Deterministic."""
import itertools, os, pickle, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "pipeline")); sys.path.insert(0, HERE)
import core, stack
from metrics import spell_f1
R = pickle.load(open(os.path.join(HERE, "ext", "feats_metrica.pkl"), "rb"))
for r in R:
    lab5 = r["label"][::core.STRIDE][:len(r["ph"])]
    r["E"] = stack.evidence(r["ball"]); r["yd"] = (lab5 == 2).astype(int); r["yh"] = (lab5 == 0).astype(int)
st = stack.fit(R)
for r in R: r["qh"], r["qd"] = stack.apply(st, r["X"], r["ph"], r["pd"], r["E"])
GRID = [dict(Ls=(a, a, d), lam_ha=lh, lam_d=ld) for a, d, lh, ld in itertools.product((8, 12, 16), (5, 10), (0.0, 1.0, 2.0), (0.3, 1.0))]
def sc(c):
    P = []
    for r in R:
        ph = np.clip(r["qh"], 1e-3, 1 - 1e-3); pd = np.clip(r["qd"], 1e-3, 1 - 1e-3)
        s = core.viterbi(np.log(np.column_stack([(1 - pd) * ph, (1 - pd) * (1 - ph), pd])), c["Ls"], c["lam_ha"], c["lam_d"])
        P.append(s[np.minimum(np.arange(r["n"]) // core.STRIDE, len(s) - 1)])
    s = spell_f1(P, [r["label"] for r in R]); return s[0] if isinstance(s, tuple) else s
if __name__ == "__main__":
    import multiprocessing as mp
    with mp.Pool(4) as pool: scores = pool.map(sc, GRID)
    best = GRID[int(np.argmax(scores))]; print("chosen", best, "in-sample spellF1 %.3f" % max(scores))
    v1 = pickle.load(open(os.path.join(HERE, "pipeline", "model.pkl"), "rb"))
    pickle.dump(dict(team=v1["team"], dead=v1["dead"], stack=st, decode=best, trained_on=v1["trained_on"],
                     stack_trained_on=["metrica_sample_1", "metrica_sample_2", "metrica_sample_3"]),
                open(os.path.join(HERE, "pipeline", "model_v2.pkl"), "wb"))
    print("saved pipeline/model_v2.pkl")
