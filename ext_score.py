import os, sys, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "pipeline")); sys.path.insert(0, HERE)
import core
from metrics import spell_f1, macro_f1
R = pickle.load(open(os.path.join(HERE, "ext", "feats_metrica.pkl"), "rb"))
def up(s, n): return s[np.minimum(np.arange(n) // core.STRIDE, len(s) - 1)]
def decode_p(ph, pd, cfg):
    ph = np.clip(ph, 1e-3, 1 - 1e-3); pd = np.clip(pd, 1e-3, 1 - 1e-3)
    lp = np.log(np.column_stack([(1 - pd) * ph, (1 - pd) * (1 - ph), pd]))
    return core.viterbi(lp, cfg["Ls"], cfg["lam_ha"], cfg["lam_d"])
def score(preds, games=None):
    sel = [i for i, r in enumerate(R) if games is None or r["game"] in games]
    P = [preds[i] for i in sel]; T = [R[i]["label"] for i in sel]
    s = spell_f1(P, T); s = s[0] if isinstance(s, tuple) else s
    npos = sum(int(np.isin(p[np.r_[0, np.where(np.diff(p) != 0)[0] + 1]], [0, 1]).sum()) for p in P)
    ntrue = sum(int(np.isin(t[np.r_[0, np.where(np.diff(t) != 0)[0] + 1]], [0, 1]).sum()) for t in T)
    dsh = np.mean(np.concatenate(P) == 2); dtr = np.mean(np.concatenate(T) == 2)
    return s, macro_f1(P, T), npos, ntrue, dsh, dtr
if __name__ == "__main__":
    preds = [up(decode_p(r["ph"], r["pd"], core.DECODE), r["n"]) for r in R]
    for g in (1, 2, 3): print("game", g, "spellF1 %.3f macroF1 %.3f poss %d/%d D %.0f%%/%.0f%%" % score(preds, [g]))
    print("ALL    spellF1 %.3f macroF1 %.3f poss %d/%d D %.0f%%/%.0f%%" % tuple(x*100 if i>3 else x for i,x in enumerate(score(preds))))
