"""v2 candidates, selected on Metrica open games only (leave-one-game-out)."""
import itertools, numpy as np
from sklearn.linear_model import LogisticRegression
from ext_score import R, up, decode_p, score, core

def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def evid(r):
    b = r["ball"].astype(float); n = len(b)
    vis = np.isfinite(b[:, 0]).astype(float)
    sp = np.hypot(np.diff(b[:, 0], prepend=np.nan), np.diff(b[:, 1], prepend=np.nan)) * 5
    inp = ((b[:, 0] > -1) & (b[:, 0] < 106) & (b[:, 1] > -1) & (b[:, 1] < 69)).astype(float)
    fast = ((sp > 8) & (inp > 0)).astype(float); mov = ((sp > 3) & (inp > 0)).astype(float)
    W = core.wmean
    cols = [W(fast, -5, 5)[:, 0], W(fast, -15, 15)[:, 0], W(mov, -5, 5)[:, 0], W(vis, -5, 5)[:, 0], W(vis, -15, 15)[:, 0],
            W(inp * vis, -5, 5)[:, 0]]
    return np.nan_to_num(np.column_stack(cols), nan=0.0)
for r in R: r["E"] = evid(r); r["yd"] = (r["label"][::core.STRIDE][:len(r["ph"])] == 2).astype(int); r["yh"] = (r["label"][::core.STRIDE][:len(r["ph"])] == 0).astype(int)

def dead_X(r): return np.column_stack([lg(r["pd"]), r["E"], r["X"][:, 94]])     # + mean player speed
def team_X(r): return np.column_stack([lg(r["ph"]), np.nan_to_num(r["X"][:, 84] - .5), np.nan_to_num(r["X"][:, 86] - .5)])
def fit_stack(tr):
    Xd = np.vstack([dead_X(r) for r in tr]); yd = np.concatenate([r["yd"] for r in tr])
    Xd = np.nan_to_num(Xd); cd = LogisticRegression(C=1.0, max_iter=500).fit(Xd, yd)
    k = np.concatenate([r["yd"] == 0 for r in tr])
    Xt = np.nan_to_num(np.vstack([team_X(r) for r in tr]))[k]; yt = np.concatenate([r["yh"] for r in tr])[k]
    ct = LogisticRegression(C=1.0, max_iter=500).fit(Xt, yt)
    return cd, ct
def apply(r, st):
    cd, ct = st
    return ct.predict_proba(np.nan_to_num(team_X(r)))[:, 1], cd.predict_proba(np.nan_to_num(dead_X(r)))[:, 1]

GRID = [dict(Ls=(a, a, d), lam_ha=lh, lam_d=ld) for a, d, lh, ld in itertools.product((3, 5, 8), (10, 30), (0.0, 0.5, 1.0, 2.0), (1.0, 3.0))]
def run(variant, cfg, idx):
    out = {}
    for i in idx:
        r = R[i]
        if variant == "v1": ph, pd = r["ph"], r["pd"]
        else: ph, pd = r["_ph"], r["_pd"]
        out[i] = up(decode_p(ph, pd, cfg), r["n"])
    return out

import multiprocessing as mp
def _tr_score(c):
    from metrics import spell_f1
    p = run(_V, c, _TRI); s = spell_f1([p[i] for i in _TRI], [R[i]["label"] for i in _TRI]); return s[0] if isinstance(s, tuple) else s

if __name__ == "__main__":
    print("start", len(GRID), "configs", flush=True)
    games = (1, 2, 3); res = {}
    for variant in ("v1", "stack"):
        tot = []
        for g in games:
            tr = [r for r in R if r["game"] != g]; te = [i for i, r in enumerate(R) if r["game"] == g]
            if variant == "stack":
                st = fit_stack(tr)
                for i, r in enumerate(R): r["_ph"], r["_pd"] = apply(r, st)
            tri = [i for i, r in enumerate(R) if r["game"] != g]
            # choose cfg on the two training games
            from metrics import spell_f1
            global _V, _TRI; _V, _TRI = variant, tri
            with mp.Pool(4) as pool: sc = list(zip(pool.map(_tr_score, GRID), GRID))
            s_tr, c = max(sc, key=lambda x: x[0])
            p = run(variant, c, te); P = [p[i] for i in te]; T = [R[i]["label"] for i in te]
            s = spell_f1(P, T); s = s[0] if isinstance(s, tuple) else s
            from metrics import macro_f1
            npos = sum(int(np.isin(q[np.r_[0, np.where(np.diff(q) != 0)[0] + 1]], [0, 1]).sum()) for q in P)
            tot.append(s); print(variant, "held-out game", g, "spellF1 %.3f macroF1 %.3f poss %d D %.0f%% | cfg %s" % (s, macro_f1(P, T), npos, 100*np.mean(np.concatenate(P)==2), c), flush=True)
        print(variant, "LOGO mean spellF1 %.3f" % np.mean(tot), flush=True)
