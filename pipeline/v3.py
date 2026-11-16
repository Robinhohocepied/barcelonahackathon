"""v3: team model with frame-level control evidence, a turnover (H<->A boundary) detector, and a Viterbi whose
H<->A switch cost drops where the detector is confident. Fitted on external games only (Metrica open + SkillCorner)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
import core, stack

def _lc(LC): return np.column_stack([(LC == 0).astype(float), (LC == 1).astype(float)])
def team_X(r, qh): return np.nan_to_num(np.column_stack([stack.lg(qh), r["C"], _lc(r["LC"]), r["X"][:, 84] - .5]))
def bnd_X(r, qh, qd):
    W = core.wmean; l = stack.lg(qh)
    d1 = W(l, 1, 5)[:, 0] - W(l, -5, -1)[:, 0]; d2 = W(l, 1, 10)[:, 0] - W(l, -10, -1)[:, 0]
    lcch = np.r_[0, (np.diff(r["LC"].astype(int)) != 0) & (r["LC"][1:] >= 0) & (r["LC"][:-1] >= 0)].astype(float)
    return np.nan_to_num(np.column_stack([np.abs(d1), np.abs(d2), d1, l, stack.lg(qd), r["C"], _lc(r["LC"]),
                                          W(lcch, -2, 2)[:, 0], W(lcch, -5, 5)[:, 0], r["E"]]))

def labels5(r):
    lab = r["label"][::core.STRIDE][:len(r["ph"])]
    yd = (lab == 2).astype(int); yh = (lab == 0).astype(int)
    b = np.zeros(len(lab), int); last = -1
    for t in range(1, len(lab)):
        if lab[t] in (0, 1) and lab[t - 1] in (0, 1) and lab[t] != lab[t - 1]:
            b[max(0, t - 2): t + 3] = 1
    return yd, yh, b

def fit(rows_team, rows_dead):
    st = stack.fit(rows_dead)                                   # v2 dead + team recalibration (Metrica only)
    for r in rows_team + rows_dead: r["qh"], r["qd"] = stack.apply(st, r["X"], r["ph"], r["pd"], r["E"])
    k = np.concatenate([r["yd"] == 0 for r in rows_team])
    Xt = np.vstack([team_X(r, r["qh"]) for r in rows_team])[k]; yt = np.concatenate([r["yh"] for r in rows_team])[k]
    ct = LogisticRegression(C=1.0, max_iter=1000).fit(Xt, yt)
    Xb = np.vstack([bnd_X(r, r["qh"], r["qd"]) for r in rows_team])[k]; yb = np.concatenate([r["yb"] for r in rows_team])[k]
    cb = HGB(max_iter=200, learning_rate=0.06, max_leaf_nodes=16, l2_regularization=1.0, random_state=0).fit(Xb, yb)
    return dict(stack=st, team3=ct, bnd=cb)

def apply(m, r):
    qh, qd = stack.apply(m["stack"], r["X"], r["ph"], r["pd"], r["E"])
    th = m["team3"].predict_proba(team_X(r, qh))[:, 1]; pb = m["bnd"].predict_proba(bnd_X(r, qh, qd))[:, 1]
    return th, qd, pb

def viterbi_tv(logp, Ls, lam_ha_t, lam_d):
    """core.viterbi with a time-varying H<->A switch cost"""
    T, L, NEG = len(logp), max(Ls), -1e18
    V = np.full((3, L), NEG)
    for s in range(3): V[s, Ls[s] - 1] = 0
    PS = np.zeros((T, 3, L), dtype=np.int8); PJ = np.zeros((T, 3, L), dtype=np.int8)
    for t in range(T):
        C = ((0, lam_ha_t[t], lam_d), (lam_ha_t[t], 0, lam_d), (lam_d, lam_d, 0))
        N = np.full((3, L), NEG); S = np.zeros((3, L), dtype=np.int8); J = np.zeros((3, L), dtype=np.int8)
        for s in range(3):
            l = Ls[s]
            if l > 1: N[s, 1:l] = V[s, :l - 1]; S[s, 1:l] = s; J[s, 1:l] = np.arange(l - 1)
            if V[s, l - 1] > N[s, l - 1]: N[s, l - 1] = V[s, l - 1]; S[s, l - 1] = s; J[s, l - 1] = l - 1
            best, kb = NEG, 0
            for k in range(3):
                if k != s:
                    v = V[k, Ls[k] - 1] - C[k][s]
                    if v > best: best, kb = v, k
            if best > N[s, 0]: N[s, 0], S[s, 0], J[s, 0] = best, kb, Ls[kb] - 1
        V = N + logp[t][:, None]; PS[t], PJ[t] = S, J
    s = int(np.argmax([V[i, Ls[i] - 1] for i in range(3)])); j = Ls[s] - 1
    out = np.empty(T, dtype=np.int8)
    for t in range(T - 1, -1, -1):
        out[t] = s; s, j = int(PS[t, s, j]), int(PJ[t, s, j])
    return out

def decode(th, qd, pb, n, cfg):
    th = np.clip(th, 1e-3, 1 - 1e-3); qd = np.clip(qd, 1e-3, 1 - 1e-3); pb = np.clip(pb, 1e-3, 1 - 1e-3)
    lam_t = np.maximum(0.0, cfg["a0"] + cfg["b"] * np.log((1 - pb) / pb))
    s = viterbi_tv(np.log(np.column_stack([(1 - qd) * th, (1 - qd) * (1 - th), qd])), cfg["Ls"], lam_t, cfg["lam_d"])
    return s[np.minimum(np.arange(n) // core.STRIDE, len(s) - 1)]


def refine(s5, th, pb, thr, gap, margin):
    """ASRF-style refinement (Ishikawa et al. 2021): inside each decoded H/A spell, cut at boundary peaks
    (local maxima of pb above thr, at least `gap` samples apart) and give each piece the team its mean th favours,
    if it is clear by `margin`. s5 at 5 Hz -> refined s5."""
    s = s5.copy(); T = len(s)
    pk = np.where((pb > thr) & (pb >= np.r_[0, pb[:-1]]) & (pb >= np.r_[pb[1:], 0]))[0]
    ch = np.r_[0, np.where(np.diff(s) != 0)[0] + 1, T]
    for i in range(len(ch) - 1):
        a, b = ch[i], ch[i + 1]
        if s[a] == 2 or b - a < 2 * gap: continue
        cuts, last = [], a
        for p in pk[(pk > a + gap) & (pk < b - gap)]:
            if p - last >= gap: cuts.append(p); last = p
        edges = [a] + cuts + [b]
        for u, v in zip(edges[:-1], edges[1:]):
            m = th[u:v].mean()
            if m > .5 + margin: s[u:v] = 0
            elif m < .5 - margin: s[u:v] = 1
    return s


def decode_refined(th, qd, pb, n, cfg):
    s = decode(th, qd, pb, len(th) * core.STRIDE, cfg)[::core.STRIDE][:len(th)]
    if cfg.get("thr") is not None: s = refine(s, th, pb, cfg["thr"], cfg["gap"], cfg["margin"])
    return s[np.minimum(np.arange(n) // core.STRIDE, len(s) - 1)]
