"""Possession pipeline core: ATD reader, ball-free + ball features, Viterbi decoding.
Self-contained (numpy only here; sklearn only for the trained models). Used identically by train.py and predict.py."""
import warnings
import xml.etree.ElementTree as ET
import numpy as np
warnings.filterwarnings("ignore")

PITCH = (105.3, 68.0)            # metres, games.py (measured from corrected tracking)
STRIDE, W = 5, 10                # features at 5 Hz, +-2 s context windows
GRADED = {"BAR-ATH": [(11185, 90651), (110808, 190197)]}     # DATA.md, video frames, inclusive
DECODE = dict(Ls=(5, 5, 30), lam_ha=1.0, lam_d=3.0)         # min spell (5 Hz samples) for H, A, D; switch costs
TEAM_COLS = np.arange(94)                                     # H/A model: shape + player speed + ball features
DEAD_COLS = np.r_[0:84, 94:97]                                # dead-ball model: ball-free only (survived leave-one-source-out)
H, A, D = 0, 1, 2


# ---------- input: Metrica ATD raw text + metadata ----------
def slot_teams(meta_path):
    """0 = home, 1 = away for every track slot (same rule as starter/baseline.py)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], dtype=np.int8)


def read_atd(raw_path, meta_path):
    """-> pos (n,P,2) metres, ball (n,2) metres, slot_team (P,), indexed by video frame (frame_id - 1). NaN = missing."""
    team = slot_teams(meta_path); P = len(team)
    L, Wd = PITCH
    ids, rows, balls = [], [], []
    with open(raw_path) as f:
        for line in f:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            v = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)[:3 * P].reshape(-1, 3)
            if len(v) < P:
                v = np.vstack([v, np.full((P - len(v), 3), np.nan, np.float32)])
            b = np.array(parts[2].split(","), dtype=np.float32)
            ids.append(int(parts[0]) - 1); rows.append(v[:, :2]); balls.append(b[:2])
    n = max(ids) + 1
    pos = np.full((n, P, 2), np.nan, np.float32); ball = np.full((n, 2), np.nan, np.float32)
    pos[ids] = np.array(rows); ball[ids] = np.array(balls)
    pos[..., 0] *= L; pos[..., 1] *= Wd; ball[:, 0] *= L; ball[:, 1] *= Wd
    return pos.astype(np.float64), ball.astype(np.float64), team


# ---------- features ----------
def wmean(a, lo, hi):
    """nan-aware mean of a[i+lo : i+hi+1] along axis 0"""
    a = a.reshape(len(a), -1); ok = np.isfinite(a)
    c = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(np.where(ok, a, 0), 0)])
    k = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(ok, 0)])
    i = np.arange(len(a)); l, h = np.clip(i + lo, 0, len(a)), np.clip(i + hi + 1, 0, len(a))
    n = k[h] - k[l]
    return np.where(n > 0, (c[h] - c[l]) / np.maximum(n, 1), np.nan)


def nearest(pos, ball, slot_team):
    """team of the nearest tracked player to the ball and the distance (m); -1 / inf where unknown"""
    dist = np.hypot(pos[:, :, 0] - ball[:, None, 0], pos[:, :, 1] - ball[:, None, 1])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    j = np.argmin(dist, axis=1); best = dist[np.arange(len(j)), j]
    team = np.where(np.isfinite(best), slot_team[np.arange(len(j)), j], -1).astype(np.int8)
    return team, best.astype(np.float32)


def pair_feats(px, py, st):
    n, P = px.shape; out = np.full((n, 4), np.nan)
    for s in range(0, n, 1500):
        x, y, t = px[s:s + 1500], py[s:s + 1500], st[s:s + 1500]
        d = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        same = (t[:, :, None] == t[:, None, :]); eye = np.eye(P, dtype=bool)[None]
        d = np.where(np.isfinite(d), d, np.inf)
        dm = np.where(same & ~eye, d, np.inf).min(2); do = np.where(~same, d, np.inf).min(2)
        dm[~np.isfinite(dm)] = np.nan; do[~np.isfinite(do)] = np.nan
        for ti in (0, 1):
            m = (t == ti) & np.isfinite(x)
            out[s:s + 1500, ti] = np.nanmean(np.where(m, dm, np.nan), 1)
            out[s:s + 1500, 2 + ti] = np.nanmean(np.where(m, do, np.nan), 1)
    return out


def features(pos, ball, slot_team):
    """pos (n,P,2) m, ball (n,2) m, slot_team (n,P) 0/1 -> X (m,103) at 5 Hz and the frame indices (relative to the input).
    cols 0:84 team shape (ball-free), 84:94 ball, 94:97 player speed, 97:103 ball speed / visibility."""
    n = len(pos); idx = np.arange(0, n, STRIDE)
    tm_full, ds_full = nearest(pos, ball, slot_team)
    p, b, st = pos[idx].copy(), ball[idx].copy(), slot_team[idx]
    cx, cy = np.nanmean(p[:, :, 0]), np.nanmean(p[:, :, 1])          # centre on the pitch (x AND y: no source-specific origin)
    p[:, :, 0] -= cx; p[:, :, 1] -= cy; b[:, 0] -= cx; b[:, 1] -= cy
    hx = np.nanmean(np.where(st == 0, p[:, :, 0], np.nan)); ax = np.nanmean(np.where(st == 1, p[:, :, 0], np.nan))
    if hx > ax:                                                        # home defends left
        p[:, :, 0] *= -1; b[:, 0] *= -1
    px, py = p[:, :, 0], p[:, :, 1]
    cols = []
    for t in (0, 1):
        m = (st == t) & np.isfinite(px); X, Y = np.where(m, px, np.nan), np.where(m, py, np.nan)
        mx, my = np.nanmean(X, 1), np.nanmean(Y, 1)
        cols += [m.sum(1), mx, my, np.nanstd(X, 1), np.nanstd(Y, 1), np.nanmin(X, 1), np.nanmax(X, 1),
                 np.nanmean(np.hypot(X - mx[:, None], Y - my[:, None]), 1)]
    pf = pair_feats(px, py, st); cols += [pf[:, i] for i in range(4)]
    base = np.column_stack(cols); base = np.column_stack([base, base[:, 1] - base[:, 9]])
    smooth = wmean(base, -W, W); delta = wmean(base, 1, W) - wmean(base, -W, -1)
    delta2 = wmean(base, 1, 3 * W) - wmean(base, -3 * W, -1)
    Xf = np.column_stack([base, smooth, delta, delta2])                               # 84
    tm, ds = tm_full[idx].astype(float), ds_full[idx]
    conf = (ds <= 3.0) & (tm >= 0); vote = np.where(conf, (tm == 0).astype(float), np.nan)
    vis = np.isfinite(b[:, 0]).astype(float)
    bcols = np.column_stack([wmean(vote, -W, W), wmean(vote, -3, 3), wmean(vote, -3 * W, 3 * W),
                             wmean(np.where(conf, ds, np.nan), -W, W), wmean(vis, -W, W),
                             b[:, 0], b[:, 1], b[:, 0] - base[:, 1], b[:, 0] - base[:, 9], wmean(b[:, 0], -W, W)])  # 10
    sp = np.hypot(np.diff(p[:, :, 0], axis=0, prepend=np.nan), np.diff(p[:, :, 1], axis=0, prepend=np.nan)) * (25 / STRIDE)
    bsp = np.hypot(np.diff(b[:, 0], prepend=np.nan), np.diff(b[:, 1], prepend=np.nan)) * (25 / STRIDE)
    psp = np.nanmean(sp, 1); ecols = []
    for a in (psp, bsp):
        ecols += [wmean(a, -W, W)[:, 0], wmean(a, -3 * W, 3 * W)[:, 0], wmean(a, -W // 2, W // 2)[:, 0]]
    ecols += [wmean(vis, -3 * W, 3 * W)[:, 0], wmean(vis, -6 * W, 6 * W)[:, 0], np.hypot(b[:, 0] - np.nanmean(b[:, 0]), b[:, 1] - np.nanmean(b[:, 1]))]
    return np.column_stack([Xf, bcols, np.column_stack(ecols)]), idx


# ---------- decoding: 3-state Viterbi with minimum spell lengths and switch costs ----------
def viterbi(logp, Ls, lam_ha, lam_d):
    T, L, NEG = len(logp), max(Ls), -1e18
    C = np.array([[0, lam_ha, lam_d], [lam_ha, 0, lam_d], [lam_d, lam_d, 0]], float)
    V = np.full((3, L), NEG)
    for s in range(3): V[s, Ls[s] - 1] = 0
    PS = np.zeros((T, 3, L), dtype=np.int8); PJ = np.zeros((T, 3, L), dtype=np.int8)
    for t in range(T):
        N = np.full((3, L), NEG); S = np.zeros((3, L), dtype=np.int8); J = np.zeros((3, L), dtype=np.int8)
        for s in range(3):
            l = Ls[s]
            if l > 1:
                N[s, 1:l] = V[s, :l - 1]; S[s, 1:l] = s; J[s, 1:l] = np.arange(l - 1)
            if V[s, l - 1] > N[s, l - 1]: N[s, l - 1] = V[s, l - 1]; S[s, l - 1] = s; J[s, l - 1] = l - 1
            best, kb = NEG, 0
            for k in range(3):
                if k != s:
                    v = V[k, Ls[k] - 1] - C[k, s]
                    if v > best: best, kb = v, k
            if best > N[s, 0]: N[s, 0], S[s, 0], J[s, 0] = best, kb, Ls[kb] - 1
        V = N + logp[t][:, None]; PS[t], PJ[t] = S, J
    s = int(np.argmax([V[i, Ls[i] - 1] for i in range(3)])); j = Ls[s] - 1
    out = np.empty(T, dtype=np.int8)
    for t in range(T - 1, -1, -1):
        out[t] = s; s, j = int(PS[t, s, j]), int(PJ[t, s, j])
    return out


def decode(X, n, team_clf, dead_clf, cfg=DECODE):
    """X: features of one half (5 Hz), n: its frame count -> label codes (n,) 0 H / 1 A / 2 D at 25 fps"""
    ph = np.clip(team_clf.predict_proba(X[:, TEAM_COLS])[:, 1], 1e-3, 1 - 1e-3)
    pd = np.clip(dead_clf.predict_proba(X[:, DEAD_COLS])[:, 1], 1e-3, 1 - 1e-3)
    lp = np.log(np.column_stack([(1 - pd) * ph, (1 - pd) * (1 - ph), pd]))
    s = viterbi(lp, cfg["Ls"], cfg["lam_ha"], cfg["lam_d"])
    return s[np.minimum(np.arange(n) // STRIDE, len(s) - 1)]
