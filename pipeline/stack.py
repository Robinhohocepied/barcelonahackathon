"""v2 second stage: recalibrates the two v1 models with ball-in-play evidence. Fitted on Metrica's open games only.
Ball evidence is positive-only by design: a visible ball that moves fast inside the pitch says 'alive'; a missing ball adds nothing."""
import numpy as np
from sklearn.linear_model import LogisticRegression
import core


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def evidence(ball5):
    """ball5: (m,2) metres at 5 Hz (NaN = not seen) -> (m,6)"""
    b = ball5.astype(float)
    vis = np.isfinite(b[:, 0]).astype(float)
    sp = np.hypot(np.diff(b[:, 0], prepend=np.nan), np.diff(b[:, 1], prepend=np.nan)) * 5
    inp = ((b[:, 0] > -1) & (b[:, 0] < 106) & (b[:, 1] > -1) & (b[:, 1] < 69)).astype(float)
    fast = ((sp > 8) & (inp > 0)).astype(float); mov = ((sp > 3) & (inp > 0)).astype(float)
    W = core.wmean
    cols = [W(fast, -5, 5)[:, 0], W(fast, -15, 15)[:, 0], W(mov, -5, 5)[:, 0], W(vis, -5, 5)[:, 0], W(vis, -15, 15)[:, 0],
            W(inp * vis, -5, 5)[:, 0]]
    return np.nan_to_num(np.column_stack(cols), nan=0.0)


def dead_X(X, pd, E): return np.nan_to_num(np.column_stack([lg(pd), E, X[:, 94]]))
def team_X(X, ph): return np.nan_to_num(np.column_stack([lg(ph), np.nan_to_num(X[:, 84] - .5), np.nan_to_num(X[:, 86] - .5)]))


def fit(rows):
    """rows: dicts with X, ph, pd, E, yd (1 = dead), yh (1 = home) at 5 Hz"""
    Xd = np.vstack([dead_X(r["X"], r["pd"], r["E"]) for r in rows]); yd = np.concatenate([r["yd"] for r in rows])
    cd = LogisticRegression(C=1.0, max_iter=500).fit(Xd, yd)
    k = yd == 0
    Xt = np.vstack([team_X(r["X"], r["ph"]) for r in rows])[k]; yt = np.concatenate([r["yh"] for r in rows])[k]
    ct = LogisticRegression(C=1.0, max_iter=500).fit(Xt, yt)
    return cd, ct


def apply(st, X, ph, pd, E):
    cd, ct = st
    return ct.predict_proba(team_X(X, ph))[:, 1], cd.predict_proba(dead_X(X, pd, E))[:, 1]
