"""v5 second stage: positive-only evidence in both directions.
alive evidence: ball visible, in the pitch and moving (stack5). dead evidence: ball visible and still, or visible at/over
a pitch line. A missing ball adds nothing either way. Learned on Metrica open games + IDSSE (whose ball is tracked during stoppages)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
import core, stack

def dead_evidence(ball5):
    b = ball5.astype(float); vis = np.isfinite(b[:, 0])
    sp = np.hypot(np.diff(b[:, 0], prepend=np.nan), np.diff(b[:, 1], prepend=np.nan)) * 5
    still = (vis & (np.nan_to_num(sp, nan=99) < 1.0)).astype(float)
    x, y = np.nan_to_num(b[:, 0], nan=50), np.nan_to_num(b[:, 1], nan=34)
    edge = (vis & ((x < 2) | (x > 103) | (y < 1.5) | (y > 66.5))).astype(float)
    out = (vis & ((x < -0.5) | (x > 105.5) | (y < -0.5) | (y > 68.5))).astype(float)
    W = core.wmean
    return np.nan_to_num(np.column_stack([W(still, -5, 5)[:, 0], W(still, -15, 15)[:, 0], W(edge, -5, 5)[:, 0], W(edge, -15, 15)[:, 0],
                                          W(out, -5, 5)[:, 0], W(out, -15, 15)[:, 0]]), nan=0.0)

def dead_X(X, pd, E, D): return np.nan_to_num(np.column_stack([stack.lg(pd), E[:, [0, 1, 2]], D, X[:, 94]]))
def fit(rows):
    Xd = np.vstack([dead_X(r["X"], r["pd"], r["E"], r["D"]) for r in rows]); yd = np.concatenate([r["yd"] for r in rows])
    cd = LogisticRegression(C=1.0, max_iter=1000).fit(Xd, yd)
    k = yd == 0
    Xt = np.vstack([stack.team_X(r["X"], r["ph"]) for r in rows])[k]; yt = np.concatenate([r["yh"] for r in rows])[k]
    return cd, LogisticRegression(C=1.0, max_iter=500).fit(Xt, yt)
def apply(st, X, ph, pd, E, D):
    cd, ct = st
    return ct.predict_proba(stack.team_X(X, ph))[:, 1], cd.predict_proba(dead_X(X, pd, E, D))[:, 1]
