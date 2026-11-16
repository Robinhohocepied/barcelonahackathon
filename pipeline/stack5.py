"""v5 second stage: truly positive-only ball evidence. The dead-ball model sees only 'ball visible, in the pitch and
moving' signals and player speed, never ball visibility itself, so a missing ball cannot push towards dead ball."""
import numpy as np
from sklearn.linear_model import LogisticRegression
import stack
POS = [0, 1, 2]                                   # stack.evidence columns: fast +-1 s, fast +-3 s, moving +-1 s

def dead_X(X, pd, E): return np.nan_to_num(np.column_stack([stack.lg(pd), E[:, POS], X[:, 94]]))
def fit(rows):
    Xd = np.vstack([dead_X(r["X"], r["pd"], r["E"]) for r in rows]); yd = np.concatenate([r["yd"] for r in rows])
    cd = LogisticRegression(C=1.0, max_iter=500).fit(Xd, yd)
    k = yd == 0
    Xt = np.vstack([stack.team_X(r["X"], r["ph"]) for r in rows])[k]; yt = np.concatenate([r["yh"] for r in rows])[k]
    return cd, LogisticRegression(C=1.0, max_iter=500).fit(Xt, yt)
def apply(st, X, ph, pd, E):
    cd, ct = st
    return ct.predict_proba(stack.team_X(X, ph))[:, 1], cd.predict_proba(dead_X(X, pd, E))[:, 1]
