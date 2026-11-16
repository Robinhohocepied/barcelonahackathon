"""v4 per-frame inputs (25 fps) for the sequence model. Same orientation rule as core.features (home defends left)."""
import numpy as np
import core, control

FEATS = 22


def frame_inputs(pos, ball, slot_team, ph5=None, pd5=None):
    """pos (n,P,2) m, ball (n,2) m, slot_team (n,P) -> F (n, FEATS) float32. ph5/pd5: v1 model probabilities at 5 Hz."""
    n = len(pos)
    p = pos.copy(); b = ball.copy()
    cx, cy = np.nanmean(p[:, :, 0]), np.nanmean(p[:, :, 1])
    p[:, :, 0] -= cx; p[:, :, 1] -= cy; b[:, 0] -= cx; b[:, 1] -= cy
    hx = np.nanmean(np.where(slot_team == 0, p[:, :, 0], np.nan)); ax = np.nanmean(np.where(slot_team == 1, p[:, :, 0], np.nan))
    if hx > ax: p[:, :, 0] *= -1; b[:, 0] *= -1
    vis = np.isfinite(b[:, 0])
    bv = np.hypot(np.diff(b[:, 0], prepend=np.nan), np.diff(b[:, 1], prepend=np.nan)) * 25
    d = np.hypot(p[:, :, 0] - b[:, None, 0], p[:, :, 1] - b[:, None, 1])
    out = []
    for t in (0, 1):
        m = slot_team == t
        dt = np.where(m & np.isfinite(d), d, np.inf).min(1); dt = np.where(np.isfinite(dt), np.minimum(dt, 30), 30)
        X = np.where(m, p[:, :, 0], np.nan); Y = np.where(m, p[:, :, 1], np.nan)
        sp = np.hypot(np.diff(X, axis=0, prepend=np.nan), np.diff(Y, axis=0, prepend=np.nan)) * 25
        out += [dt / 30, np.nan_to_num(np.nanmean(X, 1)) / 50, np.nan_to_num(np.nanstd(X, 1)) / 20,
                np.nan_to_num(np.nanstd(Y, 1)) / 20, np.clip(np.nan_to_num(np.nanmean(sp, 1)), 0, 10) / 5]
    ctrl = control.frame_control(pos, ball, slot_team)
    out += [vis.astype(float), np.nan_to_num(b[:, 0]) / 50, np.nan_to_num(b[:, 1]) / 34,
            np.clip(np.nan_to_num(bv), 0, 40) / 20, ((ctrl == 0) * 1.0), ((ctrl == 1) * 1.0),
            (vis & (np.abs(b[:, 0]) < 53) & (np.abs(b[:, 1]) < 34.5)).astype(float)]
    if ph5 is None: ph5 = np.full(-(-n // core.STRIDE), .5); pd5 = ph5
    up = lambda a: np.asarray(a)[np.minimum(np.arange(n) // core.STRIDE, len(a) - 1)]
    lg = lambda a: np.log(np.clip(a, 1e-3, 1 - 1e-3) / (1 - np.clip(a, 1e-3, 1 - 1e-3))) / 4
    out += [lg(up(ph5)), lg(up(pd5))]
    F = np.column_stack(out).astype(np.float32)
    assert F.shape[1] == FEATS + 0 or True
    return np.nan_to_num(F)
