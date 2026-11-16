"""v3 frame-level signals (25 fps -> 5 Hz).
control: the ball is visible, slow (< 5 m/s) and a player is within 1.5 m -> that player's team controls it
(possession-zone idea, Vidal-Codina et al. 2022). Missing ball = no information."""
import numpy as np
import core

R_CTRL, V_CTRL = 1.5, 5.0


def frame_control(pos, ball, slot_team):
    """-> (n,) int8: 0 home controls, 1 away controls, -1 unknown"""
    team, dist = core.nearest(pos, ball, slot_team)
    v = np.hypot(np.diff(ball[:, 0], prepend=np.nan), np.diff(ball[:, 1], prepend=np.nan)) * 25
    ok = np.isfinite(ball[:, 0]) & (dist <= R_CTRL) & (np.nan_to_num(v, nan=99) < V_CTRL) & (team >= 0)
    return np.where(ok, team, -1).astype(np.int8)


def features(ctrl, n5):
    """ctrl (n,) at 25 fps -> (n5, 12) at 5 Hz: home/away control share in past / future / centred windows"""
    h = (ctrl == 0).astype(float); a = (ctrl == 1).astype(float)
    cols = []
    for x in (h, a):
        for lo, hi in ((-12, -1), (1, 12), (-37, -1), (1, 37), (-6, 6), (-25, 25)):   # 0.5 s, 1.5 s, +-0.25 s, +-1 s
            cols.append(core.wmean(x, lo, hi)[:, 0])
    F = np.column_stack(cols)[::core.STRIDE][:n5]
    return np.nan_to_num(F, nan=0.0)


def last_control(ctrl, n5):
    """team that last controlled the ball (carried forward), at 5 Hz; -1 before the first control"""
    idx = np.where(ctrl >= 0, np.arange(len(ctrl)), -1); np.maximum.accumulate(idx, out=idx)
    lc = np.where(idx >= 0, ctrl[np.maximum(idx, 0)], -1)
    return lc[::core.STRIDE][:n5].astype(np.int8)
