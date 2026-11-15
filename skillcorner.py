"""SkillCorner open data (MIT) -> 25 fps half arrays with H/A/D labels.

Labels: no dead-ball flag exists in this data, so D is a PRIOR, not a measurement:
the ball is alive from one player_possession event to the next and the label stays with the last
team (loose ball and passes in flight keep the last owner, as CHALLENGE.md says). A gap between two
possession events longer than DEAD_GAP_S is a stoppage: D from LEAD_S after the previous event
ends until the next one starts.
"""
import json
import numpy as np
import pandas as pd

DEAD_GAP_S = 8.0
LEAD_S = 2.0
SRC_FPS, OUT_FPS = 10, 25


def _upsample(a, n_out, nearest_only=False):
    t = np.arange(n_out) / (OUT_FPS / SRC_FPS)
    i0 = np.minimum(np.floor(t).astype(int), len(a) - 1)
    i1 = np.minimum(i0 + 1, len(a) - 1)
    near = np.minimum(np.round(t).astype(int), len(a) - 1)
    if nearest_only:
        return a[near]
    f = (t - i0).reshape((-1,) + (1,) * (a.ndim - 1))
    lerp = a[i0] * (1 - f) + a[i1] * f
    both = np.isfinite(a[i0]) & np.isfinite(a[i1])
    return np.where(both, lerp, a[near])


def load(mid, root):
    """List of two half dicts: pos (n,P,2) m, ball (n,2) m (NaN = not detected), team (P,), label (n,)."""
    m = json.load(open(f"{root}/{mid}_match.json"))
    home = m["home_team"]["id"]
    team_of = {p["id"]: (0 if p["team_id"] == home else 1) for p in m["players"]}
    pids = sorted(team_of)
    col = {p: i for i, p in enumerate(pids)}
    fr, per, ball, rows = [], [], [], []
    with open(f"{root}/{mid}_tracking.jsonl") as f:
        for line in f:
            r = json.loads(line)
            b = r["ball_data"]
            fr.append(r["frame"])
            per.append(r["period"] or 0)
            ball.append((b["x"], b["y"]) if (b.get("is_detected") and b["x"] is not None) else (np.nan, np.nan))
            rows.append([(p["player_id"], p["x"], p["y"]) for p in r["player_data"]
                         if p["player_id"] in col and p["x"] is not None])
    fr, per, ball = np.array(fr), np.array(per), np.array(ball, dtype=float)
    pos = np.full((len(fr), len(pids), 2), np.nan)
    for i, rr in enumerate(rows):
        for pid, x, y in rr:
            pos[i, col[pid]] = (x, y)
    team = np.array([team_of[p] for p in pids], dtype=np.int8)

    ev = pd.read_csv(f"{root}/../de_{mid}.csv", low_memory=False)
    ev = ev[ev.event_type == "player_possession"].sort_values("frame_start")
    ev_team = np.where(ev.team_id.values == home, 0, 1)
    es, ee = ev.frame_start.values, ev.frame_end.values

    halves = []
    for p in (1, 2):
        idx = np.where(per == p)[0]
        f0, f1 = fr[idx[0]], fr[idx[-1]]
        n10 = len(idx)
        lab10 = np.full(n10, 2, dtype=np.int8)
        sel = (es >= f0) & (es <= f1)
        s, e, t = es[sel], np.minimum(ee[sel], f1), ev_team[sel]
        for k in range(len(s)):
            nxt = s[k + 1] if k + 1 < len(s) else f1 + 1
            lab10[s[k] - f0: nxt - f0] = t[k]
            gap = nxt - e[k]
            if gap > DEAD_GAP_S * SRC_FPS:
                lab10[int(e[k] + LEAD_S * SRC_FPS) - f0: nxt - f0] = 2
        n25 = int((n10 - 1) * OUT_FPS / SRC_FPS) + 1
        halves.append(dict(
            pos=_upsample(pos[idx], n25),
            ball=_upsample(ball[idx], n25),
            team=team,
            label=_upsample(lab10.astype(float), n25, nearest_only=True).astype(np.int8),
        ))
    return halves
