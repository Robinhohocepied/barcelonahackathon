"""Run this on YOUR Mac (needs internet; figshare/HuggingFace are not reachable from my workspace).

    pip install kloppy numpy
    python export_idsse.py --limit 3000        # 2-minute smoke test on one match, prints what it sees
    python export_idsse.py                     # all 7 matches -> idsse/idsse_<id>.npz (~100 MB each)

Labels follow the challenge's own recipe (CHALLENGE.md): D where the tracking says the ball is dead,
otherwise the team the tracking names as owning the ball. IDSSE is CC-BY 4.0 (DFL / Sportec Solutions,
Bassek et al. 2025, doi 10.1038/s41597-025-04505-y) -- credit it in the write-up.
"""
import argparse, os
import numpy as np
from kloppy import sportec

IDS = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]


def export(mid, out_dir, limit=None):
    ds = sportec.load_open_tracking_data(match_id=mid, only_alive=False, limit=limit)
    pd_ = ds.metadata.pitch_dimensions
    L, W = float(pd_.pitch_length), float(pd_.pitch_width)
    players = [p for t in ds.metadata.teams for p in t.players]
    slot = {p.player_id: i for i, p in enumerate(players)}
    team = np.array([0 if p.team is ds.metadata.teams[0] else 1 for p in players], dtype=np.int8)
    n = len(ds.frames)
    pos = np.full((n, len(players), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    label = np.full(n, 2, dtype=np.int8)
    period = np.zeros(n, dtype=np.int8)
    for i, fr in enumerate(ds.frames):
        period[i] = fr.period.id
        b = fr.ball_coordinates
        if b is not None and b.x is not None:
            ball[i] = (b.x * L, b.y * W)
        for p, pdata in fr.players_data.items():
            c = pdata.coordinates
            if c is not None and c.x is not None and p.player_id in slot:
                pos[i, slot[p.player_id]] = (c.x * L, c.y * W)
        if fr.ball_state.name != "DEAD":
            label[i] = 0 if fr.ball_owning_team is ds.metadata.teams[0] else 1
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "idsse_%s.npz" % mid)
    np.savez_compressed(path, pos=pos, ball=ball, team=team, label=label, period=period)
    print("%s: %d frames, %d players, pitch %.1fx%.1f | H %.1f%% A %.1f%% D %.1f%% | ball present %.1f%% | -> %s" % (
        mid, n, len(players), L, W, *(100 * np.mean(label == c) for c in (0, 1, 2)),
        100 * np.mean(np.isfinite(ball[:, 0])), path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="idsse")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    for mid in ([a.only] if a.only else (IDS[:1] if a.limit else IDS)):
        export(mid, a.out, a.limit)
