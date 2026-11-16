"""Download IDSSE (CC-BY 4.0, DFL / Sportec, mirror: huggingface.co/datasets/pysport/idsse-data) one match at a time,
convert with kloppy from LOCAL files, write idsse/idsse_<id>.npz, delete the big XML. Run on your Mac:
    python research/export_idsse_local.py            (all 7; ~400 MB temporary disk per match)
    python research/export_idsse_local.py J03WPY     (one match)"""
import os, subprocess, sys
import numpy as np
from kloppy import sportec
BASE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
IDS = {"J03WMX": "000001", "J03WN1": "000001", "J03WOH": "000002", "J03WOY": "000002", "J03WPY": "000002", "J03WQQ": "000002", "J03WR9": "000002"}

def get(name, dst):
    subprocess.run(["curl", "-sSL", "--fail", "-o", dst, BASE + name], check=True)

def export(mid, out="idsse"):
    os.makedirs(out, exist_ok=True); com = IDS[mid]
    meta = os.path.join(out, "meta_%s.xml" % mid); raw = os.path.join(out, "raw_%s.xml" % mid)
    print(mid, "downloading ...", flush=True)
    get("DFL_02_01_matchinformation_DFL-COM-%s_DFL-MAT-%s.xml" % (com, mid), meta)
    get("DFL_04_03_positions_raw_observed_DFL-COM-%s_DFL-MAT-%s.xml" % (com, mid), raw)
    print(mid, "parsing ...", flush=True)
    ds = sportec.load_tracking(meta_data=meta, raw_data=raw, only_alive=False)
    pd_ = ds.metadata.pitch_dimensions; L, W = float(pd_.pitch_length), float(pd_.pitch_width)
    players = [p for t in ds.metadata.teams for p in t.players]
    slot = {p.player_id: i for i, p in enumerate(players)}
    team = np.array([0 if p.team is ds.metadata.teams[0] else 1 for p in players], dtype=np.int8)
    n = len(ds.frames)
    pos = np.full((n, len(players), 2), np.nan, dtype=np.float32); ball = np.full((n, 2), np.nan, dtype=np.float32)
    label = np.full(n, 2, dtype=np.int8); period = np.zeros(n, dtype=np.int8)
    for i, fr in enumerate(ds.frames):
        period[i] = fr.period.id
        b = fr.ball_coordinates
        if b is not None and b.x is not None: ball[i] = (b.x * L, b.y * W)
        for p, pdata in fr.players_data.items():
            c = pdata.coordinates
            if c is not None and c.x is not None and p.player_id in slot: pos[i, slot[p.player_id]] = (c.x * L, c.y * W)
        if fr.ball_state is not None and fr.ball_state.name != "DEAD":
            label[i] = 0 if fr.ball_owning_team is ds.metadata.teams[0] else 1
    path = os.path.join(out, "idsse_%s.npz" % mid)
    np.savez_compressed(path, pos=pos, ball=ball, team=team, label=label, period=period)
    os.remove(raw); os.remove(meta)
    print("%s DONE: %d frames | H %.0f%% A %.0f%% D %.0f%% | ball %.0f%% | %s" % (mid, n, *(100 * np.mean(label == c) for c in (0, 1, 2)),
          100 * np.mean(np.isfinite(ball[:, 0])), path), flush=True)

if __name__ == "__main__":
    for mid in (sys.argv[1:] or list(IDS)):
        export(mid)
