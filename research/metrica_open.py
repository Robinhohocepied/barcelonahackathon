"""Metrica Sports open sample data (github.com/metrica-sports/sample-data, 3 anonymised games, 25 fps,
synchronised tracking + human-annotated events) -> half dicts like idsse.load(): pos/ball in metres, team, label.

Labels from the events, following CHALLENGE.md:
  - ball alive: the team of the last controlling event (PASS, CARRY, RECOVERY, SET PIECE, SHOT, BALL LOST);
    CHALLENGE and FAULT RECEIVED do not move possession, so a loose ball stays with the last owner.
  - ball dead (D): from a stoppage (BALL OUT, FAULT RECEIVED, CARD, a SHOT that is a goal) until the next SET PIECE
    starts (or, if none is logged, the next controlling event).
  - frames before the first / after the last event of a half are not graded (label 3 = X).
Fetch: research/fetch_metrica.sh  (files are NOT committed: no redistribution)."""
import json, os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

L, W = 105.0, 68.0
OWN = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST"}
STOP = {"BALL OUT", "FAULT RECEIVED", "CARD"}


def _labels(ev, n_frames):
    """ev: DataFrame [period, start, end, type, subtype, team(0/1)] -> label (n_frames,), 0/1/2/3, frame = index+1"""
    lab = np.full(n_frames, 3, np.int8)
    for per, e in ev.groupby("period"):
        e = e.sort_values(["start", "end"]).reset_index(drop=True)
        f0, f1 = int(e.start.min()), int(e[["start", "end"]].max(axis=1).max())
        cur, dead = None, False
        marks = []                                   # (frame, state) state 0/1/2
        for r in e.itertuples():
            goal = r.type == "SHOT" and isinstance(r.subtype, str) and "GOAL" in r.subtype.upper()
            if r.type in STOP:
                marks.append((int(r.start), 2)); dead = True
            elif r.type == "SET PIECE":
                marks.append((int(r.start), r.team)); dead = False; cur = r.team
            elif r.type in OWN and r.team in (0, 1):
                if not dead or r.type != "BALL LOST":
                    marks.append((int(r.start), r.team)); cur = r.team; dead = False
                if goal:
                    marks.append((int(max(r.end, r.start)), 2)); dead = True
        marks.sort(key=lambda m: m[0])
        for k, (f, s) in enumerate(marks):
            g = marks[k + 1][0] if k + 1 < len(marks) else f1 + 1
            lab[max(f, f0) - 1: min(g, f1 + 1) - 1] = s
    return lab


def _halves(pos, ball, team, lab, period):
    out = []
    for p in (1, 2):
        idx = np.where((period == p) & (lab != 3))[0]
        if len(idx):
            a, b = idx[0], idx[-1] + 1
            out.append(dict(pos=pos[a:b], ball=ball[a:b], team=team, label=lab[a:b].copy()))
    return out


def _csv_game(root, g):
    trk = {}
    for side in ("Home", "Away"):
        t = pd.read_csv(f"{root}/Sample_Game_{g}_RawTrackingData_{side}_Team.csv", skiprows=2)
        trk[side] = t
    h, a = trk["Home"], trk["Away"]
    def players(t):
        cols = [c for c in t.columns if c.startswith("Player")]
        return [(t[c].values, t[t.columns[t.columns.get_loc(c) + 1]].values) for c in cols]
    ph, pa = players(h), players(a)
    n = len(h); P = len(ph) + len(pa)
    pos = np.full((n, P, 2), np.nan)
    for j, (x, y) in enumerate(ph + pa):
        pos[:, j, 0], pos[:, j, 1] = x * L, y * W
    bc = h.columns.get_loc("Ball")
    ball = np.column_stack([h.iloc[:, bc].values * L, h.iloc[:, bc + 1].values * W])
    team = np.array([0] * len(ph) + [1] * len(pa), np.int8)
    e = pd.read_csv(f"{root}/Sample_Game_{g}_RawEventsData.csv")
    ev = pd.DataFrame(dict(period=e.Period, start=e["Start Frame"], end=e["End Frame"], type=e.Type,
                           subtype=e.Subtype, team=e.Team.map({"Home": 0, "Away": 1}).fillna(-1).astype(int)))
    lab = _labels(ev, n)
    return _halves(pos, ball, team, lab, h.Period.values)


def _epts_game(root):
    r = ET.parse(f"{root}/Sample_Game_3_metadata.xml").getroot(); t = lambda e: e.tag.split("}")[-1]
    teams = [e.get("id") for e in r.iter() if t(e) == "Team"]
    pteam = {e.get("id"): teams.index(e.get("teamId")) for e in r.iter() if t(e) == "Player"}
    chan = {e.get("id"): e.get("playerId") for e in r.iter() if t(e) == "PlayerChannel"}
    pids = sorted(pteam, key=lambda p: (pteam[p], p)); col = {p: i for i, p in enumerate(pids)}
    specs = []
    for e in r.iter():
        if t(e) == "DataFormatSpecification":
            refs = [c.get("playerChannelId") for c in e.iter() if t(c) == "PlayerChannelRef"][::2]
            specs.append((int(e.get("startFrame")), int(e.get("endFrame")), [col[chan[c]] for c in refs]))
    lines = open(f"{root}/Sample_Game_3_tracking.txt").read().split("\n")
    lines = [l for l in lines if l]
    n = max(int(l.split(":", 1)[0]) for l in lines)
    pos = np.full((n, len(pids), 2), np.nan); ball = np.full((n, 2), np.nan)
    for l in lines:
        fr, pl, bl = l.split(":"); fr = int(fr)
        slots = next(s[2] for s in specs if s[0] <= fr <= s[1])
        v = np.array(pl.replace(";", ",").split(","), float).reshape(-1, 2)
        pos[fr - 1, slots[:len(v)]] = v[:len(slots)]
        ball[fr - 1] = np.array(bl.split(",")[:2], float)
    pos[..., 0] *= L; pos[..., 1] *= W; ball[:, 0] *= L; ball[:, 1] *= W
    team = np.array([pteam[p] for p in pids], np.int8)
    d = json.load(open(f"{root}/Sample_Game_3_events.json"))["data"]
    ev = pd.DataFrame(dict(period=[x["period"] for x in d], start=[x["start"]["frame"] for x in d],
                           end=[x["end"]["frame"] for x in d], type=[x["type"]["name"] for x in d],
                           subtype=[(x["subtypes"]["name"] if isinstance(x["subtypes"], dict) else
                                     ",".join(s["name"] for s in x["subtypes"]) if x["subtypes"] else None) for x in d],
                           team=[teams.index(x["team"]["id"]) if x.get("team") else -1 for x in d]))
    lab = _labels(ev, n)
    period = np.zeros(n, np.int8)
    for p, e in ev.groupby("period"):
        period[int(e.start.min()) - 1: int(e[["start", "end"]].max(axis=1).max())] = p
    return _halves(pos, ball, team, lab, period)


def load(game, root=None):
    root = root or os.environ.get("MTAH_METRICA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ext", "metrica"))
    return _epts_game(root) if int(game) == 3 else _csv_game(root, int(game))
