"""Turn ONE external match (IDSSE npz) into the challenge's ATD format (0-1 coordinates, 46 track slots, static slot->team in the
metadata XML, NaN gaps, ball visible ~59%), with dead time before / between / after the halves, plus a full-length GT csv (X outside the halves).
    python make_test_file.py --npz ../idsse/idsse_J03WR9.npz --out testdata"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'research'))
from degrade import degrade
import idsse
ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()
L, Wd, SLOTS, PER = 105.3, 68.0, 46, 23
rng = np.random.default_rng(a.seed); halves = idsse.load(a.npz)
GAPS = [400, 900, 500, 100]                                                    # frames of nothing before, between, after (ungraded)
pos_all, ball_all, gt, ranges, t = [], [], [], [], 0
def blank(n):
    return np.full((n, SLOTS, 2), np.nan), np.full((n, 2), np.nan)
p, b = blank(GAPS[0]); pos_all.append(p); ball_all.append(b); gt.append(np.full(GAPS[0], "X")); t = GAPS[0]
for k, h in enumerate(halves):
    d = degrade(h, rng, ball_mode="markov")                                    # ATD-like: noise, track gaps, swaps, ball gaps
    n, P = d["pos"].shape[:2]; st = d["slot_team"]; tmo = h["team"]
    home_o, away_o = np.where(tmo == 0)[0], np.where(tmo == 1)[0]                # original slots keep their identity (tracklets persist)
    out = np.full((n, SLOTS, 2), np.nan)
    from degrade import to_static
    sp = to_static(d, tmo)
    out[:, :len(home_o)] = sp[:, home_o]; out[:, PER:PER + len(away_o)] = sp[:, away_o]
    pos_all.append(out); ball_all.append(d["ball"]); lab = np.array(["H", "A", "D"])[d["label"]]; gt.append(lab)
    ranges.append((t, t + n - 1)); t += n
    g2 = GAPS[1 + k]; p, b = blank(g2); pos_all.append(p); ball_all.append(b); gt.append(np.full(g2, "X")); t += g2
pos = np.concatenate(pos_all); ball = np.concatenate(ball_all); gt = np.concatenate(gt); N = len(pos)
os.makedirs(a.out, exist_ok=True)
def grp(x, y): return "NaN,NaN,NaN" if not np.isfinite(x) else "%.6f,%.6f,0.0" % (x / L, y / Wd)
with open(os.path.join(a.out, "TEST_tactical_atd.txt"), "w") as f:
    for i in range(N):
        f.write("%d:%s:%s\n" % (i + 1, ";".join(grp(*pos[i, s]) for s in range(SLOTS)), grp(*ball[i])))
with open(os.path.join(a.out, "TEST_tactical_atd.xml"), "w") as f:
    f.write('<Metadata><Teams><Team id="H"><Name>Home</Name></Team><Team id="A"><Name>Away</Name></Team></Teams><Players>')
    for s in range(SLOTS): f.write('<Player id="%d" teamId="%s"><Name>Track %d</Name></Player>' % (s + 1, "H" if s < PER else "A", s + 1))
    f.write("</Players></Metadata>")
with open(os.path.join(a.out, "TEST_gt.csv"), "w") as f:
    f.write("frame,label\n"); f.writelines("%d,%s\n" % (i, l) for i, l in enumerate(gt))
print("frames", N, "halves", ",".join("%d:%d" % r for r in ranges), "ball visible %.1f%%" % (100 * np.isfinite(ball[:, 0]).mean()))
