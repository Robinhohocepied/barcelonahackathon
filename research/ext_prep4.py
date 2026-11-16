"""v4 prep: per-frame inputs + labels for Metrica open games and SkillCorner games -> ext/f4_<src>_<g>.npz"""
import os, sys, pickle, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, "pipeline")); sys.path.insert(0, HERE)
import core, frames
from degrade import degrade, to_static
M = pickle.load(open(os.path.join(HERE, "pipeline", "model.pkl"), "rb"))
if __name__ == "__main__":
    src, g = sys.argv[1], sys.argv[2]
    if src == "metrica":
        import metrica_open; halves = metrica_open.load(int(g), os.path.join(HERE, "ext", "metrica")); rng = np.random.default_rng(1000 + int(g))
    else:
        import skillcorner; halves = skillcorner.load(int(g), os.path.join(HERE, "ext", "sk")); rng = np.random.default_rng(int(g) % 9973)
    arrs = {}
    for k, h in enumerate(halves):
        d = degrade(h, rng, ball_mode="native"); d["pos"] = to_static(d, h["team"]); st = np.broadcast_to(h["team"], d["slot_team"].shape)
        X, idx = core.features(d["pos"], d["ball"], st)
        ph = M["team"].predict_proba(X[:, core.TEAM_COLS])[:, 1]; pd = M["dead"].predict_proba(X[:, core.DEAD_COLS])[:, 1]
        F = frames.frame_inputs(d["pos"], d["ball"], st, ph, pd)
        arrs["F%d" % k] = F; arrs["y%d" % k] = h["label"].astype(np.int8)
    np.savez_compressed(os.path.join(HERE, "ext", "f4_%s_%s.npz" % (src, g)), **arrs)
    print("ok", src, g, [arrs["F%d" % k].shape for k in range(len(halves))], flush=True)
