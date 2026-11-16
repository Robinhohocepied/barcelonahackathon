"""v5 prep. variant: native (as before) | aug (extra random ball loss, independent of game state; for training)
| stress (heavier random ball loss; for robustness testing only). -> ext/p5_<variant>_<src>_<g>.pkl"""
import os, sys, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "pipeline")); sys.path.insert(0, HERE)
import core, stack, frames
from degrade import degrade, to_static, _ball_markov
M = pickle.load(open(os.path.join(HERE, "pipeline", "model.pkl"), "rb"))
EXTRA = {"native": None, "aug": 0.75, "stress": 0.55}            # share of frames where the extra mask keeps the ball
if __name__ == "__main__":
    variant, src, g = sys.argv[1:4]
    if src == "metrica":
        import metrica_open; halves = metrica_open.load(int(g), os.path.join(HERE, "ext", "metrica")); seed = 1000 + int(g)
    elif src == "idsse":
        import idsse; seed = sum(map(ord, g))
        halves = []
        for h in idsse.load(os.path.join(HERE, "idsse", "idsse_%s.npz" % g)):      # crop to first..last live frame of the half
            live = np.where(h["label"] != 2)[0]; a, b = live[0], live[-1] + 1
            halves.append(dict(pos=h["pos"][a:b], ball=h["ball"][a:b], team=h["team"], label=h["label"][a:b]))
    else:
        import skillcorner; halves = skillcorner.load(int(g), os.path.join(HERE, "ext", "sk")); seed = int(g) % 9973
    rng = np.random.default_rng(seed); rng2 = np.random.default_rng(seed + {"native": 0, "aug": 77, "stress": 99}[variant])
    out = []
    for k, h in enumerate(halves):
        d = degrade(h, rng, ball_mode="markov" if src == "idsse" else "native"); d["pos"] = to_static(d, h["team"]); st = np.broadcast_to(h["team"], d["slot_team"].shape)
        if EXTRA[variant]: d["ball"][~_ball_markov(len(d["ball"]), rng2, EXTRA[variant], 60)] = np.nan
        X, idx = core.features(d["pos"], d["ball"], st)
        ph = M["team"].predict_proba(X[:, core.TEAM_COLS])[:, 1]; pd = M["dead"].predict_proba(X[:, core.DEAD_COLS])[:, 1]
        out.append(dict(src=src, game=g, half=k + 1, variant=variant, X=X.astype(np.float32), ph=ph, pd=pd, E=stack.evidence(d["ball"][idx]), B5=d["ball"][idx].astype(np.float32),
                        F=frames.frame_inputs(d["pos"], d["ball"], st, ph, pd), label=h["label"].astype(np.int8), n=len(h["label"]),
                        vis=float(np.isfinite(d["ball"][:, 0]).mean())))
    pickle.dump(out, open(os.path.join(HERE, "ext", "p5_%s_%s_%s.pkl" % (variant, src, g)), "wb"))
    print("ok", variant, src, g, "ball visible %.0f%%" % (100 * np.mean([o["vis"] for o in out])), flush=True)
