"""v3 prep: Metrica open games + SkillCorner games -> ext/feats3_<name>.pkl (base-model probs, evidence, control features)"""
import os, sys, pickle, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, "pipeline")); sys.path.insert(0, HERE)
import core, stack, control
from degrade import degrade, to_static
M = pickle.load(open(os.path.join(HERE, "pipeline", "model.pkl"), "rb"))

def one(h, rng, mode):
    d = degrade(h, rng, ball_mode=mode); d["pos"] = to_static(d, h["team"]); st = np.broadcast_to(h["team"], d["slot_team"].shape)
    X, idx = core.features(d["pos"], d["ball"], st)
    ph = M["team"].predict_proba(X[:, core.TEAM_COLS])[:, 1]; pd = M["dead"].predict_proba(X[:, core.DEAD_COLS])[:, 1]
    ctrl = control.frame_control(d["pos"], d["ball"], st)
    return dict(X=X.astype(np.float32), ph=ph, pd=pd, E=stack.evidence(d["ball"][idx]), C=control.features(ctrl, len(idx)),
                LC=control.last_control(ctrl, len(idx)), label=h["label"], n=len(h["label"]))

if __name__ == "__main__":
    src, g = sys.argv[1], sys.argv[2]
    if src == "metrica":
        import metrica_open; halves = metrica_open.load(int(g)); rng = np.random.default_rng(1000 + int(g)); mode = "native"
    else:
        import skillcorner; halves = skillcorner.load(int(g), os.path.join(HERE, "ext", "sk")); rng = np.random.default_rng(int(g) % 9973); mode = "native"
    out = [dict(one(h, rng, mode), src=src, game=g, half=k + 1) for k, h in enumerate(halves)]
    pickle.dump(out, open(os.path.join(HERE, "ext", "feats3_%s_%s.pkl" % (src, g)), "wb")); print("ok", src, g, [o["X"].shape for o in out], flush=True)
