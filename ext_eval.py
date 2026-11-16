"""External validation on Metrica open games (never the provided match).
    python ext_eval.py prep        -> ext/feats_metrica.pkl (degraded like the ATD, features, model probabilities)"""
import os, sys, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "pipeline")); sys.path.insert(0, HERE)
import core
from degrade import degrade, to_static
import metrica_open


def prep(model_path=os.path.join(HERE, "pipeline", "model.pkl"), out=os.path.join(HERE, "ext", "feats_metrica.pkl")):
    m = pickle.load(open(model_path, "rb")); res = []
    for g in (1, 2, 3):
        rng = np.random.default_rng(1000 + g)
        for k, h in enumerate(metrica_open.load(g)):
            d = degrade(h, rng, ball_mode="native")
            d["pos"] = to_static(d, h["team"]); st = np.broadcast_to(h["team"], d["slot_team"].shape)
            X, idx = core.features(d["pos"], d["ball"], st)
            ph = m["team"].predict_proba(X[:, core.TEAM_COLS])[:, 1]; pd_ = m["dead"].predict_proba(X[:, core.DEAD_COLS])[:, 1]
            res.append(dict(game=g, half=k + 1, X=X.astype(np.float32), ph=ph, pd=pd_, label=h["label"], n=len(h["label"]),
                            ball=d["ball"][idx], pos=d["pos"][idx].astype(np.float32), team=h["team"]))
            print("game", g, "half", k + 1, X.shape, flush=True)
    pickle.dump(res, open(out, "wb"))


if __name__ == "__main__":
    if sys.argv[1] == "prep": prep()
