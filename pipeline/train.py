"""Train on EXTERNAL data only (IDSSE CC-BY, SkillCorner MIT), degraded to look like the ATD.
    python train.py prep <idsse|sk> <name-or-path>     features for one match -> feats/<name>.pkl
    python train.py cv                                  leave-one-match-out, D-F1 / spellF1 / macroF1
    python train.py fit [--holdout NAME] --out model.pkl"""
import glob, os, pickle, sys, argparse
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import core
mk = lambda: HGB(max_iter=250, learning_rate=0.06, max_leaf_nodes=24, l2_regularization=1.0, random_state=0)


def prep(kind, arg):
    from degrade import degrade
    rng = np.random.default_rng(abs(hash(os.path.basename(arg))) % 997 if False else sum(map(ord, os.path.basename(arg))))
    if kind == "idsse":
        import idsse; halves = idsse.load(arg); mode = "markov"; name = os.path.basename(arg)[6:-4]
    else:
        import skillcorner; halves = skillcorner.load(int(arg), os.environ.get('MTAH_EXT', os.path.join(os.path.dirname(HERE), 'ext', 'sk'))); mode = "native"; name = "sk" + arg
    out = []
    for h in halves:
        d = degrade(h, rng, ball_mode=mode)
        from degrade import to_static
        d["pos"] = to_static(d, h["team"]); d["slot_team"] = np.broadcast_to(h["team"], d["slot_team"].shape)   # same representation as the real ATD
        X, idx = core.features(d["pos"], d["ball"], d["slot_team"])
        out.append(dict(X=X.astype(np.float32), y=d["label"][idx], label=d["label"], n=len(d["label"])))
    os.makedirs(os.path.join(HERE, "feats"), exist_ok=True)
    pickle.dump(out, open(os.path.join(HERE, "feats", name + ".pkl"), "wb")); print("saved", name, [h["X"].shape for h in out])


def load_feats():
    return {os.path.basename(p)[:-4]: pickle.load(open(p, "rb")) for p in sorted(glob.glob(os.path.join(HERE, "feats", "*.pkl")))}


def fit(tr):
    X = np.vstack([h["X"] for h in tr]); y = np.concatenate([h["y"] for h in tr]); k = y < 2
    team = mk().fit(X[k][::2][:, core.TEAM_COLS], (y[k][::2] == 0).astype(int))
    dead = mk().fit(X[::2][:, core.DEAD_COLS], (y[::2] == 2).astype(int))
    return team, dead


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(sys.argv[2], sys.argv[3])
    elif cmd == "cv":
        sys.path.insert(0, os.path.dirname(HERE)); from metrics import spell_f1, macro_f1
        F = load_feats(); rows = []
        for m in F:
            tr = [h for k, v in F.items() if k != m for h in v]; team, dead = fit(tr)
            pred = [core.decode(h["X"], h["n"], team, dead) for h in F[m]]; truth = [h["label"] for h in F[m]]
            pc, tc = np.concatenate(pred) == 2, np.concatenate(truth) == 2
            df1 = 2 * (pc & tc).sum() / max(1, pc.sum() + tc.sum())
            rows.append((m, spell_f1(pred, truth)[0], macro_f1(pred, truth), df1)); print("%-10s spellF1 %.3f macroF1 %.3f D-F1 %.3f" % rows[-1], flush=True)
        for nm, sel in (("IDSSE", [r for r in rows if r[0][:2] != "sk"]), ("SkillCorner", [r for r in rows if r[0][:2] == "sk"])):
            a = np.array([r[1:] for r in sel]); print("%s mean (n=%d): spellF1 %.4f macroF1 %.4f D-F1 %.3f" % (nm, len(sel), *a.mean(0)))
    elif cmd == "fit":
        ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--holdout", default=None); ap.add_argument("--out", default="model.pkl"); a = ap.parse_args()
        F = load_feats(); tr = [h for k, v in F.items() if k != a.holdout for h in v]
        team, dead = fit(tr); pickle.dump(dict(team=team, dead=dead, trained_on=[k for k in F if k != a.holdout]), open(a.out, "wb")); print("saved", a.out, "trained on", [k for k in F if k != a.holdout])
