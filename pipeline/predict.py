"""Frame-by-frame possession labels from Metrica ATD tracking.
    python predict.py --atd-raw data/tactical/BAR-ATH_tactical_atd.txt --atd-meta data/tactical/BAR-ATH_tactical_atd.xml --out submission.csv
Output: frame,label with H / A / D on graded frames and X elsewhere. Needs numpy + scikit-learn (the sequence model runs in plain numpy)."""
import argparse, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--atd-raw", required=True); ap.add_argument("--atd-meta", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_v41.pkl"))
    ap.add_argument("--halves", default=None, help="graded video-frame ranges 'a:b,c:d' (inclusive); default = BAR-ATH from DATA.md")
    ap.add_argument("--n-frames", type=int, default=None, help="rows to write; default 199319 for BAR-ATH, else the last frame in the file")
    a = ap.parse_args()
    halves = [tuple(int(x) for x in h.split(":")) for h in a.halves.split(",")] if a.halves else core.GRADED["BAR-ATH"]
    m = pickle.load(open(a.model, "rb")); print("reading ATD ...", flush=True)
    pos, ball, team = core.read_atd(a.atd_raw, a.atd_meta)
    n_total = a.n_frames or (199319 if not a.halves else len(pos))
    print("frames %d, slots %d (home %d / away %d), ball visible %.1f%%" % (len(pos), len(team), (team == 0).sum(), (team == 1).sum(), 100 * np.isfinite(ball[:, 0]).mean()), flush=True)
    out = np.full(n_total, "X", dtype="<U1")
    for lo, hi in halves:
        hi = min(hi, len(pos) - 1)
        p, b = pos[lo:hi + 1], ball[lo:hi + 1]
        X, _ = core.features(p, b, np.broadcast_to(team, (len(p), len(team))))
        _ = np.arange(0, len(p), core.STRIDE)
        if "tcn" in m:                                     # v4: v2 probabilities averaged with a sequence model (TCN) read at 25 fps
            import stack, frames, tcn
            st = np.broadcast_to(team, (len(p), len(team)))
            ph = m["team"].predict_proba(X[:, core.TEAM_COLS])[:, 1]; pd = m["dead"].predict_proba(X[:, core.DEAD_COLS])[:, 1]
            if "stack5" in m:                              # v4.1: positive-only second stage (stack5.py)
                import stack5
                qh, qd = stack5.apply(m["stack5"], X, ph, pd, stack.evidence(b[_]))
            else:
                qh, qd = stack.apply(m["stack"], X, ph, pd, stack.evidence(b[_]))
            Pv = np.column_stack([(1 - qd) * qh, (1 - qd) * (1 - qh), qd])
            Pf = tcn.np_forward(m["tcn"], frames.frame_inputs(p, b, st, ph, pd))
            T5 = -(-len(Pf) // 5); Pt = np.vstack([Pf, np.repeat(Pf[-1:], T5 * 5 - len(Pf), 0)]).reshape(T5, 5, 3).mean(1)
            c = m["decode"]; k = min(len(Pt), len(Pv)); P = c["w"] * Pt[:k] + (1 - c["w"]) * Pv[:k]
            s = core.viterbi(np.log(np.clip(P, 1e-4, 1)), c["Ls"], c["lam_ha"], c["lam_d"])
            codes = s[np.minimum(np.arange(len(p)) // core.STRIDE, len(s) - 1)]
        elif "stack" in m:                                 # v2: ball-in-play evidence + recalibration (stack.py)
            import stack
            ph = m["team"].predict_proba(X[:, core.TEAM_COLS])[:, 1]; pd = m["dead"].predict_proba(X[:, core.DEAD_COLS])[:, 1]
            qh, qd = stack.apply(m["stack"], X, ph, pd, stack.evidence(b[_]))
            qh, qd = np.clip(qh, 1e-3, 1 - 1e-3), np.clip(qd, 1e-3, 1 - 1e-3)
            c = m["decode"]
            s = core.viterbi(np.log(np.column_stack([(1 - qd) * qh, (1 - qd) * (1 - qh), qd])), c["Ls"], c["lam_ha"], c["lam_d"])
            codes = s[np.minimum(np.arange(len(p)) // core.STRIDE, len(s) - 1)]
        else:
            codes = core.decode(X, len(p), m["team"], m["dead"])
        out[lo:hi + 1] = np.array(["H", "A", "D"])[codes]
        print("half %d:%d  H %d  A %d  D %d" % (lo, hi, *(int((codes == c).sum()) for c in (0, 1, 2))), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("frame,label\n"); f.writelines("%d,%s\n" % (i, l) for i, l in enumerate(out))
    print("wrote", a.out, len(out), "rows")
