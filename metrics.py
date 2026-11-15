"""spellF1 and macroF1, ported from the description in EVALUATION.md (NOT from possession_metrics.py).
Codes: 0 = H, 1 = A, 2 = D, 3 = X. One array per half; halves never join."""
import numpy as np

H, A, D, X = 0, 1, 2, 3
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def ffill_x(a):
    """Replace X with the last non-X label in the same half; back-fill the head."""
    a = a.copy()
    ok = a != X
    if not ok.any():
        return np.full_like(a, D)
    idx = np.where(ok, np.arange(len(a)), -1)
    np.maximum.accumulate(idx, out=idx)
    first = np.argmax(ok)
    idx[idx < 0] = first
    return a[idx]


def possessions(a):
    """[(start, end_exclusive, team)] for maximal runs of H or A."""
    if len(a) == 0:
        return []
    ch = np.r_[0, np.where(np.diff(a) != 0)[0] + 1, len(a)]
    return [(int(ch[i]), int(ch[i + 1]), int(a[ch[i]])) for i in range(len(ch) - 1) if a[ch[i]] in (H, A)]


def _overlaps(P, T):
    """(iou, i, j) for every same-team pair that overlaps in time (two-pointer sweep)."""
    out, j0 = [], 0
    for i, (ps, pe, pt) in enumerate(P):
        while j0 < len(T) and T[j0][1] <= ps:
            j0 += 1
        j = j0
        while j < len(T) and T[j][0] < pe:
            ts, te, tt = T[j]
            if tt == pt:
                inter = min(pe, te) - max(ps, ts)
                out.append((inter / ((pe - ps) + (te - ts) - inter), i, j))
            j += 1
    return out


def spell_f1_counts(pred_halves, true_halves):
    """Per-threshold (tp, n_pred, n_true), pooled over halves."""
    tp = {t: 0 for t in THRESHOLDS}
    npred = ntrue = 0
    for p, g in zip(pred_halves, true_halves):
        P, T = possessions(ffill_x(p)), possessions(g)
        npred += len(P)
        ntrue += len(T)
        pairs = sorted(_overlaps(P, T), reverse=True)
        for t in THRESHOLDS:
            used_p, used_t = set(), set()
            for iou, i, j in pairs:
                if iou < t:
                    break
                if i not in used_p and j not in used_t:
                    used_p.add(i)
                    used_t.add(j)
            tp[t] += len(used_p)
    return tp, npred, ntrue


def spell_f1(pred_halves, true_halves):
    tp, npred, ntrue = spell_f1_counts(pred_halves, true_halves)
    f = [2 * tp[t] / (npred + ntrue) if npred + ntrue else 0.0 for t in THRESHOLDS]
    return float(np.mean(f)), npred, ntrue


def macro_f1(pred_halves, true_halves):
    p = np.concatenate(pred_halves)
    g = np.concatenate(true_halves)
    f = []
    for c in (H, A, D):
        tp = np.sum((p == c) & (g == c))
        fp = np.sum((p == c) & (g != c))
        fn = np.sum((p != c) & (g == c))
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f))
