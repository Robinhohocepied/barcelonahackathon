"""Make external tracking look like the challenge's automatic ATD.
Everything here is calibrated on facts published in DATA.md (ball visible 59.3% of graded frames,
mean 20.5 tracked objects, identity switches, no player IDs) -- not on the provided match."""
import numpy as np

DEFAULTS = dict(
    noise_m=0.1,            # position noise, players and ball (std, metres); results barely move 0-0.3
    drop_start=0.0015,      # per-frame chance a player's track starts a gap (~7% of frames missing)
    drop_len=50,            # mean gap length in frames
    swap_p=0.025,           # per 5-frame sample, per opposite-team pair closer than swap_dist (calibrate.py)
    swap_dist=1.5,          # metres
    swap_len=500,           # mean frames a swapped identity lasts before the track is re-acquired
    ball_mode="native",     # 'native' keeps the source's own detection gaps; 'markov' simulates them
    ball_visible=0.593,     # target for 'markov' (DATA.md: graded frames)
    ball_gap_mean=60,       # mean ball gap in frames for 'markov'
)


def _ball_markov(n, rng, visible, gap_mean):
    """Two-state chain: geometric gaps (heavy enough tail via mixture) and visible runs."""
    on_mean = gap_mean * visible / (1 - visible)
    out = np.empty(n, dtype=bool)
    i, state = 0, rng.random() < visible
    while i < n:
        L = int(rng.exponential(on_mean if state else gap_mean)) + 1
        if not state and rng.random() < 0.15:
            L *= 4                                  # the occasional long occlusion
        out[i:i + L] = state
        i += L
        state = not state
    return out


def degrade(half, rng, **kw):
    p = {**DEFAULTS, **kw}
    pos = half["pos"].copy()
    ball = half["ball"].copy()
    n, P, _ = pos.shape

    pos += rng.normal(0, p["noise_m"], pos.shape)
    ball += rng.normal(0, p["noise_m"], ball.shape)

    # player gaps
    start = rng.random((n, P)) < p["drop_start"]
    length = rng.geometric(1.0 / p["drop_len"], (n, P))
    miss = np.zeros((n, P), dtype=bool)
    for f, j in np.argwhere(start):
        miss[f:f + length[f, j], j] = True
    pos[miss] = np.nan

    if p["ball_mode"] == "markov":
        ball[~_ball_markov(n, rng, p["ball_visible"], p["ball_gap_mean"])] = np.nan

    # identity switches: an opposite-team pair in close contact may swap tracks for good
    team = half["team"]
    hc, ac = np.where(team == 0)[0], np.where(team == 1)[0]
    sub = pos[::5]
    xh, xa = sub[:, hc, 0], sub[:, ac, 0]
    yh, ya = sub[:, hc, 1], sub[:, ac, 1]
    d = np.hypot(xh[:, :, None] - xa[:, None, :], yh[:, :, None] - ya[:, None, :])
    cand = np.argwhere(np.nan_to_num(d, nan=np.inf) < p["swap_dist"])
    cand = cand[rng.random(len(cand)) < p["swap_p"]]
    delta = np.zeros((n + 5, P), dtype=np.int8)
    dur = rng.exponential(p["swap_len"], len(cand)).astype(int) + 5
    for (s, h, a), L in zip(cand, dur):
        e = min(s * 5 + L, n)
        for f in (s * 5, e):
            delta[f, hc[h]] ^= 1
            delta[f, ac[a]] ^= 1
    slot_team = (team[None, :] ^ np.bitwise_xor.accumulate(delta[:n], axis=0)).astype(np.int8)
    return dict(pos=pos, ball=ball, slot_team=slot_team, label=half["label"])


def nearest(d):
    """Team of the nearest tracked player to the ball, and how far (m). -1 / inf where unknown."""
    dist = np.hypot(d["pos"][:, :, 0] - d["ball"][:, None, 0], d["pos"][:, :, 1] - d["ball"][:, None, 1])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    j = np.argmin(dist, axis=1)
    best = dist[np.arange(len(j)), j]
    team = np.where(np.isfinite(best), d["slot_team"][np.arange(len(j)), j], -1).astype(np.int8)
    return team, best.astype(np.float32)


def to_static(d, team):
    """Express identity swaps the way the real ATD does: slot -> team is FIXED (metadata), and a swap makes the two
    tracks exchange the people they follow (a position jump of <= swap_dist metres). degrade() models a swap by flipping
    both labels; here the two positions are exchanged instead. Returns pos (n,P,2); slot_team is just `team` for every frame."""
    pos = d["pos"].copy(); st = d["slot_team"]
    ho, ao = np.where(team == 0)[0], np.where(team == 1)[0]
    Fh, Fa = (st[:, ho] != 0), (st[:, ao] != 1)
    oh = np.argsort(~Fh, axis=1, kind="stable"); oa = np.argsort(~Fa, axis=1, kind="stable"); kk = np.minimum(Fh.sum(1), Fa.sum(1))
    for r in range(int(kk.max()) if len(kk) else 0):
        rows = np.where(kk > r)[0]; hs, as_ = ho[oh[rows, r]], ao[oa[rows, r]]
        tmp = pos[rows, hs].copy(); pos[rows, hs] = pos[rows, as_]; pos[rows, as_] = tmp
    return pos
