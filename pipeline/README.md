# Possession pipeline (More Than A Hack 2027, Phase 1)

> This file describes the v1 base models. The submitted version (v4) adds the v2 second stage and the v4 sequence model on top: see the top-level README and PachoAnalytics.md. The decoder settings below are v1's, not the submitted ones.

    python pipeline/predict.py --atd-raw data/tactical/BAR-ATH_tactical_atd.txt \
                               --atd-meta data/tactical/BAR-ATH_tactical_atd.xml --out submission.csv

Input: Metrica ATD raw text + metadata XML (0-1 coordinates, 46 slots, NaN gaps). Output: `frame,label`, H/A/D on the two graded halves, X elsewhere.
Needs numpy, pandas (scorer only) and scikit-learn. About 1 minute for a full match.

**How it works** (all learned on external data only: IDSSE / DFL CC-BY 4.0, SkillCorner open data MIT, degraded to look like ATD)
1. Per 0.2 s: team-shape features (centroid, width, depth, compactness, nearest-teammate / nearest-opponent distances, their trends over +-2 s / +-6 s), player speed, and ball features (nearest-player vote, visibility).
2. Two gradient-boosted models: who has the ball (H vs A) and a dead-ball detector that is **ball-free** (the only variant that held up leave-one-source-out).
3. 3-state Viterbi: minimum spell 1 s (H, A) / 6 s (D), switch costs 1 (H<->A) and 3 (to/from D).

Not used: Smart Tagging, homography, video, the broadcast feed. Nothing here was trained, fitted or tuned on the provided match or the public ground truth.

**Retrain:** `python pipeline/train.py prep idsse idsse/idsse_<id>.npz` (per match, see export_idsse.py / fetch_data.sh), then `python pipeline/train.py cv` and `python pipeline/train.py fit --out pipeline/model.pkl`.
**Format test on external data:** `python pipeline/make_test_file.py --npz idsse/idsse_J03WR9.npz --out pipeline/testdata`, then predict with `--model pipeline/model_holdout_J03WR9.pkl --halves <printed ranges>`.
