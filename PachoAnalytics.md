# PachoAnalytics: technical document

More Than A Hack 2027, Initial Challenge (team possession). Code: https://github.com/Robinhohocepied/barcelonahackathon

## 1. Approach

A learned model, trained only on other matches, that reads the tactical tracking and outputs H / A / D for every frame.

1. **Features, every 0.2 s.** Team shape for both teams (centroid, width, depth, compactness), nearest-teammate and nearest-opponent distances, how these change over ±2 s and ±6 s, player speed, and ball features (which team is nearest the ball, ball visibility).
2. **Two gradient-boosted classifiers** (scikit-learn `HistGradientBoostingClassifier`):
   - **Who has the ball** (H vs A): uses all features.
   - **Is the ball dead** (D): uses **no ball features**. A version that also used the ball did better on one data source but did not transfer to the other, so we dropped it.
3. **3-state Viterbi decoding** turns the frame probabilities into clean possession spells: minimum spell 1 s for H/A and 6 s for D, switch cost 1 between H and A and 3 into or out of D.

**Why this approach.** Hand-written rules ("nearest player to the ball") break when the ball is missing, and here it is missing for 43% of frames. Team shape reveals who is attacking even without the ball. Training on other matches is also the only way to learn this without labels on the provided game.

Frames outside the two graded halves (as published in DATA.md) are written as X.

## 2. Assets used

| Asset | Used? |
|---|---|
| Tactical tracking (ATD) + metadata | **Yes**: the only input |
| Smart Tagging | No (not tried in time) |
| Homography, tactical video | No |
| Broadcast tracking / video | No (not synchronised; no added value for this approach) |

Tried and dropped: a v1 rule (nearest player + smoothing) and a dead-ball detector that used the ball track. In leave-one-match-out validation on IDSSE, the learned dead-ball detector with Viterbi decoding gave spellF1 0.45, against 0.30 for the v1 rule.

## 3. Training data

None of it comes from the provided game.

| Dataset | Licence | Labelled by | Amount |
|---|---|---|---|
| IDSSE (DFL / Sportec Solutions, Bassek et al. 2025, doi 10.1038/s41597-025-04505-y) | CC-BY 4.0 | DFL / Sportec: official ball status (alive/dead) and ball-owning team per frame | 7 full Bundesliga matches |
| SkillCorner open data | MIT | SkillCorner: `player_possession` events | 2 full matches |

**Degradation simulator.** The external tracking is cleaner than ours, so before training we make it look like the challenge's ATD: position noise, player track gaps, a missing ball (about 59% visible, the figure in DATA.md), and identity swaps between nearby opponents in the same fixed-slot form the ATD uses. All settings come from published figures in DATA.md, not from the provided match.

## 4. Possession rules implemented

The labels follow the sources, which match CHALLENGE.md:

- **Loose balls and passes in flight** stay with the last team in possession until the other team takes the ball.
- **Dead ball (D)**: IDSSE uses the provider's own dead-ball flag. SkillCorner has no such flag, so any gap of more than 8 s between possessions counts as a stoppage, labelled D from 2 s after the last possession until play restarts.
- **Restarts** (throw-ins, free kicks, goal kicks, corners) are D until the ball is played, then belong to the team taking them.
- **Challenges**: no special rule. Possession switches when the model is confident the other team has the ball, and the decoder ignores flips shorter than 1 s.

## 5. Results and limitations

| Check | spellF1 | macroF1 |
|---|---|---|
| Leave-one-match-out, IDSSE (degraded) | 0.388 | 0.813 |
| Leave-one-match-out, SkillCorner (degraded) | 0.547 | 0.853 |
| End-to-end test: a held-out IDSSE match written in the exact challenge file format | 0.435 | 0.818 |
| Public 5 minutes, scored once as a pipeline check (no tuning) | 0.400 | 0.828 |

**Limitations**

- **It merges possessions.** In the public 5 minutes it found 7 possessions where there were 17, each about twice as long. In the external test it found 247 against 370. This is the main weakness.
- **It predicts too much dead ball.** On the full match, D is about 61% of graded time, where 40–45% is typical. The dead-ball detector may not fully transfer to this tracking system.
- **Little training variety:** 9 matches from 2 sources. More sources would help most.
- **With more time:** tune the decoder's switch costs on external data, add more external matches, and try Smart Tagging and the video as extra inputs.

## 6. Reproduction

```bash
git clone https://github.com/Robinhohocepied/barcelonahackathon && cd barcelonahackathon
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # numpy 2.2.6, scikit-learn 1.7.2, pandas 2.3.3 (Python 3.10)
# put the provided bundle's tactical/ folder in data/
./reproduce.sh
```

`reproduce.sh` runs `pipeline/predict.py` with the committed model and checks the result against the SHA-256 of the submitted `PachoAnalytics.csv`. It prints `IDENTICAL` on success. This was verified from a fresh download of the public repository. Runtime is a few seconds.

Retraining from scratch (optional): `pipeline/README.md`.
