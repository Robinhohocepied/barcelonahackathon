# PachoAnalytics: technical document

More Than A Hack 2027, Initial Challenge (team possession). Code: https://github.com/Robinhohocepied/barcelonahackathon

## 1. Approach

A learned model, trained only on other matches, that reads the tactical tracking and outputs H / A / D for every frame.

1. **Features, every 0.2 s.** Team shape for both teams (centroid, width, depth, compactness), nearest-teammate and nearest-opponent distances, how these change over ±2 s and ±6 s, player speed, and ball features (which team is nearest the ball, ball visibility).
2. **Two gradient-boosted classifiers** (scikit-learn `HistGradientBoostingClassifier`):
   - **Who has the ball** (H vs A): uses all features.
   - **Is the ball dead** (D): uses **no ball features**. A version that also used the ball did better on one data source but did not transfer to the other, so we dropped it.
3. **v2 second stage (`pipeline/stack.py`)**: two small logistic models recalibrate those probabilities using **ball-in-play evidence**: how often, in the last ±1 s and ±3 s, the ball is visible, inside the pitch and moving faster than 8 m/s. The evidence is positive-only: a fast ball on the pitch says "alive", a missing ball adds nothing. This fixes v1's main failure: it read "ball not seen + players walking" as a stoppage, so slow build-up play became dead ball.
4. **3-state Viterbi decoding** turns the probabilities into clean possession spells: minimum spell 3.2 s for H/A and 1 s for D, no extra cost for H/A switches and cost 1 into or out of D (chosen on external games).

**Why this approach.** Hand-written rules ("nearest player to the ball") break when the ball is missing, and here it is missing for 43% of frames. Team shape reveals who is attacking even without the ball. Training on other matches is also the only way to learn this without labels on the provided game.

Frames outside the two graded halves (as published in DATA.md) are written as X.

## 2. Assets used

| Asset | Used? |
|---|---|
| Tactical tracking (ATD) + metadata | **Yes**: the only input |
| Smart Tagging | No (not tried in time) |
| Homography, tactical video | No |
| Broadcast tracking / video | No (not synchronised; no added value for this approach) |

Tried and dropped: a rule (nearest player + smoothing), a dead-ball detector that used raw ball features (did not transfer across data sources), and label-shift EM (Saerens et al. 2002) on the dead-ball prior (no gain, prior estimates off by ~5 points).

## 3. Training data

None of it comes from the provided game.

| Dataset | Licence | Labelled by | Amount |
|---|---|---|---|
| IDSSE (DFL / Sportec Solutions, Bassek et al. 2025, doi 10.1038/s41597-025-04505-y) | CC-BY 4.0 | DFL / Sportec: official ball status (alive/dead) and ball-owning team per frame | 7 full Bundesliga matches |
| SkillCorner open data | MIT | SkillCorner: `player_possession` events | 2 full matches |
| Metrica Sports open sample data (github.com/metrica-sports/sample-data) | Public sample data, used with credit, not redistributed | Metrica: human-annotated events (passes, ball out, set pieces...) | 3 full anonymised matches: used **only** for the v2 second stage and the decoder settings, and as the validation set |

**Degradation simulator.** The external tracking is cleaner than ours, so before training we make it look like the challenge's ATD: position noise, player track gaps, a missing ball (about 59% visible, the figure in DATA.md), and identity swaps between nearby opponents in the same fixed-slot form the ATD uses. All settings come from published figures in DATA.md, not from the provided match.

## 4. Possession rules implemented

The labels follow the sources, which match CHALLENGE.md:

- **Loose balls and passes in flight** stay with the last team in possession until the other team takes the ball.
- **Dead ball (D)**: IDSSE uses the provider's own dead-ball flag. Metrica sample games: D from a stoppage event (ball out, foul, card, goal) until the next set piece starts. SkillCorner has no such flag, so any gap of more than 8 s between possessions counts as a stoppage, labelled D from 2 s after the last possession until play restarts.
- **Restarts** (throw-ins, free kicks, goal kicks, corners) are D until the ball is played, then belong to the team taking them.
- **Challenges**: no special rule. Possession switches when the model is confident the other team has the ball, and the decoder ignores flips shorter than 1 s.

## 5. Results and limitations

| Check | spellF1 | macroF1 |
|---|---|---|
| Leave-one-match-out, IDSSE (degraded) | 0.388 | 0.813 |
| Leave-one-match-out, SkillCorner (degraded) | 0.547 | 0.853 |
| End-to-end test: a held-out IDSSE match written in the exact challenge file format | 0.435 | 0.818 |
| Public 5 minutes, v1, scored once as a pipeline check | 0.400 | 0.828 |
| **v2, leave-one-game-out on the 3 Metrica games (v1: 0.450 / 0.758)** | **0.528** | **0.81** |
| v2, public 5 minutes, scored once as a pipeline check (decision was already made on external games) | 0.529 | 0.715 |

**Limitations**

- **Still too few possessions.** v2 finds 242 on the full match (v1: 184); external matches have ~340. Short possessions are still merged.
- **Dead ball:** v2 brings D from 61% to 40% of graded time on the full match, but on external games it still slightly over-predicts D (37% vs 31%).
- **Decoder settings sit at the edge of the tried range** and differ between folds, so they are not fully pinned down.
- **Little training variety:** 9 matches for the base models, 3 for the second stage.
- **With more time:** tune the decoder's switch costs on external data, add more external matches, and try Smart Tagging and the video as extra inputs.

## 6. Reproduction

```bash
git clone https://github.com/Robinhohocepied/barcelonahackathon && cd barcelonahackathon
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # numpy 2.2.6, scikit-learn 1.7.2, pandas 2.3.3 (Python 3.10)
# put the provided bundle's tactical/ folder in data/
./reproduce.sh
```

`reproduce.sh` runs `pipeline/predict.py` with the committed model (`model_v2.pkl`) and checks the result against the SHA-256 of the submitted `PachoAnalytics.csv`. It prints `IDENTICAL` on success. This was verified from a fresh download of the public repository. Runtime is a few seconds.

Refitting v2 (optional): `./fetch_metrica.sh && python ext_eval.py prep && python train_v2.py`. Validation: `python ext_v2.py`, `python ext_v3.py` (logs in `results/`). Base models: `pipeline/README.md`.
