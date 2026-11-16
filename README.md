# Who has the ball?

**Team PachoAnalytics** · Metrica Sports' [More Than A Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027), Initial Challenge

We had to say, for every frame of FC Barcelona vs Athletic Club, which team has the ball: home, away, or game stopped. The only input is tracking data: 22 players and a ball, seen as dots 25 times a second, with no video. And the cameras lose the ball 43% of the time.

## How we approached it

1. **Learn from matches where we already know the answer.** 16 external matches with known possession: 7 Bundesliga (DFL), 6 A-League (SkillCorner), 3 Metrica samples. Nothing was trained, fitted or tuned on the provided match or on its public ground truth.
2. **Make them messy on purpose.** Their tracking is cleaner than ours, so we add noise, gaps, identity swaps and a missing ball before training.
3. **Read the team shapes, then every split second.** A model on team shape (who pushes forward and spreads out), then a sequence model that looks at every frame.
4. **Clean up the answer.** The raw guesses flicker; a decoder turns them into realistic possessions.

## Six versions, one shipped

| Version | Score on unseen matches | With the ball hidden more | "Stopped" on the real match | Decision |
|---|---|---|---|---|
| v1 team shapes | 0.450 | | 61% | too much "stopped" |
| v2 + ball rule | 0.528 | 0.353 | 40% | learned "no ball = stopped" |
| v3 + turnover detector | 0.544 | | not run | gain too small |
| v4 + sequence model | **0.574** | 0.467 | 40% | best score, less robust |
| **v4.1 shipped** | 0.568 | **0.487** | **37%** | **most reliable** |
| v5 + Bundesliga training | other test | | 61% | rejected |

Score = possession-level F1 (spellF1), leave-one-game-out on 3 Metrica games. "Ball hidden more" re-scores the same games with only ~30% of the ball visible. Real matches with known answers spend 26–52% of the time stopped.

We picked the version that holds up on a bad day, not the one with the best average, and dropped a version that tested better (v5) because it failed a basic sanity check on the real match.

## What we did differently

- **No ball seen means "we don't know"**, not "game stopped" (v4.1 never uses ball visibility as evidence).
- **Selected on the worst case**, including extra ball loss, not on the mean.
- **Asked an outside critic to attack the work** and published the corrections in the technical document.
- **Every version had to make football sense on the real match** (share of stopped time, number of possessions, both halves alike), checked without labels.

## Limits

- **Very quick possessions** (under 3 s, 31% of all) are mostly missed: 18% found for 1–3 s, versus 76% over 8 s.
- **Stoppages with the ball still in view** (a throw-in being set up) are often read as play.
- **New tracking providers need calibration:** 0.37 on an unseen provider vs 0.57 on Metrica-style data.

Full method, numbers and corrections: [`PachoAnalytics.md`](PachoAnalytics.md).

## Reproduce the submitted file

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pinned: numpy 2.2.6, scikit-learn 1.7.2, pandas 2.3.3
# put the provided bundle's tactical/ folder in data/ (not included: the data is gated)
./reproduce.sh
```

`reproduce.sh` runs `pipeline/predict.py` with the committed model and compares the output with the SHA-256 of the submitted file in `expected/SHA256SUMS`. It prints `IDENTICAL` on success. Verified from a clean clone on Python 3.10 (Linux arm64) and Python 3.13 (Linux x86_64). Runtime: a few seconds.

## What's in the repo

| Folder | What |
|---|---|
| `pipeline/` | **The submission.** `predict.py` (ATD in, `frame,label` CSV out), `model_v41.pkl` (the submitted models), and the code it runs: features and decoder (`core.py`), sequence model (`tcn.py`, `frames.py`, `control.py`), second stage (`stack5.py`) |
| `research/` | How we got there: external data loaders and downloads, the degradation simulator, training and validation of every version, model assembly. See `pipeline/README.md` for retraining |
| `results/` | Validation logs behind every number above |
| `expected/` | SHA-256 of the submitted file, used by `reproduce.sh` |

## Data and credits

Training data: IDSSE (DFL / Sportec Solutions, CC-BY 4.0), SkillCorner open data (MIT), Metrica Sports open sample data. No match data is included in this repository.
