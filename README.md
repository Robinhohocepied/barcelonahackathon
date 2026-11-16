# PachoAnalytics: More Than A Hack 2027, Initial Challenge (possession)

Per-frame team possession (H / A / D / X) for FC Barcelona vs Athletic Club, from tracking data only.
A learned model trained on **external data only** (IDSSE / DFL, CC-BY 4.0; SkillCorner open data, MIT; Metrica Sports open sample data).
Nothing was trained, fitted or tuned on the provided match or on the public ground truth.

## Reproduce the submission

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pinned: numpy 2.2.6, scikit-learn 1.7.2, pandas 2.3.3
# put the provided bundle's tactical/ folder in data/ (not included: the data is gated)
./reproduce.sh
```

`reproduce.sh` runs `pipeline/predict.py` with the committed model and checks the output against the
SHA-256 of the submitted file in `expected/SHA256SUMS`. It prints `IDENTICAL` on success.
Verified IDENTICAL from a clean clone on Python 3.10 (Linux arm64) and Python 3.13 (Linux x86_64). Runtime is a few seconds.

## Layout

| path | what |
|---|---|
| `pipeline/predict.py` | the submission command: ATD in, `frame,label` CSV out |
| `pipeline/core.py` | ATD reader, features, 3-state Viterbi decoder |
| `pipeline/model_v41.pkl` | **the submitted models (v4.1)**: v4 with the positive-only second stage (`pipeline/stack5.py`) |
| `ext_prep5.py`, `train_v5.py`, `ext_v5eval.py`, `build_model_v41.py` | robustness validation (normal + extra ball loss) and v4.1 assembly; results in `results/validation_v5_robustness.log` |
| `pipeline/model_v4.pkl` | previous version v4: v1 models, v2 second stage, the v4 sequence model (TCN weights) and decoder settings |
| `pipeline/tcn.py`, `pipeline/frames.py`, `pipeline/control.py` | v4 sequence model (numpy inference), its per-frame inputs, ball-control signal |
| `ext_prep4.py`, `train_v4.py`, `ext_v4eval.py`, `ext_v4ens.py`, `fetch_skillcorner.sh` | v4 data prep, training (torch) and validation (`results/validation_v4.md`) |
| `pipeline/model_v2.pkl` | previous version (v2), kept for comparison |
| `pipeline/stack.py` | v2 second stage: ball-in-play evidence + recalibration |
| `metrica_open.py`, `fetch_metrica.sh` | Metrica open sample games: loader and labels from events |
| `ext_eval.py`, `ext_v2.py`, `ext_v3.py`, `train_v2.py` | external validation and fitting of v2 (results in `results/`) |
| `pipeline/train.py` | retraining and leave-one-match-out validation |
| `degrade.py` | turns clean external tracking into ATD-like tracking (gaps, missing ball, track swaps) |
| `idsse.py`, `export_idsse.py`, `skillcorner.py`, `fetch_data.sh` | external data loaders |
| `pipeline/make_test_file.py` | writes an external match in the exact challenge format, for end-to-end tests |

Retraining (optional, needs `requirements-train.txt` and the external data): see `pipeline/README.md`.
