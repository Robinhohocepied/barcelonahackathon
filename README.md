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
Tested with Python 3.10. Runtime is a few seconds.

## Layout

| path | what |
|---|---|
| `pipeline/predict.py` | the submission command: ATD in, `frame,label` CSV out |
| `pipeline/core.py` | ATD reader, features, 3-state Viterbi decoder |
| `pipeline/model_v2.pkl` | the models used for the submission (v1 models + v2 second stage + decoder settings) |
| `pipeline/stack.py` | v2 second stage: ball-in-play evidence + recalibration |
| `metrica_open.py`, `fetch_metrica.sh` | Metrica open sample games: loader and labels from events |
| `ext_eval.py`, `ext_v2.py`, `ext_v3.py`, `train_v2.py` | external validation and fitting of v2 (results in `results/`) |
| `pipeline/train.py` | retraining and leave-one-match-out validation |
| `degrade.py` | turns clean external tracking into ATD-like tracking (gaps, missing ball, track swaps) |
| `idsse.py`, `export_idsse.py`, `skillcorner.py`, `fetch_data.sh` | external data loaders |
| `pipeline/make_test_file.py` | writes an external match in the exact challenge format, for end-to-end tests |

Retraining (optional, needs `requirements-train.txt` and the external data): see `pipeline/README.md`.
