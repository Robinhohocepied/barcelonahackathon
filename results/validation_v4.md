# v4 validation (leave-one-Metrica-game-out; decoder chosen on the OTHER games' held-out outputs)

| held-out game | v2 | v4 (TCN only) | v4 submitted (TCN + v2, w = 0.5) |
|---|---|---|---|
| 1 | 0.490 | 0.557 | 0.565 |
| 2 | 0.536 | 0.526 | 0.549 |
| 3 | 0.559 | 0.600 | 0.607 |
| mean spellF1 | 0.528 | 0.561 | **0.574** |

Reproduce: `research/fetch_metrica.sh && bash research/fetch_skillcorner.sh`, then `python research/ext_prep4.py <src> <game>` for every game,
`python research/train_v4.py <1|2|3> 25`, `python research/ext_v4eval.py`, `python research/ext_prep3.py metrica <g>`, `python research/ext_v4ens.py`.
Final model: `python research/train_v4.py all 25` (needs torch; inference does not).
Note: v3 (pipeline/v3.py, research/ext_v3eval.py) was evaluated and not shipped (+0.016 spellF1).
