# 01 — Results tracker + backtest

**Goal:** after every gameweek, score each prediction against what actually happened and keep a running season table. Run a backtest over past gameweeks to get a baseline.

## Decisions

- **Actual results:**
  - FPL fixtures give the final score and each scorer's FPL player id. Player ids avoid name-matching problems.
  - ESPN gives goal minutes, matched to fixtures by date and team.
  - Add an `ESPN_TO_CANONICAL_TEAM` mapping to `config.py`.
- **Prediction JSON changes:**
  - Add `scorer_id` next to `scorer`.
  - Add a top-level `predictor` field (`"dixon-coles"`), ready for phase 3.
- **Points** (constants in `config.py`):
  - Exact score: 3. Correct result only: 1.
  - +1 for each predicted scorer who actually scores (matched as a multiset, so each actual goal can only be matched once).
  - +1 for each matched scorer whose predicted minute is within ±10 of the actual minute.
- **Outputs:**
  - `out/results/gwNN.json`: per-match breakdown.
  - `out/results/season.json`: running totals per predictor.
- **Backtest:**
  - `football-ml backtest --from 1 --to N` re-predicts each past gameweek with the model trained only on results before that gameweek.
  - The FPL API only gives season-to-date player stats, so the scorer part of the backtest uses future information. Mark it `"scorer_leak": true` in the output until weekly snapshots exist (phase 4).

- **Added while building:**
  - Championship (E1) results are now in the training data, so newly promoted teams get a rating; GW1 couldn't be predicted without it.
  - ESPN credits an own goal to the team that benefits from it.
  - Scorer names are matched loosely (accents, ß, compound or full legal names), which is safe because we only compare against one team's scorers in one match.

## Tasks

- [x] `sources/results.py`: fetch and parse actual results (FPL + ESPN), with tests on fixture payloads
- [x] `evaluate/points.py`: pure scoring function, with tests for every points rule and edge case (0-0, own goals, scorer not predicted)
- [x] Add `scorer_id` and `predictor` to the prediction JSON
- [x] `football-ml score --gameweek N` writes the gameweek and season results
- [x] `football-ml backtest --from --to`
- [x] Run the backtest for GW1–9 and record the baseline in this file

## Baseline (GW1–9, 90 matches, run 2026-11-03)

| Points | Exact scores | Correct results | Scorer hits* | Minute hits* |
|---|---|---|---|---|
| **141** | 7 (8%) | 45 (50%) | 67 | 15 |

\*Optimistic, because player stats come from the live FPL feed. 243/243 actual goals were matched to a minute.

## Done when

- GW10 can be scored automatically once it finishes
- The GW1–9 baseline points are written down here

> Superseded as the reference: score-model changes are now judged on plan 06's multi-season bench, and the GW1–10 leak-free backtest baseline is in plan 05/06.
