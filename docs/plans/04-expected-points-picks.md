# 04 — Expected-points picks + player snapshots

**Goal:** choose each match's scoreline, scorers and minutes to score the most expected points under our rules, while keeping predictions sensible enough to show in a reel. Start saving FPL player data every gameweek, because it can't be recovered later.

Why this comes first: the model's expected goals over GW1–9 total 256, against 243 actual, so the rates are fine. But the headline scorelines (likeliest result, then likeliest score) total only 188 goals. Each missing goal is a scorer pick we never made, and a wrong pick costs nothing. See `reports/Improving our match prediction model.md`.

## Decisions

- **Rules stay as they are (decided 2026-11-03).** No penalty for wrong scorers. Instead the pick has to be *plausible*.
- **Plausibility floor:** only scorelines with model probability ≥ `MIN_SCORELINE_PROBABILITY` (start at 0.05) can be picked. Without it, the best pick is about 6-5 every match. Expect picks like 2-1, 3-1 and 2-2 rather than 1-0.
- **Expected points of a pick (a, b):**
  - outcome: `P(result of a-b) + 2 × P(a, b)`, since an exact score is 3 and the correct result alone is 1
  - plus scorers: for each pick, `P(that pick is matched) × (1 + P(minute within ±10))`
  - Search every plausible (a, b), compute the total, and keep the best. If it scores no better than the current rule, keep the current rule.
- **Scorers stay the existing xG + goals weights in this plan** (plan 05 replaces them), so the change can be measured on its own. Each player's chance of scoring is approximated from their share of team xG × the team's expected goals.
- **Repeat picks:** the k-th pick of the same player is worth `P(player scores ≥ k)`. Choose greedily on that instead of the `SCORER_REPEAT_DECAY` rule.
- **Minutes:** use fixed best minutes on the elapsed scale (one goal: about 50–58'; the same player twice: non-overlapping windows such as 39' and 61'). Tune them on past seasons' ESPN goal times rather than the 90 backtest matches.
- **Switch:** `--pick-strategy modal|expected-points`. The old rule stays available for comparison.
- **Benchmark predictor:** `naive-favourite` (bookmaker favourite wins 1-0 or 2-1, never a draw). Over the last five seasons it scored 0.70–0.83 outcome points per match, against our 0.66 on GW1–9.
- **Judging results:** 90 matches means about ±13 points of noise. Report the per-match points difference with a paired bootstrap over gameweeks, and report outcome points (exact + result) separately from scorer and minute points, since scorer points use live FPL data and are optimistic.
- **Snapshots:** `predict` writes `data/snapshots/gwNN.json` before the deadline: id, team, xG, goals, minutes, starts, status, chance of playing, news and penalty order. These files are committed.

## Tasks

- [x] `predict/picks.py`: expected points of a pick, plausibility floor, best-pick search, with tests (hand-checked matrices, floor respected, falls back to modal when nothing beats it)
- [x] Repeat-pick maths and fixed minutes, with tests
- [x] `--pick-strategy` on `predict` and `backtest`
- [x] `naive-favourite` predictor (odds from the cached football-data CSVs, pre-closing columns only)
- [x] Paired-bootstrap comparison in the backtest output
- [x] `snapshots.py` plus writing a snapshot on every `predict`, tested by a round-trip
- [x] Backtest GW1–9: modal vs expected-points vs naive-favourite, with results recorded here

## Results (GW1–9, 90 matches, run 2026-11-03, availability `lagged`)

| Predictor | Points | Exact | Results | Scorer hits* | Minute hits* | Goals picked |
|---|---|---|---|---|---|---|
| dixon-coles-modal | 141 | 7 | 45 | 67 | 15 | 188 |
| **dixon-coles-expected-points** | **180** | 7 | 45 | 93 | 28 | 341 |
| naive-favourite | 156 | 6 | 43 | 77 | 24 | 242 |

Paired block bootstrap over gameweeks, per match vs modal (95% interval):

| | Total | Outcome | Scorers* | Minutes* |
|---|---|---|---|---|
| expected-points | **+0.43** [+0.14, +0.73] | 0.00 [−0.22, +0.21] | +0.29 [+0.23, +0.36] | +0.14 [+0.06, +0.24] |
| naive-favourite | +0.17 [0.00, +0.33] | −0.04 [−0.20, +0.10] | +0.11 [+0.08, +0.14] | +0.10 [0.00, +0.19] |

\*Still optimistic (live FPL stats). Expected-points picks more goals, so it gains more from the leak than modal does; the outcome column has no leak.

- **Expected-points is now the default** (`DEFAULT_PICK_STRATEGY`): +39 pts with an interval clear of zero. Modal reproduces the 141 baseline exactly.
- The gain is all scorers and minutes; outcome points are unchanged (87/90 scorelines changed, same 45 correct results). Picks are mostly 3-1 (33), 2-2 (20), 1-3, 2-1, 1-2; the floor held (lowest pick 5.0%).
- Picks total 341 goals against 243 actual (3.8 a match). The 5% floor is what keeps it there.
- Fixed minutes, tuned on 3,284 ESPN goals from 2023-24 to 2025-26: one pick 56' (26% within ±10, against about 21% for a random draw), two picks 56' + 86', three 41' + 64' + 86'. Goals by different players are nudged one minute apart, so a 3-1 reads 55', 56', 57', 58'.
- Player goals are Poisson with mean = share of the eligible squad's scoring weight × the side's expected goals. There's no own-goal or "other" bucket yet (plan 05).
- naive-favourite: 2-1 when the de-margined Avg>2.5 price says over is more likely, else 1-0. It only works in the backtest, because the current-season CSV has no odds for unplayed matches (live odds come in plan 06). Its outcome points (0.61 a match) trail modal here, unlike the 0.70–0.83 seen over five past seasons.
- LLM predictors still fall back to the modal pick (`ctx.baseline`).
- Backtest output is now named per strategy (`dixon-coles-modal.json`, `dixon-coles-expected-points.json`, `naive-favourite.json`) plus `out/backtest/comparisons.json`.
- First snapshot: `data/snapshots/gw10.json`. `predict` refreshes it on every run before the deadline and never writes one after.

## Done when

- Expected-points picks are the default only if they beat modal picks on the backtest. The plausibility floor holds on every predicted scoreline.
- A snapshot is committed for every gameweek predicted from now on.
