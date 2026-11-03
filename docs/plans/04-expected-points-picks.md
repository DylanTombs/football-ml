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

- [ ] `predict/picks.py`: expected points of a pick, plausibility floor, best-pick search, with tests (hand-checked matrices, floor respected, falls back to modal when nothing beats it)
- [ ] Repeat-pick maths and fixed minutes, with tests
- [ ] `--pick-strategy` on `predict` and `backtest`
- [ ] `naive-favourite` predictor (odds from the cached football-data CSVs, pre-closing columns only)
- [ ] Paired-bootstrap comparison in the backtest output
- [ ] `snapshots.py` plus writing a snapshot on every `predict`, tested by a round-trip
- [ ] Backtest GW1–9: modal vs expected-points vs naive-favourite, with results recorded here

## Done when

- Expected-points picks are the default only if they beat modal picks on the backtest. The plausibility floor holds on every predicted scoreline.
- A snapshot is committed for every gameweek predicted from now on.
