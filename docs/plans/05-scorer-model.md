# 05 — Scorer model

**Goal:** a proper chance-of-scoring for every player. It replaces the "0.7 × xG + 0.3 × goals" weights, so plan 04's expected-points picks work from real probabilities.

## Decisions

- **Per player, per match:**
  - expected goals `μ = team λ (non-penalty) × npxG share × expected minutes / 90`
  - plus a penalty term for the first-choice taker: `team penalty rate × 0.78`
  - chance of scoring = `1 − e^(−μ)`
- **Shrink small samples:** blend each player's npxG per 90 towards a position average, weighted by about 10–20 matches of minutes (tune by backtest). Over fewer than about 10 matches, xG per 90 is badly distorted (FPL Review).
- **Drop actual goals as a signal.** Finishing over xG barely carries over from season to season, so the 0.3 weight is mostly noise.
- **Expected minutes:** starts in the last 5 gameweeks, × `chance_of_playing / 100`, and 0 if injured or suspended. Leave a European-week rotation factor to tune; no published size exists.
- **Penalty takers:** use FPL `penalties_order` live. In backtests, infer the taker from past penalty events, because historical penalty order isn't available.
- **No leakage:** rebuild each player's stats before every gameweek from FPL `element-summary/{id}` history (per-fixture xG, minutes, starts, kickoff). Use plan 04 snapshots once they exist. This removes `scorer_leak` from the backtest.

## Tasks

- [ ] Fetch and cache per-player history, keeping only fixtures that kicked off before the gameweek
- [ ] `predict/scoring_chance.py`: shrunk npxG share, expected minutes, penalties, with tests
- [ ] Feed the probabilities into plan 04's picks and repeat-pick maths
- [ ] Backtest GW1–N without leakage: old weights vs new model, with scorer hits per pick recorded here

## Done when

- The backtest no longer needs `scorer_leak`, and the new model gets at least as many scorer hits as the old weights on the same leak-free data.
