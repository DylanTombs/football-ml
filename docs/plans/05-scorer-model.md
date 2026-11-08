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

- [x] Fetch and cache per-player history, keeping only fixtures that kicked off before the gameweek
- [x] `predict/scoring_chance.py`: shrunk npxG share, expected minutes, penalties, with tests
- [x] Feed the probabilities into plan 04's picks and repeat-pick maths
- [x] Backtest GW1–N without leakage: old weights vs new model, with scorer hits per pick recorded here

## Results (GW1–10, 100 matches, run 2026-11-08, availability `lagged`, leak-free)

The backtest now rebuilds every player's stats from `element-summary` rows before each gameweek and infers penalty takers from past penalties, for **both** scorer models (`scorer_leak: false`). These numbers replace plan 04's leaky ones as the baseline.

| Predictor | Points | Exact | Results | Scorer hits | Minute hits |
|---|---|---|---|---|---|
| dixon-coles-modal | 126 | 9 | 51 | 48 | 9 |
| dixon-coles-expected-points (weights) | **168** | 6 | 55 | 75 | 26 |
| dixon-coles-expected-points-chance | 163 | 2 | 52 | **79** | **28** |
| naive-favourite | 140 | 8 | 49 | 56 | 19 |

Chance vs weights, paired bootstrap per match: total −0.05 [−0.20, +0.09], outcome −0.11 [−0.21, −0.02], scorers +0.04 [−0.04, +0.11], minutes +0.02 [−0.02, +0.06]. 18 scorelines changed.

Isolating the scorer model (same scorelines, the weights' expected-points picks, scorers from each model):

| Scorers from | Hits / picks | Per pick | Points |
|---|---|---|---|
| weights | 75 / 345 | 0.217 | 168 |
| chance | 81 / 345 | 0.235 | 176 |

Calibration over every squad player, GW2–10 (6,048 player-matches): Brier **0.0303** (chance) vs 0.0320 (weights). Weights are overconfident at the top: players given 30–45% scored 23% of the time (chance model: 36% vs 36%). Predicted scorers in total: 215 (chance) and 218 (weights) against 196 and 180 real.

Shrinkage prior (same-scoreline scorer hits): 0 min → 72, 450 → 81, 900 → 81, 1,350 → 81, 1,800 → 80. Kept 900 (10 matches): flat from 450 to 1,800, and shrinking clearly beats none.

- **Chance is now the default** (`DEFAULT_SCORER_MODEL`). It meets the done-when (79 vs 75 scorer hits, no leak), is clearly better calibrated, and picks better scorers on the same scorelines. Total points are a statistical tie.
- **Caveat:** fed into the scoreline search, its values changed 18 scorelines and lost 11 outcome points (mostly 4 fewer exact scores, the noisiest component). Re-check on the plan 06 bench; `--scorer-model weights` keeps the old behaviour.
- **Expected minutes** = mean minutes over the team's last 5 matches × chance of playing (not starts: it also credits regular subs). New signings with no rows get 0 until they play.
- **Shares are normalised** over the available squad, so an absent player's threat passes to whoever plays; λ itself is untouched (team-level injury effects are plan 08). 3% of λ is set aside for own goals.
- **Non-penalty xG** = FPL xG − 0.78 per penalty attempt (ESPN `penaltyKick` goals + FPL `penalties_missed`). Team penalty rate is shrunk towards 0.13 attempts a match with 20 matches of prior.
- **Penalty taker:** live, the first FPL `penalties_order` player expected to play; in the backtest, the player with most attempts so far. The term is scaled by his expected minutes / 90.
- **Not done:** a European-week rotation factor (no published size; needs data).
- GW1 has no history before kickoff, so the backtest makes no scorer picks there for either model. Live predictions always have history.
- `naive-favourite` stays on the weights as a fixed benchmark. The backtest fetches ~550 player histories on first run (about 2 minutes, cached 12 h).

## Done when

- The backtest no longer needs `scorer_leak`, and the new model gets at least as many scorer hits as the old weights on the same leak-free data.
