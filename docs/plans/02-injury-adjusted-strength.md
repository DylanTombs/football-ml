# 02 — Injury-adjusted team strength

**Goal:** missing players lower their team's expected goals, not just drop off the scorer list.

## Decisions

- **Missing share** for each team: sum over players of `xG × (1 − availability)`, divided by the team's total xG. A player's availability is `chance_of_playing / 100`, or 0 if they're injured or suspended.
- **Adjusted expected goals:** `λ_adj = λ × (1 − INJURY_IMPACT × missing_share)`.
  - Start with `INJURY_IMPACT = 0.5`, because replacements still score some goals.
  - Tune it with the backtest: compare 0, 0.25, 0.5 and 0.75 and keep the best.
- **Refactor:** split `score_matrix` into `expected_goals → matrix_from_rates(λ, μ, ρ)` so the adjustment stays a pure step between the two.
- **v1 scope:** attack only. Missing defenders and keepers aren't counted yet, since xG only measures attacking output.
- **Prediction JSON:** list the missing players per team (`"missing": [{"name", "xg_share"}]`). This also gives a content angle: "City without Haaland".

- **Added while building:**
  - Players FPL marks `u` (left / loaned out) or `n` count as departed and aren't part of the team at all. `i`/`s` with no chance given count as 0.
  - FPL keeps no history of injury news, so the backtest previously applied *today's* injury list to every past gameweek. It now rebuilds availability from per-gameweek minutes (`event/N/live/`, cached for good) with `--availability`:
    - `lagged` (default): 0 minutes in the team's previous match after playing for them before. No leak.
    - `oracle`: 0 minutes in that gameweek itself. Uses the real line-up, so only an upper bound.
    - `live`: today's injury list (the old behaviour).
  - Historical availability only changes `availability`, not scorer eligibility, so scorer points stay comparable with the baseline.
  - `missing` is listed per side: `"missing": {"home": [...], "away": [...]}`. It's always written, even with the adjustment off.

## Tasks

- [x] `predict/availability.py`: compute `missing_share`, with tests covering full squad, one star out, and a doubtful player at 50%
- [x] Refactor the Dixon-Coles functions to expose `matrix_from_rates`, keeping the existing tests green
- [x] Wire it into the pipeline and add `missing` to the JSON
- [x] Backtest-sweep `INJURY_IMPACT` and record the results here

## Results (GW1–9, 90 matches, run 2026-11-03)

| Impact | lagged | oracle | live |
|---|---|---|---|
| **0 (baseline)** | **141** | **141** | **141** |
| 0.25 | 134 | 138 | 133 |
| 0.5 | 123 | 126 | 129 |
| 0.75 | 120 | 118 | 126 |

- **Decision: `INJURY_IMPACT = 0.0` (off).** Every value scores fewer points, even with the oracle's perfect line-up knowledge.
- Almost all of the loss is scorer and minute points. Results and exact scores barely move (oracle 0.25: 46 results vs 45; exact scores 6–7 throughout).
- Why: the *headline* scores add up to too few goals (188 predicted vs 243 actual over GW1–9). Lowering λ drops that to 155, so there are fewer scorer guesses, and a wrong scorer guess costs nothing. Any change that lowers headline goal counts loses points under these rules.
  - Correction (2026-11-03): the model's expected goals add up to 256, against 243 actual, so the rates aren't too low. The shortfall comes from picking each match's single most likely score, which favours low scores. The fix is choosing picks to maximise expected points, not calibrating the rates.
- Ratings already absorb long injuries partly, because recent results reflect the weaker side.
- Worth trying later: choosing scorelines and scorers to maximise expected points, or using injuries to shift *who wins* while keeping goal counts.

## Done when

- The adjustment is on by default, but only if the backtest shows it doesn't score fewer points than the unadjusted model. **It does score fewer, so it ships off; the plumbing and `missing` JSON are in place.**
