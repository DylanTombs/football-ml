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

## Tasks

- [ ] `predict/availability.py`: compute `missing_share`, with tests covering full squad, one star out, and a doubtful player at 50%
- [ ] Refactor the Dixon-Coles functions to expose `matrix_from_rates`, keeping the existing tests green
- [ ] Wire it into the pipeline and add `missing` to the JSON
- [ ] Backtest-sweep `INJURY_IMPACT` and record the results here

## Done when

- The adjustment is on by default, but only if the backtest shows it doesn't score fewer points than the unadjusted model
