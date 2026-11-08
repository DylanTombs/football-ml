# 06 — Multi-season test bench + bookmaker odds

**Goal:** judge changes to the score model on five seasons instead of 90 matches, then blend bookmaker odds into λ and μ, which is the strongest information we can get.

## Decisions

- **Walk-forward over past seasons:** refit before every gameweek using only earlier matches. Tune on 2021-22 to 2024-25 (1,520 matches). Hold out 2025-26 and test on it once. Add 2020-21 so the first tuning season has a warm-up year.
- **Metric for tuning:** out-of-sample log-likelihood of the exact score (log loss). Report 1X2 log loss, RPS and outcome points alongside. Points are too noisy to tune on: even near-identical variants need about a 16-point gap over 90 matches to be told apart.
- **Choosing settings:** a small grid (time decay, ridge penalty). Pick the simplest setting within one standard error of the best.
- **Odds to rates:**
  - remove the bookmaker margin with Shin's method
  - fit λ and μ to the 1X2 and over/under 2.5 probabilities on a Dixon-Coles grid (as `penaltyblog`'s `goal_expectancy` does)
  - blend: `λ = w × model + (1 − w) × market`, with `w` tuned on the walk-forward. Expect a small weight on the model.
- **Odds data:**
  - Use pre-closing odds only (columns without the `C` suffix), never closing odds. They're collected on Friday afternoon (by about 17:00 UK) for weekend games and on Tuesday for midweek games.
  - The live source is football-data's `fixtures.csv`. It's overwritten every round, so save a copy with each frozen prediction.
  - If a match has no odds, fall back to the model alone.
- **Paired comparison against the plan 01 / 04 baseline** for every change, as the CLAUDE.md rule requires.

## Tasks

- [x] Add 2020-21 to the training data; walk-forward runner over past seasons (score part only)
- [x] Log loss, RPS and calibration output (predicted vs actual draw rate and total goals)
- [x] Tune time decay and the ridge penalty, with results recorded here
- [x] `sources/odds.py`: pre-closing odds from season CSVs and `fixtures.csv`, Shin margin removal, with tests
- [x] Odds → λ, μ; blend weight tuned on the walk-forward
- [x] Save `fixtures.csv` odds alongside each gameweek's frozen prediction

## Results (run 2026-11-08)

**Bench baseline** (current settings xi 0.0019, l2 0.5; 2021-22 to 2024-25, 1,520 PL matches, weekly refits): exact-score log loss **3.0090**, 1X2 log loss 0.9693, RPS 0.2000, modal outcome points 1,123; goals 4,338 predicted vs 4,516 actual (−4%), draws 348 vs 350. `football-ml bench` reproduces it in about 3 s.

**xi / l2 grid** (`bench --grid`, `out/bench/grid-tune.json`): best xi 0.003, l2 0.1 (exact 3.0051, RPS 0.1987, 1,139 pts). The current setting was not within one SE. Below l2 0.1 the gain is under 0.0004.

**Odds blend** on the tuning seasons, xi 0.003 / l2 0.1, all 1,520 matches with odds:

| Model weight w | Exact log loss | 1X2 log loss | RPS | Outcome pts |
|---|---|---|---|---|
| 0.0 (market rates) | **2.9684** | **0.9454** | **0.1923** | 1,170 |
| 0.1 | 2.9692 | 0.9464 | 0.1926 | 1,173 |
| 0.5 | 2.9785 | 0.9524 | 0.1945 | 1,165 |
| 1.0 (model) | 3.0051 | 0.9656 | 0.1987 | 1,139 |

Loss rises steadily with w; only w = 0 is within one SE of the best.

**2025-26 holdout** (380 matches, run once):

| | Exact log loss | vs old settings | 1X2 log loss | RPS | Outcome pts |
|---|---|---|---|---|---|
| old settings (xi 0.0019, l2 0.5) | 2.8964 | – | 1.0395 | 0.2131 | 247 |
| tuned (xi 0.003, l2 0.1) | 2.9044 | +0.0081 ± 0.0036 | 1.0394 | 0.2129 | 236 |
| tuned + market (w = 0) | **2.8696** | **−0.0268 ± 0.0108** | **1.0164** | **0.2056** | **274** |

**Current season** (`backtest` GW1–10, default predictor, odds blend vs plan 05's model-only run, paired): **+11 pts** (174 vs 163) [+0.01, +0.21 per match], outcome +8, scorers +2, minutes +1; 35 scorelines changed. Modal: −2 [−0.23, +0.15].

- **Odds removed entirely (Dylan, 2026-11-08).** The market rates forecast better (tables above), but they would make the predictions the bookmakers' instead of a statistical model's, which defeats the series. All odds code (Shin, market rates, `fixtures.csv`, `naive-favourite`) was deleted; the numbers stay here as a record. Improvements come from the model and from Claude's adjustments (plans 07-08).
- **xi / l2 stay at 0.0019 / 0.5.** The tuned values won on 2021-25 but lost on the holdout (2.2 SE worse, −11 points), so they didn't generalise.
- **GW1–10 backtest, model only** (after this plan): expected-points + chance scorers 164, weights 170, modal 127 (plan 05: 163 / 168 / 126; the small shift is the 2020-21 warm-up season).
- **Faster fitting:** the fit now uses an exact gradient (0.036 s instead of 2 s, same optimum, checked by test). 2020-21 is a warm-up season (about 1% weight live).
- Shin de-margining on both 1X2 and over/under; (λ, μ) solved by Nelder-Mead on the squared error of H/D/A/over 2.5 with the fitted ρ. Matches without over/under odds fit the 1X2 alone.
- Calibration left for plan 07: the model under-predicts goals by 3-4% on the bench, and 2025-26 had more draws (104) than any setting predicted (89-93).

## Done when

- The model's settings are chosen on the multi-season walk-forward. ~~The odds blend is on by default only if it beats the model alone on held-out 2025-26 log loss without losing outcome points.~~ Odds removed (2026-11-08).
