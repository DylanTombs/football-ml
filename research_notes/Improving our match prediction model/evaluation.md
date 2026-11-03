# Evaluating and tuning a football score-prediction model when the headline metric is noisy prediction-game points

Note on own computations: figures marked **[computed]** were calculated during this research from data already in the repo. Per-match points come from `out/backtest/results/gw01..gw09/dixon-coles.json` (the plan-01 baseline: 90 matches, 141 pts). Bookmaker figures come from the cached football-data.co.uk CSVs `data/cache/football_data_E0_2122..2627.csv` ([football-data.co.uk](https://football-data.co.uk/mmz4281/2425/E0.csv)). The scripts were throwaway and are not in the repo. Margins were removed by normalising 1/odds (proportional method). "PSC" means Pinnacle closing odds and "AvgC" means market-average closing odds.

## 1. How noisy are prediction-game points, and what difference can be detected at 90 / 380 / 1000+ matches?

### Takeaway
Points per match have an SD of about 1.36, so the 90-match total of 141 has a standard error of about ±13 points (±0.14 per match). At 90 matches, only very large differences between models can be detected, and a single season is still weak evidence. Compare variants on the same matches (paired), test the per-match differences with a paired bootstrap, permutation test or Diebold-Mariano test, and choose variants with a proper scoring rule on 1,500–2,000 past-season matches rather than with points.

### Cited findings
- **[computed]** Distribution of baseline per-match total points over 90 matches: 0 pts in 26 matches, 1 in 19, 2 in 25, 3 in 10, 4 in 8 and 5 in 2. Mean 1.567, SD 1.359. Components:

  | Component | Mean | SD |
  |---|---|---|
  | Outcome (0/1/3) | 0.656 | 0.832 |
  | Scorers | 0.744 | 0.783 |
  | Minutes | 0.167 | 0.373 |

  Outcome points were 0 in 45 matches, 1 in 38 and 3 in 7. Source: `out/backtest/results/*/dixon-coles.json` (local).
- **[computed]** Standard error of mean points per match (SD/√n), using SD 1.36:

  | Matches | SE per match | SE scaled to a 380-match season total |
  |---|---|---|
  | 90 | 0.143 | ±54 |
  | 380 | 0.070 | ±27 |
  | 1,000 | 0.043 | ±16 |
  | 1,900 | 0.031 | ±12 |

  For the score part alone (outcome points, SD 0.83), the SE is 0.088 at 90 matches, 0.043 at 380, 0.026 at 1,000 and 0.017 at 2,280.
- **[computed]** Minimum detectable difference (MDD) between two models at 80% power with a two-sided 5% test. The formula is MDD ≈ 2.8·SD_d/√n, where SD_d = 1.36·√(2(1−ρ)) and ρ is the per-match correlation of the two models' points. Each cell shows the difference per match, then the same difference as total points over n matches:

  | ρ | SD_d | 90 matches | 380 matches | 1,000 matches | 1,900 matches |
  |---|---|---|---|---|---|
  | 0 (unpaired / independent) | 1.92 | 0.57 (51 pts) | 0.28 (105 pts) | 0.17 (170 pts) | 0.12 (235 pts) |
  | 0.5 | 1.36 | 0.40 (36 pts) | 0.20 (74 pts) | 0.12 (120 pts) | 0.087 (166 pts) |
  | 0.8 | 0.86 | 0.25 (23 pts) | 0.12 (47 pts) | 0.076 (76 pts) | 0.055 (105 pts) |
  | 0.9 (near-identical variants) | 0.61 | 0.18 (16 pts) | 0.087 (33 pts) | 0.054 (54 pts) | 0.039 (74 pts) |

  This is a normal approximation. Points are discrete and lumpy, so a bootstrap is the safer test in practice.
- The Diebold-Mariano (1995) test compares two forecasts of the same series using the mean of their loss differential d_t = L(model A) − L(model B), standardised by a HAC (Newey-West) long-run variance. That standardisation is needed because the loss differential can be serially correlated — [statsmodels DM test docs](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.diebold_mariano_test.html); [macroforecast DM docs](https://macroforecast.readthedocs.io/en/stable/encyclopedia/l6/equal_predictive_test/dm_diebold_mariano.html)
- The original DM test can be badly over-sized (rejects too often) in small samples. The Harvey, Leybourne & Newbold (1997, IJF 13(2)) correction rescales the statistic and compares it against a Student-t with T−1 degrees of freedom. statsmodels exposes this as `harvey_adj` — [statsmodels](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.diebold_mariano_test.html); [Real Statistics DM page](https://real-statistics.com/diebold-mariano-test/)
- DM is described as invalid for nested models; the Clark-West test is the usual alternative there — [MetricGate DM notes](https://metricgate.com/docs/diebold-mariano-equal-accuracy/) (secondary source)
- Wheatcroft compares scoring rules by the proportion of forecast pairs in which each rule picks the "perfect" system as n grows. He judges significance with 95% resampling intervals on paired differences, which is a paired-resampling approach to choosing between forecast systems. For small n, the rules differ little — [Wheatcroft, arXiv 1908.08980 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1908.08980)

### Inferences
- A 141 vs 150 difference over 90 matches (0.1 per match) cannot be told apart from noise unless the two variants make almost the same predictions. Even then, the paired MDD is about 16 points.
- Be careful with variants that share most predictions. A high ρ makes the test sharper only because the variants differ on few matches, so the difference rests on a handful of fixtures. Report how many matches changed prediction alongside the points difference.
- Recommended paired test for points: compute d_i = pts_A − pts_B per match.
  - Bootstrap by gameweek blocks (resample the 38 gameweeks with replacement, or season-gameweeks), because results within a gameweek share conditions. Take the 95% CI of mean d.
  - Or run a sign-flip permutation test on d_i.
  - DM with the HLN correction is equivalent in spirit. Serial correlation across matches is probably weak, so HAC lags of 0–1 are enough.
- The scorer and minute components add variance (SDs 0.78 and 0.37) that has nothing to do with Dixon-Coles. Test score-model changes on outcome + exact points only, so scorer noise does not swamp them.
- Prediction-game points score a point forecast with a coarse 0/1/3 rule, so they discard most of the information in the probabilities. A proper scoring rule on the full probability forecast has lower relative noise. Section 2 gives the numbers: the bookmaker RPS SE at 380 matches is about 0.007 on a mean of 0.19.

### Gaps
- I did not find a published estimate of per-match point variance for 3/1-type prediction games. The figures above come from this project's own 90 matches.
- ρ between real variants is unknown. Compute it from the plan-02 injury variant's backtest, which is a direct way to calibrate the MDD table.

## 2. Proper scoring rules: RPS vs log loss (ignorance) vs Brier, and what to tune on

### Takeaway
RPS has been the football default since Constantinou & Fenton (2012). Wheatcroft (JQAS 2021) shows that the ignorance score (log loss) picks the better forecaster more reliably. Tune Dixon-Coles hyperparameters on out-of-sample log loss: the 1X2 log loss, and ideally the exact-scoreline log-likelihood, which is also what Dixon & Coles themselves used for the decay parameter. Report RPS alongside for comparability with the literature and bookmakers.

### Cited findings
- Constantinou & Fenton, "Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models", *JQAS* 8(1), 2012, DOI 10.1515/1559-0410.1418 — [RePEc](https://ideas.repec.org/a/bpj/jqsprt/v8y2012i1n12.html); [QMUL repository](https://qmro.qmul.ac.uk/xmlui/handle/123456789/10783)
  - They argue that only distance-sensitive rules should be used for H/D/A outcomes, which makes the RPS the natural choice. They illustrate this with five hypothetical matches — as summarised in [Wheatcroft, arXiv 1908.08980](https://arxiv.org/pdf/1908.08980)
- Wheatcroft, "Evaluating probabilistic forecasts of football matches: the case against the ranked probability score", *JQAS* 17(4):273–287, 2021, DOI 10.1515/jqas-2019-0089 — [RePEc](https://ideas.repec.org/a/bpj/jqsprt/v17y2021i4p273-287n1.html)
  - He argues that distance sensitivity "adds nothing in terms of the usual aims of using scoring rules".
  - In two simulation experiments, the ignorance score (local, strictly proper) beats both Brier and RPS at identifying the better forecast. This casts doubt on the value of non-locality and distance sensitivity.
  - Experiment two builds perfect and imperfect forecast pairs from bookmaker probabilities. At small perturbations (δ = 0.01 and 0.025) the ordering is Ignorance > Brier > RPS. At δ = 0.05 and 0.1, Brier and RPS are not significantly different, while Ignorance still wins — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/1908.08980)
- penaltyblog provides `rps_average`, `rps_array`, `ignorance_score` and `multiclass_brier_score`. Its docs describe the ignorance score as log loss: it is strictly proper and more sensitive to extreme probabilities than Brier. For example, p = 0.1 on the realised outcome scores about 2.3 and p = 0.9 scores about 0.1 — [penaltyblog ignorance](https://penaltyblog.readthedocs.io/en/master/metrics/ignorance.html); [penaltyblog RPS](https://penaltyblog.readthedocs.io/en/master/metrics/rps.html); [penaltyblog metrics index](https://penaltyblog.readthedocs.io/en/master/metrics/index.html)
- Dixon & Coles (1997) chose the time-decay ξ to maximise a predictive log-probability score on match outcomes (H/D/A), not the fit to scorelines. They found ξ = 0.0065 per half-week, a half-life of about one year. Choosing ξ is awkward because the weighted likelihoods for successive times are not independent — as described in [arXiv 2508.05891](https://arxiv.org/html/2508.05891v1); see also [penaltyblog models overview](https://penaltyblog.readthedocs.io/en/latest/models/overview.html), whose ξ is on a per-day scale (example ξ = 0.001).
- **[computed]** Noise in the metric itself, using Pinnacle closing odds on 1,730 EPL matches (2021-22 to 2025-26):
  - Per-match RPS: mean 0.1918, SD 0.135. SE is 0.0143 at 90 matches, 0.0069 at 380 and 0.0031 at 1,900.
  - Per-match log loss: SD 0.49. SE is 0.052 at 90 matches, 0.025 at 380 and 0.011 at 1,900.
  - Paired differences between two similar models will have a much smaller SD than these.

### Inferences
- **Tuning objective:** Use the mean negative log-likelihood of the realised exact scoreline under the model's score matrix. It is local and strictly proper for the full scoreline distribution. It directly rewards getting the exact-score and outcome probabilities right, and it is what Dixon-Coles maximises in-sample anyway.
- **Secondary metrics:** Report 1X2 log loss and RPS for comparison with bookmakers and the literature, plus a total-goals log loss or a Poisson deviance on home and away goals.
- **Turning probabilities into points:** Points are a decision layer on top of the probabilities. Because points are not a proper score of the probabilities, first make the probability model good (on log loss), then choose the scoreline that maximises expected points:

  E[pts | pick s] = 3·P(s) + 1·[P(result(s)) − P(s)]

  This is often not the modal scoreline. Check the expected-points pick against points in the backtest.
- With 1,500–2,000 out-of-sample matches, a paired log-loss difference of about 0.005–0.01 per match should be detectable. That is far finer than points allow.

### Gaps
- I found no paper that directly tests exact-scoreline log-likelihood as a model-selection metric against RPS. The recommendation above is reasoning from the Wheatcroft locality argument.
- I could not access the full Constantinou & Fenton 2012 text, so its content is cited via summaries.
- The search for opisthokonta (Jonas Lindstrøm) blog posts on choosing ξ returned no results.

## 3. Recommended protocol: walk-forward backtesting, hyperparameter tuning without overfitting, holdout seasons

### Takeaway
Use rolling-origin (walk-forward) evaluation, refitting before each gameweek using only data available before kickoff. Tune a small number of hyperparameters (ξ, regularisation or shrinkage, promoted-team priors) on a tuning block of seasons, using exact-scoreline log loss. Then lock them in and confirm once on a holdout season that was never used for tuning. The current season is the live test.

### Cited findings
- Time-series cross-validation, also called rolling forecasting origin or walk-forward evaluation (Hyndman & Athanasopoulos; Tashman 2000), refits the model on y_1..y_t for each origin t and records the error at t+h. `forecast::tsCV` has `initial` (a warm-up period with no evaluation) and `window` (a fixed-length rolling training window) options — [tsCV docs](https://search.r-project.org/CRAN/refmans/forecast/html/tsCV.html); [r-bloggers / Hyndsight post](https://www.r-bloggers.com/2016/12/cross-validation-for-time-series/)
- A rolling origin moves test data into training at later steps, so if it is used incorrectly it is "naturally susceptible to data leakage" — [Forecast Evaluation for Data Scientists, arXiv 2203.10716](https://arxiv.org/pdf/2203.10716)
- Dixon & Coles tuned ξ by out-of-sample predictive log score on outcomes — [arXiv 2508.05891](https://arxiv.org/html/2508.05891v1)
- Hobbyist walk-forward EPL backtests report per-season RPS spread across seasons, for example a best ensemble at about 0.198 with a worst season at 0.220. This shows season-to-season variation is large relative to differences between models — [GitHub Ficabral10](https://github.com/Ficabral10/Premier-League-Predictions) (non-peer-reviewed)

### Inferences (proposed protocol)
1. **Data split:**
   - Warm-up: 2020-21, plus the E1 data already used for promoted teams.
   - Tuning: 2021-22 to 2024-25 (1,520 matches).
   - Holdout: 2025-26 (380 matches), touched once per plan.
   - Live: 2026-27, where predictions are frozen.
   - The repo cache currently starts at 2021-22, so adding 2020-21 to the season list would give the first tuning season a warm-up year.
2. **Walk-forward:** For each gameweek (or each matchday date), fit on matches with kickoff before the first kickoff of that gameweek, predict the whole gameweek, and store the score matrix. This mirrors the existing `backtest` command, extended across seasons.
3. **Tune by grid, not by heavy search:**
   - ξ: about 6 values spanning half-lives of 3 months to 2 years.
   - Shrinkage or L2 strength: about 4 values.
   - Promoted-team prior: a few values.
   - The objective is mean exact-scoreline log loss over the tuning seasons.
   - Prefer the simplest setting within 1 SE of the best (the one-SE rule). Use a paired block bootstrap over gameweeks to get that SE.
4. **Report for every candidate:**
   - Exact-score log loss, 1X2 log loss, RPS and Brier.
   - Points on the score part, with paired-bootstrap CIs against the baseline.
   - Points per season, to check that the gain is consistent across seasons and not driven by one season.
5. **Holdout:** Evaluate the chosen configuration once on 2025-26 and record the result in the plan file. If the holdout is used to choose between options, it stops being a holdout. Then roll the holdout forward a season.
6. **Speed:** Refitting per gameweek costs 38 fits × 5 seasons × the grid size. Warm-starting the optimiser from the previous week's parameters keeps this cheap.

### Gaps
- I found no football-specific peer-reviewed guidance on how many tuning seasons are enough, or on how to guard against overfitting ξ. The protocol above is a standard forecasting practice adapted to this pipeline.

## 4. Historical player data for backtesting the scorer part without leakage

### Takeaway
vaastav/Fantasy-Premier-League (MIT licence) has per-player, per-fixture rows for 2016-17 to 2026-27. `expected_goals` and `starts` are present from 2022-23 onwards, so a scorer model can be backtested on about four past seasons. Neither vaastav nor FPL has goal minutes, so the minute component needs another source. The live FPL `element-summary` history does give per-fixture `expected_goals` for the current season.

### Cited findings
- The vaastav `data/` folder contains `2016-17` … `2026-27`, plus `cleaned_merged_seasons.csv` and `cleaned_merged_seasons_team_aggregated.csv` — **[verified via GitHub API]** [repo contents](https://github.com/vaastav/Fantasy-Premier-League/tree/master/data). The README says weekly updates stopped after 2024-25 and that it moved to "three major releases per season". It also asks users to link back to the repo if they publish work using the data — [README](https://github.com/vaastav/Fantasy-Premier-League)
- **Licence:** MIT, "Copyright (c) 2017-19 Vaastav Anand" — [LICENSE](https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/LICENSE). The underlying data is scraped from the FPL API, so the Premier League's terms on that data are a separate question (not checked).
- **Files:** `season/gws/gwN.csv`, `season/gws/merged_gw.csv` (all players by gameweek), `season/players/<name>/gws.csv` and `history.csv`, and `cleaned_players.csv` — [README](https://github.com/vaastav/Fantasy-Premier-League)
- **[verified headers]** `merged_gw.csv` columns:
  - **2021-22:** name, position, team, xP, assists, bonus, bps, clean_sheets, creativity, element, fixture, goals_conceded, goals_scored, ict_index, influence, kickoff_time, minutes, opponent_team, own_goals, penalties_missed, red_cards, round, saves, selected, team_a_score, team_h_score, threat, total_points, value, was_home, yellow_cards, GW and others. There is **no** xG and no `starts`.
  - **2022-23:** adds `expected_goals`, `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded` and `starts`.
  - **2024-25:** adds manager columns (`mng_*`) as well.
  - No season has a goal-minute column — [2021-22](https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/gws/merged_gw.csv), [2022-23](https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2022-23/gws/merged_gw.csv), [2024-25](https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv)
- **Leakage warning from the README:** `xP` (from the API's `ep_this`) may be a post-match value, so ML users should shift or drop it. GW35 xP values are all zero — [README](https://github.com/vaastav/Fantasy-Premier-League)
- **[verified live]** `https://fantasy.premierleague.com/api/element-summary/{id}/` returns `fixtures`, `history` and `history_past`. Each `history` row is one fixture this season and includes `fixture`, `kickoff_time`, `minutes`, `starts`, `goals_scored`, `assists`, `expected_goals`, `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded`, `threat`, `bps` and `defensive_contribution`. Example: element 1, GW1 2026-27, `expected_goals` "0.00", `minutes` 90 — [FPL API element-summary](https://fantasy.premierleague.com/api/element-summary/1/). `history_past` holds only season totals.

### Inferences
- **Leakage-free scorer backtest:**
  - For a match in gameweek g, build player features only from rows with `round < g` (or `kickoff_time` before the match): rolling minutes share, starts, goals and xG per 90.
  - Never use the same-fixture row's `minutes` or `starts` to decide who plays. The current baseline's "optimistic" scorer hits come from exactly this kind of leakage, because they use the live feed.
  - For lineup uncertainty, use the prior gameweeks' starts rate.
- **Which seasons:** xG-based scorer features can be backtested on 2022-23 to 2025-26 (about 1,520 matches). 2021-22 only supports goals and minutes features.
- **Team mapping:** vaastav's `team` and `opponent_team` IDs are per-season. The repo's `master_team_list.csv` maps IDs to names, which will need mapping to football-data names.
- **Goal minutes:** Historical goal minutes need a match-event source. Options are ESPN, which the repo already uses for minutes, or FBref / Understat shot data. Without one, the minute component cannot be backtested historically and should be reported separately.
- **Weight on the scorer component:** Scorer points have SD 0.78 per match. With about 1,500 matches, a paired test can detect scorer-model differences of roughly 0.05–0.08 points per match.

### Gaps
- I did not verify whether ESPN's historical match summaries give goal minutes reliably for 2021-22 to 2025-26.
- I did not check the Premier League's FPL data terms of use.
- I did not check whether vaastav's 2025-26 and 2026-27 folders are complete, given the move to periodic releases.

## 5. Calibration checks

### Takeaway
Plot reliability diagrams for P(home), P(draw) and P(away), with consistency bars from resampling, so that 90–380 matches are not over-read. Also compare predicted with actual distributions of total goals and goal difference. Draw calibration and the low-scoring tail are where Poisson-type models usually go wrong.

### Cited findings
- Bröcker & Smith, "Increasing the reliability of reliability diagrams", *Weather and Forecasting* (2007; LSE preprint 2006) — [LSE PDF](https://www.lse.ac.uk/CATS/Assets/PDFs/Publications/Papers/2007/73-IncreasingReliabilityDiagrams-2006-Brocker-Smith.pdf)
  - Consistency bars, obtained by resampling or from the binomial distribution, show how far observed frequencies can stray from the diagonal even if the forecast is perfectly calibrated.
  - With fixed-width bins, perfectly calibrated forecasts can still appear off-diagonal by up to half the bin width — as summarised in [arXiv 1902.06977](https://arxiv.org/pdf/1902.06977) and [arXiv 2112.10327](https://arxiv.org/pdf/2112.10327)
- The CORP approach (isotonic-regression reliability curves with consistency bands covering 90% of curves under calibration) is a more modern alternative — [arXiv 2108.03210](https://arxiv.org/pdf/2108.03210)
- **[computed]** EPL base rates useful as calibration targets (football-data.co.uk, local cache):

  | Season | Draw rate | Goals per match |
  |---|---|---|
  | 2021-22 | 0.232 | 2.82 |
  | 2022-23 | 0.229 | 2.85 |
  | 2023-24 | 0.216 | 3.28 |
  | 2024-25 | 0.245 | 2.93 |
  | 2025-26 | 0.274 | 2.75 |
  | 2026-27 GW1–9 | 0.278 | 2.70 |

  Goals per match vary by more than 0.5 between seasons.

### Inferences
- **Checks per backtest:**
  - (a) A reliability diagram per outcome, using about 10 quantile bins and consistency bars.
  - (b) Mean predicted vs actual draw rate.
  - (c) Mean predicted λ_home + λ_away vs actual goals per match, by season and by month.
  - (d) Predicted P(0-0), P(1-1) and P(1-0) vs observed frequencies.
  - (e) A PIT or histogram check of total goals.
- Because draw rates jumped in 2025-26 and 2026-27, check whether time decay lets the model adapt. This is also a reason not to over-tune ξ to older seasons.

### Gaps
- I found no published EPL-specific calibration study of Dixon-Coles to quote as a benchmark.

## 6. Benchmarks: bookmaker RPS, accuracy and exact-score hit rates

### Takeaway
Closing EPL bookmaker odds achieve RPS of about 0.18–0.20 per season (Pinnacle 0.180–0.211 in 2020-21 to 2024-25), log loss of about 0.90–0.99, and 51–60% accuracy picking the favourite. A strong model should approach but rarely beat the closing market. The single most common EPL scoreline (usually 1-1) occurs in 10–12% of matches, so an exact-score hit rate of about 10–12% is a realistic ceiling-ish target. The baseline's 8% (7/90) is within noise of that.

### Cited findings
- Pena.lt (penaltyblog author) analysed 250M betting lines from 1.25M fixtures. EPL RPS by bookmaker:

  | Bookmaker | 2020-21 | 2021-22 | 2022-23 | 2023-24 | 2024-25 |
  |---|---|---|---|---|---|
  | Pinnacle | 0.211 | 0.192 | 0.197 | 0.180 | 0.192 |
  | bet365 | 0.212 | 0.189 | 0.196 | 0.180 | 0.193 |
  | William Hill | 0.211 | 0.188 | 0.198 | 0.181 | 0.193 |

  EPL, La Liga and the Bundesliga are generally 0.18–0.20. The Eredivisie, Liga Portugal and Serie A are about 0.175–0.185. Lower tiers are often above 0.21 — [pena.lt, July 2025](https://pena.lt/y/2025/07/16/how-accurate-are-soccer-odds/)
- **[computed]** Pinnacle closing odds (PSC), from local football-data.co.uk data:

  | Season | RPS | Log loss | Brier | Favourite accuracy |
  |---|---|---|---|---|
  | 2021-22 | 0.1890 | 0.937 | 0.554 | 58.4% |
  | 2022-23 | 0.1975 | 0.962 | 0.571 | 54.7% |
  | 2023-24 | 0.1805 | 0.900 | 0.526 | 59.7% |
  | 2024-25 | 0.1961 | 0.966 | 0.575 | 55.3% |
  | 2025-26 | 0.1994 | 0.988 | 0.591 | 51.4% |

  These match pena.lt's figures for overlapping seasons. Market-average closing (AvgC) RPS for **2026-27 GW1–9 (the same 90 matches as the baseline) is 0.2058**, so this is the bar to compare the model's RPS against on the current backtest.
- **[computed]** Naive points baselines on the score part (3/1 scoring, Pinnacle-closing favourite, never predicting a draw):

  | Pick for the favourite | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
  |---|---|---|---|---|---|
  | 1-0 | 0.83 | 0.77 | 0.74 | 0.75 | 0.70 |
  | 2-1 | 0.76 | 0.75 | 0.79 | 0.76 | 0.70 |

  Values are points per match. For comparison, the Dixon-Coles baseline's score-part points over GW1–9 2026-27 are 0.656 per match (exact 7, results 45).
- **[computed]** EPL scoreline frequencies over 1,990 matches (2021-22 to 2026-27 GW9):

  | Scoreline | Share |
  |---|---|
  | 1-1 | 11.1% |
  | 2-1 | 8.4% |
  | 1-0 | 8.0% |
  | 0-1 | 7.1% |
  | 2-0 | 6.9% |
  | 2-2 | 6.7% |
  | 1-2 | 6.4% |
  | 0-2 | 5.5% |

  The most common scoreline's share per season ranged from 10.0% to 12.4%.
- A University of Reading blog reported a model hitting 4/40 (10%) correct scorelines in one Football League weekend, "a little lower than what we would expect the model to achieve" — [Reading econscorecast](https://blogs.reading.ac.uk/econscorecast/2018/08/19/reality-check/) (anecdotal, small sample)
- Reade, Singleton & Vaughan Williams (*Economic Issues* vol. 25) studied two EPL seasons. Forecast-encompassing regressions found model scoreline probabilities better than bookmaker-implied ones, but a betting strategy earned no consistent returns. Correct-score odds show significant favourite-longshot bias, which is generally absent in result markets. This matters if bookmaker correct-score odds are used as a scoreline benchmark — [Reading Football Research](https://research.reading.ac.uk/football-research/new-journal-article-betting-markets-for-english-premier-league-results-and-scorelines-evaluating-a-forecasting-model/); [discussion paper PDF](https://www.reading.ac.uk/web/files/economics/emdp202003.pdf)
- Hobbyist EPL models (non-peer-reviewed) report walk-forward RPS of about 0.198–0.21 and still trail closing odds — [Ficabral10](https://github.com/Ficabral10/Premier-League-Predictions); [KyanSoni](https://github.com/KyanSoni/premier-league-forecasting); [lawrencemartin05](https://github.com/lawrencemartin05/EPL-Market-Efficiency) (a Poisson model at RPS 0.203 lost to the market)

### Inferences
- **Score-model targets** for a walk-forward multi-season backtest: RPS within about 0.005–0.01 of closing odds on the same matches (around 0.195–0.205), 1X2 log loss under about 0.98, and 50–55% result accuracy. Compare against the bookmaker on identical matches with a paired test, never against a headline number from another season.
- **Points game:**
  - The naive "favourite wins 1-0 / 2-1" rule scored 0.70–0.83 points per match over five seasons, which beats the baseline's 0.656 on GW1–9. The 90-match sample is too small to be conclusive, but this must be included as a benchmark predictor in every backtest.
  - The baseline's 50% result accuracy is roughly at the market favourite's level (51–60%).
  - An exact-score rate of 8% (7/90, 95% CI roughly 3–15%) is consistent with the expected 9–12%.
- **Early-season difficulty:** Market RPS on GW1–9 2026-27 (0.206) is worse than any full-season figure since 2020-21. Early-season or this-season matches are harder, which is another reason not to judge on 90 matches.

### Gaps
- I found no authoritative published exact-score hit rate for a large EPL Dixon-Coles backtest. The 10–12% figure is inferred from the most common scoreline's frequency.
- I did not find the closing-odds market's own prediction-game points under 3/1 scoring, for example by using the market-implied modal scoreline. That would need correct-score odds, which football-data.co.uk does not provide.
