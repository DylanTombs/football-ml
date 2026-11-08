# Overview

## Data flow

```
football-data.co.uk ─┐ results (PL + Championship, 7 seasons), pre-closing odds (season CSVs + fixtures.csv)
FPL API ─────────────┼─► load_season_inputs() ─► SeasonInputs ─► gameweek_context(gw) ─► each predictor ─► out/predictions/gwNN/<predictor>.json
ESPN scoreboard ─────┘ fixtures, squads,                                │
                       availability, goal events                         ▼
                                       build_actual_matches(gw) ─► score_gameweek ─► out/results/gwNN/<predictor>.json + season.json
```

Per fixture: Dixon-Coles rates λ, μ → injury adjustment (`INJURY_IMPACT`, off by default) → score matrix → pick (`--pick-strategy`):

- **expected-points** (default): the scoreline with probability ≥ `MIN_SCORELINE_PROBABILITY` worth the most expected points; scorers greedy on P(player scores ≥ k), with player goals Poisson. `--scorer-model chance` (default, plan 05): shrunk non-penalty xG per 90 × expected minutes, shared out of the side's open-play rate, plus the penalty taker's term. `weights`: (0.7 xG + 0.3 goals) share × side's expected goals. Fixed minutes from `REPEAT_PICK_MINUTES`.
- **modal**: likeliest result, then the likeliest score within it; scorers greedy by xG + goals with a repeat decay; minutes sampled from this season's goal times, seeded by fixture id. This is `ctx.baseline`, which the LLM predictors fall back to.

`predict` also writes `data/snapshots/gwNN.json` (FPL player state, before the deadline only). These files are committed.

## Modules (`src/football_ml/`)

| Module | Role |
|---|---|
| `config.py` | Every constant, URL, team-name map and points value |
| `sources/` | `http` (disk cache, falls back to stale cache), `football_data`, `fpl`, `player_history` (per-player match rows), `espn`, `results` (actual matches) |
| `model/market.py` | Shin de-margining; (λ, μ) implied by 1X2 + over 2.5 odds (plan 06). Benchmark only: odds never feed predictions (`ODDS_MODEL_WEIGHT` = 1) |
| `model/dixon_coles.py` | Fit (time decay, ridge penalty, ρ correction) and score matrix |
| `predict/` | `scoreline`, `scorers`, `timing` (incl. fixed minutes), `picks` (expected points, floor, P(X ≥ k) scorers), `scoring_chance` (plan 05 scorer model), `availability` (missing xG share per team) |
| `evaluate/` | `points` (one match), `season` (gameweek totals + season table), `compare` (paired block bootstrap over gameweeks) |
| `predictors/` | `base` (`Predictor`, `FixtureContext`, `MatchPrediction`, JSON), `context` (Dixon-Coles baseline + form per fixture), `dixon_coles` (pick strategy), `expected_points`, `naive_favourite` (backtest benchmark), `llm` + `llm_prompt` (validation, retry, fallback), `registry` (names → predictors, API key checks) |
| `llm/` | `LLMProvider` protocol, `anthropic` (SDK, structured outputs), `openrouter` (GPT, Gemini…), `cache` |
| `pipeline.py` | `SeasonInputs`, `gameweek_context`, `predictor_report`, `run_gameweek` |
| `bench.py` | Multi-season walk-forward (weekly refits, PL matches only), exact-score log loss / 1X2 log loss / RPS / calibration, xi-l2 grid, odds blend weight (plan 06) |
| `backtest.py` | Replays past gameweeks without access to later results: modal, the chosen strategy (weights), the chance scorer model and naive-favourite, compared with modal (and chance vs weights) |
| `snapshots.py` | Pre-deadline FPL player snapshots and loading them back as `Player`s |
| `historical_availability.py` | Who was missing in a past gameweek, from FPL minutes (`lagged` / `oracle` / `live`), and players' season stats as of a gameweek (`players_before`) |
| `cli.py` | `predict`, `score`, `backtest` |

## Data source quirks

- **Team names:** the canonical name is football-data's. FPL and ESPN names are mapped in `config.py`, so a newly promoted team may need a new entry.
- **football-data.co.uk:** `www.` redirects to the bare domain. Only pre-closing odds (`AvgH/D/A`, `Avg>2.5`) are used; the season CSV has no odds for unplayed matches, so upcoming odds come from `fixtures.csv` (all leagues, filtered to `Div` E0). It is overwritten every round and still shows the last round until the new odds go up (Friday afternoon, about 17:00 UK, for weekend games), (used only for benchmarks). Each `predict` saves the matching rows to `data/odds/gwNN.csv` (committed). Files carry a UTF-8 BOM but are served as Latin-1, so the first header arrives as `ï»¿Div`. The current season's CSV is refetched every 6 hours; past seasons are cached for good.
- **FPL:** `bootstrap-static` player stats are season-to-date only. `element-summary/{id}` has per-match rows (xG, minutes, starts, missed penalties), one request per player (~550, cached 12 h). The backtest rebuilds stats from these before each gameweek, so it no longer leaks. FPL xG includes penalties; non-penalty xG subtracts 0.78 per attempt (ESPN `penaltyKick` goals + FPL misses). Weekly snapshots exist from GW10 (plan 04).
- **FPL own goals:** listed under the side of the player who scored them.
- **FPL finished flags:** `finished` waits for bonus points (often the next morning); scoring also accepts `finished_provisional`, when the score and scorers are already final.
- **FPL availability:** only the current injury news exists, with no history. Status `u` means left or loaned out. `event/N/live/` gives per-gameweek minutes, which the backtest uses to infer past absences.
- **ESPN:** no date-range queries, so it's one request per day, cached for good once the day is over. Own goals are credited to the team that benefits. Clocks look like `90'+3'`.
- **Understat:** unreachable from here, so it isn't used.

## Prediction JSON (key fields)

`gameweek`, `predictor`, `predictor_config{type, model}`, `model{trained_until, injury_impact, …}`, `matches[]{fixture_id, home, away, prediction{score, goals[]{minute, team, scorer, scorer_id}, fallback, reasoning?}, prediction_probability, baseline_score, probabilities, expected_goals, top_scorelines, missing{home[], away[]}{name, xg_share}}`

`probabilities` / `expected_goals` / `top_scorelines` / `missing` are always the Dixon-Coles model's, whichever predictor wrote the file.

## LLM predictors

- Keys come from `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY`, checked up front for the enabled predictors only. Models live in `PREDICTOR_SPECS` in `config.py`.
- One call per fixture. Validated responses are cached in `data/cache/llm/`, so reruns are free.
- Never backtested: the models may have seen the real results.
