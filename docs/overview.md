# Overview

## Data flow

```
football-data.co.uk ─┐ results (PL + Championship, 6 seasons)
FPL API ─────────────┼─► load_season_inputs() ─► SeasonInputs ─► gameweek_context(gw) ─► each predictor ─► out/predictions/gwNN/<predictor>.json
ESPN scoreboard ─────┘ fixtures, squads,                                │
                       availability, goal events                         ▼
                                       build_actual_matches(gw) ─► score_gameweek ─► out/results/gwNN/<predictor>.json + season.json
```

Per fixture: Dixon-Coles rates λ, μ → injury adjustment (`INJURY_IMPACT`, off by default) → score matrix → headline score (likeliest result, then the likeliest score within it) → scorers (greedy by xG + goals, with less weight for players already given a goal) → minutes (stratified sample of this season's real goal times, seeded by fixture id).

## Modules (`src/football_ml/`)

| Module | Role |
|---|---|
| `config.py` | Every constant, URL, team-name map and points value |
| `sources/` | `http` (disk cache, falls back to stale cache), `football_data`, `fpl`, `espn`, `results` (actual matches) |
| `model/dixon_coles.py` | Fit (time decay, ridge penalty, ρ correction) and score matrix |
| `predict/` | `scoreline`, `scorers`, `timing`, `availability` (missing xG share per team) |
| `evaluate/` | `points` (one match), `season` (gameweek totals + season table) |
| `predictors/` | `base` (`Predictor`, `FixtureContext`, `MatchPrediction`, JSON), `context` (Dixon-Coles baseline + form per fixture), `dixon_coles`, `llm` + `llm_prompt` (validation, retry, fallback), `registry` (names → predictors, API key checks) |
| `llm/` | `LLMProvider` protocol, `anthropic` (SDK, structured outputs), `openrouter` (GPT, Gemini…), `cache` |
| `pipeline.py` | `SeasonInputs`, `gameweek_context`, `predictor_report`, `run_gameweek` |
| `backtest.py` | Replays past gameweeks without access to later results |
| `historical_availability.py` | Who was missing in a past gameweek, from FPL minutes (`lagged` / `oracle` / `live`) |
| `cli.py` | `predict`, `score`, `backtest` |

## Data source quirks

- **Team names:** the canonical name is football-data's. FPL and ESPN names are mapped in `config.py`, so a newly promoted team may need a new entry.
- **football-data.co.uk:** `www.` redirects to the bare domain. The current season's CSV is refetched every 6 hours; past seasons are cached for good.
- **FPL:** player stats are season-to-date only. This leaks future information into the scorer part of the backtest until plan 04 adds weekly snapshots.
- **FPL own goals:** listed under the side of the player who scored them.
- **FPL availability:** only the current injury news exists, with no history. Status `u` means left or loaned out. `event/N/live/` gives per-gameweek minutes, which the backtest uses to infer past absences.
- **ESPN:** no date-range queries, so it's one request per day, cached for good once the day is over. Own goals are credited to the team that benefits. Clocks look like `90'+3'`.
- **Understat:** unreachable from here, so it isn't used.

## Prediction JSON (key fields)

`gameweek`, `predictor`, `predictor_config{type, model}`, `model{trained_until, injury_impact, …}`, `matches[]{fixture_id, home, away, prediction{score, goals[]{minute, team, scorer, scorer_id}, fallback, reasoning?}, baseline_score, probabilities, expected_goals, top_scorelines, missing{home[], away[]}{name, xg_share}}`

`probabilities` / `expected_goals` / `top_scorelines` / `missing` are always the Dixon-Coles model's, whichever predictor wrote the file.

## LLM predictors

- Keys come from `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY`, checked up front for the enabled predictors only. Models live in `PREDICTOR_SPECS` in `config.py`.
- One call per fixture. Validated responses are cached in `data/cache/llm/`, so reruns are free.
- Never backtested: the models may have seen the real results.
