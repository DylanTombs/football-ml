# football-ml

Premier League predictions: exact scoreline, goalscorers and goal minutes for every fixture in a gameweek.

```bash
uv run football-ml predict              # next gameweek
uv run football-ml predict --gameweek 12
uv run football-ml score --gameweek 10   # after the gameweek: points vs actual results
uv run football-ml backtest --from 1 --to 9
uv run pytest --cov=football_ml
```

Output:
- `out/predictions/gwNN.json`: predictions, committed before kickoff
- `out/results/`: points per gameweek, plus `season.json`
- `out/backtest/`: replayed past gameweeks

Downloads are cached in `data/cache/`.

## How it works

| Step | Source | What |
|---|---|---|
| Scoreline | football-data.co.uk (6 seasons, PL + Championship) | Dixon-Coles Poisson model, time-decayed. Picks the likeliest result (home win, draw or away win), then the likeliest score within it |
| Scorers | FPL API | Available players weighted by season xG + goals, with diminishing weight for repeat scorers |
| Minutes | ESPN scoreboard | Sampled from this season's real goal-time distribution, seeded by fixture id |

Scoring: exact score 3, correct result 1, +1 for each correct scorer, +1 for each correct scorer whose goal is within ±10 minutes.

All tunable values live in `src/football_ml/config.py`.

## Roadmap

See [docs/plans](docs/plans/README.md): results tracker → injury-adjusted strength → LLM predictors → weekly automation. Architecture and data source quirks: [docs/overview.md](docs/overview.md).
