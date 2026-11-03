# football-ml

Premier League predictions: exact scoreline, goalscorers and goal minutes for every fixture in a gameweek.

```bash
uv run football-ml predict              # next gameweek
uv run football-ml predict --gameweek 12
uv run pytest --cov=football_ml
```

Output: `out/predictions/gwNN.json` (one file per gameweek). Downloads are cached in `data/cache/`.

## How it works

| Step | Source | What |
|---|---|---|
| Scoreline | football-data.co.uk (6 seasons) | Dixon-Coles Poisson model, time-decayed. Picks the likeliest result (home win, draw or away win), then the likeliest score within it |
| Scorers | FPL API | Available players weighted by season xG + goals, with diminishing weight for repeat scorers |
| Minutes | ESPN scoreboard | Sampled from this season's real goal-time distribution, seeded by fixture id |

All tunable values live in `src/football_ml/config.py`.

## Next

- Results tracker: compare each gameweek's predictions against the actual results
- Better scorer model (minutes-adjusted, penalties, opponent strength)
- Backtest Dixon-Coles against bookmaker odds
