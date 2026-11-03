# football-ml

Premier League predictions (exact score, goalscorers, goal minutes) for every fixture in a gameweek, marked against real results each week. It feeds a recurring content series on Dylan's CS/AI reels, so a working pipeline matters more than accuracy for now. Accuracy is improved plan by plan.

`~/CLAUDE.md` (Remotion reel animations) also loads here because it's in a parent folder. Ignore it in this repo.

## Start of every session

1. Read `docs/plans/README.md` for status. The first plan not marked done is the next piece of work. Dylan decides when to start it.
2. Read `docs/overview.md` for how the system fits together and the quirks of each data source.

## Commands

```bash
uv run football-ml predict [--gameweek N] [--predictors dixon-coles,claude,gpt,gemini] [--pick-strategy expected-points|modal]
                                            # writes out/predictions/gwNN/<predictor>.json + data/snapshots/gwNN.json
uv run football-ml score --gameweek N       # scores every predictor: out/results/gwNN/<predictor>.json + season.json
uv run football-ml backtest --from 1 --to N [--pick-strategy S]  # modal vs S vs naive-favourite → out/backtest/
uv run pytest --cov=football_ml             # keep ≥ 80%
```

## Rules

- **Commits:** one commit per plan (`feat: <summary> (plan NN)`), pushed to `origin main` (private repo DylanTombs/football-ml). No Claude attribution or `Co-Authored-By` lines.
- **TDD:** write the test first, see it fail, then implement. Tests never hit the network; mock `fetch_text` or pass `SeasonInputs`.
- **Predictions are frozen:** never regenerate or edit `out/predictions/gwNN/*.json` after that gameweek's first kickoff.
- **Judge by backtest:** any change to how predictions are made must be compared against the baseline points in `docs/plans/01-results-tracker.md` (from plan 06, the multi-season bench) with a paired comparison, since 90 matches carries about ±13 points of noise. Record the new numbers in that phase's plan file.
- **Plausible picks:** predicted scorelines must stay plausible enough for a reel (plan 04's probability floor). Don't game the rules with extreme scores.
- **Code:** constants only in `config.py`. Frozen dataclasses and no mutation. Prediction code does no IO; fetching happens in `load_season_inputs()`.
- **Finishing a plan:** tick its tasks, write down the results and decisions made along the way (a few bullets, no essays), and set the status in `docs/plans/README.md`.
