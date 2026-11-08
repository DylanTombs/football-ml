# football-ml

Premier League predictions (exact score, goalscorers, goal minutes) for every fixture in a gameweek, marked against real results each week. It feeds a recurring content series on Dylan's CS/AI reels, so a working pipeline matters more than accuracy for now. Accuracy is improved plan by plan.

`~/CLAUDE.md` (Remotion reel animations) also loads here because it's in a parent folder. Ignore it in this repo.

## Start of every session

1. Read `docs/plans/README.md` for status. The first plan not marked done is the next piece of work. Dylan decides when to start it.
2. Read `docs/overview.md` for how the system fits together and the quirks of each data source.

## Commands

```bash
uv run football-ml predict [--gameweek N] [--predictors dixon-coles,claude,gpt,gemini] [--pick-strategy expected-points|modal] [--scorer-model chance|weights]
                                            # writes out/predictions/gwNN/<predictor>.json + data/snapshots/gwNN.json
uv run football-ml score --gameweek N       # scores every predictor: out/results/gwNN/<predictor>.json + season.json
uv run football-ml backtest --from 1 --to N [--pick-strategy S]  # modal vs S vs chance scorers vs naive-favourite, leak-free → out/backtest/
uv run football-ml bench [--seasons tune|holdout] [--xi X --l2 L | --grid]  # walk-forward log loss over past seasons → out/bench/
uv run pytest --cov=football_ml             # keep ≥ 80%
```

## Rules

- **Commits:** one commit per plan (`feat: <summary> (plan NN)`), pushed to `origin main` (private repo DylanTombs/football-ml). No Claude attribution or `Co-Authored-By` lines.
- **TDD:** write the test first, see it fail, then implement. Tests never hit the network; mock `fetch_text` or pass `SeasonInputs`.
- **Predictions are frozen:** never regenerate or edit `out/predictions/gwNN/*.json` after that gameweek's first kickoff.
- **Judge by backtest:** score-model changes are judged on the multi-season bench (`bench`, exact-score log loss on 2021-25, baseline in `docs/plans/06-test-bench-and-odds.md`), then once on the 2025-26 holdout. Pick and scorer changes are judged on the GW1–N `backtest` with a paired comparison, since 100 matches carries about ±13 points of noise. Record the new numbers in that phase's plan file.
- **No bookmaker odds as a model input (decided 2026-11-08).** The series is about a statistical model (Dixon-Coles + FPL data + Claude's adjustments). Odds are only a benchmark (`naive-favourite`, "how close are we to the bookies"). `ODDS_MODEL_WEIGHT` stays 1.0.
- **Plausible picks:** predicted scorelines must stay plausible enough for a reel (plan 04's probability floor). Don't game the rules with extreme scores.
- **Code:** constants only in `config.py`. Frozen dataclasses and no mutation. Prediction code does no IO; fetching happens in `load_season_inputs()`.
- **Finishing a plan:** tick its tasks, write down the results and decisions made along the way (a few bullets, no essays), and set the status in `docs/plans/README.md`.
