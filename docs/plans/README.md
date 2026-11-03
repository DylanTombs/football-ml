# Roadmap

Build in this order. Each phase relies on the one before it.

| # | Plan | Why it comes here | Status |
|---|---|---|---|
| 1 | [Results tracker + backtest](01-results-tracker.md) | Without a score we can't tell whether any later change helps | done: baseline 141 pts |
| 2 | [Injury-adjusted team strength](02-injury-adjusted-strength.md) | Injuries currently change only who scores, not the scoreline | todo |
| 3 | [Predictor interface + LLM adapters](03-predictors-and-llm.md) | Claude / OpenRouter models compete against Dixon-Coles | todo |
| 4 | [Weekly command + automation](04-weekly-command.md) | One command (later a schedule) each gameweek | todo |

Rules that apply to every phase:
- Any change to how we predict is judged by backtest points from phase 1, not by how plausible it looks
- Predictions get committed before the first kickoff and are never edited afterwards
