# Roadmap

Build in this order. Each phase relies on the one before it. The research behind plans 04–08 is in `reports/Improving our match prediction model.md`.

| # | Plan | Why it comes here | Status |
|---|---|---|---|
| 1 | [Results tracker + backtest](01-results-tracker.md) | Without a score we can't tell whether any later change helps | done: baseline 141 pts |
| 2 | [Injury-adjusted team strength](02-injury-adjusted-strength.md) | Injuries currently change only who scores, not the scoreline | done: off by default (weak evidence, within noise) |
| 3 | [Predictor interface + LLM adapters](03-predictors-and-llm.md) | One interface for the pipeline and benchmarks; LLM plumbing for plan 08 | done: AI-vs-AI comparison dropped |
| 4 | [Expected-points picks + player snapshots](04-expected-points-picks.md) | The biggest gain: modal picks leave goals, and so scorer points, on the table. Snapshots can't be recovered later | todo |
| 5 | [Scorer model](05-scorer-model.md) | Real chances of scoring for plan 04's picks; removes the scorer leak | todo |
| 6 | [Multi-season test bench + bookmaker odds](06-test-bench-and-odds.md) | 90 matches can't rank changes; odds are the strongest input for λ and μ | todo |
| 7 | [Dixon-Coles structure tweaks](07-model-structure.md) | Small model gains, measurable only on the plan 06 bench | todo |
| 8 | [Availability and LLM adjustments](08-llm-adjustments.md) | Team news via fitted, capped adjustments; the LLM only extracts facts | todo |
| 9 | [Weekly command + automation](09-weekly-command.md) | One command (later a schedule) each gameweek | todo |

Rules that apply to every phase:
- Any change to how we predict is judged by backtest, not by how plausible it looks. From plan 06, use the multi-season bench and a paired comparison. 90 matches carries about ±13 points of noise.
- Predicted scorelines must be plausible enough to show in a reel (plan 04's probability floor). We don't game the rules with 6-5 picks.
- Predictions get committed before the first kickoff and are never edited afterwards
