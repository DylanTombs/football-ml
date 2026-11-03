# 03 — Predictor interface + LLM adapters

**Goal:** every predictor (Dixon-Coles, Claude, GPT, Gemini…) sits behind one interface, so all of them get scored the same way and can be ranked against each other. This also sets up the "AI vs stats model" series.

## Decisions

- **`Predictor` protocol:** `name` plus `predict(ctx: FixtureContext) -> MatchPrediction`. `MatchPrediction` is a frozen dataclass holding the score, goals (scorer id, name, minute) and an optional `reasoning`.
- **`FixtureContext`:** everything a predictor may use:
  - the fixture itself
  - Dixon-Coles probabilities, top scorelines and expected goals
  - squads with availability and missing players (from phase 2)
  - each team's last 5 results
- **`LLMProvider` protocol:** `complete_json(system, user, schema) -> dict`, with two implementations:
  - `AnthropicProvider`: Claude API directly, using structured or tool output. Check the current SDK docs and model ids when implementing.
  - `OpenRouterProvider`: OpenAI-compatible endpoint, giving GPT, Gemini, Llama and others through one key.
- **`LLMPredictor(provider, model)`:** gives the LLM the Dixon-Coles baseline as context and asks it to adjust using its own judgement (form, manager change, rotation, derby). It should adjust the baseline, not replace it.
- **Validating LLM output:**
  - scores must be integers from 0 to 10
  - scorers must be available players in that team's squad
  - minutes must be valid clock values
  - number of goals must equal the score
  - On failure, retry once, then fall back to the Dixon-Coles prediction marked `"fallback": true`.
- **Cost:**
  - One call per fixture.
  - Responses cached on disk by `(gameweek, fixture, predictor)`, so reruns are free.
  - Tests use a fake provider and never make live calls.
- **Config:**
  - API keys come from the `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY` env vars. Check they're set at startup, but only for the predictors that are enabled.
  - The predictor list lives in `config.py` as `(name, provider, model)`.
- **Output:** `out/predictions/gwNN/<predictor>.json`. Move the existing `gw10.json` to `gw10/dixon-coles.json`.

- **Added while building:**
  - Models (checked 2026-11-03): `claude` → `claude-opus-5-5` (Anthropic SDK, effort `high`), `gpt` → `openai/gpt-6.1-sol`, `gemini` → `google/gemini-4-argon` (both via OpenRouter, both support strict JSON schema).
  - Claude uses the official `anthropic` SDK with structured outputs (`output_config.format`) and the server-side refusal fallback (`fallbacks: "default"`), so a declined request is re-run on another Claude model rather than failing. Tests inject a fake client.
  - Failure types: a bad key, unknown model or rejected request (`LLMConfigError`) stops the run, because falling back silently on all 10 fixtures would hide it. Timeouts, 5xx errors, refusals and invalid output retry once (with the error fed back), then fall back to Dixon-Coles.
  - Only validated responses are cached (`data/cache/llm/gwNN/<predictor>/<fixture>.json`), so a rerun retries fallbacks.
  - Scorer choices are available, non-departed players with minutes. Clocks must be 1–90' with stoppage only after 45' or 90' (max +15).
  - The prompt includes the points rules, since that's the game every predictor is playing.
  - Every match's `probabilities`, `expected_goals`, `top_scorelines` and `missing` are always Dixon-Coles's, whatever the predictor. `baseline_score` shows what the LLM changed from.
  - Results are now `out/results/gwNN/<predictor>.json`, and `season.json` holds every predictor. `score` and `backtest` print a leaderboard, and fallbacks are counted per predictor.
  - The backtest stays Dixon-Coles only. LLMs can't be backtested fairly, because their training data may include the real results.
  - Refactor check: the backtest still scores 141, and the regenerated GW10 Dixon-Coles predictions match the old ones exactly.

## Tasks

- [x] `predictors/base.py`: `Predictor`, `FixtureContext`, `MatchPrediction` and shared JSON serialisation
- [x] Move the current logic into `DixonColesPredictor`, keeping the existing tests green
- [x] `llm/base.py`, `llm/anthropic.py`, `llm/openrouter.py`, tested against mocked HTTP
- [x] `predictors/llm.py`: prompt, schema, validation, retry and fallback, tested with a fake provider (including bad output)
- [x] Make the phase 1 results tracker and season table handle multiple predictors
- [x] `football-ml predict --predictors dixon-coles,claude,gpt`
- [x] ~~Live run: GW10 with dixon-coles, claude, gpt and gemini~~ dropped (see below)

## Outcome (2026-11-03)

- **Scope change:** Dylan doesn't want an "AI vs AI" comparison. The series is about how well *our* pipeline predicts. So LLM predictors stay in the code but aren't run each week, and the live run was dropped.
- What carries forward: the `Predictor` interface, `FixtureContext`, per-predictor outputs and leaderboard (useful for benchmarks like the naive favourite rule in plan 04), and the LLM providers, cache and validation, which plan 08 reuses to extract facts.
- Research behind the new roadmap: `reports/Improving our match prediction model.md`.

## Done when

- ~~One gameweek is predicted by at least 3 predictors and scored on the same leaderboard~~ Superseded by the scope change above. Done when the interface and providers are in place and tested, which they are.
