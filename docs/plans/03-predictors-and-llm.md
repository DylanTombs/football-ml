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

## Tasks

- [ ] `predictors/base.py`: `Predictor`, `FixtureContext`, `MatchPrediction` and shared JSON serialisation
- [ ] Move the current logic into `DixonColesPredictor`, keeping the existing tests green
- [ ] `llm/base.py`, `llm/anthropic.py`, `llm/openrouter.py`, tested against mocked HTTP
- [ ] `predictors/llm.py`: prompt, schema, validation, retry and fallback, tested with a fake provider (including bad output)
- [ ] Make the phase 1 results tracker and season table handle multiple predictors
- [ ] `football-ml predict --predictors dixon-coles,claude,gpt`

## Done when

- One gameweek is predicted by at least 3 predictors and scored on the same leaderboard
