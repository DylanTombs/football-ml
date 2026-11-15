# 08 — Availability and LLM adjustments

**Goal:** use team news (injuries, returns, new managers, rotation) to adjust the model's λ and μ. The LLM extracts the facts; a fitted model decides how much each fact matters.

## Decisions

- **Two steps, both tested:**
  1. **Availability features without an LLM** first, from FPL flags and snapshots: a key attacker out, a key player back, missing defenders on the *opponent*. Effects are fitted on data and shrunk towards zero.
  2. **LLM fact extraction** next, for what FPL doesn't capture: a new manager, a European game midweek, a cup game, a derby. The LLM fills a fixed schema of facts only. It never chooses multipliers or scores.
- **Rules for any adjustment:**
  - capped at about ±15%
  - only for *new* news since the ratings window, because recent results already reflect long-term injuries
  - works both ways (a return raises λ, and missing defenders raise the opponent's λ)
  - re-centred across each gameweek so it doesn't drain goals overall (plan 02's mistake)
- **Evidence so far:** combining player-level and team-level ratings beats either on its own (Arntzen & Hvattum 2021). Judgemental adjustments only help when they're large and rare (Fildes & Goodwin). In 2026 World Cup benchmarks, no LLM beat the betting market, and LLMs under-predicted draws and goals.
- **Testing the LLM part:** forward only. Backtests on past matches leak, because the model may know the results. Run it in shadow mode: write the adjusted prediction alongside the real one, frozen before kickoff, with a pinned model version and logged prompts. Turn it on only after enough matches (several hundred) show a paired gain.
- **Inputs:** FPL `news` and `chance_of_playing` from snapshots, plus saved preview text rather than live web search, so runs can be reproduced. LLM providers, caching and validation come from plan 03. Every player named is checked against the FPL squad.

## Decisions made 2026-11-16

- **LLM layer first.** It can only be tested forward, so every gameweek it misses is lost from the record. The availability fit waits for next week's data.
- **Fixed small priors to start.** Each fact moves λ/μ by a fixed effect set in `config.py` (about ±3–8%, capped at ±15% overall, re-centred per gameweek). These priors affect the shadow only and get replaced by fitted values once there's data. Claude still only reports facts.
- **Sources by fact:**
  - Players out, doubtful or back: FPL `status`, `chance_of_playing_*`, `news` and `news_added`, from snapshots. Claude turns the `news` text into facts.
  - Fixture congestion (European or cup games): ESPN team schedules across all competitions. Rest days are computed, with no LLM.
  - Derbies: a fixed list in `config.py`.
  - New manager: a date check against a small table of managerial changes.
  - Press-conference text: added later (Guardian Open Platform API if its licence fits, otherwise pasted into `data/previews/gwNN/`).
- All text Claude reads is saved to disk before the run, so every shadow prediction can be repeated.

## Tasks

- [x] Facts schema + Claude extractor over FPL `news` (reusing plan 03's `llm/`), with validation tests
- [ ] Structured facts: ESPN congestion, derby list, manager changes
- [ ] Fixed-prior adjustment (capped, re-centred) → `out/predictions/gwNN/pipeline-shadow.json`, scored by `score`
- [ ] Press-conference text source
- [ ] Availability features from snapshots, fitted with shrinkage, backtested against the plan 06 baseline (next week's data)
- [ ] Forward-test log, with results recorded here every few gameweeks

## Done when

- The availability features are on or written off by backtest. The LLM layer has a shadow-mode run of at least one full half-season and a recorded go/no-go decision.
