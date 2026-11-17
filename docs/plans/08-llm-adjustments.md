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

## Notes as built

- **News facts:** one Claude call per gameweek covers every player with news who hasn't left (86 at GW10). Claude returns a reason (including `returned`) and an expected return date. Each answer is checked against the items sent, with one retry; if that fails too, there are no facts. The prompt and response are cached together.
- **Congestion:** ESPN `soccer/all/teams/{id}/schedule` (played) plus `?fixture=true` (upcoming) covers league, cups and Europe. Rest days and days to the next game are worked out per side, with friendlies skipped.
- **Derbies** and **manager changes** are tables in `config.py`. A manager counts as new for 56 days. The table is empty at GW12: nine changes were made in the summer, none confirmed in-season (Palace reported to be close to replacing Sage).
- **Shadow adjustment:** the fixed priors live in `FACT_EFFECTS`:
  - key attacker out −6%, scaled by the chance he misses (0 if Claude's return date is before kickoff);
  - key attacker back +4%;
  - regular defender out +3% to the opponent;
  - short rest (<3.5 days) −3%;
  - European game within 3.5 days −3%;
  - new manager +4%;
  - derby −3% to both sides.
  
  Only news from the last 14 days counts. Each side is capped at ±15%, then everything is re-centred so the gameweek's total expected goals don't change. The shadow uses the main pick strategy on the adjusted rates, which `score` marks like any other file (points and log loss). It runs when `ANTHROPIC_API_KEY` is set and never blocks the main prediction. Turn it off with `--no-shadow`.
- **Dry run on live GW11 data**, with a stand-in for Claude that reads FPL status: 7 of 10 fixtures adjusted, all multipliers between ×0.95 and ×1.03, no predicted scoreline changed. Small priors were intended, so the forward record will mostly show log-loss differences, not points.

## Tasks

- [x] Facts schema + Claude extractor over FPL `news` (reusing plan 03's `llm/`), with validation tests
- [x] Structured facts: ESPN congestion, derby list, manager changes
- [x] Fixed-prior adjustment (capped, re-centred) → `out/predictions/gwNN/pipeline-shadow.json`, scored by `score`
- [ ] Press-conference text source
- [ ] Availability features from snapshots, fitted with shrinkage, backtested against the plan 06 baseline (next week's data)
- [ ] Forward-test log, with results recorded here every few gameweeks

## Done when

- The availability features are on or written off by backtest. The LLM layer has a shadow-mode run of at least one full half-season and a recorded go/no-go decision.
