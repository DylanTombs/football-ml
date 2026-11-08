# 07 — Dixon-Coles structure tweaks

**Goal:** small, evidence-backed changes to the team model, each tested on the plan 06 bench. Expect small gains. Published comparisons show goals-only models (Poisson variants, Weibull, pi-ratings) are close to tied.

## Decisions

- **Separate intercept and home advantage for PL and Championship.** Promoted teams' xG falls about a third after promotion, and one shared intercept blurs the two leagues.
- **Home advantage that varies by season**, or decays faster than the ratings. PL home advantage is at a record low (36% home wins in 2026-27 so far). This mainly shifts the home/draw/away split.
- **Shots blend:** add shots and shots on target, already in football-data CSVs, alongside goals. Wheatcroft (2020) found shot-based ratings add information beyond goals and even beyond odds. Pure xG ratings aren't proven to beat Dixon-Coles in the PL.
- **Not worth doing:** a total-goals multiplier (the rates are already calibrated at 256 vs 243), or swapping to a different goal-count model just for the tails.
- Each tweak is judged separately on held-out log loss and outcome points.

## Tasks

- [x] Per-league intercept and home advantage
- [x] Season-varying home advantage (plus a season scoring level)
- [x] Shots-based component and blend weight
- [ ] Results for each tweak recorded here: forward, from GW11 (see below)

## How it was built (2026-11-08)

**Approach changed (Dylan):** tweaks aren't written off on the bench. Each is a named **shadow variant** that runs every gameweek next to the main prediction, frozen and scored like any predictor, and one is promoted only once a few weeks show a pattern.

- **Variants** (`model/structure.py` `VARIANTS`), each saved as `out/predictions/gwNN/dixon-coles-<name>.json` with the same picks and scorers as the main prediction, so only the team model differs:

| Variant | What changes |
|---|---|
| `per-league` | Championship gets its own scoring level and home advantage (`league:E1`, `home:E1`) |
| `season-home` | home advantage per season, shrunk to the shared one (`SEASON_HOME_L2` = 10) |
| `season-level` | scoring level per season, shrunk (`SEASON_LEVEL_L2` = 10) |
| `shots` | goals model blended with a shots-on-target model: log λ = 0.5 log λ_goals + 0.5 log(SOT rate × goals per SOT) (`SHOTS_WEIGHT`) |

- Extras are design-matrix columns with their own ridge penalty, fitted jointly with the exact gradient (checked against finite differences). The shots model reuses the same fit on `HST`/`AST` with ρ fixed at 0; ρ always comes from the goals model.
- **Forward record:** every report now saves each match's (λ, μ, ρ), and `score` records −log P(actual score) per match and the mean in the leaderboard (`log loss`). Weekly points are noisy over 10 matches; log loss separates variants faster. GW10 was predicted before this, so it has no log loss.
- A variant that fails to fit is logged and skipped; it never blocks the main prediction.
- `predict` prints only the matches where a variant's scoreline differs from the main one (useful as a reel hook).
- Early bench check (tuning seasons 2021-25, before the switch to forward tracking): `per-league` +0.0023 ± 0.0014 exact log loss vs the plain model; `season-home` −0.0012 to −0.0025 (l2 0.1 / 1 / 10), with predicted goals closer to actual (4,416 vs 4,338, actual 4,516). `season-level` and `shots` weren't benched; `bench --season-level` / `--per-league` / `--season-home` can run them if wanted.

## Done when

- ~~Each tweak is either on (it beat the plan 06 baseline on held-out data) or written off with its numbers.~~ Each tweak runs as a shadow variant; promote one to the main model once several gameweeks of forward results show a pattern (Dylan, 2026-11-08).
