# 07 — Dixon-Coles structure tweaks

**Goal:** small, evidence-backed changes to the team model, each tested on the plan 06 bench. Expect small gains. Published comparisons show goals-only models (Poisson variants, Weibull, pi-ratings) are close to tied.

## Decisions

- **Separate intercept and home advantage for PL and Championship.** Promoted teams' xG falls about a third after promotion, and one shared intercept blurs the two leagues.
- **Home advantage that varies by season**, or decays faster than the ratings. PL home advantage is at a record low (36% home wins in 2026-27 so far). This mainly shifts the home/draw/away split.
- **Shots blend:** add shots and shots on target, already in football-data CSVs, alongside goals. Wheatcroft (2020) found shot-based ratings add information beyond goals and even beyond odds. Pure xG ratings aren't proven to beat Dixon-Coles in the PL.
- **Not worth doing:** a total-goals multiplier (the rates are already calibrated at 256 vs 243), or swapping to a different goal-count model just for the tails.
- Each tweak is judged separately on held-out log loss and outcome points.

## Tasks

- [ ] Per-league intercept and home advantage
- [ ] Season-varying home advantage
- [ ] Shots-based component and blend weight
- [ ] Results for each tweak recorded here

## Done when

- Each tweak is either on (it beat the plan 06 baseline on held-out data) or written off with its numbers.
