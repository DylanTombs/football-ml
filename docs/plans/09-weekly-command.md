# 09 — Weekly command + automation

**Goal:** one command each gameweek scores the last round, saves snapshots of player data and odds, and predicts the next round. Later it runs on a schedule.

## Decisions

- **`football-ml weekly`** does, in order:
  1. Score the last gameweek, if all its fixtures have finished.
  2. Snapshot FPL player data (plan 04) and pre-match odds (plan 06) for the next gameweek.
  3. Predict the next gameweek with the pipeline, plus any benchmark predictors.
  4. Print a summary that's ready for a reel.
- **No editing after kickoff:** if the gameweek's first match has started, refuse to overwrite existing predictions (`--force` exists only for local testing). This keeps predictions credible.
- **Snapshots** are built in plan 04. Until this plan lands, they're taken by hand each week by `predict`.
- **Schedule:**
  - A GitHub Action runs on Friday, after press conferences and once FPL availability has been updated. Once odds are used (plan 06), it has to run after football-data.co.uk posts them (around 17:00 UK) but before a Friday 20:00 kickoff, so about 17:30 UK. For midweek rounds, run on Tuesday afternoon.
  - API keys come from repo secrets.
  - It commits the predictions and results with a `data:` commit.
  - It skips the run if the next deadline is more than 4 days away (international breaks).

## Tasks

- [ ] `weekly` command, including the no-overwrite-after-kickoff check (with tests)
- [ ] `.github/workflows/weekly.yml`, with a test run done by hand before enabling the schedule

## Done when

- The weekly run does a full gameweek (score → snapshot → predict → commit) with no manual steps
