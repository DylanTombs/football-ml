# 04 — Weekly command + automation

**Goal:** one command each gameweek scores the last round, saves a snapshot of player data, and predicts the next round with every predictor. Later it runs on a schedule.

## Decisions

- **`football-ml weekly`** does, in order:
  1. Score the last gameweek, if all its fixtures have finished.
  2. Snapshot FPL player data (id, team, xG, goals, minutes, availability only) to `data/snapshots/gwNN.json`. This file is committed.
  3. Predict the next gameweek with every enabled predictor.
  4. Print a summary that's ready for a reel.
- **Snapshots:** from here on, backtests read the snapshot instead of live FPL data, which removes the scorer leak from phase 1.
- **No editing after kickoff:** if the gameweek's first match has started, refuse to overwrite existing predictions (`--force` exists only for local testing). This keeps predictions credible.
- **Schedule:**
  - A GitHub Action runs Friday at 14:00 UTC, after press conferences, once FPL availability has been updated.
  - API keys come from repo secrets.
  - It commits the predictions and results with a `data:` commit.
  - It skips the run if the next deadline is more than 4 days away (international breaks).

## Tasks

- [ ] `snapshots.py`: write and read player snapshots, tested by a round-trip
- [ ] Backtest reads a snapshot when one exists
- [ ] `weekly` command, including the no-overwrite-after-kickoff check (with tests)
- [ ] `.github/workflows/weekly.yml`, with a test run done by hand before enabling the schedule

## Done when

- The weekly run does a full gameweek (score → snapshot → predict → commit) with no manual steps
