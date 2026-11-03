# 09 — Weekly command

**Goal:** one command, run by hand each gameweek, that scores the last round, saves snapshots of player data and odds, and predicts the next round.

## Decisions

- **`football-ml weekly`** does, in order:
  1. Score the last gameweek, if all its fixtures have finished.
  2. Snapshot FPL player data (plan 04) and pre-match odds (plan 06) for the next gameweek.
  3. Predict the next gameweek with the pipeline, plus any benchmark predictors.
  4. Print a summary that's ready for a reel.
- **No editing after kickoff:** if the gameweek's first match has started, refuse to overwrite existing predictions (`--force` exists only for local testing). This keeps predictions credible.
- **Snapshots** are built in plan 04. Until this plan lands, they're taken by hand each week by `predict`.
- **Run by hand, no schedule (decided 2026-11-03).** Dylan runs `football-ml weekly` himself whenever there's a gameweek, watches the results come out, and makes the video in his own time.
  - Run it after the last gameweek finishes and before the next first kickoff. Friday afternoon is ideal: after press conferences, and after football-data.co.uk posts odds (around 17:00 UK) once plan 06 uses them.
  - If the next round's first match has already kicked off, it refuses to predict and only scores (see above).
  - It prints the files to commit and the suggested `data:` commit message. Committing stays manual.

## Tasks

- [ ] `weekly` command, including the no-overwrite-after-kickoff check (with tests)
- [ ] Reel-ready summary: last round's points (by match and the season total) and the next round's predictions

## Done when

- One `weekly` run does a full gameweek (score → snapshot → predict → summary) with no other commands needed
