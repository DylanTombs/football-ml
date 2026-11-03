"""Pre-deadline snapshots of FPL player state (xG, minutes, injury news, penalty
order). The FPL API only ever shows the present, so these are the only record
of what we knew before a gameweek."""

import json
import logging
from pathlib import Path

import pandas as pd

from football_ml.config import SNAPSHOT_FIELDS, SNAPSHOTS_DIR
from football_ml.predict.scorers import Player
from football_ml.sources.fpl import parse_players

log = logging.getLogger(__name__)


def gameweek_deadline(bootstrap: dict, gameweek: int) -> pd.Timestamp:
    for event in bootstrap.get("events", []):
        if event.get("id") == gameweek and event.get("deadline_time"):
            return pd.Timestamp(event["deadline_time"])
    raise ValueError(f"FPL has no deadline for gameweek {gameweek}")


def build_snapshot(bootstrap: dict, gameweek: int, taken_at: pd.Timestamp) -> dict:
    return {
        "gameweek": gameweek,
        "taken_at": taken_at.isoformat(),
        "deadline": gameweek_deadline(bootstrap, gameweek).isoformat(),
        "teams": [{"id": t["id"], "name": t["name"]} for t in bootstrap["teams"]],
        "elements": [{f: e.get(f) for f in SNAPSHOT_FIELDS} for e in bootstrap["elements"]],
    }


def snapshot_players(snapshot: dict) -> tuple[Player, ...]:
    return parse_players(snapshot)


def snapshot_path(gameweek: int, snapshots_dir: Path = SNAPSHOTS_DIR) -> Path:
    return snapshots_dir / f"gw{gameweek:02d}.json"


def load_snapshot(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_snapshot(
    bootstrap: dict, gameweek: int, now: pd.Timestamp, snapshots_dir: Path = SNAPSHOTS_DIR
) -> Path | None:
    """Write (or refresh) the gameweek's snapshot. After the deadline the data
    no longer shows the pre-gameweek state, so nothing is written."""
    deadline = gameweek_deadline(bootstrap, gameweek)
    if now >= deadline:
        log.warning("GW%d deadline (%s) has passed; not writing a snapshot", gameweek, deadline)
        return None
    path = snapshot_path(gameweek, snapshots_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = build_snapshot(bootstrap, gameweek, taken_at=now)
    path.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False), encoding="utf-8")
    return path
