"""When the goals go in, and in what order."""

import random
import re
from dataclasses import dataclass

_CLOCK = re.compile(r"^(\d+)'(?:\+(\d+)')?$")


@dataclass(frozen=True)
class GoalTime:
    sort_key: float  # 45+2' -> 45.02 so it sorts before 46'
    label: str


def parse_clock(raw: str) -> GoalTime:
    """Parse ESPN clocks like "15'" or "90'+4'"."""
    match = _CLOCK.match(raw.strip())
    if not match:
        raise ValueError(f"Unrecognised match clock: {raw!r}")
    base, extra = int(match.group(1)), int(match.group(2) or 0)
    label = f"{base}+{extra}'" if extra else f"{base}'"
    return GoalTime(sort_key=base + extra / 100, label=label)


def goal_times(
    n_goals: int, distribution: tuple[GoalTime, ...], seed: int = 0
) -> tuple[GoalTime, ...]:
    """Stratified sample from the real goal-time distribution: goal k is drawn
    from quantile band [k/n, (k+1)/n), so goals stay spread across the match
    but vary between fixtures. Same seed → same times."""
    if n_goals == 0:
        return ()
    if not distribution:
        raise ValueError("Goal-time distribution is empty")
    ordered = sorted(distribution, key=lambda t: t.sort_key)
    rng = random.Random(seed)
    return tuple(
        ordered[min(int((k + rng.random()) / n_goals * len(ordered)), len(ordered) - 1)]
        for k in range(n_goals)
    )


def order_goals(home_goals: int, away_goals: int) -> tuple[str, ...]:
    """Interleave each side's goals evenly across the match (home first on ties)."""
    slots = [((i + 0.5) / home_goals, 0, "home") for i in range(home_goals)] + [
        ((i + 0.5) / away_goals, 1, "away") for i in range(away_goals)
    ]
    return tuple(side for _, _, side in sorted(slots))
