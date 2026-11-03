"""When the goals go in, and in what order."""

import random
import re
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

_CLOCK = re.compile(r"^(\d+)'?(?:\+(\d+)'?)?$")


@dataclass(frozen=True)
class GoalTime:
    sort_key: float  # 45+2' -> 45.02 so it sorts before 46'
    label: str

    def elapsed(self) -> int:
        """Minutes played including stoppage time: 90+4' -> 94."""
        base = int(self.sort_key)
        return base + round((self.sort_key - base) * 100)


def parse_clock(raw: str) -> GoalTime:
    """Parse clocks like "15'", ESPN's "90'+4'" or our own "90+4'"."""
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


# --- Fixed minutes (plan 04) ---------------------------------------------------
# Elapsed scale, as scored: 45+2' → 47, 90+4' → 94.
_CANDIDATE_MINUTES = range(1, 101)
_REGULATION = 90


def window_coverage(elapsed: Sequence[int], minute: int, tolerance: int) -> float:
    """Share of goals within ±tolerance of `minute`."""
    if not elapsed:
        return 0.0
    return sum(abs(e - minute) <= tolerance for e in elapsed) / len(elapsed)


def best_minutes(elapsed: Sequence[int], k: int, tolerance: int) -> tuple[int, ...]:
    """k minutes whose ±tolerance windows don't overlap and together cover the
    most goals: where to put k picks of the same player. Ties → earliest."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    cover = {m: window_coverage(elapsed, m, tolerance) for m in _CANDIDATE_MINUTES}
    apart = 2 * tolerance
    options = (
        combo for combo in combinations(_CANDIDATE_MINUTES, k)
        if all(b - a > apart for a, b in zip(combo, combo[1:]))
    )
    return max(options, key=lambda combo: (sum(cover[m] for m in combo), [-m for m in combo]))


def _spread(k: int) -> tuple[int, ...]:
    return tuple(round(_REGULATION * (j + 0.5) / k) for j in range(k))


def _nudge(minute: int, used: set[int]) -> int:
    """Nearest free minute: m, m+1, m−1, m+2, …"""
    for step in range(_REGULATION):
        for candidate in (minute + step, minute - step):
            if candidate >= 1 and candidate not in used:
                return candidate
    raise ValueError("No free minute left")


def pick_minutes(
    scorers: Sequence[Hashable | None], table: Mapping[int, tuple[int, ...]]
) -> tuple[int, ...]:
    """Fixed elapsed minute for each predicted goal, in the order given. A player
    picked k times uses `table[k]` (tiled windows); unknown scorers (None) count
    as different players. Goals by different players never share a minute."""
    keys = [s if s is not None else ("unknown", i) for i, s in enumerate(scorers)]
    counts = Counter(keys)
    windows = {key: list(table.get(n) or _spread(n)) for key, n in counts.items()}
    used: set[int] = set()
    minutes = []
    for key in keys:
        minute = _nudge(windows[key].pop(0), used)
        used.add(minute)
        minutes.append(minute)
    return tuple(minutes)


def elapsed_goal_time(minute: int) -> GoalTime:
    """GoalTime for an elapsed minute; past 90 becomes stoppage time (93 → 90+3')."""
    if minute <= _REGULATION:
        return GoalTime(float(minute), f"{minute}'")
    extra = minute - _REGULATION
    return GoalTime(_REGULATION + extra / 100, f"{_REGULATION}+{extra}'")
