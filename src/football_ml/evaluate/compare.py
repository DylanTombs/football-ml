"""Paired comparison of two predictors on the same matches.

Matches within a gameweek share conditions, so the bootstrap resamples whole
gameweeks (block bootstrap) rather than single matches."""

import numpy as np

from football_ml.config import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, POINT_COMPONENTS


def _rows(results: list[dict]) -> dict[int, tuple[int, dict]]:
    return {row["fixture_id"]: (gw["gameweek"], row) for gw in results for row in gw["matches"]}


def _name(results: list[dict]) -> str:
    return results[0]["predictor"] if results else "?"


def _component(diffs: np.ndarray, counts: np.ndarray, samples: np.ndarray) -> dict:
    """diffs/counts per gameweek; samples = resampled gameweek indices."""
    total = int(diffs.sum())
    resampled = diffs[samples].sum(axis=1) / counts[samples].sum(axis=1)
    lo, hi = np.percentile(resampled, [2.5, 97.5])
    return {
        "difference": total,
        "per_match": round(total / int(counts.sum()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
    }


def paired_comparison(
    reference: list[dict],
    challenger: list[dict],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """challenger − reference points on their shared matches, per component,
    with a 95% interval for the per-match difference."""
    ref, chal = _rows(reference), _rows(challenger)
    shared = sorted(set(ref) & set(chal))
    if not shared:
        raise ValueError("The two predictors have no shared matches to compare")

    gameweeks = sorted({ref[fid][0] for fid in shared})
    index = {gw: i for i, gw in enumerate(gameweeks)}
    counts = np.zeros(len(gameweeks))
    diffs = {c: np.zeros(len(gameweeks)) for c in POINT_COMPONENTS}
    for fid in shared:
        i = index[ref[fid][0]]
        counts[i] += 1
        for c in POINT_COMPONENTS:
            diffs[c][i] += chal[fid][1]["points"][c] - ref[fid][1]["points"][c]

    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(gameweeks), size=(resamples, len(gameweeks)))
    return {
        "reference": _name(reference),
        "challenger": _name(challenger),
        "matches": len(shared),
        "gameweeks": len(gameweeks),
        "changed_scorelines": sum(ref[f][1]["predicted"] != chal[f][1]["predicted"] for f in shared),
        "components": {c: _component(diffs[c], counts, samples) for c in POINT_COMPONENTS},
    }
