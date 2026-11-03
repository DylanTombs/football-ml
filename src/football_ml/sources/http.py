"""HTTP fetching with a simple on-disk cache."""

import logging
import time
from pathlib import Path

import requests

from football_ml.config import CACHE_DIR, HTTP_HEADERS, HTTP_TIMEOUT_SECONDS

log = logging.getLogger(__name__)


class DataSourceError(RuntimeError):
    """A data source could not be reached or returned something unusable."""


def _is_fresh(path: Path, max_age_hours: float | None) -> bool:
    if not path.exists():
        return False
    if max_age_hours is None:  # cache forever (historical data)
        return True
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < max_age_hours


def fetch_text(url: str, cache_name: str, max_age_hours: float | None) -> str:
    """Return the body of `url`, served from cache when fresh enough."""
    path = CACHE_DIR / cache_name
    if _is_fresh(path, max_age_hours):
        return path.read_text(encoding="utf-8")

    try:
        response = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        if path.exists():
            log.warning("Fetch failed for %s (%s); using stale cache", url, exc)
            return path.read_text(encoding="utf-8")
        raise DataSourceError(f"Could not fetch {url}: {exc}") from exc

    response.encoding = response.encoding or "utf-8"
    text = response.text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
