"""LLM responses on disk, so reruns of the same gameweek are free."""

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_SAFE_PART = re.compile(r"^[A-Za-z0-9_.-]+$")


class JsonFileCache:
    def __init__(self, root: Path):
        self._root = root

    def _path(self, key: tuple[str, ...]) -> Path:
        if not key or any(not _SAFE_PART.match(part) or part in (".", "..") for part in key):
            raise ValueError(f"Unsafe cache key: {key!r}")
        return self._root.joinpath(*key[:-1], f"{key[-1]}.json")

    def get(self, key: tuple[str, ...]) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Ignoring corrupt LLM cache entry %s", path)
            return None

    def put(self, key: tuple[str, ...], value: dict) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
