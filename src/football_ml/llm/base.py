"""One interface over every LLM vendor: send a prompt, get JSON matching a schema."""

from typing import Protocol


class LLMError(RuntimeError):
    """The call failed or returned something unusable; worth retrying or falling back."""


class LLMConfigError(LLMError):
    """Bad key, unknown model or rejected request: retrying won't help, so stop the run."""


class LLMProvider(Protocol):
    model: str

    def complete_json(self, system: str, user: str, schema: dict) -> dict: ...
