"""Claude via the official Anthropic SDK, with structured JSON output."""

import json

import anthropic

from football_ml.config import ANTHROPIC_EFFORT, ANTHROPIC_FALLBACK_BETA, LLM_MAX_TOKENS, LLM_TIMEOUT_SECONDS
from football_ml.llm.base import LLMConfigError, LLMError

_FATAL = (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
          anthropic.NotFoundError, anthropic.BadRequestError)


class AnthropicProvider:
    def __init__(self, model: str, client=None, effort: str = ANTHROPIC_EFFORT, max_tokens: int = LLM_MAX_TOKENS):
        self.model = model
        self._client = client or anthropic.Anthropic(timeout=LLM_TIMEOUT_SECONDS)
        self._effort = effort
        self._max_tokens = max_tokens

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                betas=[ANTHROPIC_FALLBACK_BETA],
                fallbacks="default",
                output_config={"effort": self._effort, "format": {"type": "json_schema", "schema": schema}},
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except _FATAL as exc:
            raise LLMConfigError(f"Anthropic rejected the request ({type(exc).__name__}): {exc}") from exc
        except anthropic.APIError as exc:  # rate limits, 5xx, network: already retried by the SDK
            raise LLMError(f"Anthropic call failed ({type(exc).__name__}): {exc}") from exc

        if response.stop_reason in ("refusal", "max_tokens"):
            details = getattr(response, "stop_details", None)
            raise LLMError(f"Claude stopped with {response.stop_reason}: {details}")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("Claude returned no text block")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Claude returned invalid JSON: {exc}") from exc
