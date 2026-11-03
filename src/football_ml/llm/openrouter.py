"""GPT, Gemini and others through OpenRouter's OpenAI-compatible endpoint."""

import json

import requests

from football_ml.config import LLM_MAX_TOKENS, LLM_TIMEOUT_SECONDS, OPENROUTER_URL
from football_ml.llm.base import LLMConfigError, LLMError

_FATAL_STATUSES = (400, 401, 402, 403, 404)


class OpenRouterProvider:
    def __init__(self, model: str, api_key: str, max_tokens: int = LLM_MAX_TOKENS):
        self.model = model
        self._api_key = api_key
        self._max_tokens = max_tokens

    def _post(self, body: dict) -> dict:
        try:
            response = requests.post(
                OPENROUTER_URL,
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=LLM_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise LLMError(f"OpenRouter request failed: {exc}") from exc
        if response.status_code in _FATAL_STATUSES:
            raise LLMConfigError(f"OpenRouter rejected the request ({response.status_code}): {response.text[:300]}")
        if response.status_code >= 400:
            raise LLMError(f"OpenRouter error {response.status_code}: {response.text[:300]}")
        try:
            return response.json()
        except ValueError as exc:
            raise LLMError(f"OpenRouter returned a non-JSON body: {exc}") from exc

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        data = self._post({
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "match_prediction", "strict": True, "schema": schema},
            },
        })
        if "error" in data:
            raise LLMError(f"OpenRouter provider error: {data['error']}")
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"OpenRouter response missing a message: {exc!r}") from exc
        if choice.get("finish_reason") == "length":
            raise LLMError("OpenRouter response cut off at max_tokens")
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMError(f"{self.model} returned invalid JSON: {exc}") from exc
