from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest
import requests

from football_ml.llm import openrouter
from football_ml.llm.anthropic import AnthropicProvider
from football_ml.llm.base import LLMConfigError, LLMError
from football_ml.llm.cache import JsonFileCache
from football_ml.llm.openrouter import OpenRouterProvider

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"],
          "additionalProperties": False}


class _FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _client(messages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def _message(text, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, stop_details=None,
                           content=[SimpleNamespace(type="thinking", thinking=""),
                                    SimpleNamespace(type="text", text=text)])


def _status_error(cls, status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


def test_anthropic_sends_schema_and_parses_json():
    messages = _FakeMessages(_message('{"x": 3}'))
    provider = AnthropicProvider("claude-opus-5-5", client=_client(messages))
    assert provider.complete_json("sys", "user", SCHEMA) == {"x": 3}
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "user"}]
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert call["output_config"]["effort"] == "high"
    assert call["fallbacks"] == "default"


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_anthropic_unusable_stop_reasons_raise(stop_reason):
    provider = AnthropicProvider("m", client=_client(_FakeMessages(_message("{}", stop_reason))))
    with pytest.raises(LLMError, match=stop_reason):
        provider.complete_json("s", "u", SCHEMA)


def test_anthropic_invalid_json_raises():
    provider = AnthropicProvider("m", client=_client(_FakeMessages(_message("not json"))))
    with pytest.raises(LLMError):
        provider.complete_json("s", "u", SCHEMA)


def test_anthropic_auth_error_is_a_config_error():
    error = _status_error(anthropic.AuthenticationError, 401)
    provider = AnthropicProvider("m", client=_client(_FakeMessages(error=error)))
    with pytest.raises(LLMConfigError):
        provider.complete_json("s", "u", SCHEMA)


def test_anthropic_server_error_is_retryable():
    error = _status_error(anthropic.InternalServerError, 500)
    provider = AnthropicProvider("m", client=_client(_FakeMessages(error=error)))
    with pytest.raises(LLMError) as info:
        provider.complete_json("s", "u", SCHEMA)
    assert not isinstance(info.value, LLMConfigError)


class _Response:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _openrouter_body(content, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": content}}]}


def test_openrouter_posts_json_schema_request(monkeypatch):
    calls = []
    monkeypatch.setattr(openrouter.requests, "post",
                        lambda url, **kw: calls.append((url, kw)) or _Response(200, _openrouter_body('{"x": 1}')))
    provider = OpenRouterProvider("openai/gpt-6.1-sol", api_key="k")
    assert provider.complete_json("sys", "user", SCHEMA) == {"x": 1}
    url, kwargs = calls[0]
    assert url.endswith("/chat/completions")
    assert kwargs["headers"]["Authorization"] == "Bearer k"
    body = kwargs["json"]
    assert body["model"] == "openai/gpt-6.1-sol"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["response_format"]["json_schema"]["strict"] is True


@pytest.mark.parametrize("status, error", [(401, LLMConfigError), (404, LLMConfigError), (502, LLMError)])
def test_openrouter_http_errors(monkeypatch, status, error):
    monkeypatch.setattr(openrouter.requests, "post", lambda url, **kw: _Response(status, {"error": "x"}))
    with pytest.raises(error):
        OpenRouterProvider("m", api_key="k").complete_json("s", "u", SCHEMA)


def test_openrouter_connection_error_is_retryable(monkeypatch):
    def boom(url, **kw):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(openrouter.requests, "post", boom)
    with pytest.raises(LLMError, match="offline"):
        OpenRouterProvider("m", api_key="k").complete_json("s", "u", SCHEMA)


@pytest.mark.parametrize("body", [
    {"error": {"message": "provider down"}},
    _openrouter_body('{"x": 1}', finish="length"),
    _openrouter_body("not json"),
    {"choices": []},
    ValueError("bad body"),
])
def test_openrouter_unusable_bodies_raise(monkeypatch, body):
    monkeypatch.setattr(openrouter.requests, "post", lambda url, **kw: _Response(200, body))
    with pytest.raises(LLMError):
        OpenRouterProvider("m", api_key="k").complete_json("s", "u", SCHEMA)


def test_json_file_cache_round_trip(tmp_path):
    cache = JsonFileCache(tmp_path)
    assert cache.get(("gw10", "claude", "100")) is None
    cache.put(("gw10", "claude", "100"), {"x": 1})
    assert cache.get(("gw10", "claude", "100")) == {"x": 1}
    assert (tmp_path / "gw10" / "claude" / "100.json").exists()


def test_json_file_cache_ignores_corrupt_entries(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.json").write_text("{oops")
    assert JsonFileCache(tmp_path).get(("a", "b")) is None


def test_json_file_cache_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        JsonFileCache(tmp_path).get(("..", "x"))
