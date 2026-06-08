"""Tests for AI client and providers."""

import asyncio
import os

import pytest

from training_data_bot.ai import AIClient, AIClientError, AIResponse, GeminiProvider, MockAIProvider
from training_data_bot.core.exceptions import AIProviderConfigurationError


def run(coro):
    return asyncio.run(coro)


class FlakyProvider:
    def __init__(self, failures=1):
        self.calls = 0
        self.failures = failures

    async def generate(self, prompt, *, timeout=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary")
        return AIResponse(text="ok", token_usage=3)

    async def close(self):
        return None


class SlowProvider:
    async def generate(self, prompt, *, timeout=None):
        await asyncio.sleep(0.05)
        return AIResponse(text="too late")

    async def close(self):
        return None


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class FakeHttpClient:
    def __init__(self):
        self.calls = []
        self.closed = False

    async def post(self, url, *, params, json, timeout):
        self.calls.append({"url": url, "params": params, "json": json, "timeout": timeout})
        return FakeResponse({"candidates": [{"content": {"parts": [{"text": "gemini text"}]}}]})

    async def aclose(self):
        self.closed = True


def test_mock_provider_returns_deterministic_text():
    response = run(MockAIProvider().generate("Hello   world"))

    assert response.text == "Mock response: Hello world"
    assert response.token_usage > 0
    assert response.raw_response == {"provider": "mock"}


def test_ai_client_retries_transient_errors_and_succeeds():
    provider = FlakyProvider(failures=2)
    response = run(AIClient(provider=provider, max_retries=2).generate("prompt"))

    assert response.text == "ok"
    assert provider.calls == 3


def test_ai_client_fails_after_max_retries():
    provider = FlakyProvider(failures=3)

    with pytest.raises(AIClientError):
        run(AIClient(provider=provider, max_retries=1).generate("prompt"))

    assert provider.calls == 2


def test_ai_client_enforces_timeout():
    with pytest.raises(AIClientError):
        run(AIClient(provider=SlowProvider(), timeout=1, max_retries=0).generate("prompt", timeout=0.01))


def test_ai_client_rejects_empty_prompt():
    with pytest.raises(AIClientError, match="Prompt"):
        run(AIClient().generate("  "))


def test_ai_client_from_env_defaults_to_mock_provider():
    client = AIClient.from_env()

    assert isinstance(client.provider, MockAIProvider)


def test_gemini_provider_requires_environment_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(AIProviderConfigurationError):
        GeminiProvider.from_env()


def test_gemini_provider_reads_key_from_environment_and_builds_request(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    fake_client = FakeHttpClient()
    provider = GeminiProvider.from_env(model="gemini-test-model", client=fake_client)

    response = run(provider.generate("rendered prompt", timeout=7))

    assert response.text == "gemini text"
    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["url"].endswith("/models/gemini-test-model:generateContent")
    assert call["params"] == {"key": "test-key"}
    assert call["json"] == {"contents": [{"parts": [{"text": "rendered prompt"}]}]}
    assert call["timeout"] == 7


def test_gemini_provider_empty_response_raises(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class EmptyClient(FakeHttpClient):
        async def post(self, url, *, params, json, timeout):
            return FakeResponse({"candidates": [{"content": {"parts": [{"text": ""}]}}]})

    with pytest.raises(AIClientError, match="empty"):
        run(GeminiProvider.from_env(client=EmptyClient()).generate("prompt"))


def test_tests_do_not_require_real_gemini_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert os.environ.get("GEMINI_API_KEY") is None
    response = run(AIClient().generate("offline prompt"))
    assert response.text.startswith("Mock response:")
