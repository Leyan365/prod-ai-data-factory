"""
AI client abstraction and provider implementations.
"""

import asyncio
from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Protocol

from .core.config import settings
from .core.exceptions import AIClientError, AIProviderConfigurationError


@dataclass
class AIResponse:
    """Structured response from an AI provider."""

    text: str
    token_usage: int = 0
    cost: Optional[float] = None
    raw_response: Optional[Dict[str, Any]] = None


class AIProvider(Protocol):
    """Provider protocol used by AIClient."""

    async def generate(self, prompt: str, *, timeout: Optional[int] = None) -> AIResponse:
        """Generate text for a rendered prompt."""

    async def close(self) -> None:
        """Close provider resources."""


class MockAIProvider:
    """Deterministic offline provider for tests and local development."""

    async def generate(self, prompt: str, *, timeout: Optional[int] = None) -> AIResponse:
        normalized = " ".join(prompt.split())
        return AIResponse(
            text=f"Mock response: {normalized[:500]}",
            token_usage=max(1, len(prompt) // 4),
            raw_response={"provider": "mock"},
        )

    async def close(self) -> None:
        return None


class GeminiProvider:
    """Gemini REST provider that reads credentials from GEMINI_API_KEY only."""

    def __init__(self, *, model: str = settings.gemini_model, client: Any = None):
        self.model = model
        self._client = client
        self._owns_client = client is None
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise AIProviderConfigurationError(
                "GEMINI_API_KEY environment variable is required for GeminiProvider"
            )

    @classmethod
    def from_env(cls, *, model: str = settings.gemini_model, client: Any = None) -> "GeminiProvider":
        return cls(model=model, client=client)

    async def generate(self, prompt: str, *, timeout: Optional[int] = None) -> AIResponse:
        client = await self._get_client()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        params = {"key": self.api_key}

        response = await client.post(url, params=params, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        text = self._extract_text(data)
        return AIResponse(text=text, raw_response=data)

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient()
        return self._client

    def _extract_text(self, data: Dict[str, Any]) -> str:
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AIClientError("Gemini response did not include text output") from exc

        if not isinstance(text, str) or not text.strip():
            raise AIClientError("Gemini response text was empty")
        return text


class AIClient:
    """Retrying client wrapper around an AI provider."""

    def __init__(
        self,
        provider: Optional[AIProvider] = None,
        *,
        timeout: int = settings.ai_timeout,
        max_retries: int = settings.ai_max_retries,
    ):
        self.provider = provider or MockAIProvider()
        self.timeout = timeout
        self.max_retries = max_retries

        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

    @classmethod
    def from_env(cls, provider_name: str = settings.default_ai_provider, **kwargs: Any) -> "AIClient":
        normalized = provider_name.lower().strip()
        if normalized == "mock":
            return cls(provider=MockAIProvider(), **kwargs)
        if normalized == "gemini":
            return cls(provider=GeminiProvider.from_env(), **kwargs)
        raise AIProviderConfigurationError(f"Unknown AI provider: {provider_name}")

    async def generate(self, prompt: str, *, timeout: Optional[int] = None) -> AIResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise AIClientError("Prompt must be a non-empty string")

        effective_timeout = timeout or self.timeout
        attempts = self.max_retries + 1
        last_error: Optional[BaseException] = None

        for _ in range(attempts):
            try:
                return await asyncio.wait_for(
                    self.provider.generate(prompt, timeout=effective_timeout),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError as exc:
                last_error = exc
            except AIProviderConfigurationError:
                raise
            except Exception as exc:
                last_error = exc

        raise AIClientError("AI provider failed after retries", cause=last_error)

    async def close(self) -> None:
        await self.provider.close()
