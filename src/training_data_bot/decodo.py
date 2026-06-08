"""
Minimal Decodo client placeholder.

The real production integration can replace this class when API credentials and
network behavior are defined.
"""

from typing import Any, Dict


class DecodoClient:
    """Small async scraping client compatible with WebLoader."""

    def __init__(self, **kwargs: Any):
        self.options = kwargs
        self._client = None

    async def scrape_url(self, url: str, **kwargs: Any) -> Dict[str, str]:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(follow_redirects=True, timeout=15.0)

        response = await self._client.get(url, headers={"User-Agent": "TrainingDataBot/0.1"})
        response.raise_for_status()
        return {"content": response.text}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
