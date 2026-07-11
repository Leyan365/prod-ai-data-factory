"""
Fail-closed placeholder for a future policy-aware Decodo adapter.
"""

import ipaddress
import socket
from typing import Any, Dict
from urllib.parse import urlparse

from .core.config import RemoteFetchPolicy, settings
from .core.exceptions import DocumentLoadingError


class DecodoClient:
    """Direct use is rejected unless an explicit remote policy is supplied."""

    def __init__(self, *, remote_fetch_policy: RemoteFetchPolicy | None = None, **kwargs: Any):
        self.options = kwargs
        self.remote_fetch_policy = remote_fetch_policy or settings.remote_fetch
        self._client = None

    async def scrape_url(self, url: str, **kwargs: Any) -> Dict[str, str]:
        parsed = urlparse(url)
        policy = self.remote_fetch_policy
        if not policy.enabled:
            raise DocumentLoadingError("Direct Decodo remote calls are disabled by policy")
        if parsed.scheme not in {"https", "http"} or (parsed.scheme == "http" and not policy.allow_http):
            raise DocumentLoadingError("URL scheme is not permitted by remote fetch policy")
        if parsed.username or parsed.password or not parsed.hostname:
            raise DocumentLoadingError("URL credentials and missing hosts are not permitted")
        host = parsed.hostname.rstrip(".").lower()
        if host not in {item.rstrip(".").lower() for item in policy.allowed_hosts}:
            raise DocumentLoadingError(f"URL host is not approved: {host}")
        try:
            addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
        except OSError as exc:
            raise DocumentLoadingError(f"Unable to resolve approved URL host: {host}") from exc
        if any(not ipaddress.ip_address(address[0]).is_global for *_, address in addresses):
            raise DocumentLoadingError("URL resolved to a non-public IP address")

        import httpx
        if self._client is None:
            self._client = httpx.AsyncClient(
                follow_redirects=False,
                timeout=settings.resource_limits.request_timeout_seconds,
            )
        max_bytes = settings.resource_limits.max_remote_bytes
        async with self._client.stream("GET", url, headers={"User-Agent": "TrainingDataBot/0.1"}) as response:
            if 300 <= response.status_code < 400:
                raise DocumentLoadingError("Direct Decodo redirects are disabled; use WebLoader policy handling")
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise DocumentLoadingError("Remote response exceeds configured size limit")
            return {"content": bytes(body).decode(response.encoding or "utf-8", errors="replace")}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
