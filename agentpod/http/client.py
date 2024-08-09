import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional

import httpx

from .retry_transport import RetryTransport

# Get the version from the package metadata with a fallback
try:
    __version__ = version("agentpod")
except PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback version


def retryable_httpx(timeout: Optional[httpx.Timeout] = None, **kwargs) -> httpx.AsyncClient:
    headers = kwargs.pop("headers", {})
    if "User-Agent" not in headers:
        headers["User-Agent"] = f"agentpod-python/{__version__}"

    timeout = timeout or httpx.Timeout(5.0, read=30.0, write=30.0, connect=5.0, pool=10.0)

    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout,
        transport=RetryTransport(wrapped_transport=httpx.AsyncHTTPTransport()),
        **kwargs,
    )
