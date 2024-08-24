import asyncio
from typing import List, Optional

from pydantic import BaseModel, Field

from agentpod.http.client import retryable_httpx
from agentpod.utils.contentizer import contentize
from agentpod.utils.file_cache import file_cache
from agentpod.utils.tracker import SearchMeta, UsageTracker


class BingSearchResult(BaseModel):
    title: str = Field(..., description="The title of the search result")
    url: str = Field(..., description="The URL of the search result")
    snippet: str = Field(..., description="A short description or snippet of the search result")
    content: Optional[str] = Field(None, description="The content of the search result if fetch_content is True")


class AsyncBingSearch:
    def __init__(
        self,
        api_key,
        endpoint="https://api.bing.microsoft.com/v7.0/search",
        enable_cache=False,
        usage_tracker: Optional[UsageTracker] = None,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.enable_cache = enable_cache
        self.usage_tracker = usage_tracker

    async def search(self, query, count=3, fetch_content=False) -> List[BingSearchResult]:
        if self.enable_cache:
            cache_enabled_func = file_cache()(self._search_with_retry)
            return await cache_enabled_func(query, count, fetch_content)
        else:
            return await self._search_with_retry(query, count, fetch_content)

    async def _search_with_retry(self, query, count=3, fetch_content=False) -> List[BingSearchResult]:
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {"q": query, "count": count}

        async with retryable_httpx() as client:
            response = await client.get(self.endpoint, headers=headers, params=params)
            response.raise_for_status()
            search_results = response.json()

        if self.usage_tracker:
            await self.usage_tracker.update_search_cost(search_engine=SearchMeta.BING)

        results: List[BingSearchResult] = []
        for result in search_results.get("webPages", {}).get("value", []):
            results.append(
                BingSearchResult(title=result["name"], url=result["url"], snippet=result["snippet"], content=None)
            )

        if fetch_content:
            contents = await asyncio.gather(*[contentize(b.url) for b in results])
            for bing_result, content in zip(results, contents):
                bing_result.content = content

        return results
