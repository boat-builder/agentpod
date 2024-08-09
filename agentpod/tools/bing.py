import asyncio
from typing import List, Optional

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from agentpod.utils.file_cache import file_cache
from agentpod.utils.tracker import SearchMeta, UsageTracker


class BingSearchResult(BaseModel):
    title: str = Field(..., description="The title of the search result")
    url: str = Field(..., description="The URL of the search result")
    snippet: str = Field(..., description="A short description or snippet of the search result")


class BingSearch:
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

    async def _asearch_with_retry(self, query, count=3, retries=3) -> List[BingSearchResult]:
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {"q": query, "count": count}

        attempt = 0
        while attempt < retries:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(self.endpoint, headers=headers, params=params)
                    response.raise_for_status()
                    search_results = response.json()
                if self.usage_tracker:
                    await self.usage_tracker.update_search_cost(search_engine=SearchMeta.BING)

                results = [
                    BingSearchResult(title=result["name"], url=result["url"], snippet=result["snippet"])
                    for result in search_results.get("webPages", {}).get("value", [])
                ]
                return results
            except Exception as e:
                logger.error(f"Error searching Bing ({attempt}/{retries} retry): {e} \nquery: {query}")
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)  # exponential backoff
                else:
                    logger.error("Maximum retries reached, returning empty data.")
                    return []
            attempt += 1
        raise Exception("Maximum retries reached")

    async def asearch(self, query, count=3, retries=3) -> List[BingSearchResult]:
        if self.enable_cache:
            cache_enabled_func = file_cache()(self._asearch_with_retry)
            return await cache_enabled_func(query, count, retries)
        return await self._asearch_with_retry(query, count, retries)
