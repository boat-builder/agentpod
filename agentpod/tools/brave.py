import time

from brave import AsyncBrave
from loguru import logger

from agentpod.utils import file_cache

COST_PER_SEARCH = 0.005


class SearchUsageTracker:
    def __init__(self):
        self.search_count = 0
        self.active = False

    async def __aenter__(self):
        self.active = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.reset()
        self.active = False

    def update(self):
        if self.active:
            self.search_count += 1

    def reset(self):
        self.search_count = 0

    @property
    def total_cost(self):
        return self.search_count * COST_PER_SEARCH

    def __repr__(self):
        return f"SearchUsageTracker(search_count={self.search_count}, total_cost={self.total_cost:.2f})"


class BraveSearch:
    def __init__(self, api_key, enable_cache=False):
        self.api_key = api_key
        self.enable_cache = enable_cache
        self.client = AsyncBrave(api_key=self.api_key)
        self.usage_tracker = SearchUsageTracker()

    async def _asearch_with_retry(self, query, retries=3):
        attempt = 0
        while attempt < retries:
            try:
                result = await self.client.search(q=query, count=5)
                if self.usage_tracker.active:
                    self.usage_tracker.update()
                return result
            except Exception as e:
                logger.error(f"Error searching brave ({attempt}/{retries} retry): {e} \nquery: {query}")
                if "ValidationError" in str(e) and "video.views" in str(e):
                    logger.error("Validation error for video views, returning empty data.")
                    dummy = object()  # Return a simple dummy object instead of None
                    dummy.web_results = []
                    return dummy
                if attempt < retries - 1:
                    time.sleep(2**attempt)  # poor man's exponential backoff
                else:
                    logger.error("Maximum retries reached, returning empty data.")
                    dummy = object()  # Return a simple dummy object instead of None
                    dummy.web_results = []
                    return dummy
            attempt += 1

    async def asearch(self, query, retries=3):
        if self.enable_cache:
            cache_enabled_func = file_cache()(self._asearch_with_retry)
            return await cache_enabled_func(query, retries)
        return await self._asearch_with_retry(query, retries)
