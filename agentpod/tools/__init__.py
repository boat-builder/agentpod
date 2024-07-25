import time

from loguru import logger

from agentpod.tools.brave import AsyncBrave
from agentpod.utils import file_cache
from agentpod.utils.tracker import UsageTracker


class BraveSearch:
    def __init__(self, api_key, enable_cache=False, usage_tracker: UsageTracker = None):
        self.api_key = api_key
        self.enable_cache = enable_cache
        self.client = AsyncBrave(api_key=self.api_key)
        self.usage_tracker = usage_tracker

    async def _asearch_with_retry(self, query, retries=3):
        attempt = 0
        while attempt < retries:
            try:
                result = await self.client.search(q=query, count=5)
                if self.usage_tracker:
                    self.usage_tracker.update_search_cost()
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
