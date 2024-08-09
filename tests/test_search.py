import os

import pytest

from agentpod import UsageTracker
from agentpod.tools import BingSearch
from agentpod.tools.bing import BingSearchResult


@pytest.mark.asyncio
async def test_bing_search_client():
    # Get the Bing API key from environment variable
    bing_api_key = os.getenv("BING_SEARCH_API_KEY")
    if not bing_api_key:
        pytest.skip("BING_SEARCH_API_KEY environment variable not set")

    tracker = UsageTracker()

    # Initialize the BingSearch
    bing_client = BingSearch(api_key=bing_api_key, usage_tracker=tracker)

    # Perform a search
    search_query = "How can I use Localportal for accessing remote jupyter?"
    search_results = await bing_client.asearch(query=search_query, count=3)

    # Assert that we got a response
    assert isinstance(search_results, list)
    assert len(search_results) <= 3

    # Check the structure of the first result
    if search_results:
        first_result = search_results[0]
        assert isinstance(first_result, BingSearchResult)
        assert hasattr(first_result, "url")
        assert hasattr(first_result, "snippet")
        assert hasattr(first_result, "title")
    # Verify that the search results are related to the query
    assert any(
        "jupyter" in result.snippet.lower() or "localportal" in result.snippet.lower() for result in search_results
    )
    print(tracker)

    # Test tracker values
    assert tracker.total_llm_cost == 0.000000
    assert tracker.total_search_cost == 0.005000
    assert tracker.total_cost == 0.005000
    assert tracker.total_search_count == 1
