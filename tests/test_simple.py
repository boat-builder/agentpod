import os

import pytest

from agentpod import AsyncClient, LLMMeta, Message, UsageTracker
from agentpod.tools import BingSearch
from agentpod.tools.bing import BingSearchResult


@pytest.mark.asyncio
async def test_async_client():
    # Get the OpenAI API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    tracker = UsageTracker()

    # Initialize the AsyncClient
    client = AsyncClient(api_key=api_key, model=LLMMeta.GPT_4O_MINI, usage_tracker=tracker)

    # Create a simple message
    messages = [Message(role="user", content="Say 'Hello, World!'")]

    # Invoke the client
    response = await client.invoke(messages)

    # Assert that we got a response
    assert isinstance(response, Message)
    assert response.role == "assistant"
    assert "Hello, World!" in response.content

    # Test tracker values
    assert 0.000003 <= tracker.total_llm_cost <= 0.000006
    assert tracker.total_search_cost == 0.000000
    assert 0.000003 <= tracker.total_cost <= 0.000006
    assert tracker.total_search_count == 0


@pytest.mark.asyncio
async def test_bing_search_client():
    # Get the Bing API key from environment variable
    bing_api_key = os.getenv("BING_SEARCH_API_KEY")
    if not bing_api_key:
        pytest.skip("BING_SEARCH_API_KEY environment variable not set")

    # Initialize the BingSearch
    bing_client = BingSearch(api_key=bing_api_key)

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

    # Verify that the search results are related to the query
    assert any(
        "jupyter" in result.snippet.lower() or "localportal" in result.snippet.lower() for result in search_results
    )
