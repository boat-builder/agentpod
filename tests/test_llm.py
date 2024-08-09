import os

import pytest

from agentpod import AsyncClient, LLMMeta, Message, UsageTracker


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
