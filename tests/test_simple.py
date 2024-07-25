import os

import pytest

from agentpod.client.client import AsyncClient, LLMMeta, Message


@pytest.mark.asyncio
async def test_async_client():
    # Get the OpenAI API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    # Initialize the AsyncClient
    client = AsyncClient(api_key=api_key, model=LLMMeta.GPT_4O_MINI)

    # Create a simple message
    messages = [Message(role="user", content="Say 'Hello, World!'")]

    # Invoke the client
    response = await client.invoke(messages)

    # Assert that we got a response
    assert isinstance(response, Message)
    assert response.role == "assistant"
    assert "Hello, World!" in response.content
