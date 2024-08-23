import os

import pytest
from pydantic import BaseModel

from agentpod import (
    AsyncClient,
    ImageContent,
    LLMMeta,
    Message,
    TextContent,
    UsageTracker,
)


class IsBlackAndWhiteResponse(BaseModel):
    is_black_and_white: bool


@pytest.mark.asyncio
async def test_vision_and_output_type():
    # Get the OpenAI API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    tracker = UsageTracker()

    # Initialize the AsyncClient
    client = AsyncClient(api_key=api_key, model=LLMMeta.GPT_4O_MINI, usage_tracker=tracker)

    # Create a simple message
    messages = [
        Message(
            role="user",
            content=[
                TextContent(text="is the image black and white?"),
                ImageContent(
                    url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                ),
            ],
        )
    ]

    # Invoke the client
    response: IsBlackAndWhiteResponse = await client.invoke(messages, output_type=IsBlackAndWhiteResponse)

    # Assert that we got a response
    assert isinstance(response, IsBlackAndWhiteResponse)

    print(tracker)

    # Test tracker values
    assert 0.003 <= tracker.total_llm_cost <= 0.006
    assert tracker.total_search_cost == 0.000000
