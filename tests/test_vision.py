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


class AltTag(BaseModel):
    alt_tag: str


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
                TextContent(
                    text="Give me a detailed description of what is in the image, the camera angle etc so I can use that as the alt tag for the image to make it ADA compliant"
                ),
                ImageContent(
                    url="https://s3.amazonaws.com/zcom-media/sites/a0iE000000GVzCxIAL/media/catalog/product/1/0/1009543.jpg"
                ),
            ],
        )
    ]

    # Invoke the client
    response: AltTag = await client.invoke(messages, output_type=AltTag)
    print(response)

    # Assert that we got a response
    assert isinstance(response, AltTag)

    print(tracker)

    # Test tracker values
    assert 0.003 <= tracker.total_llm_cost <= 0.006
    assert tracker.total_search_cost == 0.000000
