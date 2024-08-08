import os

import pytest
from pydantic import BaseModel, Field

from agentpod import AsyncClient, LLMMeta, Message, UsageTracker


class GreetingResponse(BaseModel):
    greeting: str


@pytest.mark.asyncio
async def test_simple_output_type():
    # Get the OpenAI API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    tracker = UsageTracker()

    # Initialize the AsyncClient
    client = AsyncClient(api_key=api_key, model=LLMMeta.GPT_4O_MINI, usage_tracker=tracker)

    # Create a simple message
    messages = [Message(role="user", content="Generate a greeting")]

    # Invoke the client with a response model
    response = await client.invoke(messages, output_type=GreetingResponse)

    # Assert that we got a response
    assert isinstance(response, GreetingResponse)
    assert isinstance(response.greeting, str)
    assert len(response.greeting) > 0


class GreetingResponseConstrained(BaseModel):
    greeting: str = Field(description="A greeting message that starts with 'Hai Boatbuilder'")


@pytest.mark.asyncio
async def test_output_type_constrained():
    # Get the OpenAI API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    tracker = UsageTracker()

    # Initialize the AsyncClient
    client = AsyncClient(api_key=api_key, model=LLMMeta.GPT_4O_MINI, usage_tracker=tracker)

    # Create a simple message
    messages = [Message(role="user", content="Generate a greeting")]

    # Invoke the client with a response model
    response = await client.invoke(messages, output_type=GreetingResponseConstrained)

    # Assert that we got a response
    assert isinstance(response, GreetingResponseConstrained)
    assert isinstance(response.greeting, str)
    assert response.greeting.startswith("Hai Boatbuilder")
