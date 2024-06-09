import asyncio
import os
from enum import Enum
from typing import AsyncGenerator, Literal, Optional, Type, Union

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from agentpod.client.structured.custom_async_openai import CustomAsyncOpenAI
from agentpod.client.structured.mode import Mode
from agentpod.client.structured.patch import patch


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

    def to_dict(self) -> dict[
        Literal[
            "role",
            "content",
        ],
        str,
    ]:
        return self.model_dump()


MODEL_COSTS = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-32k": {"input": 60.00, "output": 120.00},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-instruct": {"input": 1.50, "output": 2.00},
}


class LLMMeta(Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"
    GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"

    @classmethod
    def get_model_cost(cls, model):
        return MODEL_COSTS[model.value]


class LLMUsageTracker:
    def __init__(self):
        self.completion_tokens: int = 0
        self.prompt_tokens: int = 0
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.active: bool = False

    async def __aenter__(self):
        self.active = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.reset()
        self.active = False

    def update(self, usage, provider: str, model: LLMMeta):
        if provider.lower() != "openai":
            raise ValueError("Currently, only 'openai' provider is supported.")

        model_costs = LLMMeta.get_model_cost(model)

        self.completion_tokens += usage.completion_tokens
        self.prompt_tokens += usage.prompt_tokens
        self.total_tokens += usage.total_tokens

        input_cost_per_token = model_costs["input"] / 1_000_000
        output_cost_per_token = model_costs["output"] / 1_000_000

        self.total_cost += (usage.prompt_tokens * input_cost_per_token) + (
            usage.completion_tokens * output_cost_per_token
        )

    def reset(self):
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def __repr__(self):
        return (
            f"UsageTracker(completion_tokens={self.completion_tokens}, "
            f"prompt_tokens={self.prompt_tokens}, total_tokens={self.total_tokens}, "
            f"total_cost={self.total_cost:.6f})"
        )


class AsyncClient:
    def __init__(
        self,
        api_key: Optional[str] = "",
        provider: Optional[str] = "openai",
        model: Union[str, LLMMeta] = LLMMeta.GPT_3_5_TURBO_INSTRUCT,
    ):
        if provider.lower() != "openai":
            raise ValueError("Currently, only 'openai' provider is supported.")
        self.provider = provider

        api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        if not api_key:
            raise ValueError(
                "API key must be provided either as an argument or through the OPENAI_API_KEY environment variable."
            )

        self._native_client = AsyncOpenAI(api_key=api_key)
        self._structured_client = CustomAsyncOpenAI(
            client=self._native_client,
            create=patch(create=self._native_client.chat.completions.create, mode=Mode.TOOLS),
            mode=Mode.TOOLS,
            provider=provider,
        )

        if isinstance(model, str):
            try:
                self.model = LLMMeta[model.upper().replace("-", "_")]
            except KeyError:
                raise ValueError(f"Unsupported model type: {model}")
        else:
            self.model = model

        self.usage_tracker = LLMUsageTracker()  # Initialize the usage tracker here

    async def invoke(
        self, messages: list[Message], output_type: Optional[Type[BaseModel]] = None, max_retries: Optional[int] = 3
    ) -> Message | BaseModel:
        if output_type:
            response = await self._structured_client.chat.completions.create(
                model=self.model.value,
                messages=[message.to_dict() for message in messages],
                response_model=output_type,
                stream=False,
                raw_processor_fn=lambda original: (
                    (
                        self.usage_tracker.update(original.usage, self.provider, self.model)
                        if original.usage and self.usage_tracker.active
                        else None
                    ),
                ),
                max_retries=max_retries,
            )
            return response
        else:
            response = await self._native_client.chat.completions.create(
                model=self.model.value,
                messages=[message.to_dict() for message in messages],
                stream=False,
            )
            if response.usage and self.usage_tracker.active:
                self.usage_tracker.update(response.usage, self.provider, self.model)

            # Craft a Message response from the response variable
            choice = response.choices[0]
            return Message(role=choice.message.role, content=choice.message.content)

    async def stream(
        self,
        messages: list[Message],
        output_type: Optional[Type[BaseModel]] = None,
        partial: Optional[bool] = False,
        max_retries: Optional[int] = 3,
    ) -> AsyncGenerator[Message, None]:
        if output_type:
            # TODO use max retries and partial. For partial, create a structured.Partial type and pass it. Rest is handled internally
            raise NotImplementedError
        else:
            response = await self._native_client.chat.completions.create(
                model=self.model.value,
                messages=[message.to_dict() for message in messages],
                stream=True,
                stream_options={"include_usage": True},
            )
            first_chunk = True
            role = None
            async for chunk in response:
                if chunk.usage and not chunk.choices and self.usage_tracker.active:
                    self.usage_tracker.update(chunk.usage, self.provider, self.model)
                if chunk.choices:
                    choice = chunk.choices[0]
                    if first_chunk:
                        role = choice.delta.role
                        first_chunk = False
                    content = choice.delta.content if choice.delta.content else ""
                    yield Message(role=role, content=content)


if __name__ == "__main__":
    client = AsyncClient(model=LLMMeta.GPT_3_5_TURBO_0125)
    sample_messages = [
        Message(
            role="system",
            content="You are a helpful assistant that knows the distance between two points",
        ),
        Message(role="user", content="Hello, how are you?"),
        Message(
            role="assistant",
            content="Hello! I'm just a computer program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?",
        ),
        Message(role="user", content="How far is Paris from Newyork?"),
    ]

    class Distance(BaseModel):

        distance: float = Field(
            description="Distance in miles between two points",
        )

    async def stream_example():
        async with client.usage_tracker as tracker:
            astream = client.stream(sample_messages, output_type=Distance, partial=True, max_retries=2)
            async for response in astream:
                print(response)
            print(tracker)

    async def invoke_example():
        async with client.usage_tracker as tracker:
            response = await client.invoke(sample_messages, output_type=Distance, max_retries=2)
            print(response)
            print(tracker)

    asyncio.run(invoke_example())
    # asyncio.run(stream_example())
