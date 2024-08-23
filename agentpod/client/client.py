import asyncio
import os
from typing import AsyncGenerator, Dict, List, Literal, Optional, Type, Union

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from agentpod.client.structured.custom_async_openai import CustomAsyncOpenAI
from agentpod.client.structured.mode import Mode
from agentpod.utils.tracker import LLMMeta, UsageTracker


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    url: Optional[str] = None
    base64: Optional[str] = (
        None  # f"data:image/jpeg;base64,{base64_image}" - we should have a utility to read from file
    )
    detail: Optional[str] = "auto"  # "auto", "high", "low"

    def model_dump(self) -> Dict[str, Union[str, Dict[str, str]]]:
        result = {"type": "image_url"}
        image_url = {}
        if self.url:
            image_url["url"] = self.url
        elif self.base64:
            image_url["url"] = self.base64
        if self.detail:
            image_url["detail"] = self.detail
        result["image_url"] = image_url
        return result


ContentType = Union[str, List[Union[TextContent, ImageContent]]]


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: ContentType

    def to_dict(self) -> Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]:
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        else:
            return {"role": self.role, "content": [item.model_dump() for item in self.content]}


class AsyncClient:
    def __init__(
        self,
        api_key: str = "",
        provider: str = "openai",
        model: Union[str, LLMMeta] = LLMMeta.GPT_4O_MINI,
        usage_tracker: UsageTracker | None = None,
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

        self._usage_tracker = usage_tracker

    async def invoke(
        self, messages: list[Message], output_type: Optional[Type[BaseModel]] = None, max_retries: Optional[int] = 3
    ) -> Message | BaseModel:
        if output_type:
            response, original = await self._structured_client.create(
                model=self.model.value,
                messages=[message.to_dict() for message in messages],
                response_model=output_type,
                stream=False,
                max_retries=max_retries,
            )
            if original.usage and self._usage_tracker:
                await self._usage_tracker.update_llm_cost(original.usage, self.provider, self.model)
            return response
        else:
            response = await self._native_client.chat.completions.create(
                model=self.model.value,
                messages=[message.to_dict() for message in messages],
                stream=False,
            )
            if response.usage and self._usage_tracker:
                await self._usage_tracker.update_llm_cost(response.usage, self.provider, self.model)

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
                if chunk.usage and not chunk.choices and self._usage_tracker:
                    self._usage_tracker.update_llm_cost(chunk.usage, self.provider, self.model)
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
