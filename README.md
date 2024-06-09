# AgentPod

AgentPod is a simple framework to build agents on top of LLMs. It currently supports OpenAI and will soon support Ollama. We do not plan to extend support beyond these at this point. Since it involves many API calls, we built it on top of asyncio and do not plan to create a sync client. If you need to use it in a synchronous program, you can use asyncio.run.

AgentPod supports both structured output with pydantic and unstructured output. It provides a reliable way to calculate the cost of API calls, with an easy-to-use API to get this cost at a detailed level. You can also access the raw responses from the LLM.

Our goal is to create a reliable, lightweight, and minimalistic framework to interact with LLMs. We are not focusing on building many integrations because the field is changing quickly. There are many similar client packages available, but AgentPod was created from our frustration with existing frameworks, which are often non-flexible, do too much behind the scenes, change APIs often, and have complex codebases. We are an AI agency, and we use AgentPod in production for all our agents.

## Installation

```
pip install agentpod
```

## Usage

```python
import asyncio
from pydantic import BaseModel, Field
from agentpod import AsyncClient, Message, LLMMeta


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
asyncio.run(stream_example())
```

## Acknowledgements

This project includes code from [Instructor](https://github.com/jxnl/instructor), which is licensed under the MIT License.

- Original Project: [Instructor](https://github.com/jxnl/instructor)
- Copyright (c) 2024 Jason Liu

