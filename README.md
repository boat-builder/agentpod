# AgentPod

AgentPod is a simple framework to build agents on top of LLMs. It currently supports OpenAI and will support Ollama at somepoint in the near future. We do not plan to extend support beyond these at this point. Since it is just a wrapper over a bunch of http calls, we built it on top of asyncio. And, at this point, we do not plan to create a sync client.

Agentpod supports both structured & unstructured output (even with Vision API). It provides a reliable way to calculate the cost of API calls, with an easy-to-use API to get this cost at a detailed level. You can also access the raw responses from the LLM. 

Our goal is to create a reliable, lightweight, and minimalistic framework to interact with LLMs. We are not focusing on building any integrations that isn't used at production by our users. There are many similar client packages available, but Agentpod was created from our frustration with existing frameworks, which are often non-flexible, do too much behind the scenes, change APIs often, and have complex codebases. We are an AI agency, and we use Agentpod in production for all our agents.

## Installation

```
pip install agentpod
```

## Examples

Examples can be found at our tests directory [tests/](tests/).

## Acknowledgements

This project includes code from [Instructor](https://github.com/jxnl/instructor), which is licensed under the MIT License.
