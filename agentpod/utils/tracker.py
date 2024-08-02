import asyncio
from enum import Enum

COST_PER_SEARCH = 0.005

MODEL_COSTS = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
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
    GPT_4O_MINI = "gpt-4o-mini"
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


SEARCH_COSTS = {
    "bing": 0.005,
    "brave": 0.005,
}


class SearchMeta(Enum):
    BING = "bing"
    BRAVE = "brave"

    @classmethod
    def get_search_cost(cls, search_engine):
        return SEARCH_COSTS[search_engine.value]


class UsageTracker:
    """
    A class to track the usage of LLM (Large Language Models) and search operations, and calculate the associated costs.

    This class is designed to be thread-safe for asynchronous operations using asyncio.Lock.
    Note that it is not thread-safe for synchronous operations because asyncio.Lock is
    specifically designed for use with asyncio's event loop and does not provide protection
    against concurrent access from multiple threads.
    """

    def __init__(self):
        self.completion_tokens: int = 0
        self.prompt_tokens: int = 0
        self.total_tokens: int = 0
        self.total_search_count: int = 0
        self.total_llm_cost: float = 0.0
        self.total_search_cost: float = 0.0
        self._lock = asyncio.Lock()  # Add a lock for thread safety

    async def update_llm_cost(self, usage, provider: str, model: LLMMeta):
        if provider.lower() != "openai":
            raise ValueError("Currently, only 'openai' provider is supported.")

        async with self._lock:
            model_costs = LLMMeta.get_model_cost(model)
            self.completion_tokens += usage.completion_tokens
            self.prompt_tokens += usage.prompt_tokens
            self.total_tokens += usage.total_tokens

            input_cost_per_token = model_costs["input"] / 1_000_000
            output_cost_per_token = model_costs["output"] / 1_000_000
            cost = (usage.prompt_tokens * input_cost_per_token) + (usage.completion_tokens * output_cost_per_token)
            self.total_llm_cost += cost

    async def update_search_cost(self, num_search: int = 1, search_engine: SearchMeta = SearchMeta.BING):
        search_cost = SearchMeta.get_search_cost(search_engine)
        async with self._lock:
            self.total_search_count += num_search
            self.total_search_cost += num_search * search_cost

    @property
    def total_cost(self) -> float:
        return self.total_llm_cost + self.total_search_cost

    def __repr__(self):
        return (
            f"UsageTracker(completion_tokens={self.completion_tokens}, "
            f"prompt_tokens={self.prompt_tokens}, total_tokens={self.total_tokens}, "
            f"total_llm_cost={self.total_llm_cost:.6f}, total_search_cost={self.total_search_cost:.6f}, "
            f"total_cost={self.total_cost:.6f}, total_search_count={self.total_search_count})"
        )
