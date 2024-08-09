# type: ignore[all]
from __future__ import annotations

import logging
from json import JSONDecodeError
from typing import Any, Callable, TypeVar

from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, RetryError, Retrying, stop_after_attempt
from typing_extensions import ParamSpec

from .exceptions import InstructorRetryException
from .mode import Mode
from .process_response import process_response_async
from .utils import dump_message, merge_consecutive_messages

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


def reask_messages(response: ChatCompletion, mode: Mode, exception: Exception):
    yield dump_message(response.choices[0].message)
    # TODO: Give users more control on configuration
    if mode == Mode.TOOLS:
        for tool_call in response.choices[0].message.tool_calls:
            yield {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors",
            }
    else:
        yield {
            "role": "user",
            "content": f"Recall the function correctly, fix the errors, exceptions found\n{exception}",
        }


async def retry_async(
    func: Callable[T_ParamSpec, T_Retval],
    response_model: type[T] | None,
    validation_context: dict[str, Any] | None,
    args: Any,
    kwargs: Any,
    max_retries: int | AsyncRetrying = 1,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
) -> T:
    # If max_retries is int, then create a AsyncRetrying object
    if isinstance(max_retries, int):
        logger.debug(f"max_retries: {max_retries}")
        max_retries = AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            reraise=True,
        )
    if not isinstance(max_retries, (AsyncRetrying, Retrying)):
        raise ValueError("max_retries must be an `int` or a `tenacity.AsyncRetrying` object")

    try:
        async for attempt in max_retries:
            logger.debug(f"Retrying, attempt: {attempt}")
            with attempt:
                try:
                    response: ChatCompletion = await func(*args, **kwargs)
                    stream = kwargs.get("stream", False)
                    processed = await process_response_async(
                        response,
                        response_model=response_model,
                        stream=stream,
                        validation_context=validation_context,
                        strict=strict,
                        mode=mode,
                    )
                    # my hack - always returning processed and original
                    return processed, response
                except (ValidationError, JSONDecodeError) as e:
                    logger.debug(f"Error response: {response}", e)
                    kwargs["messages"].extend(reask_messages(response, mode, e))
                    if mode in {Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_JSON}:
                        kwargs["messages"] = merge_consecutive_messages(kwargs["messages"])
                    raise InstructorRetryException(
                        e,
                        last_completion=response,
                        n_attempts=attempt.retry_state.attempt_number,
                        messages=kwargs["messages"],
                        total_usage=total_usage,
                    ) from e
    except RetryError as e:
        logger.exception(f"Failed after retries: {e.last_attempt.exception}")
        raise InstructorRetryException(
            e,
            last_completion=response,
            n_attempts=attempt.retry_state.attempt_number,
            messages=kwargs["messages"],
            total_usage=total_usage,
        ) from e
