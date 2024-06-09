# type: ignore[all]
import logging
from functools import wraps
from typing import Callable, Protocol, TypeVar, Union

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from typing_extensions import ParamSpec

from .mode import Mode
from .process_response import handle_response_model
from .retry import retry_async

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")


class AsyncInstructorChatCompletionCreate(Protocol):
    async def __call__(
        self,
        response_model: type[T_Model] = None,
        validation_context: dict = None,
        max_retries: int = 1,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> T_Model: ...


def patch(
    create: Callable[T_ParamSpec, T_Retval] = None,
    mode: Mode = Mode.TOOLS,
) -> Union[OpenAI, AsyncOpenAI]:
    """
    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """

    logger.debug(f"Patching `client.chat.completions.create` with {mode=}")

    if create is None:
        raise ValueError("create must be provided")

    @wraps(create)
    async def new_create_async(
        response_model: type[T_Model] = None,
        validation_context: dict = None,
        max_retries: int = 1,
        strict: bool = True,
        raw_processor_fn: Callable | None = None,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> T_Model:
        response_model, new_kwargs = handle_response_model(response_model=response_model, mode=mode, **kwargs)
        response = await retry_async(
            func=create,
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            strict=strict,
            raw_processor_fn=raw_processor_fn,
            mode=mode,
        )
        return response

    return new_create_async
