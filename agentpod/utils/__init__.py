import time
from typing import Literal, Union

from pydantic import BaseModel


def normalize_to_id(input_string: str) -> str:
    """
    Converts a normal sentence into a normalized ID format by replacing spaces with underscores,
    removing special characters, converting to lowercase, and appending a timestamp to ensure uniqueness.

    Args:
        input_string (str): The input string to be normalized.

    Returns:
        str: A normalized string suitable for use as an ID.
    """
    import re
    import time

    # Replace spaces with underscores
    input_string = input_string.replace(" ", "_")
    # Remove all non-alphanumeric characters (except underscores) and convert to lowercase
    normalized_string = re.sub(r"[^\w_]+", "", input_string).lower()
    # Append current timestamp
    timestamp = int(time.time())
    unique_normalized_string = f"{normalized_string}_{timestamp}"
    return unique_normalized_string


def escape_curly_braces(input_string: str) -> str:
    """
    Escapes curly braces in the input string so it can be safely used in an f-string formatter.

    Args:
        input_string (str): The input string that may contain curly braces.

    Returns:
        str: The input string with curly braces escaped.
    """
    return input_string.replace("{", "{{").replace("}", "}}")


class CostInfo(BaseModel):
    value: float


class ProgressInfo(BaseModel):
    message: str
    percentage: float


class ResponseStruct(BaseModel):
    type: Literal["cost", "progress"]
    details: Union[ProgressInfo, CostInfo]
