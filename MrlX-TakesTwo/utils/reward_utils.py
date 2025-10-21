"""
Module: reward_utils
--------------------
Provides utility functions for model-based scoring and dialogue processing.

Features:
    - Retry decorator for async functions with backoff
    - Calls DeepSeek-R1 model with retry
    - Extracts
<think class="done">
 content and answers
    - Converts messages to dialogue format
    - Parses JSON score outputs
    - Extracts diagnosis and recommendation from LLM responses
"""

import re
import json
import asyncio
from typing import Tuple, List, Dict, Callable, TypeVar, Any
from functools import wraps

T = TypeVar("T")


def async_retry_with_backoff(
    max_retries: int = 200,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    log_after: int = 5,
    context_name: str = "API call",
    non_retryable_errors: Tuple[str, ...] = (
        "authentication",
        "api key",
        "unauthorized",
    ),
    default_return: Any = None,
):
    """
    Decorator for async functions to add retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        log_after: Start logging after this many retries
        context_name: Name to display in log messages
        non_retryable_errors: Tuple of error keywords that should not trigger retries
        default_return: Value to return on failure instead of raising (None = raise exception)

    Returns:
        Decorated function with retry logic

    Example:
        @async_retry_with_backoff(max_retries=100, context_name="DeepSeek-R1", default_return=("", ""))
        async def call_api(messages):
            response = await client.chat.completions.create(...)
            return response
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for i in range(max_retries):
                try:
                    return await func(*args, **kwargs)

                except Exception as error:
                    # Check for non-retryable errors
                    error_msg = str(error).lower()
                    if any(keyword in error_msg for keyword in non_retryable_errors):
                        print(f"{context_name} - Non-retryable error: {error}")
                        if default_return is not None:
                            return default_return
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** min(i, 10)), max_delay)

                    # Log retries after threshold
                    if i >= log_after:
                        print(
                            f"{context_name} Retry #{i+1}/{max_retries}, "
                            f"error: {error}, waiting {delay:.1f}s"
                        )

                    # Last retry - don't sleep and handle failure
                    if i == max_retries - 1:
                        print(f"{context_name} failed after {max_retries} retries.")
                        if default_return is not None:
                            return default_return
                        raise

                    await asyncio.sleep(delay)

            # This should never be reached due to the return/raise above
            if default_return is not None:
                return default_return
            raise RuntimeError(f"{context_name} failed after {max_retries} retries.")

        return wrapper

    return decorator


@async_retry_with_backoff(
    max_retries=200,
    base_delay=1.0,
    max_delay=60.0,
    log_after=5,
    context_name="DeepSeek-R1",
    default_return=("", ""),
)
async def call_r1_model_async(
    client,
    messages: List[Dict[str, str]],
    model: str = "DeepSeek-R1",
    temperature: float = 0.0,
) -> Tuple[str, str]:
    """
    Generic async R1 model call function with automatic retry

    Args:
        client: AsyncOpenAI client instance
        messages: Conversation message list
        model: Model name (default "DeepSeek-R1")
        temperature: Generation temperature (default 0.0)

    Returns:
        (think, answer) tuple, returns ("", "") on failure
    """
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    resp = completion.choices[0].message.content
    think, answer = extract_text(resp)
    return think, answer


def extract_text(response_text: str) -> Tuple[str, str]:
    """Split model response into think and answer."""
    if "</think>" in response_text:
        think, answer = response_text.split("</think>")
        return think.strip(), answer.strip()
    return "", response_text.strip()


def convert_messages_to_dialogue(messages: List[Dict[str, str]]) -> str:
    """Converts message list to string dialogue."""
    dialogue = []
    for item in messages:
        if item["role"] == "user":
            dialogue.append("Doctor：" + item["content"])
        elif item["role"] == "assistant":
            # Remove the <think> block from the assistant's response
            patient = re.sub(
                r"<think>.*?</think>", "", item["content"], flags=re.DOTALL
            ).strip()
            dialogue.append("Patient：" + patient)
    return "\n".join(dialogue)


def parse_score(result_tuple: Tuple[str, str]):
    """Helper function to parse the JSON content from the model response."""
    _, text = result_tuple
    json_content = re.sub(r"```json\n|\n```", "", text, flags=re.IGNORECASE)
    return json.loads(json_content)


def extract_diagosis_recommendation(messages):
    """Extracts diagnosis and recommendation from assistant's last message."""
    diagnosis, recommendation = "", ""
    if messages[-1]["role"] == "assistant":
        last_response = messages[-1]["content"]

        answer = last_response.split("</think>")[-1].strip()

        diagnosis_match = re.search(
            r"Diagnosis[:：](.*?)(?=Recommendation[:：]|$)",
            answer,
            re.DOTALL,
        )
        recommendation_match = re.search(
            r"Recommendation[:：](.*?)(?=\n|$)", answer, re.DOTALL
        )

        if diagnosis_match:
            diagnosis = diagnosis_match.group(1).strip()
        if recommendation_match:
            recommendation = recommendation_match.group(1).strip()

    return diagnosis, recommendation
