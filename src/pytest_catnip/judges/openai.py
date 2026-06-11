from collections.abc import Awaitable, Callable

from pytest_catnip.judges.models import JudgeResult
from pytest_catnip.judges.prompt import get_prompt

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError("Install pytest-catnip[judge-openai] to use the OpenAI judge.") from e


def openai_judge(api_key: str, model: str) -> Callable[[str, str], Awaitable[tuple[bool, str]]]:
    client = AsyncOpenAI(api_key=api_key)

    async def judge(actual_reply: str, expected_judge_prompt: str) -> tuple[bool, str]:
        completion = await client.responses.parse(
            model=model,
            input=get_prompt(actual=actual_reply, expected=expected_judge_prompt),
            text_format=JudgeResult,
        )
        result = completion.output_parsed
        if not result:
            raise ValueError("Failed to parse judge result from OpenAI response.")

        return result.passed, result.reason

    return judge
