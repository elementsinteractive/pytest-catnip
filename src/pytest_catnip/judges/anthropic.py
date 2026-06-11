from collections.abc import Awaitable, Callable

from pytest_catnip.judges.models import JudgeResult
from pytest_catnip.judges.prompt import get_prompt

try:
    from anthropic import AsyncAnthropic
    from anthropic.types import MessageParam, OutputConfigParam
except ImportError as e:
    raise ImportError("Install pytest-catnip[judge-anthropic] to use the Anthropic judge.") from e


def anthropic_judge(api_key: str, model: str) -> Callable[[str, str], Awaitable[tuple[bool, str]]]:
    client = AsyncAnthropic(api_key=api_key)

    async def judge(actual_reply: str, expected_judge_prompt: str) -> tuple[bool, str]:
        messages: list[MessageParam] = [
            {"role": "user", "content": get_prompt(actual=actual_reply, expected=expected_judge_prompt)},
        ]
        output_config: OutputConfigParam = {
            "format": {
                "type": "json_schema",
                "schema": JudgeResult.model_json_schema(),
            }
        }

        response = await client.messages.create(
            model=model,
            max_tokens=512,
            messages=messages,
            output_config=output_config,
        )

        content_block = response.content[0] if response.content else None
        if content_block is None or content_block.type != "text":
            raise ValueError("Anthropic response did not contain a single text block.")

        result = JudgeResult.model_validate_json(content_block.text)
        return result.passed, result.reason

    return judge
