from collections.abc import Awaitable, Callable

from pytest_catnip.judges.models import JudgeResult
from pytest_catnip.judges.prompt import get_prompt

try:
    from google import genai
    from google.genai import types
except ImportError as e:
    raise ImportError("Install pytest-catnip[judge-gemini] to use the Gemini judge.") from e


def gemini_judge(api_key: str, model: str) -> Callable[[str, str], Awaitable[tuple[bool, str]]]:
    client = genai.Client(api_key=api_key)

    async def judge(actual_reply: str, expected_judge_prompt: str) -> tuple[bool, str]:
        response = await client.aio.models.generate_content(
            model=model,
            contents=get_prompt(actual=actual_reply, expected=expected_judge_prompt),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=JudgeResult,
                temperature=0.0,
            ),
        )
        result = JudgeResult.model_validate_json(str(response.text))
        return result.passed, result.reason

    return judge
