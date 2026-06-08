import os
from collections.abc import Awaitable, Callable
from typing import Annotated

import pytest
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    passed: Annotated[bool, Field(description="True if the bot reply meets the expectation, False otherwise.")]
    reason: Annotated[str, Field(description="A short explanation of why it passed or failed.")]


@pytest.fixture(scope="session")
def llm_judge_function() -> Callable[[str, str], Awaitable[tuple[bool, str]]]:
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    async def judge(actual_reply: str, expected_judge_prompt: str) -> tuple[bool, str]:
        prompt = (
            f"You are a strict QA testing judge.\n"
            f"Expectation: {expected_judge_prompt}\n"
            f"Actual Bot Reply: {actual_reply}\n\n"
            f"Evaluate if the bot reply satisfies the expectation."
        )

        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=JudgeResult,
                temperature=0.0,
            ),
        )

        result = JudgeResult.model_validate_json(str(response.text))

        return result.passed, result.reason

    return judge
