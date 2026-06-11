import os
from collections.abc import Awaitable, Callable

import pytest
from pytest_catnip.judges import gemini_judge


@pytest.fixture(scope="session")
def llm_judge_function() -> Callable[[str, str], Awaitable[tuple[bool, str]]]:
    return gemini_judge(
        api_key=os.getenv("GOOGLE_API_KEY", ""),
        model="gemini-2.5-flash",
    )
