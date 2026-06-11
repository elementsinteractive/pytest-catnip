from collections.abc import Awaitable, Callable
from typing import Protocol

_JudgeFunctionType = Callable[[str, str], Awaitable[tuple[bool, str]]]


class _JudgeFactory(Protocol):
    def __call__(self, api_key: str, model: str) -> _JudgeFunctionType: ...


def __getattr__(name: str) -> _JudgeFactory:
    """Dynamically import judge functions when accessed as attributes of the module.

    This allows for users to import directly from `pytest_catnip.judges` without incurring in import errors
    for judge implementations they have not installed.
    """
    if name == "openai_judge":
        from pytest_catnip.judges.openai import openai_judge

        return openai_judge
    if name == "gemini_judge":
        from pytest_catnip.judges.gemini import gemini_judge

        return gemini_judge
    if name == "anthropic_judge":
        from pytest_catnip.judges.anthropic import anthropic_judge

        return anthropic_judge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
