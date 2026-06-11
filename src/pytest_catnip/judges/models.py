from typing import Annotated

from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    passed: Annotated[bool, Field(description="True if the bot reply meets the expectation, False otherwise.")]
    reason: Annotated[str, Field(description="A short explanation of why it passed or failed.")]
