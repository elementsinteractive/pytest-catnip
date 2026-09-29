import pathlib
import re
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ToolExpectation(BaseModel):
    """Expectation for a single tool call within a phase.

    Entries in ``expect_tools`` may be a plain tool name (string) or an object
    with a ``name`` and optional ``args``. ``args`` is matched as a *partial*
    (subset) match against the arguments the tool was actually called with: only
    the keys listed here are checked, and each value must compare equal to the
    corresponding actual argument. When ``args`` is ``None`` only the tool name
    is asserted.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_shorthand(cls, value: Any) -> Any:  # noqa: ANN401
        # Allow a plain string entry as shorthand for ``{"name": <string>}``.
        if isinstance(value, str):
            return {"name": value}
        return value


class BargeIn(BaseModel):
    """Interrupt the bot's previous reply mid-stream, then send this phase's utterance.

    Exactly one trigger must be set. ``after`` fires at the first streamed chunk where the
    accumulated bot text matches the regex (case-insensitive); ``after_words`` fires once the
    reply reaches that many words. The reply is cut right after the trigger.
    """

    model_config = ConfigDict(extra="forbid")

    after: str | None = None
    after_words: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_shorthand(cls, value: Any) -> Any:  # noqa: ANN401
        # ``barge_in: true`` interrupts after the first word.
        if value is True:
            return {"after_words": 1}
        return value

    @model_validator(mode="after")
    def _validate_trigger(self) -> "BargeIn":
        if (self.after is None) == (self.after_words is None):
            raise ValueError("barge_in needs exactly one of 'after' or 'after_words' (or use 'barge_in: true')")
        if self.after is not None:
            if not self.after.strip():
                raise ValueError("barge_in.after must not be empty")
            try:
                re.compile(self.after)
            except re.error as e:
                raise ValueError(f"barge_in.after is not a valid regex ({self.after!r}): {e}") from None
        return self


class CatnipTestPhaseData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    send: str
    barge_in: BargeIn | None = None
    expect_tools: list[ToolExpectation] | None = None
    expect_reply_contains: list[str] = []
    expect_llm_judge: list[str] | None = None
    expect_flow_state: str | None = None

    @field_validator("barge_in", mode="before")
    @classmethod
    def _barge_in_false_is_none(cls, value: Any) -> Any:  # noqa: ANN401
        return None if value is False else value


class ReliabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: Annotated[int, Field(ge=2)]
    min_pass: Annotated[int, Field(ge=1)]


class CatnipTestCaseData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    max_auto_confirm: Annotated[int, Field(ge=0)] = 0
    turn_timeout: Annotated[int, Field(ge=0)] = 15
    skip: bool = False
    phases: list[CatnipTestPhaseData]
    source_path: pathlib.Path
    reliability: ReliabilityConfig | None = None

    # Pytest integration
    markers: list[str] = []
    fixtures: list[str] = []

    # Expectations
    expect_flow_nodes: list[str] | None = None
    expect_llm_judge: list[str] | None = None

    @model_validator(mode="after")
    def _validate_barge_in_position(self) -> "CatnipTestCaseData":
        if self.phases and self.phases[0].barge_in is not None:
            raise ValueError("phases -> 0 -> barge_in: the first phase has no previous bot reply to interrupt")
        return self
