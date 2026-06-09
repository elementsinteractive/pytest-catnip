import pathlib
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class CatnipTestPhaseData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    send: str
    expect_tools: list[ToolExpectation] | None = None
    expect_reply_contains: list[str] = []
    expect_llm_judge: list[str] | None = None
    expect_flow_state: str | None = None


class ReliabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: Annotated[int, Field(ge=2)]
    min_pass: Annotated[int, Field(ge=1)]


class CatnipTestCaseData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    max_auto_confirm: Annotated[int, Field(ge=0)] = 3
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
