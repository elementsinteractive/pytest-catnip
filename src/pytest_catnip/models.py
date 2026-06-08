import pathlib
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class CatnipTestPhaseData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    send: str
    expect_tools: list[str] | None = None
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
    markers: list[str] = []
    phases: list[CatnipTestPhaseData]
    source_path: pathlib.Path
    reliability: ReliabilityConfig | None = None
