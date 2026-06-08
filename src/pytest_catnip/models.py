import pathlib
from typing import Annotated

from pydantic import BaseModel, Field


class CatnipTestPhaseData(BaseModel):
    send: str
    expect_tools: list[str] | None = None
    expect_reply_contains: list[str] = []
    expect_flow_state: str | None = None


class ReliabilityConfig(BaseModel):
    runs: Annotated[int, Field(ge=2)]
    min_pass: Annotated[int, Field(ge=1)]


class CatnipTestCaseData(BaseModel):
    name: str
    description: str = ""
    max_auto_confirm: Annotated[int, Field(ge=0)] = 3
    turn_timeout: Annotated[int, Field(ge=0)] = 15
    skip: bool = False
    markers: list[str] = []
    phases: list[CatnipTestPhaseData]
    source_path: pathlib.Path
    reliability: ReliabilityConfig | None = None
