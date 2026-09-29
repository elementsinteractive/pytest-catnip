"""Barge-in as seen by custom frame processors, run through catnip's own runner with the real LLM pipeline."""

import os
import pathlib
from collections.abc import Callable
from typing import override

import pytest
from dotenv import load_dotenv
from pipecat.frames.frames import Frame, InterruptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.google.llm import GoogleLLMService
from pytest_catnip import CatnipTurnTracker
from pytest_catnip.models import CatnipTestCaseData
from pytest_catnip.runner import CatnipAssertionManager, CatnipTestInitializer, CatnipTestRunner

load_dotenv()

_CASE = CatnipTestCaseData(
    name="barge-in-processors",
    source_path=pathlib.Path("barge-in-processors.yaml"),
    phases=[
        {"send": "Explain in detail how volcanoes form."},  # type: ignore[list-item]
        {"send": "Stop. What is 2 + 2?", "barge_in": {"after_words": 5}},  # type: ignore[list-item]
    ],
)


class FrameRecorder(FrameProcessor):
    """Records every frame of interest with the direction it travelled."""

    def __init__(self, log: list[tuple[str, FrameDirection]]) -> None:
        super().__init__()
        self._log = log

    @override
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self._log.append((type(frame).__name__, direction))
        await self.push_frame(frame, direction)


def _pipeline_factory(before_llm: FrameProcessor) -> Callable[[], Pipeline]:
    def factory() -> Pipeline:
        context = LLMContext(
            messages=[{"role": "system", "content": "You are a verbose assistant. Explain things in 6+ sentences."}]
        )
        llm = GoogleLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            settings=GoogleLLMService.Settings(model="gemini-2.5-flash"),
        )
        user_agg, assistant_agg = LLMContextAggregatorPair(context)
        tracker = CatnipTurnTracker()
        return Pipeline([user_agg, before_llm, llm, tracker, assistant_agg])

    return factory


async def _run(factory: Callable[[], Pipeline]) -> None:
    async def judge(_: str, __: str) -> tuple[bool, str]:
        return True, ""

    runner = CatnipTestRunner(
        test_initializer=CatnipTestInitializer(pipeline_factory=factory),
        assertion_manager=CatnipAssertionManager(judge),
        timeout=60,
        autoconfirm_response="Yes.",
        is_confirmation_question=lambda _: False,
    )
    _, exc = await runner.run_once(_CASE)
    if exc is not None:
        raise exc


@pytest.mark.asyncio
async def test_interruption_enters_at_the_source() -> None:
    before_llm: list[tuple[str, FrameDirection]] = []
    await _run(_pipeline_factory(FrameRecorder(before_llm)))

    # A processor in front of the LLM sees the interruption flowing downstream, like from a real transport.
    assert ("InterruptionFrame", FrameDirection.DOWNSTREAM) in before_llm
