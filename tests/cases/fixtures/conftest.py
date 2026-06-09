import os
from collections.abc import Callable, Iterable
from unittest.mock import patch

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pytest_catnip import CatnipTurnTracker


async def handle_get_best_character(params: FunctionCallParams) -> None:
    await params.result_callback({"status": "ok", "result": "Shadowheart"})


@pytest.fixture
def mock_get_best_character() -> Iterable[None]:
    async def mock_tool(params: FunctionCallParams) -> None:
        await params.result_callback({"status": "ok", "result": "Astarion"})

    with patch(f"{__name__}.handle_get_best_character", side_effect=mock_tool):
        yield


@pytest.fixture
def catnip_pipeline() -> Callable[[], Pipeline]:
    def factory() -> Pipeline:
        order_tool = FunctionSchema(
            name="get_best_character",
            description="Get who's the best character in Baldur's Gate 3.",
            properties={},
            required=[],
        )
        context = LLMContext(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions about Baldur's Gate 3. "
                        "Always call get_best_character and return its result. Be extremely brief."
                    ),
                }
            ],
            tools=ToolsSchema(standard_tools=[order_tool]),
        )
        llm = GoogleLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            settings=GoogleLLMService.Settings(model="gemini-2.5-flash"),
        )
        llm.register_function("get_best_character", handle_get_best_character)

        ctx_agg = LLMContextAggregatorPair(context)
        turn_tracker = CatnipTurnTracker()
        return Pipeline([ctx_agg.user(), llm, turn_tracker, ctx_agg.assistant()])

    return factory
