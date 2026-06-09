"""Pipeline fixture for tool-argument assertion test cases.

Scoped to tests/cases/tools/, this conftest provides a pipecat pipeline
exposing a tool that takes *arguments* (constrained to enums so the LLM's output
is deterministic). It exists to exercise the ``expect_tools`` argument-matching
behaviour from YAML. Cases elsewhere are unaffected.
"""

import os
from collections.abc import Callable

import pytest
from dotenv import load_dotenv
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pytest_catnip import CatnipTurnTracker

load_dotenv()


@pytest.fixture
def catnip_pipeline() -> Callable[[], Pipeline]:
    def factory() -> Pipeline:
        order_tool = FunctionSchema(
            name="order_pizza",
            description="Record a pizza order with the requested size (and optional crust).",
            properties={
                "size": {"type": "string", "enum": ["small", "medium", "large"]},
                "crust": {"type": "string", "enum": ["thin", "thick"]},
            },
            required=["size"],
        )
        context = LLMContext(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a pizza-ordering assistant. Keep replies very short. "
                        "As soon as the user tells you a size, call order_pizza with the "
                        "matching size. If they also mention a crust, include it. "
                        "Do not ask follow-up questions."
                    ),
                }
            ],
            tools=ToolsSchema(standard_tools=[order_tool]),
        )
        llm = GoogleLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            settings=GoogleLLMService.Settings(model="gemini-2.5-flash"),
        )

        async def handle_order_pizza(params: FunctionCallParams) -> None:
            await params.result_callback({"status": "ok"})

        llm.register_function("order_pizza", handle_order_pizza)

        ctx_agg = LLMContextAggregatorPair(context)
        turn_tracker = CatnipTurnTracker()
        return Pipeline([ctx_agg.user(), llm, turn_tracker, ctx_agg.assistant()])

    return factory
