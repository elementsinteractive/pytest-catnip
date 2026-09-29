"""Pipeline and judge fixtures for barge-in test cases.

Scoped to tests/cases/barge_in/. The bot is told to give long, enumerated answers so there is
always something left to interrupt.
"""

import os
from collections.abc import Awaitable, Callable

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
from pytest_catnip.judges import gemini_judge

load_dotenv()


@pytest.fixture
def catnip_pipeline() -> Callable[[], Pipeline]:
    def factory() -> Pipeline:
        price_tool = FunctionSchema(
            name="lookup_price",
            description="Look up the price of a pizza size.",
            properties={"size": {"type": "string", "enum": ["small", "medium", "large"]}},
            required=["size"],
        )
        context = LLMContext(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a verbose assistant. When asked to explain or list something, answer with "
                        "at least six full sentences, numbering the points 'Point one', 'Point two', and so on. "
                        "If a message of yours ends abruptly mid-sentence, the user interrupted you: never resume, "
                        "repeat or summarise that earlier answer, and answer only the user's new message, briefly. "
                        "When asked for a price, call lookup_price."
                    ),
                }
            ],
            tools=ToolsSchema(standard_tools=[price_tool]),
        )
        llm = GoogleLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            settings=GoogleLLMService.Settings(model="gemini-2.5-flash"),
        )

        async def handle_lookup_price(params: FunctionCallParams) -> None:
            await params.result_callback({"price_eur": 12})

        llm.register_function("lookup_price", handle_lookup_price)

        ctx_agg = LLMContextAggregatorPair(context)
        return Pipeline([ctx_agg.user(), llm, CatnipTurnTracker(), ctx_agg.assistant()])

    return factory


@pytest.fixture(scope="session")
def llm_judge_function() -> Callable[[str, str], Awaitable[tuple[bool, str]]]:
    return gemini_judge(api_key=os.getenv("GOOGLE_API_KEY", ""), model="gemini-2.5-flash")
