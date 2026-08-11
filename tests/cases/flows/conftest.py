"""Pipeline fixture for pipecat-flows test cases.

This conftest is scoped to tests/cases/flows/ so the CatnipFlowBundle
it provides only affects YAML cases in this subdirectory.
The existing cases in tests/cases/ continue to use the plain-Pipeline
fixture from tests/conftest.py and are completely unaffected.
"""

import os
from collections.abc import Callable

import pytest
from dotenv import load_dotenv
from pipecat.flows import FlowManager, FlowsFunctionSchema
from pipecat.flows.types import FlowArgs, NodeConfig
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.google.llm import GoogleLLMService
from pytest_catnip import CatnipTurnTracker
from pytest_catnip.flows import CatnipFlowBundle, CatnipFlowTracker

load_dotenv()


def _farewell_node() -> NodeConfig:
    return NodeConfig(
        name="farewell",
        task_messages=[
            {
                "role": "system",
                "content": "The user has said goodbye. Wish them farewell briefly.",
            }
        ],
        functions=[],
    )


def _greeting_node() -> NodeConfig:
    async def handle_say_farewell(args: FlowArgs, fm: FlowManager) -> tuple[None, NodeConfig]:
        return None, _farewell_node()

    return NodeConfig(
        name="greeting",
        respond_immediately=False,
        task_messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. You must answer questions, including mathematical calculations. Keep replies very short. "
                    "When the user says goodbye or wants to end the conversation, "
                    "call say_farewell."
                ),
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="say_farewell",
                description="Call this when the user says goodbye or wants to end the conversation.",
                properties={},
                required=[],
                handler=handle_say_farewell,
            )
        ],
    )


@pytest.fixture
def catnip_pipeline() -> Callable[[], CatnipFlowBundle]:
    def factory() -> CatnipFlowBundle:
        context = LLMContext(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Keep replies very short.",
                }
            ]
        )
        llm = GoogleLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            settings=GoogleLLMService.Settings(model="gemini-2.5-flash"),
        )
        ctx_agg = LLMContextAggregatorPair(context)
        turn_tracker = CatnipTurnTracker()
        pipeline = Pipeline([ctx_agg.user(), llm, turn_tracker, ctx_agg.assistant()])

        async def init_flow(worker: PipelineWorker) -> CatnipFlowTracker:
            tracker = CatnipFlowTracker(worker=worker, llm=llm, context_aggregator=ctx_agg)
            await tracker.initialize(_greeting_node())
            return tracker

        return CatnipFlowBundle(pipeline=pipeline, init_flow=init_flow)

    return factory
