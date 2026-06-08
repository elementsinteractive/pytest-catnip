from collections.abc import Callable

import pytest
from dotenv import load_dotenv
from loguru import logger as _loguru_logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.google.llm import GoogleLLMService
from pytest_catnip import CatnipTurnTracker

load_dotenv()
_loguru_logger.remove()  # silence all pipecat/loguru output during tests


@pytest.fixture
def catnip_pipeline() -> Callable[[], Pipeline]:
    def factory() -> Pipeline:
        context = LLMContext(
            messages=[
                {
                    "role": "user",
                    "content": "You are a helpful assistant. Keep replies short.",
                },
            ]
        )
        import os

        llm = GoogleLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            settings=GoogleLLMService.Settings(model="gemini-2.5-flash"),
        )
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
        turn_tracker = CatnipTurnTracker()
        return Pipeline(
            [
                user_aggregator,
                llm,
                turn_tracker,
                assistant_aggregator,
            ]
        )

    return factory
