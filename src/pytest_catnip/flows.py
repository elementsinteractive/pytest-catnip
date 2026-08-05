from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from typing import Any, override

from pipecat.flows import FlowManager
from pipecat.flows.types import NodeConfig
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker


class CatnipFlowTracker(FlowManager):
    """``FlowManager`` subclass that records every node transition.

    Use this inside your ``CatnipFlowBundle.init_flow`` factory.
    ``expect_flow_state`` assertions compare against ``current_node``,
    and ``node_history`` is included in failure messages for easy debugging.
    """

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)
        self.node_history: list[str] = []

    @override
    async def _set_node(self, node_id: str, node_config: NodeConfig) -> None:
        self.node_history.append(node_id)
        await super()._set_node(node_id, node_config)


@dataclasses.dataclass(frozen=True)
class CatnipFlowBundle:
    """Bundles a ``Pipeline`` with a ``CatnipFlowTracker`` factory.

    Return this from your ``catnip_pipeline`` fixture factory (instead of a
    bare ``Pipeline``) when you want ``expect_flow_state`` assertions.

    ``init_flow`` is called once per run (after the ``PipelineWorker`` is
    created and running), so every run — including each reliability run —
    gets a fresh, consistent set of pipeline and flow-manager objects.

    Use ``respond_immediately=False`` on your initial ``NodeConfig`` so the
    bot doesn't produce a greeting before the first user message arrives.
    """

    pipeline: Pipeline
    init_flow: Callable[[PipelineWorker], Awaitable[CatnipFlowTracker]]
