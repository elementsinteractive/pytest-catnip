import asyncio
import logging
import pathlib
import re
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import pytest
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.workers.runner import WorkerRunner
from pytest_catnip.harness import CatnipSession, CatnipTurnTracker, TurnLogKind, _ToolCall
from pytest_catnip.models import CatnipTestCaseData, CatnipTestPhaseData, ToolExpectation

logger = logging.getLogger(__name__)


@dataclass
class _Attempt:
    run_idx: int
    passed: bool
    elapsed: float
    conv_log: list[tuple[TurnLogKind, str]]
    error: str


def create_test_function(
    case_data: CatnipTestCaseData,
) -> Callable[..., Coroutine[Any, Any, None]]:
    """Create an actual pytest test function that will execute the test case logic when called."""

    @pytest.mark.asyncio
    async def test_function(
        request: pytest.FixtureRequest,
        catnip_pipeline: Callable[[], Pipeline],
        autoconfirm_response: str,
        is_confirmation_question: Callable[[str], bool],
        llm_judge_function: Callable[[str, str], Awaitable[tuple[bool, str]]],
    ) -> None:
        if case_data.skip:
            pytest.skip("Test case marked as skipped")

        reliability_mode: bool = request.config.getoption("--catnip-reliability", default=False) or bool(
            request.config.getini("catnip_reliability")
        )
        overall_timeout = float(
            request.config.getoption("--catnip-overall-timeout", default=None)
            or request.config.getini("catnip_overall_timeout")
        )

        if reliability_mode and case_data.reliability is not None:
            await _run_reliability(
                request=request,
                case_data=case_data,
                pipeline_factory=catnip_pipeline,
                autoconfirm_response=autoconfirm_response,
                is_confirmation_question=is_confirmation_question,
                llm_judge_function=llm_judge_function,
                overall_timeout=overall_timeout,
            )
        else:
            conv_log, exc = await _run_once(
                case_data=case_data,
                pipeline_factory=catnip_pipeline,
                autoconfirm_response=autoconfirm_response,
                is_confirmation_question=is_confirmation_question,
                llm_judge_function=llm_judge_function,
                overall_timeout=overall_timeout,
            )
            request.node._catnip_attempts = [
                _Attempt(run_idx=0, passed=exc is None, elapsed=0.0, conv_log=conv_log, error=str(exc) if exc else "")
            ]
            if exc is not None:
                raise exc

    # Add pytest marks based on case_data.markers
    for marker_name in case_data.markers:
        marker_decorator = getattr(pytest.mark, marker_name)
        test_function = marker_decorator(test_function)
    return test_function


async def _run_once(
    case_data: CatnipTestCaseData,
    pipeline_factory: Callable[[], Any],
    autoconfirm_response: str,
    is_confirmation_question: Callable[[str], bool],
    llm_judge_function: Callable[[str, str], Awaitable[tuple[bool, str]]],
    overall_timeout: float,
) -> tuple[list[tuple[TurnLogKind, str]], Exception | None]:
    """Run all phases once on a fresh pipeline. Returns (conv_log, exception_or_None)."""
    bundle = pipeline_factory()

    # Support CatnipFlowBundle (pipecat-flows integration) or a plain Pipeline.
    # The import is lazy so pipecat-ai-flows remains an optional dependency.
    try:
        from pytest_catnip.flows import CatnipFlowBundle

        if isinstance(bundle, CatnipFlowBundle):
            pipeline: Pipeline = bundle.pipeline
            _init_flow = bundle.init_flow
        else:
            pipeline = bundle
            _init_flow = None
    except ImportError:
        pipeline = bundle
        _init_flow = None

    turn_tracker = next((p for p in pipeline.processors if isinstance(p, CatnipTurnTracker)), None)
    if turn_tracker is None:
        raise pytest.UsageError(
            f"[{case_data.name}] The 'catnip_pipeline' fixture must include "
            f"an instance of 'CatnipTurnTracker' in its processor list."
        )

    logger.debug(
        "[%s] Starting scenario (%d phases): %s",
        case_data.name,
        len(case_data.phases),
        case_data.description,
    )

    worker = PipelineWorker(pipeline)
    session = CatnipSession(worker=worker, tracker=turn_tracker)

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    runner_bg = asyncio.create_task(runner.run())

    flow_manager: Any = None
    if _init_flow is not None:
        flow_manager = await _init_flow(worker)

    exc: Exception | None = None
    try:
        async with asyncio.timeout(overall_timeout):
            for phase_idx, phase in enumerate(case_data.phases):
                reply, triggered_calls = await _execute_phase_exchanges(
                    phase=phase,
                    phase_idx=phase_idx,
                    case_data=case_data,
                    session=session,
                    tracker=turn_tracker,
                    autoconfirm_response=autoconfirm_response,
                    is_confirmation_question=is_confirmation_question,
                )
                failure = None if phase.expect_tools is None else _match_toolcalls(phase.expect_tools, triggered_calls)
                if failure is None:
                    await _assert_post_phase(
                        phase, phase_idx, case_data.source_path, llm_judge_function, reply, flow_manager
                    )
                    logger.debug(
                        "[%s] phase=%d ok  expected=%s called=%s",
                        case_data.name,
                        phase_idx,
                        phase.expect_tools,
                        triggered_calls,
                    )
                else:
                    _raise_phase_failure(
                        case_data.source_path,
                        phase,
                        phase_idx,
                        triggered_calls,
                        reply,
                        failure,
                        case_data.max_auto_confirm,
                        is_confirmation_question,
                    )

            logger.debug("[%s] Scenario complete.", case_data.name)

    except Exception as e:  # noqa: BLE001
        exc = e
    finally:
        await worker.cancel()
        await runner_bg

    return session.transcript, exc


async def _execute_phase_exchanges(
    phase: CatnipTestPhaseData,
    phase_idx: int,
    case_data: CatnipTestCaseData,
    session: CatnipSession,
    tracker: CatnipTurnTracker,
    autoconfirm_response: str,
    is_confirmation_question: Callable[[str], bool],
) -> tuple[str, list[_ToolCall]]:
    """Send the phase utterance, auto-confirming bot questions up to the budget.

    Returns ``(final_reply, triggered_tool_calls)``.
    """
    tools_snapshot = len(tracker.tool_calls)
    reply = await session.send(phase.send, timeout=case_data.turn_timeout)
    logger.debug("[%s] phase=%d BOT: %s", case_data.name, phase_idx, reply)

    confirm_count = 0
    while confirm_count < case_data.max_auto_confirm and is_confirmation_question(reply):
        confirm_count += 1
        logger.debug(
            "[%s] phase=%d auto-confirm %d/%d -> %r",
            case_data.name,
            phase_idx,
            confirm_count,
            case_data.max_auto_confirm,
            autoconfirm_response,
        )
        reply = await session.send(autoconfirm_response, timeout=case_data.turn_timeout)
        logger.debug("[%s] phase=%d BOT (after confirm): %s", case_data.name, phase_idx, reply)

    triggered_calls = list(tracker.tool_calls[tools_snapshot:])
    return reply, triggered_calls


async def _run_reliability(
    request: pytest.FixtureRequest,
    case_data: CatnipTestCaseData,
    pipeline_factory: Callable[[], Any],
    autoconfirm_response: str,
    is_confirmation_question: Callable[[str], bool],
    llm_judge_function: Callable[[str, str], Awaitable[tuple[bool, str]]],
    overall_timeout: float,
) -> None:
    """Run the scenario N times on fresh pipelines, require at least M passes."""
    if case_data.reliability is None:
        raise pytest.UsageError(f"[{case_data.name}] _run_reliability called without a reliability config")
    cfg = case_data.reliability
    attempts: list[_Attempt] = []

    for run_idx in range(cfg.runs):
        t0 = time.monotonic()
        conv_log, exc = await _run_once(
            case_data=case_data,
            pipeline_factory=pipeline_factory,
            autoconfirm_response=autoconfirm_response,
            is_confirmation_question=is_confirmation_question,
            llm_judge_function=llm_judge_function,
            overall_timeout=overall_timeout,
        )
        attempts.append(
            _Attempt(
                run_idx=run_idx,
                passed=exc is None,
                elapsed=time.monotonic() - t0,
                conv_log=conv_log,
                error=str(exc)[:400] if exc else "",
            )
        )

    request.node._catnip_attempts = attempts

    passes = sum(1 for a in attempts if a.passed)
    if passes < cfg.min_pass:
        fail_lines = "\n".join(f"  run {a.run_idx + 1}: {a.error.splitlines()[0]}" for a in attempts if not a.passed)
        pytest.fail(
            f"[{case_data.name}] Reliability: {passes}/{cfg.runs} passed (need {cfg.min_pass}).\n"
            f"Failures:\n{fail_lines}"
        )


def _tool_matches(expectation: ToolExpectation, call: _ToolCall) -> bool:
    """Return True if *call* satisfies *expectation* (name + partial arg match)."""
    if call.name != expectation.name:
        return False
    if expectation.args is None:
        return True
    return all(key in call.arguments and call.arguments[key] == value for key, value in expectation.args.items())


def _match_toolcalls(
    expectations: list[ToolExpectation],
    triggered_calls: list[_ToolCall],
) -> str | None:
    """Validate triggered tool calls against expectations.

    Returns ``None`` on success, otherwise a human-readable failure reason.

    Rules:
      * The set of called tool names must equal the set of expected names.
      * Every expectation must be satisfied by at least one actual call, where
        ``args`` is matched as a subset (partial) of the call's arguments.
    """
    expected_names = {e.name for e in expectations}
    called_names = {c.name for c in triggered_calls}
    if expected_names != called_names:
        return f"tool name set mismatch — expected {sorted(expected_names)}, called {sorted(called_names)}"

    for expectation in expectations:
        if not any(_tool_matches(expectation, call) for call in triggered_calls):
            return f"no call matched {expectation.name!r} with args {expectation.args!r}"

    return None


def _raise_phase_failure(
    case_path: pathlib.Path,
    phase: CatnipTestPhaseData,
    phase_idx: int,
    triggered_calls: list[_ToolCall],
    reply: str,
    reason: str,
    max_auto_confirm: int,
    is_confirmation_question: Callable[[str], bool],
) -> None:
    hint = (
        "  Hint: bot didn't ask a question — check expect_tools or add more phases.\n"
        if not is_confirmation_question(reply)
        else f"  Hint: max auto-confirm budget ({max_auto_confirm}) exhausted — increase max_auto_confirm?\n"
    )
    expected = [{"name": e.name, "args": e.args} for e in (phase.expect_tools or [])]
    actual = [{"name": c.name, "args": c.arguments} for c in triggered_calls]
    raise AssertionError(
        f"\n[{case_path}] Phase {phase_idx} failed.\n"
        f"  Reason:         {reason}\n"
        f"  Sent:           {phase.send!r}\n"
        f"  Expected tools: {expected}\n"
        f"  Tools called:   {actual}\n"
        f"  Bot reply:      {reply}\n" + hint
    )


async def _assert_post_phase(
    phase: CatnipTestPhaseData,
    phase_idx: int,
    case_path: pathlib.Path,
    llm_judge_function: Callable[[str, str], Awaitable[tuple[bool, str]]],
    reply: str,
    flow_manager: object = None,  # CatnipFlowTracker or None; typed as object to avoid optional dep
) -> None:
    """Run assertions that apply after a phase completes."""
    for pattern in phase.expect_reply_contains:
        if not re.search(pattern, reply, re.IGNORECASE):
            raise AssertionError(
                f"\n[{case_path}] Phase {phase_idx}: reply doesn't match {pattern!r}.\n  Reply: {reply[:300]!r}"
            )

    judge_coros = []
    for judge_prompt in phase.expect_llm_judge or []:
        judge_coros.append(llm_judge_function(reply, judge_prompt))
    if judge_coros:
        results = await asyncio.gather(*judge_coros)
        for passed, reason in results:
            if not passed:
                raise AssertionError(
                    f"\n[{case_path}] Phase {phase_idx}: LLM judge failed.\n"
                    f"  Judge prompt: {judge_prompt!r}\n"
                    f"  Bot reply:   {reply!r}\n"
                    f"  Judge reason: {reason!r}"
                )

    if phase.expect_flow_state is not None:
        if flow_manager is None:
            raise pytest.UsageError(
                f"[{case_path}] Phase {phase_idx}: 'expect_flow_state' requires the "
                f"'catnip_pipeline' fixture to return a 'CatnipFlowBundle'. "
                f"See pytest_catnip.flows for instructions."
            )
        actual = getattr(flow_manager, "current_node", None)
        if actual != phase.expect_flow_state:
            history = getattr(flow_manager, "node_history", "unavailable")
            raise AssertionError(
                f"\n[{case_path}] Phase {phase_idx}: expected flow state {phase.expect_flow_state!r}, "
                f"got {actual!r}.\n  node_history: {history}"
            )
