import pathlib
from collections.abc import Callable, Generator

import pytest
from _pytest.terminal import TerminalReporter
from pipecat.pipeline.pipeline import Pipeline
from pytest_catnip.collector import CatnipFileCollector
from pytest_catnip.harness import CatnipTurnTracker, TurnLogKind
from pytest_catnip.runner import _Attempt


def pytest_addoption(parser: pytest.Parser, pluginmanager: pytest.PytestPluginManager) -> None:
    parser.addoption(
        "--catnip-show-conversation",
        action="store_true",
        default=None,
        help="Print the conversation turns for each test case after it runs (for debugging failures).",
    )
    parser.addini(
        "catnip_show_conversation",
        "Print the conversation turns for each test case after it runs (for debugging failures).",
        type="bool",
        default=False,
    )
    parser.addoption(
        "--catnip-reliability",
        action="store_true",
        default=None,
        help="Enable reliability mode: run scenarios with a 'reliability' block N times and assert M passes.",
    )
    parser.addini(
        "catnip_reliability",
        "Enable reliability mode: run scenarios with a 'reliability' block N times and assert M passes.",
        type="bool",
        default=False,
    )
    parser.addoption(
        "--catnip-overall-timeout",
        type=float,
        default=None,
        help="Maximum seconds for a single scenario run across all phases. Default: 120.",
    )
    parser.addini(
        "catnip_overall_timeout",
        "Maximum seconds for a single scenario run across all phases.",
        type="string",
        default="120",
    )
    parser.addini(
        "catnip_cases_directory",
        "Directory where pytest-catnip should look for test case files.",
        default="tests/cases",
    )


def pytest_configure(config: pytest.Config) -> None:
    cases_dir = config.getini("catnip_cases_directory")
    if not cases_dir or not pathlib.Path(cases_dir).is_dir():
        raise pytest.UsageError(f"Invalid pytest-catnip cases directory: {cases_dir!r}")

    config.pluginmanager.register(_CatnipConversationPlugin(config), "catnip-conversation")


def pytest_collect_file(parent: pytest.Collector, file_path: pathlib.Path) -> pytest.File | None:
    resolved_path = file_path.resolve()
    cases_dir: str = parent.config.getini("catnip_cases_directory")
    if (
        resolved_path.is_relative_to(pathlib.Path(cases_dir).resolve())
        and resolved_path.suffix in (".yaml", ".yml")
        and resolved_path.name.startswith("test_")
    ):
        return CatnipFileCollector.from_parent(parent, path=file_path)
    return None


@pytest.fixture
def catnip_pipeline() -> Callable[[], Pipeline]:
    """Fixture that provides the pipeline factory used by each test scenario.

    The factory is called once per run (including each reliability run), so every
    run gets a completely fresh set of stateful pipecat objects.

    Override this fixture in your ``conftest.py``. Two return types are supported:

    **Plain Pipeline** — for bots without pipecat-flows::

        @pytest.fixture
        def catnip_pipeline() -> Callable[[], Pipeline]:
            def factory() -> Pipeline:
                context = LLMContext(messages=[...])
                llm = OpenAILLMService(...)
                ctx_agg = LLMContextAggregatorPair(context)
                turn_tracker = CatnipTurnTracker()
                return Pipeline([ctx_agg.user(), llm, turn_tracker, ctx_agg.assistant()])
            return factory

    **CatnipFlowBundle** — for bots that use pipecat-flows and want
    ``expect_flow_state`` assertions in their YAML test cases::

        from pytest_catnip.flows import CatnipFlowBundle, CatnipFlowTracker

        @pytest.fixture
        def catnip_pipeline() -> Callable[[], CatnipFlowBundle]:
            def factory() -> CatnipFlowBundle:
                context = LLMContext(messages=[...])
                llm = OpenAILLMService(...)
                ctx_agg = LLMContextAggregatorPair(context)
                turn_tracker = CatnipTurnTracker()
                pipeline = Pipeline([ctx_agg.user(), llm, turn_tracker, ctx_agg.assistant()])

                async def init_flow(worker: PipelineWorker) -> CatnipFlowTracker:
                    tracker = CatnipFlowTracker(worker=worker, llm=llm, context_aggregator=ctx_agg)
                    await tracker.initialize(create_initial_node())
                    return tracker

                return CatnipFlowBundle(pipeline=pipeline, init_flow=init_flow)
            return factory

    ``CatnipTurnTracker`` must always be present in the pipeline so pytest-catnip
    can track turns and tool calls. Do not add STT or TTS processors.
    """

    def _default_factory() -> Pipeline:
        return Pipeline([CatnipTurnTracker()])

    return _default_factory


@pytest.fixture
def autoconfirm_response() -> str:
    """Response to give the bot when it asks for confirmation (if the test case allows auto-confirming).

    Override this fixture to use a different response.
    """
    return "Yes."


@pytest.fixture
def is_confirmation_question() -> Callable[[str], bool]:
    """Predicate that returns True when the bot's reply is asking for confirmation.

    Override this fixture to match your bot's confirmation phrasing.
    The default treats any reply ending in '?' as a confirmation question.
    """

    def _default(text: str) -> bool:
        return text.rstrip().endswith("?")

    return _default


class _CatnipConversationPlugin:
    """Prints conversation logs inline in the terminal when --catnip-show-conversation is passed."""

    _ENTRY_LABELS: dict[TurnLogKind, str] = {
        TurnLogKind.USER: "USER",
        TurnLogKind.BOT: " BOT",
        TurnLogKind.TOOL: "TOOL",
    }
    _ENTRY_MARKUP: dict[TurnLogKind, dict[str, bool]] = {
        TurnLogKind.USER: {"blue": True, "bold": True},
        TurnLogKind.BOT: {"purple": True, "bold": True},
        TurnLogKind.TOOL: {"light": True},
    }

    def __init__(self, config: pytest.Config) -> None:
        self._config = config
        self._pending: dict[str, list[_Attempt]] = {}

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(
        self, item: pytest.Item, call: pytest.CallInfo[None]
    ) -> Generator[None, pytest.TestReport, pytest.TestReport]:
        report = yield
        if report.when == "call":
            attempts: list[_Attempt] | None = getattr(item, "_catnip_attempts", None)
            if attempts is not None:
                self._pending[report.nodeid] = attempts
        return report

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "call":
            return
        cli = self._config.getoption("--catnip-show-conversation", default=None)
        show = cli if cli is not None else self._config.getini("catnip_show_conversation")
        if not show:
            return
        attempts = self._pending.pop(report.nodeid, None)
        if attempts is None:
            return
        tr: TerminalReporter | None = self._config.pluginmanager.get_plugin("terminalreporter")
        if tr is None:
            return
        if len(attempts) == 1:
            self._print_single(tr, attempts[0])
        else:
            self._print_reliability(tr, attempts)

    def _print_single(self, tr: TerminalReporter, attempt: _Attempt) -> None:
        tr.write_sep("-", "conversation")
        for entry_type, text in attempt.conv_log:
            label = self._ENTRY_LABELS[entry_type]
            markup = self._ENTRY_MARKUP.get(entry_type, {})
            tr.write_line(f"  {label}  {text}", **markup)
        tr.write_sep("-", "")

    def _print_reliability(self, tr: TerminalReporter, attempts: list[_Attempt]) -> None:
        n = len(attempts)
        tr.write_sep("-", f"reliability ({n} runs)")
        for a in attempts:
            status = "PASSED" if a.passed else "FAILED"
            line = f"  run {a.run_idx + 1}/{n}  {status}  {a.elapsed:.1f}s"
            tr.write_line(line, green=a.passed, red=not a.passed, bold=True)
            for entry_type, text in a.conv_log:
                label = self._ENTRY_LABELS[entry_type]
                tr.write_line(f"    {label}  {text}")
            if not a.passed and a.error:
                tr.write_line(f"    ERROR  {a.error.splitlines()[0]}", red=True)
        passes = sum(1 for a in attempts if a.passed)
        tr.write_line(f"  {passes}/{n} passed", bold=True)
        tr.write_sep("-", "")
