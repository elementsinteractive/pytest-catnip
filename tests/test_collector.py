import pathlib

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture
def cases_dir(pytester: pytest.Pytester) -> pathlib.Path:
    """Create a temporary cases directory and register it in the sub-session config."""
    d = pytester.mkdir("cases")
    pytester.makeini(
        "[pytest]\n"
        "catnip_cases_directory = cases\n"
        "addopts = -p no:cov\n"
        "asyncio_mode = strict\n"
        "asyncio_default_fixture_loop_scope = function\n"
    )
    return d


def test_empty_yaml_produces_no_tests(cases_dir: pathlib.Path, pytester: pytest.Pytester) -> None:
    """An empty YAML file is silently ignored — no tests collected, no errors."""
    (cases_dir / "test_empty.yaml").write_text("")
    result = pytester.runpytest("--collect-only")
    assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED
    result.stdout.no_fnmatch_line("*ERROR*")


def test_malformed_yaml_reports_usage_error(cases_dir: pathlib.Path, pytester: pytest.Pytester) -> None:
    """Invalid YAML syntax raises a UsageError mentioning the filename."""
    (cases_dir / "test_bad.yaml").write_text("{ unclosed: [")
    result = pytester.runpytest("--collect-only")
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.stdout.fnmatch_lines(["*Malformed YAML*test_bad.yaml*"])


def test_unknown_field_reports_usage_error(cases_dir: pathlib.Path, pytester: pytest.Pytester) -> None:
    """An unrecognised field raises a UsageError mentioning 'Unknown field'."""
    (cases_dir / "test_bad.yaml").write_text("name: my-test\nphases:\n  - send: hi\nunknown_field: oops\n")
    result = pytester.runpytest("--collect-only")
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.stdout.fnmatch_lines(["*Unknown field*"])


def test_missing_required_field_reports_usage_error(cases_dir: pathlib.Path, pytester: pytest.Pytester) -> None:
    """A YAML missing the required 'name' field raises a UsageError."""
    (cases_dir / "test_missing.yaml").write_text("phases:\n  - send: hi\n")
    result = pytester.runpytest("--collect-only")
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.stdout.fnmatch_lines(["*Invalid test case schema*"])


def test_valid_yaml_collects_test_function(cases_dir: pathlib.Path, pytester: pytest.Pytester) -> None:
    """A valid YAML is collected under the correct test function name."""
    (cases_dir / "test_example.yaml").write_text(
        "name: my-example\nphases:\n  - send: hello\n    expect_reply_contains:\n      - hi\n"
    )
    result = pytester.runpytest("--collect-only")
    assert result.ret == pytest.ExitCode.OK
    result.stdout.fnmatch_lines(["*test_my_example*"])


def test_name_without_test_prefix_gets_prefix(cases_dir: pathlib.Path, pytester: pytest.Pytester) -> None:
    """A test whose 'name' doesn't start with 'test_' still gets a valid function name."""
    (cases_dir / "test_no_prefix.yaml").write_text("name: no-prefix-here\nphases:\n  - send: hello\n")
    result = pytester.runpytest("--collect-only")
    assert result.ret == pytest.ExitCode.OK
    result.stdout.fnmatch_lines(["*test_no_prefix_here*"])


def test_invalid_cases_directory_raises_usage_error(pytester: pytest.Pytester) -> None:
    """A missing or invalid catnip_cases_directory raises a UsageError at configure time."""
    pytester.makeini("[pytest]\ncatnip_cases_directory = /nonexistent/path/that/does/not/exist\naddopts = -p no:cov\n")
    result = pytester.runpytest("--collect-only")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*Invalid pytest-catnip cases directory*"])


def test_markers_applied_to_test_function(cases_dir: pathlib.Path, pytester: pytest.Pytester) -> None:
    """A YAML with markers: causes the generated test function to have those markers applied."""
    (cases_dir / "test_marked.yaml").write_text("name: marked-test\nmarkers:\n  - slow\nphases:\n  - send: hello\n")
    result = pytester.runpytest("--collect-only", "-m", "not slow")
    assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED


def test_missing_turn_tracker_raises_usage_error(cases_dir: pathlib.Path, pytester: pytest.Pytester) -> None:
    """A pipeline without CatnipTurnTracker raises a UsageError when the test runs."""
    (cases_dir / "test_no_tracker.yaml").write_text("name: no-tracker\nphases:\n  - send: hello\n")
    pytester.makepyfile(
        conftest="""
        import pytest
        from pipecat.pipeline.pipeline import Pipeline

        @pytest.fixture
        def catnip_pipeline():
            return lambda: Pipeline([])
        """
    )
    result = pytester.runpytest()
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*CatnipTurnTracker*"])
