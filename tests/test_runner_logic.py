from pytest_catnip.harness import _ToolCall
from pytest_catnip.models import ToolExpectation
from pytest_catnip.runner import CatnipAssertionManager


async def mock_llm_judge(text: str, judge_prompt: str) -> tuple[bool, str]:
    return True, "mocked"


def test_tool_matches_wrong_name() -> None:
    exp = ToolExpectation(name="foo")
    call = _ToolCall(name="bar")
    assert CatnipAssertionManager.tool_matches(exp, call) is False


def test_tool_matches_correct_name_no_args() -> None:
    exp = ToolExpectation(name="foo")
    call = _ToolCall(name="foo", arguments={"x": 1})
    assert CatnipAssertionManager.tool_matches(exp, call) is True


def test_tool_matches_args_subset_match() -> None:
    exp = ToolExpectation(name="foo", args={"size": "large"})
    call = _ToolCall(name="foo", arguments={"size": "large", "crust": "thin"})
    assert CatnipAssertionManager.tool_matches(exp, call) is True


def test_tool_matches_args_wrong_value() -> None:
    exp = ToolExpectation(name="foo", args={"size": "large"})
    call = _ToolCall(name="foo", arguments={"size": "small"})
    assert CatnipAssertionManager.tool_matches(exp, call) is False


def test_tool_matches_args_missing_key() -> None:
    exp = ToolExpectation(name="foo", args={"size": "large"})
    call = _ToolCall(name="foo", arguments={})
    assert CatnipAssertionManager.tool_matches(exp, call) is False


def test_match_toolcalls_success() -> None:
    exps = [ToolExpectation(name="foo")]
    calls = [_ToolCall(name="foo")]
    assert CatnipAssertionManager(llm_judge_function=mock_llm_judge).match_toolcalls(exps, calls) is None


def test_match_toolcalls_name_mismatch() -> None:
    exps = [ToolExpectation(name="foo")]
    calls = [_ToolCall(name="bar")]
    result = CatnipAssertionManager(llm_judge_function=mock_llm_judge).match_toolcalls(exps, calls)
    assert result is not None
    assert "foo" in result
    assert "bar" in result


def test_match_toolcalls_extra_call() -> None:
    exps = [ToolExpectation(name="foo")]
    calls = [_ToolCall(name="foo"), _ToolCall(name="bar")]
    result = CatnipAssertionManager(llm_judge_function=mock_llm_judge).match_toolcalls(exps, calls)
    assert result is not None
    assert "mismatch" in result


def test_match_toolcalls_missing_call() -> None:
    exps = [ToolExpectation(name="foo"), ToolExpectation(name="bar")]
    calls = [_ToolCall(name="foo")]
    result = CatnipAssertionManager(llm_judge_function=mock_llm_judge).match_toolcalls(exps, calls)
    assert result is not None


def test_match_toolcalls_args_no_match() -> None:
    exps = [ToolExpectation(name="foo", args={"size": "large"})]
    calls = [_ToolCall(name="foo", arguments={"size": "small"})]
    result = CatnipAssertionManager(llm_judge_function=mock_llm_judge).match_toolcalls(exps, calls)
    assert result is not None
    assert "foo" in result


def test_match_toolcalls_empty_both() -> None:
    assert CatnipAssertionManager(llm_judge_function=mock_llm_judge).match_toolcalls([], []) is None
