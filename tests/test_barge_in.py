from typing import Any

import pytest
from pydantic import ValidationError
from pytest_catnip.harness import CatnipSession, _find_cut
from pytest_catnip.models import BargeIn, CatnipTestCaseData, CatnipTestPhaseData

# ---------------------------------------------------------------- models


def _case(*phases: dict[str, Any]) -> CatnipTestCaseData:
    return CatnipTestCaseData(name="x", phases=list(phases), source_path="x.yaml")  # type: ignore[arg-type]


def test_barge_in_after_regex() -> None:
    assert BargeIn(after="foo|bar").after == "foo|bar"


def test_barge_in_after_words() -> None:
    assert BargeIn(after_words=3).after_words == 3


def test_barge_in_true_means_first_word() -> None:
    phase = CatnipTestPhaseData(send="hi", barge_in=True)  # type: ignore[arg-type]
    assert phase.barge_in == BargeIn(after_words=1)


def test_barge_in_false_and_missing_are_none() -> None:
    assert CatnipTestPhaseData(send="hi", barge_in=False).barge_in is None  # type: ignore[arg-type]
    assert CatnipTestPhaseData(send="hi").barge_in is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"after": "x", "after_words": 2}, "exactly one"),
        ({}, "exactly one"),
        ({"after_words": 0}, "greater than or equal to 1"),
        ({"after_words": -3}, "greater than or equal to 1"),
        ({"after": "("}, "not a valid regex"),
        ({"after": "   "}, "must not be empty"),
        ({"after": ""}, "must not be empty"),
    ],
)
def test_barge_in_invalid(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        BargeIn(**kwargs)


def test_barge_in_unknown_key() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        BargeIn(after_word=3)  # type: ignore[call-arg]


def test_barge_in_on_first_phase_rejected() -> None:
    with pytest.raises(ValidationError, match="first phase"):
        _case({"send": "a", "barge_in": True})


def test_barge_in_on_later_phase_accepted() -> None:
    case = _case({"send": "a"}, {"send": "b", "barge_in": {"after_words": 2}})
    assert case.phases[1].barge_in == BargeIn(after_words=2)


# ---------------------------------------------------------------- _find_cut


def test_find_cut_regex_is_case_insensitive_and_cuts_at_match_end() -> None:
    assert _find_cut(BargeIn(after="hello"), "Oh HELLO there", at_end=False) == len("Oh HELLO")


def test_find_cut_regex_not_found() -> None:
    assert _find_cut(BargeIn(after="zzz"), "Oh hello there", at_end=True) is None


def test_find_cut_words_waits_for_whitespace_mid_stream() -> None:
    barge_in = BargeIn(after_words=2)
    assert _find_cut(barge_in, "one tw", at_end=False) is None
    assert _find_cut(barge_in, "one two", at_end=False) is None  # "two" could still grow
    assert _find_cut(barge_in, "one two ", at_end=False) == len("one two")


def test_find_cut_words_at_end_counts_last_word() -> None:
    assert _find_cut(BargeIn(after_words=2), "one two", at_end=True) == len("one two")


def test_find_cut_words_too_short() -> None:
    assert _find_cut(BargeIn(after_words=5), "one two", at_end=True) is None


class _StubContext:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages

    def get_messages(self) -> list[dict[str, Any]]:
        return self._messages


class _StubAggregator:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.context = _StubContext(messages)


def _session_with_context(messages: list[dict[str, Any]]) -> CatnipSession:
    return CatnipSession(
        worker=None,  # type: ignore[arg-type]
        tracker=None,  # type: ignore[arg-type]
        assistant_aggregator=_StubAggregator(messages),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_verify_interrupted_context_accepts_spoken_text() -> None:
    session = _session_with_context([{"role": "assistant", "content": "one,  two\nthree"}])
    assert await session.verify_interrupted_context("one, two three", timeout=0.1) is None


@pytest.mark.asyncio
async def test_verify_interrupted_context_accepts_content_parts() -> None:
    session = _session_with_context([{"role": "assistant", "content": [{"type": "text", "text": "one two"}]}])
    assert await session.verify_interrupted_context("one two", timeout=0.1) is None


@pytest.mark.asyncio
async def test_verify_interrupted_context_reports_missing_assistant_message() -> None:
    session = _session_with_context([{"role": "user", "content": "hi"}])
    problem = await session.verify_interrupted_context("one two", timeout=0.1)
    assert problem is not None
    assert "None" in problem


@pytest.mark.asyncio
async def test_verify_interrupted_context_reports_untruncated_text() -> None:
    session = _session_with_context([{"role": "assistant", "content": "one two three four"}])
    problem = await session.verify_interrupted_context("one two", timeout=0.1)
    assert problem is not None
    assert "one two three four" in problem
