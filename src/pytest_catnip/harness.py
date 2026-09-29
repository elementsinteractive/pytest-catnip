import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, override

from pipecat.frames.frames import (
    Frame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pytest_catnip.models import BargeIn

logger = logging.getLogger(__name__)

DEFAULT_TURN_TIMEOUT = 15  # seconds


@dataclass
class _ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class TurnLogKind(StrEnum):
    USER = "USER"
    BOT = "BOT"
    TOOL = "TOOL"


def _last_assistant_text(aggregator: LLMAssistantAggregator) -> str | None:
    """Whitespace-normalised text of the last assistant message in the aggregator's context."""
    for message in reversed(aggregator.context.get_messages()):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return " ".join(str(content or "").split())
    return None


class BargeInMissedError(AssertionError):
    """The bot's reply ended before the barge-in trigger was reached."""

    def __init__(self, reply: str, barge_in: BargeIn, reason: str) -> None:
        super().__init__(reason)
        self.reply = reply
        self.barge_in = barge_in
        self.reason = reason


@dataclass
class _TurnResult:
    text: str
    interrupted: bool = False
    barge_in_miss: str | None = None


def _find_cut(barge_in: BargeIn, text: str, *, at_end: bool) -> int | None:
    """Return the index in *text* right after the barge-in trigger, or ``None`` if not reached yet.

    Mid-stream, the last word is only counted once whitespace follows it (it may still be growing).
    At the end of the reply every word is complete.
    """
    if barge_in.after is not None:
        match = re.search(barge_in.after, text, re.IGNORECASE)
        return match.end() if match else None
    words = list(re.finditer(r"\S+", text))
    if barge_in.after_words is None or len(words) < barge_in.after_words:
        return None
    last = words[barge_in.after_words - 1]
    if last.end() == len(text) and not at_end:
        return None
    return last.end()


class CatnipTurnTracker(FrameProcessor):
    """Frame processor that signals when a complete LLM turn is ready.

    Place between the LLM and the assistant context aggregator in the tested
    pipeline. Completed turn text is placed on an internal queue; callers
    await ``next_reply()`` to receive each response in order.
    """

    def __init__(self) -> None:
        super().__init__()
        self._interrupter: Callable[[], Awaitable[None]] | None = None
        self._interrupted_response_ended = asyncio.Event()
        self._reply_queue: asyncio.Queue[_TurnResult] = asyncio.Queue()
        self._inflight: int = 0
        self._chunks: list[str] = []
        self.tool_calls: list[_ToolCall] = []
        self._barge_in: BargeIn | None = None
        self._swallowing: bool = False
        self._assistant_aggregator: LLMAssistantAggregator | None = None

    def watch_assistant_aggregator(self, aggregator: LLMAssistantAggregator | None) -> None:
        """Let barge-in wait for *aggregator* to consume the "spoken" text before interrupting it.

        System frames such as ``InterruptionFrame`` overtake queued data frames, so without this the
        aggregator would be interrupted before it has seen the text that was supposedly spoken.
        """
        self._assistant_aggregator = aggregator

    async def _wait_until_aggregated(self, spoken: str, timeout: float = 2.0) -> None:
        if self._assistant_aggregator is None:
            return
        expected = " ".join(spoken.split())
        try:
            async with asyncio.timeout(timeout):
                while " ".join(self._assistant_aggregator.aggregation_string().split()) != expected:
                    await asyncio.sleep(0.005)
        except TimeoutError:
            logger.warning("[tracker] assistant aggregator never consumed the spoken text %r", expected)

    def set_interrupter(self, interrupter: Callable[[], Awaitable[None]] | None) -> None:
        """Set how a barge-in enters the pipeline.

        Real interruptions start at the source (the transport) and flow downstream through every
        processor. Without an *interrupter* the tracker pushes the frame both ways from its own position.
        """
        self._interrupter = interrupter

    def arm_barge_in(self, barge_in: BargeIn | None) -> None:
        """Interrupt the next bot reply once *barge_in*'s trigger is reached.

        Streamed text is simulated as "spoken" up to the trigger point; anything after it is
        dropped and an interruption is broadcast through the pipeline. Call before ``prepare_for_input``.
        """
        self._barge_in = barge_in

    def prepare_for_input(self) -> None:
        """Reset per-turn accumulation state before sending a new user message."""
        self._chunks = []
        self._inflight = 0

    async def _interrupt_at(self, cut: int, *, response_ended: bool = False) -> None:
        """Finish the current reply at *cut*, then interrupt the rest of the pipeline."""
        self._interrupted_response_ended.clear()
        if response_ended:
            self._interrupted_response_ended.set()
        text = "".join(self._chunks)[:cut].strip()
        self._chunks = []
        self._barge_in = None
        self._swallowing = True
        logger.debug("[tracker] barge-in fired, reply cut at %d chars", len(text))
        # Run apart from this frame's processing: the interruption cancels the tracker's own process task.
        self.create_task(self._deliver_interruption(text), name="catnip-barge-in")

    async def _deliver_interruption(self, spoken: str) -> None:
        """Interrupt the pipeline; the turn's result is only released once everything has settled.

        Releasing it last keeps the next user message from racing the tail of this interruption.
        """
        try:
            await self._wait_until_aggregated(spoken)
            if self._interrupter is not None:
                await self._interrupter()
            else:
                await self.push_frame(InterruptionFrame(), FrameDirection.UPSTREAM)
                await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
            # A cancelled LLM still ends its response, sometimes late. If that End frame crossed the next
            # response's Start it would be taken for that response's end, so wait for it (bounded: a
            # service may not send one).
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._interrupted_response_ended.wait(), timeout=2.0)
        finally:
            await self._reply_queue.put(_TurnResult(spoken, interrupted=True))

    async def _gate_text(self, frame: LLMTextFrame, barge_in: BargeIn) -> None:
        """Forward only the part of *frame* that comes before the barge-in trigger."""
        before = "".join(self._chunks)
        self._chunks.append(frame.text)
        cut = _find_cut(barge_in, "".join(self._chunks), at_end=False)
        if cut is None:
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)
            return
        head = frame.text[: max(cut - len(before), 0)]
        if head:
            await self.push_frame(LLMTextFrame(text=head), FrameDirection.DOWNSTREAM)
        await self._interrupt_at(cut)

    async def _gate_end(self, frame: LLMFullResponseEndFrame, barge_in: BargeIn) -> None:
        text = "".join(self._chunks)
        cut = _find_cut(barge_in, text, at_end=True)
        if cut is not None:
            await self._interrupt_at(cut, response_ended=True)
            return
        self._barge_in = None
        self._chunks = []
        if not text.strip():
            reason = "the bot streamed no text to interrupt"
        elif barge_in.after is not None:
            reason = f"barge_in.after {barge_in.after!r} not found in the bot reply"
        else:
            reason = f"barge_in.after_words is {barge_in.after_words} but the bot reply had {len(text.split())} words"
        await self._reply_queue.put(_TurnResult(text.strip(), barge_in_miss=reason))
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    @override
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:  # noqa: C901
        await super().process_frame(frame, direction)

        if isinstance(frame, FunctionCallsStartedFrame):
            names = [f.function_name for f in frame.function_calls]
            self._inflight += len(frame.function_calls)
            for fc in frame.function_calls:
                self.tool_calls.append(_ToolCall(name=fc.function_name, arguments=dict(fc.arguments or {})))
            logger.debug("[tracker] tool calls started: %s (inflight=%d)", names, self._inflight)

        elif isinstance(frame, FunctionCallResultFrame):
            is_final = frame.properties.is_final if frame.properties else True
            if is_final:
                self._inflight = max(0, self._inflight - 1)
            logger.debug(
                "[tracker] tool result: fn=%s is_final=%s inflight=%d",
                frame.function_name,
                is_final,
                self._inflight,
            )

        elif self._swallowing and isinstance(frame, LLMFullResponseStartFrame):
            # A new response begins: the interrupted one is over.
            self._swallowing = False

        elif self._swallowing and isinstance(frame, LLMTextFrame | LLMFullResponseEndFrame):
            # The rest of an interrupted reply is never "spoken". Its end frame may arrive after the
            # next user message was sent, so keep swallowing until it (or the next response) shows up.
            self._swallowing = not isinstance(frame, LLMFullResponseEndFrame)
            if isinstance(frame, LLMFullResponseEndFrame):
                self._interrupted_response_ended.set()
            return

        elif isinstance(frame, LLMTextFrame):
            if self._barge_in is not None:
                await self._gate_text(frame, self._barge_in)
                return
            self._chunks.append(frame.text)

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._inflight <= 0:
                # All tool calls resolved — this is the final response for this turn.
                if self._barge_in is not None:
                    await self._gate_end(frame, self._barge_in)
                    return
                text = "".join(self._chunks).strip()
                self._chunks = []
                await self._reply_queue.put(_TurnResult(text))
                logger.debug("[tracker] turn complete, queued reply (len=%d)", len(text))
            else:
                # LLM emitted an intermediate response before tool results arrive.
                # Drop it and wait for the post-tool continuation.
                logger.debug("[tracker] intermediate response discarded (inflight=%d)", self._inflight)
                self._chunks = []

        await self.push_frame(frame, direction)

    async def next_turn(self, timeout: float = DEFAULT_TURN_TIMEOUT) -> _TurnResult:
        """Wait for the next completed (or interrupted) turn.

        Raises ``TimeoutError`` if no reply arrives within *timeout* seconds.
        """
        return await asyncio.wait_for(self._reply_queue.get(), timeout=timeout)

    async def next_reply(self, timeout: float = DEFAULT_TURN_TIMEOUT) -> str:
        """Wait for the next completed turn and return the reply text.

        Raises ``TimeoutError`` if no reply arrives within *timeout* seconds.
        """
        return (await self.next_turn(timeout=timeout)).text


class CatnipSession:
    """Drive a headless pipeline conversation turn by turn.

    Inject user messages via ``send()`` and receive the bot's reply. A full
    transcript of the exchange is accumulated in ``transcript``.
    """

    def __init__(
        self,
        worker: PipelineWorker,
        tracker: CatnipTurnTracker,
        assistant_aggregator: LLMAssistantAggregator | None = None,
    ) -> None:
        self._worker = worker
        self._tracker = tracker
        self._assistant_aggregator = assistant_aggregator
        self.transcript: list[tuple[TurnLogKind, str]] = []
        self.last_reply_interrupted: bool = False

    @property
    def log(self) -> list[tuple[TurnLogKind, str]]:
        """Alias for ``transcript``."""
        return self.transcript

    @property
    def formatted_log(self) -> str:
        """Return a human-readable formatted transcript."""
        return "\n".join(f"[{kind.name}] {text}" for kind, text in self.transcript)

    async def verify_interrupted_context(self, spoken: str, timeout: float = 2.0) -> str | None:
        """Check the LLM context ends with exactly the *spoken* part of an interrupted reply.

        Returns ``None`` when it does (or there is no assistant aggregator to inspect), otherwise
        a description of what the context holds instead. The aggregator commits on the
        interruption asynchronously, so this polls briefly.
        """
        if self._assistant_aggregator is None:
            return None
        expected = " ".join(spoken.split())
        found: str | None = None
        try:
            async with asyncio.timeout(timeout):
                while True:
                    found = _last_assistant_text(self._assistant_aggregator)
                    if found == expected:
                        return None
                    await asyncio.sleep(0.02)
        except TimeoutError:
            return f"the last assistant message in the LLM context is {found!r}, expected {expected!r}"

    async def send(
        self,
        text: str,
        timeout: float = DEFAULT_TURN_TIMEOUT,
        *,
        interrupt_reply_with: BargeIn | None = None,
    ) -> str:
        """Inject *text* as a user utterance and block until the bot replies.

        If *interrupt_reply_with* is given, the bot's reply is cut at that trigger (see
        ``CatnipTurnTracker.arm_barge_in``) and ``last_reply_interrupted`` is set. Raises
        ``BargeInMissedError`` if the reply ended before the trigger.

        Returns the bot's reply text (the truncated text when interrupted).
        """
        t0 = time.monotonic()
        logger.debug("[session] USER -> %r", text)
        self.transcript.append((TurnLogKind.USER, text))
        self._tracker.arm_barge_in(interrupt_reply_with)
        self._tracker.prepare_for_input()
        tools_before = len(self._tracker.tool_calls)
        await self._worker.queue_frames(
            [
                TranscriptionFrame(text=text, user_id="tester", timestamp=""),
                UserStoppedSpeakingFrame(),
            ]
        )
        turn = await self._tracker.next_turn(timeout=timeout)
        reply = turn.text
        self.last_reply_interrupted = turn.interrupted
        elapsed = time.monotonic() - t0
        logger.debug("[session] BOT  <- (%.1fs) %r", elapsed, reply)
        self.transcript.append((TurnLogKind.BOT, f"{reply} [interrupted by user]" if turn.interrupted else reply))
        for call in self._tracker.tool_calls[tools_before:]:
            entry = f"{call.name} {call.arguments}" if call.arguments else call.name
            self.transcript.append((TurnLogKind.TOOL, entry))
        if turn.barge_in_miss is not None and interrupt_reply_with is not None:
            raise BargeInMissedError(reply, interrupt_reply_with, turn.barge_in_miss)
        return reply
