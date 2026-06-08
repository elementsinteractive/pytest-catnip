import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, override

from pipecat.frames.frames import (
    Frame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

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


# Backwards-compatible alias
LogEntryType = TurnLogKind


class CatnipTurnTracker(FrameProcessor):
    """Frame processor that signals when a complete LLM turn is ready.

    Place between the LLM and the assistant context aggregator in the tested
    pipeline. Completed turn text is placed on an internal queue; callers
    await ``next_reply()`` to receive each response in order.
    """

    def __init__(self) -> None:
        super().__init__()
        self._reply_queue: asyncio.Queue[str] = asyncio.Queue()
        self._inflight: int = 0
        self._chunks: list[str] = []
        self.tool_calls: list[_ToolCall] = []

    def prepare_for_input(self) -> None:
        """Reset per-turn accumulation state before sending a new user message."""
        self._chunks = []
        self._inflight = 0

    @override
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:  # noqa: C901
        await super().process_frame(frame, direction)

        if isinstance(frame, FunctionCallsStartedFrame):
            names = [f.function_name for f in frame.function_calls]
            self._inflight += len(frame.function_calls)
            for fc in frame.function_calls:
                self.tool_calls.append(_ToolCall(name=fc.function_name))
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

        elif isinstance(frame, LLMTextFrame):
            self._chunks.append(frame.text)

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._inflight <= 0:
                # All tool calls resolved — this is the final response for this turn.
                text = "".join(self._chunks).strip()
                self._chunks = []
                await self._reply_queue.put(text)
                logger.debug("[tracker] turn complete, queued reply (len=%d)", len(text))
            else:
                # LLM emitted an intermediate response before tool results arrive.
                # Drop it and wait for the post-tool continuation.
                logger.debug("[tracker] intermediate response discarded (inflight=%d)", self._inflight)
                self._chunks = []

        await self.push_frame(frame, direction)

    async def next_reply(self, timeout: float = DEFAULT_TURN_TIMEOUT) -> str:
        """Wait for the next completed turn and return the reply text.

        Raises ``TimeoutError`` if no reply arrives within *timeout* seconds.
        """
        return await asyncio.wait_for(self._reply_queue.get(), timeout=timeout)


class CatnipSession:
    """Drive a headless pipeline conversation turn by turn.

    Inject user messages via ``send()`` and receive the bot's reply. A full
    transcript of the exchange is accumulated in ``transcript``.
    """

    def __init__(self, worker: PipelineWorker, tracker: CatnipTurnTracker) -> None:
        self._worker = worker
        self._tracker = tracker
        self.transcript: list[tuple[TurnLogKind, str]] = []

    @property
    def log(self) -> list[tuple[TurnLogKind, str]]:
        """Alias for ``transcript``."""
        return self.transcript

    async def send(self, text: str, timeout: float = DEFAULT_TURN_TIMEOUT) -> str:
        """Inject *text* as a user utterance and block until the bot replies.

        Returns the bot's reply text.
        """
        t0 = time.monotonic()
        logger.debug("[session] USER -> %r", text)
        self.transcript.append((TurnLogKind.USER, text))
        self._tracker.prepare_for_input()
        tools_before = len(self._tracker.tool_calls)
        await self._worker.queue_frames(
            [
                TranscriptionFrame(text=text, user_id="tester", timestamp=""),
                UserStoppedSpeakingFrame(),
            ]
        )
        reply = await self._tracker.next_reply(timeout=timeout)
        elapsed = time.monotonic() - t0
        logger.debug("[session] BOT  <- (%.1fs) %r", elapsed, reply)
        self.transcript.append((TurnLogKind.BOT, reply))
        for call in self._tracker.tool_calls[tools_before:]:
            self.transcript.append((TurnLogKind.TOOL, call.name))
        return reply
