"""Continuous batching scheduler with async request queue and preemption."""
from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch

from src.config import EngineConfig

logger = logging.getLogger(__name__)


class SeqStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


@dataclass
class SequenceGroup:
    """Represents one request with its sequence state."""

    request_id: str
    input_ids: torch.Tensor
    max_tokens: int
    temperature: float = 0.7
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    generated_tokens: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    completion_future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    arrival_time: float = field(default_factory=time.time)

    @property
    def total_len(self) -> int:
        return self.input_ids.shape[1] + len(self.generated_tokens)

    @property
    def prompt_len(self) -> int:
        return self.input_ids.shape[1]


@dataclass
class SchedulerOutput:
    """Output of the scheduler's schedule() call."""

    seq_groups: list[SequenceGroup] = field(default_factory=list)
    preempted: list[SequenceGroup] = field(default_factory=list)
    swapped: list[SequenceGroup] = field(default_factory=list)


class Scheduler:
    """Continuous-batching scheduler with preemption support.

    Policy:
        1. Schedule waiting sequences if resources allow.
        2. If not enough resources, preempt the longest-running seq.
        3. Swapped sequences are re-added to waiting queue when space is freed.
    """

    def __init__(self, config: EngineConfig, kv_cache, block_manager):
        self.config = config
        self.kv_cache = kv_cache
        self.block_manager = block_manager
        self.waiting: deque[SequenceGroup] = deque()
        self.running: list[SequenceGroup] = []
        self.preempted: list[SequenceGroup] = []

    async def add_request(self, seq_group: SequenceGroup) -> None:
        self.waiting.append(seq_group)
        logger.debug("Request %s added (waiting=%d running=%d)", seq_group.request_id, len(self.waiting), len(self.running))
        # Kick off scheduling loop
        asyncio.get_event_loop().call_soon(self._schedule_tick)

    def _schedule_tick(self) -> None:
        """Single scheduling pass — called from event loop."""
        self._preempt()
        self._schedule()
        self._run_step()

    # ------------------------------------------------------------------
    # Scheduling logic
    # ------------------------------------------------------------------

    def _can_add(self, seq: SequenceGroup) -> bool:
        """Check if we can fit this sequence given current resources."""
        needed = (seq.prompt_len + len(seq.generated_tokens) + 1 + self.config.block_size - 1) // self.config.block_size
        free = self.block_manager.free_blocks
        if self.config.max_num_seqs and len(self.running) >= self.config.max_num_seqs:
            return False
        return free >= needed

    def _preempt(self) -> None:
        """Recompute preemption: evict longest-running sequences."""
        while self.waiting and not self._can_add(self.waiting[0]):
            if not self.running:
                break
            victim = max(self.running, key=lambda s: s.total_len)
            if self.config.preemption_mode == "swap":
                self.block_manager.swap_out(victim.block_ids)
                victim.status = SeqStatus.PREEMPTED
            else:
                self.block_manager.free(victim.block_ids)
            victim.block_ids.clear()
            self.running.remove(victim)
            victim.status = SeqStatus.PREEMPTED
            self.preempted.append(victim)
            logger.info("Preempted seq %s (total_len=%d)", victim.request_id, victim.total_len)

    def _schedule(self) -> None:
        """Schedule waiting sequences."""
        while self.waiting and self._can_add(self.waiting[0]):
            seq = self.waiting.popleft()
            num_blocks = (seq.total_len + self.config.block_size - 1) // self.config.block_size
            seq.block_ids = self.block_manager.allocate(num_blocks)
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            logger.debug("Scheduled seq %s (blocks=%d)", seq.request_id, num_blocks)

    def _run_step(self) -> None:
        """Execute a single forward step for all running sequences."""
        if not self.running:
            return

        # In a real implementation this would batch all sequences through the model
        # For this reference implementation, we simulate generation
        for seq in list(self.running):
            if len(seq.generated_tokens) >= seq.max_tokens:
                self._finish(seq, reason="max_tokens")
                continue

            # Simulated token generation
            token = 0  # placeholder
            seq.generated_tokens.append(token)

            # Stop check
            text_so_far = ""
            for stop_seq in seq.stop:
                if stop_seq and stop_seq in text_so_far:
                    self._finish(seq, reason="stop")
                    break
        else:
            # All sequences continue — schedule next tick
            asyncio.get_event_loop().call_later(0.001, self._schedule_tick)

    def _finish(self, seq: SequenceGroup, reason: str = "stop") -> None:
        """Mark sequence as finished, free resources."""
        seq.status = SeqStatus.FINISHED
        self.running.remove(seq)
        self.block_manager.free(seq.block_ids)
        seq.block_ids.clear()
        if not seq.completion_future.done():
            seq.completion_future.set_result(seq.generated_tokens)
        logger.debug("Finished seq %s reason=%s tokens=%d", seq.request_id, reason, len(seq.generated_tokens))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    @property
    def num_preempted(self) -> int:
        return len(self.preempted)
