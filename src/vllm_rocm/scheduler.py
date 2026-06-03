"""Continuous batch scheduler for vLLM-ROCm."""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    PREEMPTED = "preempted"


@dataclass
class Sequence:
    seq_id: int
    prompt_tokens: List[int]
    generated_tokens: List[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    created_at: float = field(default_factory=time.time)
    max_tokens: int = 512
    temperature: float = 1.0
    priority: int = 0

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)

    @property
    def is_finished(self) -> bool:
        return (
            self.status == RequestStatus.FINISHED
            or len(self.generated_tokens) >= self.max_tokens
        )


class ContinuousBatchScheduler:
    """Schedules requests with continuous batching and preemption."""

    def __init__(
        self,
        max_batch_size: int = 32,
        max_num_seqs: int = 256,
        max_total_tokens: int = 8192,
        preemption_budget: float = 0.8,
    ):
        self.max_batch_size = max_batch_size
        self.max_num_seqs = max_num_seqs
        self.max_total_tokens = max_total_tokens
        self.preemption_budget = preemption_budget

        self.waiting: List[Sequence] = []
        self.running: Dict[int, Sequence] = {}
        self.finished: List[Sequence] = []
        self._lock = threading.Lock()
        self._next_seq_id = 0

    def add_request(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 512,
        temperature: float = 1.0,
        priority: int = 0,
    ) -> int:
        with self._lock:
            seq = Sequence(
                seq_id=self._next_seq_id,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                priority=priority,
            )
            self._next_seq_id += 1
            self.waiting.append(seq)
            return seq.seq_id

    def schedule(self) -> List[Sequence]:
        """Select sequences for the next forward pass."""
        with self._lock:
            batch: List[Sequence] = list(self.running.values())

            self.waiting.sort(key=lambda s: -s.priority)

            for seq in self.waiting[:]:
                if len(batch) >= self.max_batch_size:
                    break
                if len(batch) + 1 > self.max_num_seqs:
                    break

                projected_tokens = sum(s.total_tokens for s in batch) + seq.total_tokens
                if projected_tokens > self.max_total_tokens:
                    self._maybe_preempt(batch)
                    if sum(s.total_tokens for s in batch) + seq.total_tokens > self.max_total_tokens:
                        break

                seq.status = RequestStatus.RUNNING
                self.running[seq.seq_id] = seq
                batch.append(seq)
                self.waiting.remove(seq)

            return batch

    def _maybe_preempt(self, batch: List[Sequence]) -> None:
        """Preempt low-priority running sequences if over budget."""
        total = sum(s.total_tokens for s in batch)
        budget = self.max_total_tokens * self.preemption_budget

        if total <= budget:
            return

        sorted_running = sorted(batch, key=lambda s: (s.priority, -s.created_at))

        while total > budget and sorted_running:
            victim = sorted_running.pop(0)
            victim.status = RequestStatus.PREEMPTED
            if victim.seq_id in self.running:
                del self.running[victim.seq_id]
            batch.remove(victim)
            self.waiting.append(victim)
            total -= victim.total_tokens

    def update_generated_tokens(self, seq_id: int, num_tokens: int = 1) -> Optional[Sequence]:
        """Mark tokens generated. Returns sequence if finished."""
        with self._lock:
            seq = self.running.get(seq_id)
            if seq is None:
                return None

            seq.generated_tokens.extend([0] * num_tokens)

            if seq.is_finished or len(seq.generated_tokens) >= seq.max_tokens:
                seq.status = RequestStatus.FINISHED
                self.finished.append(seq)
                del self.running[seq_id]
                return seq
            return None

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "waiting": len(self.waiting),
                "running": len(self.running),
                "finished": len(self.finished),
                "total_tokens": sum(s.total_tokens for s in self.running.values()),
                "max_total_tokens": self.max_total_tokens,
            }
