"""GPU memory block manager for PagedAttention KV-cache allocation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import EngineConfig

logger = logging.getLogger(__name__)


@dataclass
class Block:
    """A single memory block."""

    block_id: int
    is_free: bool = True
    ref_count: int = 0
    seq_id: str | None = None  # owning sequence


class BlockManager:
    """Manages GPU memory blocks for PagedAttention.

    Blocks are fixed-size units of GPU memory allocated for KV-cache storage.
    Supports allocation, freeing, swap-in/swap-out, and prefix sharing.

    Memory model:
        - Total GPU memory → num_blocks * block_size * block_element_size
        - Free list tracks available blocks
        - Ref counting for prefix sharing (multiple sequences sharing KV prefix)
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.block_size = config.block_size

        # Calculate number of blocks
        if config.num_kv_cache_blocks > 0:
            self.total_blocks = config.num_kv_cache_blocks
        else:
            # Estimate from GPU memory
            self.total_blocks = self._estimate_num_blocks()

        self._blocks = [Block(block_id=i) for i in range(self.total_blocks)]
        self._free_block_ids: list[int] = list(range(self.total_blocks))

        # CPU swap space
        self._swap_blocks: dict[int, list] = {}  # block_id -> stored data

        logger.info(
            "BlockManager: %d blocks (block_size=%d, total=%d GB)",
            self.total_blocks,
            self.block_size,
            self.total_blocks * self.block_size * 2 / (1024 ** 3),
        )

    def _estimate_num_blocks(self) -> int:
        """Estimate number of blocks from available GPU memory."""
        try:
            import torch
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_mem
                # Reserve memory for model weights and activations
                available = int(total_mem * self.config.gpu_memory_utilization)
                # Each block: block_size * 2 (k+v) * 2 (bytes per fp16 element) * 2 (layers estimate)
                block_element_size = self.block_size * 2 * 2 * 80  # estimate 80 layers
                return max(available // block_element_size, 64)
        except Exception:
            pass
        return 1024  # fallback

    @property
    def free_blocks(self) -> int:
        return len(self._free_block_ids)

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - self.free_blocks

    def allocate(self, num_blocks: int, seq_id: str | None = None) -> list[int]:
        """Allocate contiguous blocks.

        Args:
            num_blocks: Number of blocks to allocate.
            seq_id: Owning sequence ID.

        Returns:
            List of allocated block IDs.

        Raises:
            ValueError: If not enough free blocks.
        """
        if num_blocks > self.free_blocks:
            raise ValueError(f"Cannot allocate {num_blocks} blocks (only {self.free_blocks} free)")

        allocated = []
        for _ in range(num_blocks):
            bid = self._free_block_ids.pop()
            self._blocks[bid].is_free = False
            self._blocks[bid].ref_count = 1
            self._blocks[bid].seq_id = seq_id
            allocated.append(bid)

        logger.debug("Allocated %d blocks (free remaining: %d)", len(allocated), self.free_blocks)
        return allocated

    def free(self, block_ids: list[int]) -> None:
        """Free a list of blocks back to the pool."""
        for bid in block_ids:
            if bid < 0 or bid >= self.total_blocks:
                continue
            self._blocks[bid].ref_count -= 1
            if self._blocks[bid].ref_count <= 0:
                self._blocks[bid].is_free = True
                self._blocks[bid].seq_id = None
                self._free_block_ids.append(bid)

    def swap_out(self, block_ids: list[int]) -> None:
        """Move blocks to CPU swap space."""
        for bid in block_ids:
            self._swap_blocks[bid] = [None]  # placeholder for CPU data
            self._blocks[bid].is_free = True
            self._free_block_ids.append(bid)

    def swap_in(self, block_ids: list[int]) -> list[int]:
        """Move blocks back to GPU from swap space."""
        new_ids = []
        for bid in block_ids:
            if bid in self._swap_blocks:
                del self._swap_blocks[bid]
                self._blocks[bid].is_free = False
                self._free_block_ids.remove(bid)
                new_ids.append(bid)
        return new_ids

    def allocate_prefix(self, num_blocks: int) -> list[int]:
        """Allocate blocks with reference counting for prefix sharing."""
        blocks = self.allocate(num_blocks)
        for bid in blocks:
            self._blocks[bid].ref_count += 1
        return blocks

    def can_allocate(self, num_blocks: int) -> bool:
        return num_blocks <= self.free_blocks

    def stats(self) -> dict:
        return {
            "total_blocks": self.total_blocks,
            "free_blocks": self.free_blocks,
            "used_blocks": self.used_blocks,
            "block_size": self.block_size,
            "utilization": round(self.used_blocks / self.total_blocks * 100, 2) if self.total_blocks else 0,
        }
