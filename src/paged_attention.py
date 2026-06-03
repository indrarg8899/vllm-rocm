#!/usr/bin/env python3
"""Paged attention manager for memory-efficient KV cache."""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class PagedBlock:
    """A single block in the KV cache."""
    block_id: int
    ref_count: int = 0
    sequence_id: Optional[str] = None


class PagedAttentionManager:
    """
    Manages paged KV cache for memory-efficient attention.
    
    Allocates GPU memory in fixed-size blocks, allowing:
    - Non-contiguous KV cache storage
    - Efficient memory reuse across sequences
    - Near-zero memory waste from padding
    """

    def __init__(
        self,
        block_size: int = 16,
        max_num_blocks: int = 1024,
        num_layers: int = 80,
        num_heads: int = 64,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ):
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device or torch.device("cuda:0")

        # Free block list
        self.free_blocks: List[int] = list(range(max_num_blocks))

        # Block allocation table: seq_id -> list of block IDs
        self.block_tables: dict[str, List[int]] = {}

        # Physical KV cache (allocated on demand)
        self.kv_cache: Optional[torch.Tensor] = None
        self._allocate_cache()

    def _allocate_cache(self):
        """Pre-allocate the KV cache tensor."""
        # Shape: [2, num_layers, max_num_blocks, block_size, num_heads, head_dim]
        # 2 for K and V
        cache_size = (
            2 * self.num_layers * self.max_num_blocks *
            self.block_size * self.num_heads * self.head_dim
        )
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        total_bytes = cache_size * element_size

        print(f"Allocating KV cache: {total_bytes / 1e9:.2f} GB "
              f"({self.max_num_blocks} blocks × {self.block_size} tokens)")

        self.kv_cache = torch.zeros(
            (2, self.num_layers, self.max_num_blocks,
             self.block_size, self.num_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

    def allocate_sequence(self, seq_id: str, num_tokens: int) -> List[int]:
        """Allocate blocks for a new sequence."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if num_blocks_needed > len(self.free_blocks):
            raise RuntimeError(
                f"Not enough blocks: need {num_blocks_needed}, "
                f"have {len(self.free_blocks)} free"
            )

        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            allocated.append(block_id)

        self.block_tables[seq_id] = allocated
        return allocated

    def free_sequence(self, seq_id: str):
        """Free blocks allocated to a sequence."""
        if seq_id in self.block_tables:
            blocks = self.block_tables.pop(seq_id)
            self.free_blocks.extend(blocks)

    def get_kv_cache_for_seq(self, seq_id: str, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get K and V cache views for a specific sequence and layer."""
        if seq_id not in self.block_tables:
            raise KeyError(f"Sequence {seq_id} not found")

        blocks = self.block_tables[seq_id]
        k_cache = self.kv_cache[0, layer_idx, blocks]
        v_cache = self.kv_cache[1, layer_idx, blocks]

        return k_cache, v_cache

    def write_to_cache(self, seq_id: str, layer_idx: int, position: int,
                       k: torch.Tensor, v: torch.Tensor):
        """Write K/V data to the appropriate cache blocks."""
        blocks = self.block_tables[seq_id]
        block_idx = position // self.block_size
        offset = position % self.block_size

        if block_idx >= len(blocks):
            return

        block_id = blocks[block_idx]
        self.kv_cache[0, layer_idx, block_id, offset] = k
        self.kv_cache[1, layer_idx, block_id, offset] = v

    def get_block_usage(self) -> dict:
        """Get cache utilization statistics."""
        used = self.max_num_blocks - len(self.free_blocks)
        return {
            "total_blocks": self.max_num_blocks,
            "used_blocks": used,
            "free_blocks": len(self.free_blocks),
            "utilization": used / self.max_num_blocks if self.max_num_blocks > 0 else 0,
            "active_sequences": len(self.block_tables),
        }
