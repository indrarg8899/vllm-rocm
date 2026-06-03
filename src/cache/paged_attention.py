"""PagedAttention KV-cache manager.

Implements virtual-memory-inspired PagedAttention for efficient GPU memory
management of key-value cache during autoregressive generation.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from src.config import EngineConfig
from src.cache.block_manager import BlockManager

logger = logging.getLogger(__name__)


@dataclass
class KVBlock:
    """Single KV-cache block holding keys and values for a sequence of tokens."""

    block_id: int
    num_tokens: int  # tokens stored in this block
    key_cache: torch.Tensor  # (num_heads, block_size, head_dim)
    value_cache: torch.Tensor  # (num_heads, block_size, head_dim)


from dataclasses import dataclass, field


@dataclass
class KVCacheSlot:
    """Represents the KV-cache allocation for a sequence."""

    block_ids: list[int]
    seq_len: int


class PagedAttentionKVCacheManager:
    """PagedAttention KV-cache management.

    Each block stores a fixed number of tokens' K/V tensors.
    Block assignment is done via the BlockManager.
    Supports preemption via swap-in/swap-out.
    """

    def __init__(self, config: EngineConfig, block_manager: BlockManager):
        self.config = config
        self.block_manager = block_manager
        self.block_size = config.block_size
        self.num_layers = 0
        self.num_heads = 0
        self.head_dim = 0

        # Physical cache storage: block_id -> (key, value) tensors
        self._key_blocks: dict[int, torch.Tensor] = {}
        self._value_blocks: dict[int, torch.Tensor] = {}
        self._swap_buffer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def init_cache(self, num_layers: int, num_heads: int, head_dim: int, device: str = "cuda") -> None:
        """Pre-allocate physical cache blocks on GPU.

        Args:
            num_layers: Number of transformer layers.
            num_heads: Number of KV attention heads.
            head_dim: Dimension per head.
            device: Device string.
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        num_blocks = self.block_manager.total_blocks
        for bid in range(num_blocks):
            key = torch.zeros(num_heads, self.block_size, head_dim, dtype=torch.float16, device=device)
            val = torch.zeros(num_heads, self.block_size, head_dim, dtype=torch.float16, device=device)
            self._key_blocks[bid] = key
            self._value_blocks[bid] = val

        logger.info(
            "Initialized KV cache: %d blocks x %d layers x %d heads x %d head_dim = %.1f GB",
            num_blocks, num_layers, num_heads, head_dim,
            num_blocks * num_layers * num_heads * self.block_size * head_dim * 2 * 2 / 1e9,
        )

    def write_kv(self, block_ids: list[int], position: int, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write K/V tensors for a token at the given position.

        Args:
            block_ids: Allocated block IDs for the sequence.
            position: Token position in the sequence.
            layer_idx: Transformer layer index.
            key: Key tensor of shape (num_heads, 1, head_dim).
            value: Value tensor of shape (num_heads, 1, head_dim).
        """
        block_idx = position // self.block_size
        token_idx = position % self.block_size
        bid = block_ids[block_idx]
        self._key_blocks[bid][:, token_idx, :] = key.squeeze(1)
        self._value_blocks[bid][:, token_idx, :] = value.squeeze(1)

    def read_kv(self, block_ids: list[int], seq_len: int, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read all K/V tensors for a sequence.

        Args:
            block_ids: Allocated block IDs.
            seq_len: Sequence length (tokens to read).
            layer_idx: Transformer layer index.

        Returns:
            Tuple of (keys, values) each (num_heads, seq_len, head_dim).
        """
        keys_list = []
        values_list = []
        remaining = seq_len
        for bid in block_ids:
            tokens_in_block = min(remaining, self.block_size)
            keys_list.append(self._key_blocks[bid][:, :tokens_in_block, :])
            values_list.append(self._value_blocks[bid][:, :tokens_in_block, :])
            remaining -= tokens_in_block
            if remaining <= 0:
                break
        return torch.cat(keys_list, dim=1), torch.cat(values_list, dim=1)

    def swap_out(self, block_ids: list[int], seq_id: int) -> None:
        """Swap KV blocks to CPU for preemption."""
        for bid in block_ids:
            self._swap_buffer[f"{seq_id}_{bid}"] = (
                self._key_blocks[bid].clone().cpu(),
                self._value_blocks[bid].clone().cpu(),
            )

    def swap_in(self, block_ids: list[int], seq_id: int) -> None:
        """Swap KV blocks back to GPU."""
        for bid in block_ids:
            key = f"{seq_id}_{bid}"
            if key in self._swap_buffer:
                self._key_blocks[bid] = self._swap_buffer[key][0].cuda()
                self._value_blocks[bid] = self._swap_buffer[key][1].cuda()
                del self._swap_buffer[key]

    def copy_block(self, src_block: int, dst_block: int) -> None:
        """Copy one block's KV data to another (for beam search)."""
        self._key_blocks[dst_block].copy_(self._key_blocks[src_block])
        self._value_blocks[dst_block].copy_(self._value_blocks[src_block])
