"""PagedAttention kernel and block manager for ROCm."""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class AttentionBlock:
    """A paged KV cache block."""
    key_cache: Optional[torch.Tensor] = None
    value_cache: Optional[torch.Tensor] = None
    ref_count: int = 0
    sequence_id: int = -1


class PagedAttentionManager:
    """Manages paged KV cache on GPU memory."""

    def __init__(
        self,
        num_blocks: int = 1024,
        block_size: int = 16,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.key_cache = torch.zeros(
            num_blocks, num_heads, block_size, head_dim,
            dtype=dtype, device=device,
        )
        self.value_cache = torch.zeros(
            num_blocks, num_heads, block_size, head_dim,
            dtype=dtype, device=device,
        )

        self.free_blocks: List[int] = list(range(num_blocks))
        self.seq_blocks: Dict[int, List[int]] = {}
        self.block_ref_count: List[int] = [0] * num_blocks

    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("No free KV cache blocks available")
        block_id = self.free_blocks.pop()
        self.block_ref_count[block_id] = 1
        return block_id

    def free_block(self, block_id: int) -> None:
        self.block_ref_count[block_id] -= 1
        if self.block_ref_count[block_id] <= 0:
            self.free_blocks.append(block_id)

    def allocate_sequence(self, seq_id: int) -> int:
        block_id = self.allocate_block()
        self.seq_blocks[seq_id] = [block_id]
        return block_id

    def append_token(
        self, seq_id: int, position: int, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        """Append a token's KV to the cache."""
        blocks = self.seq_blocks.get(seq_id, [])
        if not blocks:
            self.allocate_sequence(seq_id)
            blocks = self.seq_blocks[seq_id]

        token_in_block = position % self.block_size
        block_idx = position // self.block_size

        if block_idx >= len(blocks):
            new_block = self.allocate_block()
            self.seq_blocks[seq_id].append(new_block)
            blocks = self.seq_blocks[seq_id]

        block_id = blocks[block_idx]
        self.key_cache[block_id, :, token_in_block, :] = key.view(self.num_heads, self.head_dim)
        self.value_cache[block_id, :, token_in_block, :] = value.view(self.num_heads, self.head_dim)

    def get_key_value(
        self, seq_id: int, num_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch KV cache for a sequence."""
        blocks = self.seq_blocks.get(seq_id, [])
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        key_list = []
        value_list = []

        for i in range(min(num_blocks_needed, len(blocks))):
            block_id = blocks[i]
            k = self.key_cache[block_id]
            v = self.value_cache[block_id]
            key_list.append(k)
            value_list.append(v)

        if not key_list:
            empty = torch.zeros(
                self.num_heads, 0, self.head_dim, dtype=self.dtype, device=self.device
            )
            return empty, empty

        return torch.cat(key_list, dim=1), torch.cat(value_list, dim=1)

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks for a sequence."""
        blocks = self.seq_blocks.pop(seq_id, [])
        for block_id in blocks:
            self.free_block(block_id)

    @property
    def utilization(self) -> float:
        total = self.num_blocks
        used = total - len(self.free_blocks)
        return used / total if total > 0 else 0.0
