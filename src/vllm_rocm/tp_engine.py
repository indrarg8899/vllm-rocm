"""Tensor parallelism engine for multi-GROCm inference."""

import os
import torch
import torch.distributed as dist
from typing import List, Optional


class TensorParallelEngine:
    """Distributes model tensors across multiple AMD GPUs via RCCL."""

    def __init__(self, world_size: int = 1, backend: str = "nccl"):
        self.world_size = world_size
        self.backend = backend
        self.rank = 0
        self.local_rank = 0
        self._initialized = False

    def init_distributed(self) -> None:
        """Initialize distributed process group for ROCm."""
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
            )

        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)
        self._initialized = True

    def shard_weights(
        self,
        weight: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        """Shard a weight tensor across TP ranks."""
        if not self._initialized:
            return weight

        size = weight.size(dim)
        chunk_size = size // self.world_size
        start = self.rank * chunk_size
        end = start + chunk_size
        indices = list(range(start, min(end, size)))

        if dim == 0:
            return weight[indices, :]
        elif dim == 1:
            return weight[:, indices]
        else:
            raise ValueError(f"Unsupported shard dim: {dim}")

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce tensor across TP ranks."""
        if self._initialized and self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Gather tensors from all ranks."""
        if not self._initialized or self.world_size == 1:
            return tensor

        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=dim)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from src rank."""
        if self._initialized and self.world_size > 1:
            dist.broadcast(tensor, src=src)
        return tensor

    def cleanup(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
            self._initialized = False
