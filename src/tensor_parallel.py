#!/usr/bin/env python3
"""Tensor parallelism manager for multi-GPU inference."""

import os
from typing import Optional

import torch
import torch.nn as nn

try:
    import torch.distributed as dist
    HAS_DIST = True
except ImportError:
    HAS_DIST = False


class TensorParallelManager:
    """
    Manages tensor parallelism across multiple AMD GPUs.
    
    Splits model layers and attention heads across GPUs,
    synchronizing activations and gradients via RCCL.
    """

    def __init__(
        self,
        world_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ):
        self.world_size = world_size
        self.dtype = dtype
        self.rank = 0
        self.initialized = False

        if world_size > 1:
            self._init_distributed()

    def _init_distributed(self):
        """Initialize distributed communication group."""
        if not HAS_DIST:
            raise RuntimeError("torch.distributed not available")

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                world_size=self.world_size,
                rank=int(os.environ.get("RANK", 0)),
            )
            self.rank = dist.get_rank()

        # Set device for this rank
        torch.cuda.set_device(self.rank)
        self.initialized = True

    def parallelize(self, model: nn.Module) -> nn.Module:
        """Apply tensor parallelism to a model."""
        if self.world_size == 1:
            return model

        # Split linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._split_linear(module, name)

            # Split attention heads
            if "attention" in name.lower() and hasattr(module, "num_heads"):
                self._split_attention(module)

        # Place each shard on its GPU
        device = torch.device(f"cuda:{self.rank}")
        model = model.to(device, dtype=self.dtype)

        return model

    def _split_linear(self, module: nn.Linear, name: str):
        """Split linear layer weight across TP ranks."""
        weight = module.weight.data
        out_features, in_features = weight.shape

        if out_features % self.world_size != 0:
            return

        chunk_size = out_features // self.world_size
        start = self.rank * chunk_size
        end = start + chunk_size

        module.weight = nn.Parameter(weight[start:end].clone())
        if module.bias is not None:
            module.bias = nn.Parameter(module.bias.data[start:end].clone())

    def _split_attention(self, module):
        """Split attention heads across TP ranks."""
        if hasattr(module, "num_heads"):
            total_heads = module.num_heads
            if total_heads % self.world_size == 0:
                heads_per_rank = total_heads // self.world_size
                module.num_heads = heads_per_rank
                if hasattr(module, "head_dim"):
                    module.head_dim = module.head_dim  # Keep full head dim

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce tensor across all TP ranks."""
        if self.world_size == 1 or not self.initialized:
            return tensor

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensor from all ranks."""
        if self.world_size == 1 or not self.initialized:
            return tensor

        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=-1)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from src rank."""
        if self.world_size == 1 or not self.initialized:
            return tensor

        dist.broadcast(tensor, src)
        return tensor

    def get_device(self) -> torch.device:
        """Get device for current rank."""
        return torch.device(f"cuda:{self.rank}")

    def cleanup(self):
        """Clean up distributed state."""
        if self.initialized and dist.is_initialized():
            dist.destroy_process_group()
