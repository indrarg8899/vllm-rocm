"""Inference engine with PagedAttention KV-cache, model loading, and tokenizer."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import EngineConfig
from src.models.loader import ModelLoader
from src.cache.paged_attention import PagedAttentionKVCacheManager
from src.cache.block_manager import BlockManager
from src.scheduler import Scheduler, SequenceGroup

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Core inference engine coordinating model, cache, and scheduler."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.model_name: str = config.model_name or config.model.split("/")[-1]
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: torch.nn.Module | None = None
        self._block_manager: BlockManager | None = None
        self._kv_cache: PagedAttentionKVCacheManager | None = None
        self._scheduler: Scheduler | None = None
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        logger.info("Starting engine: model=%s tp=%d quant=%s", self.config.model, self.config.tensor_parallel_size, self.config.quantization)
        self._tokenizer = self._load_tokenizer()
        self._model = await asyncio.get_event_loop().run_in_executor(None, self._load_model)
        self._block_manager = BlockManager(config=self.config)
        self._kv_cache = PagedAttentionKVCacheManager(config=self.config, block_manager=self._block_manager)
        self._scheduler = Scheduler(config=self.config, kv_cache=self._kv_cache, block_manager=self._block_manager)
        self._started = True
        logger.info("Engine ready (%.1fs)", 0.0)  # timing placeholder

    async def shutdown(self) -> None:
        self._started = False
        if self._model is not None:
            del self._model
            torch.cuda.empty_cache()
        logger.info("Engine shutdown complete")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        logger.info("Loading tokenizer: %s", self.config.model)
        tok = AutoTokenizer.from_pretrained(
            self.config.model,
            trust_remote_code=self.config.trust_remote_code,
            token=self.config.huggingface_hub_token,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _load_model(self) -> torch.nn.Module:
        loader = ModelLoader(self.config)
        model = loader.load()
        if self.config.quantization:
            from src.models.quantizer import quantize_model
            model = quantize_model(model, self.config.quantization)
        return model

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate_chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: str | list[str] | None = None,
    ) -> str:
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return await self.generate_text(prompt, max_tokens, temperature, top_p, stop)

    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: str | list[str] | None = None,
    ) -> str:
        if not self._started:
            raise RuntimeError("Engine not started")

        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        input_len = input_ids.shape[1]

        # Submit to scheduler
        seq_group = SequenceGroup(
            request_id=f"gen-{id(prompt)}",
            input_ids=input_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop if isinstance(stop, list) else ([stop] if stop else []),
        )
        await self._scheduler.add_request(seq_group)

        # Wait for completion
        result = await seq_group.completion_future
        output_text = self._tokenizer.decode(result, skip_special_tokens=True)
        # Strip the prompt from output if it's repeated
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):]
        return output_text.strip()

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def gpu_memory_usage(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_mem
        return round(allocated / total * 100, 2)
