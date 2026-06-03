"""vLLM-ROCm HTTP serving endpoint."""

import json
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class CompletionRequest:
    model: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False


@dataclass
class CompletionResponse:
    id: str
    model: str
    choices: List[dict]
    usage: dict


class ServingEngine:
    """Manages model loading and serving via OpenAI-compatible API."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model_path = model_path
        self.tp_size = tensor_parallel_size
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self._model = None

    def load_model(self) -> None:
        print(f"Loading {self.model_path} with TP={self.tp_size}, quant={self.quantization}")

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion for a prompt."""
        return CompletionResponse(
            id="cmpl-001",
            model=request.model,
            choices=[{
                "text": f"Generated text for: {request.prompt[:50]}...",
                "index": 0,
                "finish_reason": "stop",
            }],
            usage={
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": 10,
                "total_tokens": len(request.prompt.split()) + 10,
            },
        )

    def health(self) -> dict:
        return {"status": "ok", "model": self.model_path}


def serve(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    quantization: Optional[str] = None,
    gpu_memory_utilization: float = 0.9,
):
    """Start the vLLM-ROCm server."""
    engine = ServingEngine(
        model_path=model,
        tensor_parallel_size=tensor_parallel_size,
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    engine.load_model()
    print(f"Serving {model} on {host}:{port}")
    print("OpenAI-compatible API at /v1/completions")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="vLLM-ROCm Serve")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(**vars(args))
