"""OpenAI-compatible API server for vLLM-ROCm."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.config import EngineConfig, ServerConfig
from src.engine import InferenceEngine

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    cfg = EngineConfig.from_cli()
    engine = InferenceEngine(cfg)
    await engine.start()
    yield
    await engine.shutdown()


app = FastAPI(title="vLLM-ROCm", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "gpu_memory_used_pct": engine.gpu_memory_usage() if engine else 0}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": engine.model_name,
                "object": "model",
                "owned_by": "vllm-rocm",
                "created": int(time.time()),
            }
        ],
    }


def _make_chat_response(request_id: str, model: str, text: str, usage: dict) -> dict:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


async def _stream_chat(request_id: str, model: str, text: str) -> AsyncGenerator[str, None]:
    for i, token in enumerate(text.split()):
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token + " "},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)
    done = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if engine is None:
        raise HTTPException(503, "Engine not ready")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    text = await engine.generate_chat(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
    )
    prompt_tokens = sum(len(m["content"].split()) for m in messages)
    completion_tokens = len(text.split())
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}

    if request.stream:
        return StreamingResponse(_stream_chat(request_id, request.model, text), media_type="text/event-stream")

    return JSONResponse(_make_chat_response(request_id, request.model, text, usage))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if engine is None:
        raise HTTPException(503, "Engine not ready")

    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
    results = await asyncio.gather(
        *[engine.generate_text(p, request.max_tokens, request.temperature, request.top_p, request.stop) for p in prompts]
    )

    choices = [
        {"index": i, "text": r, "finish_reason": "stop"} for i, r in enumerate(results)
    ]
    total_completion = sum(len(r.split()) for r in results)
    total_prompt = sum(len(p.split()) for p in prompts)

    return JSONResponse(
        {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": choices,
            "usage": {"prompt_tokens": total_prompt, "completion_tokens": total_completion, "total_tokens": total_prompt + total_completion},
        }
    )


if __name__ == "__main__":
    import uvicorn
    cfg = ServerConfig.from_cli()
    uvicorn.run("src.api_server:app", host=cfg.host, port=cfg.port, workers=1)
