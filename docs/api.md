# API Reference

## POST /v1/completions

```json
{
  "model": "meta-llama/Llama-3-70B-Instruct",
  "prompt": "Hello, world!",
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

## POST /v1/chat/completions

Standard OpenAI chat format with messages array.

## GET /v1/models

Returns available models.

## GET /health

Returns `{"status": "ok"}` when model is ready.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| model | string | required | Model name or path |
| max_tokens | int | 512 | Max output tokens |
| temperature | float | 1.0 | Sampling temperature |
| top_p | float | 1.0 | Nucleus sampling |
| stream | bool | false | Enable streaming |
| stop | list | null | Stop sequences |
| frequency_penalty | float | 0.0 | Repetition penalty |
