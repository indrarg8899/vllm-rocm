# Docker Deployment

## Pre-built Images

```bash
docker pull ghcr.io/indrarg8899/vllm-rocm:latest
docker pull ghcr.io/indrarg8899/vllm-rocm:v0.1.0
```

## Run

```bash
docker run -d \
    --name vllm-server \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --shm-size=16g \
    -v /models:/models \
    -p 8000:8000 \
    ghcr.io/indrarg8899/vllm-rocm:latest \
    --model /models/Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --quantization fp8
```

## Build from Source

```bash
git clone https://github.com/indrarg8899/vllm-rocm.git
cd vllm-rocm
docker build -t vllm-rocm:local .
```

## Docker Compose

```yaml
services:
  vllm:
    image: ghcr.io/indrarg8899/vllm-rocm:latest
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
      - render
    volumes:
      - ./models:/models
    ports:
      - "8000:8000"
    command: >
      --model /models/Llama-3-70B-Instruct
      --tensor-parallel-size 4
```
