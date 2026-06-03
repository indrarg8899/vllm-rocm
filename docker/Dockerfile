FROM rocm/pytorch:rocm6.3-pytorch2.4

# Environment
ENV DEBIAN_FRONTEND=noninteractive
ENV ROCM_VERSION=6.3
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl vim \
    build-essential cmake \
    libnuma-dev libamdhip64-dev \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM-ROCm dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Ports
EXPOSE 8000

# Default: start API server
ENTRYPOINT ["python", "-m", "src.api_server"]
CMD ["--model", "meta-llama/Meta-Llama-3-8B-Instruct"]
