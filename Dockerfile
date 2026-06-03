FROM rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.2.0

WORKDIR /workspace/vllm-rocm
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV ROCM_PATH=/opt/rocm
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV HSA_ENABLE_SDMA=0
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

EXPOSE 8000 9090

ENTRYPOINT ["python", "-m", "src.api_server"]
CMD ["--model", "meta-llama/Llama-3.1-70B-Instruct", "--tensor-parallel-size", "4"]
