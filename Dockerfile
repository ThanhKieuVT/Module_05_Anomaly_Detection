# Inference service image. CPU by default; for GPU build with
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 -t fscad-api .
FROM python:3.11-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    FSCAD_PRODUCTS_DIR=/app/artifacts/serving

WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX}

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

COPY configs ./configs
COPY serving ./serving

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
