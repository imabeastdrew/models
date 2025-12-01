FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

ENV PATH="/workspace/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY src ./src

RUN uv pip install -e . --no-deps

VOLUME ["/workspace/checkpoints"]

ENTRYPOINT ["musicagent-train"]
