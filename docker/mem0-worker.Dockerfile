# syntax=docker/dockerfile:1.7

ARG PYTHON_BASE_IMAGE=python:3.11-slim
ARG PYTHON_BASE_DIGEST
FROM ${PYTHON_BASE_IMAGE}@${PYTHON_BASE_DIGEST}

ARG UV_VERSION=0.8.0

ENV DEBIAN_FRONTEND=noninteractive \
    MEM0_TELEMETRY=False \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_FIND_LINKS=/opt/wheelhouse \
    UV_NO_INDEX=true

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/wheelhouse/ /opt/wheelhouse/
RUN python -m pip install \
      --no-index \
      --find-links=/opt/wheelhouse \
      "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY datasets/releases/ ./datasets/releases/
COPY runs/vertical/software_mem0_v2/ ./runs/vertical/software_mem0_v2/

RUN uv sync --frozen --offline --no-dev --extra qualification \
    && /app/.venv/bin/python -m lhmsb.qualification --help >/dev/null

RUN groupadd --gid 10001 lhmsb \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin lhmsb \
    && mkdir -p /data/lhmsb \
    && chown -R lhmsb:lhmsb /app /data/lhmsb

USER lhmsb

ENTRYPOINT ["/app/.venv/bin/python", "-m", "lhmsb.qualification"]
CMD ["--help"]
