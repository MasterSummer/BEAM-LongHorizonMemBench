# syntax=docker/dockerfile:1.7

ARG PYTHON_BASE_IMAGE=python:3.11-slim
ARG PYTHON_BASE_DIGEST
FROM ${PYTHON_BASE_IMAGE}@${PYTHON_BASE_DIGEST}

ARG UV_VERSION=0.8.0
ARG SOURCE_COMMIT=unknown
ARG SOURCE_REF=detached
ARG SOURCE_DIRTY=false

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

RUN SOURCE_COMMIT="${SOURCE_COMMIT}" \
    SOURCE_REF="${SOURCE_REF}" \
    SOURCE_DIRTY="${SOURCE_DIRTY}" \
    python -c 'import json, os, pathlib; pathlib.Path("/app/BUILD.json").write_text(json.dumps({"commit": os.environ["SOURCE_COMMIT"], "dirty": os.environ["SOURCE_DIRTY"].lower() == "true", "ref": os.environ["SOURCE_REF"]}, sort_keys=True) + "\n", encoding="utf-8")'

RUN uv sync --frozen --offline --no-dev --extra qualification \
    && /app/.venv/bin/python -m lhmsb.qualification --help >/dev/null

RUN groupadd --gid 10001 lhmsb \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin lhmsb \
    && mkdir -p /data/lhmsb \
    && chown -R lhmsb:lhmsb /app /data/lhmsb

USER lhmsb

ENTRYPOINT ["/app/.venv/bin/python", "-m", "lhmsb.qualification"]
CMD ["--help"]
