# syntax=docker/dockerfile:1.7

ARG PYTHON_BASE_IMAGE=python:3.11-slim
ARG PYTHON_BASE_DIGEST
FROM ${PYTHON_BASE_IMAGE}@${PYTHON_BASE_DIGEST}

ARG SOURCE_COMMIT=unknown
ARG SOURCE_REF=detached
ARG SOURCE_DIRTY=false
ARG AMEM_SOURCE_COMMIT=ceffb860f0712bbae97b184d440df62bc910ca8d

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MEM0_TELEMETRY=False \
    LHMSB_AMEM_SOURCE_COMMIT=${AMEM_SOURCE_COMMIT}

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY docker/wheelhouse/ /opt/wheelhouse/
COPY docker/locks/amem-requirements.txt docker/locks/amem-wheelhouse-manifest.json /app/docker/locks/
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY datasets/releases/ ./datasets/releases/

RUN SOURCE_COMMIT="${SOURCE_COMMIT}" SOURCE_REF="${SOURCE_REF}" \
    SOURCE_DIRTY="${SOURCE_DIRTY}" \
    python -c 'import json, os, pathlib; pathlib.Path("/app/BUILD.json").write_text(json.dumps({"commit": os.environ["SOURCE_COMMIT"], "dirty": os.environ["SOURCE_DIRTY"].lower() == "true", "ref": os.environ["SOURCE_REF"], "amem_source_commit": os.environ["AMEM_SOURCE_COMMIT"]}, sort_keys=True) + "\n", encoding="utf-8")'

RUN python -m venv /app/.venv \
    && /app/.venv/bin/python -m pip install --no-index --find-links=/opt/wheelhouse \
      "lhmsb[qualification]==0.1.0" \
    && /app/.venv/bin/python -m pip install --no-index --require-hashes \
      --find-links=/opt/wheelhouse -r /app/docker/locks/amem-requirements.txt \
    && /app/.venv/bin/python -m lhmsb.qualification --help >/dev/null

RUN groupadd --gid 10001 lhmsb \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin lhmsb \
    && mkdir -p /data/lhmsb /tmp/lhmsb \
    && chown -R lhmsb:lhmsb /app /data/lhmsb /tmp/lhmsb

USER lhmsb
ENV HOME=/tmp/lhmsb \
    LHMSB_AMEM_PACKAGE_IDENTITY=agentic-memory

ENTRYPOINT ["/app/.venv/bin/python", "-m", "lhmsb.qualification"]
CMD ["--help"]
