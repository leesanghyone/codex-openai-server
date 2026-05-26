# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.14

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /build

RUN python -m venv /opt/venv

COPY pyproject.toml README.md LICENSE ./
COPY codex_openai_server ./codex_openai_server

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
  && pip install .

FROM python:${PYTHON_VERSION}-slim AS runtime

ARG UID=10001
ARG GID=10001
ARG VERSION=dev
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    OPENAI_COMPAT_HOST=0.0.0.0 \
    OPENAI_COMPAT_PORT=8000 \
    CODEX_HOME=/home/app/.codex

LABEL org.opencontainers.image.title="Codex OpenAI Server" \
      org.opencontainers.image.description="OpenAI-compatible HTTP server backed by locally authenticated ChatGPT Codex." \
      org.opencontainers.image.created="${BUILD_DATE}" \
    org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${VERSION}"

RUN groupadd --gid "${GID}" app \
    && useradd --uid "${UID}" --gid app --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /workspace /home/app/.codex \
    && chown -R app:app /workspace /home/app

WORKDIR /workspace

COPY --from=builder /opt/venv /opt/venv

USER app

EXPOSE 8000

ENTRYPOINT ["codex_openai_server"]

CMD []
