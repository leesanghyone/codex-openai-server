# codex-openai-server

In [OpenAI's announcement for GPT-5.5](https://openai.com/index/introducing-gpt-5-5/), they said "We'll bring GPT‑5.5 and GPT‑5.5 Pro to the API very soon."

Well, soon is now.

`codex_openai_server` is an OpenAI-compatible FastAPI server that connects to ChatGPT Codex using the local Codex CLI auth file.

```pwsh
codex login [--device-auth]
docker compose up

$env:COPILOT_PROVIDER_TYPE="openai"
$env:COPILOT_PROVIDER_BASE_URL="http://127.0.0.1:8000/v1"
$env:COPILOT_PROVIDER_API_KEY="your-super-secret-key-from-.env"
$env:COPILOT_MODEL="gpt-5.5"
$env:COPILOT_PROVIDER_WIRE_API="responses"

copilot --disable-builtin-mcps -p "Who are you?"
```

```
● Checking my documentation
  └ # GitHub Copilot CLI Documentation

I’m **GitHub Copilot CLI**, a terminal-native AI coding assistant. I can help build, edit, debug, refactor, and understand code from your command line, with GitHub and MCP-powered integrations.

I’m powered by **gpt-5.5** in this session.


Changes   +0 -0
Duration  16s
Tokens    ↑ 67.0k • ↓ 157 • 0 (cached) • 37 (reasoning)
```

`codex_openai_server` is intended for local use on a trusted machine or private network segment. It is not hardened as a public internet-facing multi-tenant service.

It is intended to look like a regular OpenAI endpoint to clients. The server exposes:

- `/v1/models`
- `/v1/responses`
- `/v1/chat/completions`

Features:

- async transport to the upstream Codex backend using `httpx`
- streaming SSE support for Responses API and Chat Completions
- tool call translation between Chat Completions and Codex Responses payloads
- local API key protection for your compatibility server
- optional Docker and Docker Compose local deployment

## Requirements

- Python 3.10+
- Python 3.14 is the preferred version for local development
- a local Codex CLI login with `auth.json`

This project assumes you already trust the clients that can reach it. There is no built-in rate limiting or request-size enforcement, so do not expose it directly to the public internet without adding your own edge controls.

The server reads Codex credentials from `CODEX_HOME/auth.json`. By default that resolves to your local `.codex` directory.

## Stability and security posture

The package metadata currently classifies this project as alpha, and that is the right expectation for upgrades and automation around it. Keep version pins explicit if you depend on exact request or deployment behavior.

This server is meant for local use on a trusted machine or private network segment. Treat `auth.json`, your local compatibility API key, and any logged payloads as sensitive material. Routine bugs and feature requests can go through the issue tracker; suspected vulnerabilities should follow [SECURITY.md](SECURITY.md) instead of a public issue.

## Installation

For a published install:

```bash
python -m pip install codex-openai-server
```

For local development, create your virtual environment with Python 3.14 if you have it available. The package and CI still target Python 3.10+ compatibility.

```bash
python -m pip install -e .[dev]
```

## Local configuration

Create a local `.env` file from `.env.example`.

Required values:

- `OPENAI_COMPAT_API_KEY`: bearer token clients must send to your local compatibility server

Optional values:

- `OPENAI_COMPAT_HOST`
- `OPENAI_COMPAT_PORT`
- `OPENAI_COMPAT_PUBLISHED_HOST`
- `OPENAI_COMPAT_PUBLISHED_PORT`
- `LOCAL_CODEX_HOME`
- `OPENAI_COMPAT_LOG_LEVEL`
- `OPENAI_COMPAT_LOG_FORMAT`
- `OPENAI_COMPAT_DEBUG_LOGGING`
- `OPENAI_COMPAT_LOG_PAYLOADS`
- `OPENAI_COMPAT_LOG_UPSTREAM_BODY_LIMIT`

## Run locally

```bash
python -m codex_openai_server
```

Or use the installed console script:

```bash
codex_openai_server
```

## Use with OpenAI clients

Point your client at `http://127.0.0.1:8000/v1` and use the local API key you set in `.env`.

```python
import openai

client = openai.OpenAI(
    api_key="your-local-server-key",
    base_url="http://127.0.0.1:8000/v1",
)

response = client.responses.create(
    model="gpt-5.5",
    input="Reply with exactly: hello",
)

print(response.output_text)
```

## Use with Copilot CLI

For GPT-5 series models, configure the Copilot CLI to use the Responses wire API. The examples below use `COPILOT_MODEL` consistently for the model selection value.

```powershell
$env:COPILOT_PROVIDER_TYPE = "openai"
$env:COPILOT_PROVIDER_BASE_URL = "http://127.0.0.1:8000/v1"
$env:COPILOT_PROVIDER_API_KEY = "your-local-server-key"
$env:COPILOT_MODEL = "gpt-5.5"
$env:COPILOT_PROVIDER_WIRE_API = "responses"

copilot -p "who are you?" --disable-builtin-mcps --allow-all-tools --stream on
```

Without `COPILOT_PROVIDER_WIRE_API=responses`, the Copilot CLI may default to the wrong wire format for GPT-5 models.

## Logging

The server supports env-controlled proxy logging for debugging upstream compatibility issues.

```dotenv
OPENAI_COMPAT_LOG_LEVEL=INFO
OPENAI_COMPAT_LOG_FORMAT=text
OPENAI_COMPAT_DEBUG_LOGGING=false
OPENAI_COMPAT_LOG_PAYLOADS=false
OPENAI_COMPAT_LOG_UPSTREAM_BODY_LIMIT=4000
```

Set `OPENAI_COMPAT_DEBUG_LOGGING=true` to log request summaries for `/v1/responses` and `/v1/chat/completions`, plus upstream request and error diagnostics.
Set `OPENAI_COMPAT_LOG_FORMAT=json` to emit structured JSON log lines for the `codex_openai_server.*` loggers.
Set `OPENAI_COMPAT_LOG_PAYLOADS=true` only when you explicitly want full request bodies in logs.
The proxy automatically retries a `/responses` call once with a forced auth refresh after upstream `401`, `403`, or `404` responses. If that retry succeeds, the upstream logger emits `Upstream Codex request succeeded after auth refresh retry` with structured fields including `recovered_from_status_code` and `retried_with_fresh_auth=true`. If upstream still returns `404` after the retry, the proxy surfaces it as a `502` because that failure is treated as an upstream/auth state problem rather than a client payload problem.

## Docker Compose

The Compose setup mounts your local Codex auth directory read-only into the container.
By default it pulls the published GHCR image for `codex_openai_server` pinned to the current release tag.
The default auth mount path can use `~/.codex`; on this Windows machine, `docker compose config` resolved that to the actual user home directory correctly.

Set `LOCAL_CODEX_HOME` in `.env` to your real Codex directory, for example:

```dotenv
LOCAL_CODEX_HOME=~/.codex
```

Then run:

```bash
docker compose up
```

If you want to override the published image version explicitly:

```dotenv
CODEX_OPENAI_SERVER_IMAGE_VERSION=v0.1.0
```

The default published-image tag is managed in the repo and updated by `bumpver`, so the default `pull_policy` is `missing` instead of `always`. That avoids re-pulling on every run while still giving you a version-pinned default.

If you want to build and use the image locally, use the tracked override file:

```bash
docker compose -f docker-compose.yaml -f docker-compose.local.yaml up --build
```

The repository Dockerfile defaults to Python 3.14 for local image builds. If you want to verify the minimum supported runtime explicitly, override the build arg, for example:

```bash
docker build --build-arg PYTHON_VERSION=3.10 -t codex_openai_server:py310 .
```

That override switches the image tag to `codex_openai_server:local`, sets `pull_policy: never`, bind-mounts the repository into `/workspace`, and runs `uvicorn --reload` with `PYTHONPATH=/workspace` so Python code changes are picked up without rebuilding the image.

The first run still needs `--build` so the local image exists. After that, you can usually use:

```bash
docker compose -f docker-compose.yaml -f docker-compose.local.yaml up
```

On Docker Desktop for Windows, the local override forces polling-based reloads so file changes inside the bind mount are detected reliably.

## Release management

This project uses `bumpver` for version and tag management.

Preview the next patch release:

```bash
bumpver update --patch --dry --no-fetch
```

Create the version commit and `vX.Y.Z` tag locally with the direct bumpver flow:

```bash
bumpver update --patch --no-push
```

That version bump also updates the default published Docker tag used in Compose.

The GitHub Actions release workflows are set up to publish Python artifacts to PyPI and Docker images to GHCR from version tags. The Docker publish workflow also smoke-tests the built image before it pushes release tags.

For publish-facing changes, keep [CHANGELOG.md](CHANGELOG.md), [README.md](README.md), and package metadata in sync so PyPI and GitHub release surfaces tell the same story.

## Development

Run checks locally:

```bash
pre-commit run --all-files
python -m pytest
```

Contributor workflow notes are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

Thanks to Simon Willison for the blog post [A pelican for GPT-5.5 via the semi-official Codex backdoor API](https://simonwillison.net/2026/Apr/23/gpt-5-5/) and for publishing [llm-openai-via-codex](https://github.com/simonw/llm-openai-via-codex), which helped inspire this OpenAI-compatible Codex proxy.
