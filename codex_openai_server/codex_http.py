from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from .codex_auth import CODEX_BASE_URL, borrow_codex_key

DEFAULT_INSTRUCTIONS = "You are a helpful assistant."
MODELS_CLIENT_VERSION = "1.0.0"
LOGGER = logging.getLogger("codex_openai_server.upstream")
UPSTREAM_BODY_LIMIT_ENV_VAR = "OPENAI_COMPAT_LOG_UPSTREAM_BODY_LIMIT"
DEBUG_LOGGING_ENV_VAR = "OPENAI_COMPAT_DEBUG_LOGGING"
LOG_PAYLOADS_ENV_VAR = "OPENAI_COMPAT_LOG_PAYLOADS"
AUTH_REFRESH_RETRY_STATUS_CODES = frozenset({401, 403, 404})


@dataclass(slots=True)
class ParsedResponse:
    text: str
    delta_text: str
    response: dict[str, Any]
    output_items: list[dict[str, Any]]
    usage: dict[str, Any] | None


@dataclass(slots=True)
class UpstreamHTTPError(Exception):
    status_code: int
    message: str
    response_text: str
    request_url: str

    def __str__(self) -> str:
        return self.message


def normalize_input(input_value: Any) -> list[dict[str, Any]]:
    if isinstance(input_value, list):
        return input_value
    if isinstance(input_value, dict):
        return [input_value]
    return [{"role": "user", "content": input_value}]


def build_responses_request(
    *,
    model: str,
    input: Any,
    instructions: str | None = None,
    stream: bool | None = None,
    store: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    del stream

    payload = {
        "model": model,
        "input": normalize_input(input),
        "instructions": instructions or DEFAULT_INSTRUCTIONS,
        "store": store,
        "stream": True,
    }
    for key, value in kwargs.items():
        if value is not None:
            payload[key] = value
    return payload


def debug_logging_enabled() -> bool:
    return _env_flag(DEBUG_LOGGING_ENV_VAR)


def payload_logging_enabled() -> bool:
    return _env_flag(LOG_PAYLOADS_ENV_VAR)


def upstream_body_limit() -> int:
    raw_value = os.environ.get(UPSTREAM_BODY_LIMIT_ENV_VAR, "4000")
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 4000


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model": payload.get("model"),
        "stream": payload.get("stream"),
        "store": payload.get("store"),
        "tool_count": len(payload.get("tools") or []),
        "input": summarize_input(payload.get("input")),
    }
    if payload_logging_enabled():
        summary["payload"] = payload
    return summary


def summarize_input(input_value: Any) -> Any:
    if isinstance(input_value, str):
        return {"kind": "string", "length": len(input_value)}
    if isinstance(input_value, dict):
        return {
            "kind": "object",
            "keys": sorted(input_value.keys()),
            "role": input_value.get("role"),
            "type": input_value.get("type"),
        }
    if isinstance(input_value, list):
        items = []
        for item in input_value[:10]:
            if isinstance(item, dict):
                items.append(
                    {
                        "role": item.get("role"),
                        "type": item.get("type"),
                        "content_type": type(item.get("content")).__name__,
                    },
                )
            else:
                items.append({"type": type(item).__name__})
        summary = {"kind": "list", "count": len(input_value), "items": items}
        if len(input_value) > 10:
            summary["truncated"] = True
        return summary
    return {"kind": type(input_value).__name__}


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted = dict(headers)
    if "Authorization" in redacted:
        redacted["Authorization"] = "Bearer ***"
    if "ChatGPT-Account-ID" in redacted:
        redacted["ChatGPT-Account-ID"] = "***"
    return redacted


async def raise_for_status_with_context(
    response: httpx.Response,
    *,
    payload: dict[str, Any],
) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        response_text = await response.aread()
        decoded_text = response_text.decode("utf-8", errors="replace")
        excerpt = decoded_text[: upstream_body_limit()]
        LOGGER.error(
            "Upstream Codex request failed",
            extra={
                "url": str(response.request.url),
                "status_code": response.status_code,
                "payload_summary": summarize_payload(payload),
                "response_excerpt": excerpt,
            },
        )
        raise UpstreamHTTPError(
            status_code=response.status_code,
            message=(
                f"Upstream Codex /responses request failed with status {response.status_code}."
            ),
            response_text=excerpt,
            request_url=str(response.request.url),
        ) from exc


def iter_sse_events(
    lines: Iterable[bytes | str],
) -> Iterator[tuple[str, dict[str, Any]]]:
    event_type = "message"
    data_lines: list[str] = []

    for raw_line in lines:
        line = (
            raw_line.decode("utf-8", errors="replace")
            if isinstance(raw_line, bytes)
            else raw_line
        )
        line = line.rstrip("\r\n")

        if not line:
            if data_lines:
                yield event_type, _parse_sse_event_data(event_type, data_lines)
            event_type = "message"
            data_lines = []
            continue

        if line.startswith("event:"):
            event_type = line[6:].strip()
            continue

        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        yield event_type, _parse_sse_event_data(event_type, data_lines)


def parse_response_stream(lines: Iterable[bytes | str]) -> ParsedResponse:
    output_items: list[dict[str, Any]] = []
    delta_chunks: list[str] = []
    response: dict[str, Any] | None = None

    for event_type, payload in iter_sse_events(lines):
        if event_type == "response.output_text.delta":
            delta_chunks.append(_optional_event_string(payload, "delta", event_type))
        elif event_type == "response.output_item.done":
            output_items.append(_require_event_object(payload, "item", event_type))
        elif event_type == "response.completed":
            response = _require_event_object(payload, "response", event_type)
            break

    if response is None:
        raise ValueError("response.completed event not found in stream")

    delta_text = "".join(delta_chunks)
    text = (
        _text_from_output_items(output_items)
        or response.get("output_text")
        or delta_text
    )
    return ParsedResponse(
        text=text,
        delta_text=delta_text,
        response=response,
        output_items=output_items,
        usage=response.get("usage"),
    )


async def parse_response_stream_async(
    lines: AsyncIterable[bytes | str],
) -> ParsedResponse:
    output_items: list[dict[str, Any]] = []
    delta_chunks: list[str] = []
    response: dict[str, Any] | None = None

    async for event_type, payload in iter_sse_events_async(lines):
        if event_type == "response.output_text.delta":
            delta_chunks.append(_optional_event_string(payload, "delta", event_type))
        elif event_type == "response.output_item.done":
            output_items.append(_require_event_object(payload, "item", event_type))
        elif event_type == "response.completed":
            response = _require_event_object(payload, "response", event_type)
            break

    if response is None:
        raise ValueError("response.completed event not found in stream")

    delta_text = "".join(delta_chunks)
    text = (
        _text_from_output_items(output_items)
        or response.get("output_text")
        or delta_text
    )
    return ParsedResponse(
        text=text,
        delta_text=delta_text,
        response=response,
        output_items=output_items,
        usage=response.get("usage"),
    )


async def create_response(
    *,
    model: str,
    input: Any,
    base_url: str = CODEX_BASE_URL,
    client: httpx.AsyncClient | None = None,
    **kwargs: Any,
) -> ParsedResponse:
    owns_client = client is None
    async_client = client if client is not None else httpx.AsyncClient()
    retry_status_code: int | None = None

    try:
        for force_auth_refresh in (False, True):
            request_kwargs = await build_request_kwargs(
                model=model,
                input=input,
                base_url=base_url,
                client=async_client,
                force_auth_refresh=force_auth_refresh,
                **kwargs,
            )
            log_upstream_request(request_kwargs)
            async with async_client.stream(
                "POST",
                request_kwargs["url"],
                headers=request_kwargs["headers"],
                json=request_kwargs["json"],
            ) as response:
                try:
                    await raise_for_status_with_context(
                        response,
                        payload=request_kwargs["json"],
                    )
                except UpstreamHTTPError as exc:
                    if _should_retry_with_fresh_auth(
                        exc,
                        force_auth_refresh=force_auth_refresh,
                    ):
                        retry_status_code = exc.status_code
                        _log_auth_refresh_retry(exc)
                        continue
                    raise _normalize_upstream_error(exc) from exc
                if retry_status_code is not None:
                    _log_auth_refresh_retry_success(
                        request_kwargs,
                        retry_status_code=retry_status_code,
                    )
                return await parse_response_stream_async(response.aiter_lines())
        raise RuntimeError("unreachable")
    finally:
        if owns_client:
            await async_client.aclose()


async def stream_response(
    *,
    model: str,
    input: Any,
    base_url: str = CODEX_BASE_URL,
    client: httpx.AsyncClient | None = None,
    **kwargs: Any,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    owns_client = client is None
    async_client = client if client is not None else httpx.AsyncClient()
    retry_status_code: int | None = None

    try:
        for force_auth_refresh in (False, True):
            request_kwargs = await build_request_kwargs(
                model=model,
                input=input,
                base_url=base_url,
                client=async_client,
                force_auth_refresh=force_auth_refresh,
                **kwargs,
            )
            log_upstream_request(request_kwargs)
            async with async_client.stream(
                "POST",
                request_kwargs["url"],
                headers=request_kwargs["headers"],
                json=request_kwargs["json"],
            ) as response:
                try:
                    await raise_for_status_with_context(
                        response,
                        payload=request_kwargs["json"],
                    )
                except UpstreamHTTPError as exc:
                    if _should_retry_with_fresh_auth(
                        exc,
                        force_auth_refresh=force_auth_refresh,
                    ):
                        retry_status_code = exc.status_code
                        _log_auth_refresh_retry(exc)
                        continue
                    raise _normalize_upstream_error(exc) from exc
                if retry_status_code is not None:
                    _log_auth_refresh_retry_success(
                        request_kwargs,
                        retry_status_code=retry_status_code,
                    )
                async for event in iter_sse_events_async(response.aiter_lines()):
                    yield event
                return
    finally:
        if owns_client:
            await async_client.aclose()


async def list_models(
    base_url: str = CODEX_BASE_URL,
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    owns_client = client is None
    async_client = client if client is not None else httpx.AsyncClient()

    try:
        token, account_id = await _borrow_codex_key(
            async_client,
            force_refresh=False,
        )
        headers = {"Authorization": f"Bearer {token}"}
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id

        response = await async_client.get(
            f"{base_url}/models",
            headers=headers,
            params={"client_version": MODELS_CLIENT_VERSION},
        )
        await raise_for_status_with_context(
            response,
            payload={
                "operation": "list_models",
                "client_version": MODELS_CLIENT_VERSION,
            },
        )
        payload = response.json()
    finally:
        if owns_client:
            await async_client.aclose()

    return [
        model["slug"]
        for model in payload.get("models", [])
        if model.get("supported_in_api") and model.get("visibility") == "list"
    ]


def _text_from_output_items(output_items: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in output_items:
        if item.get("type") != "message":
            continue
        for content_item in item.get("content", []):
            if content_item.get("type") == "output_text":
                chunks.append(content_item.get("text", ""))
    return "".join(chunks)


def _parse_sse_event_data(
    event_type: str,
    data_lines: list[str],
) -> dict[str, Any]:
    raw_payload = "\n".join(data_lines)
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed SSE JSON for event '{event_type}': {exc.msg}",
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"SSE event '{event_type}' payload must be a JSON object",
        )

    return payload


def _require_event_object(
    payload: dict[str, Any],
    key: str,
    event_type: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(
            f"SSE event '{event_type}' missing required object field '{key}'",
        )
    return value


def _optional_event_string(
    payload: dict[str, Any],
    key: str,
    event_type: str,
) -> str:
    value = payload.get(key, "")
    if isinstance(value, str):
        return value
    raise ValueError(
        f"SSE event '{event_type}' field '{key}' must be a string",
    )


async def build_request_kwargs(
    *,
    model: str,
    input: Any,
    base_url: str = CODEX_BASE_URL,
    client: httpx.AsyncClient | None = None,
    force_auth_refresh: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    token, account_id = await _borrow_codex_key(
        client,
        force_refresh=force_auth_refresh,
    )
    payload = build_responses_request(model=model, input=input, **kwargs)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id

    return {
        "url": f"{base_url}/responses",
        "headers": headers,
        "json": payload,
    }


def log_upstream_request(request_kwargs: dict[str, Any]) -> None:
    if not debug_logging_enabled():
        return

    LOGGER.info(
        "Forwarding request to upstream Codex endpoint",
        extra={
            "url": request_kwargs["url"],
            "headers": redact_headers(request_kwargs["headers"]),
            "payload_summary": summarize_payload(request_kwargs["json"]),
        },
    )


async def _borrow_codex_key(
    client: httpx.AsyncClient | None,
    *,
    force_refresh: bool,
) -> tuple[str, str | None]:
    try:
        return await borrow_codex_key(
            client=client,
            force_refresh=force_refresh,
        )
    except TypeError:
        try:
            return await borrow_codex_key(client=client)
        except TypeError:
            return await borrow_codex_key()


def _should_retry_with_fresh_auth(
    exc: UpstreamHTTPError,
    *,
    force_auth_refresh: bool,
) -> bool:
    return not force_auth_refresh and exc.status_code in AUTH_REFRESH_RETRY_STATUS_CODES


def _normalize_upstream_error(exc: UpstreamHTTPError) -> UpstreamHTTPError:
    if exc.status_code != 404:
        return exc

    return UpstreamHTTPError(
        status_code=502,
        message=(
            "Upstream Codex /responses returned 404 after retrying with a refreshed auth token."
        ),
        response_text=exc.response_text,
        request_url=exc.request_url,
    )


def _log_auth_refresh_retry(exc: UpstreamHTTPError) -> None:
    LOGGER.warning(
        "Retrying upstream Codex request with refreshed auth",
        extra={
            "url": exc.request_url,
            "status_code": exc.status_code,
        },
    )


def _log_auth_refresh_retry_success(
    request_kwargs: dict[str, Any],
    *,
    retry_status_code: int,
) -> None:
    LOGGER.info(
        "Upstream Codex request succeeded after auth refresh retry",
        extra={
            "url": request_kwargs["url"],
            "recovered_from_status_code": retry_status_code,
            "retried_with_fresh_auth": True,
            "payload_summary": summarize_payload(request_kwargs["json"]),
        },
    )


async def iter_sse_events_async(
    lines: AsyncIterable[bytes | str],
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event_type = "message"
    data_lines: list[str] = []

    async for raw_line in lines:
        line = (
            raw_line.decode("utf-8", errors="replace")
            if isinstance(raw_line, bytes)
            else raw_line
        )
        line = line.rstrip("\r\n")

        if not line:
            if data_lines:
                yield event_type, _parse_sse_event_data(event_type, data_lines)
            event_type = "message"
            data_lines = []
            continue

        if line.startswith("event:"):
            event_type = line[6:].strip()
            continue

        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        yield event_type, _parse_sse_event_data(event_type, data_lines)
