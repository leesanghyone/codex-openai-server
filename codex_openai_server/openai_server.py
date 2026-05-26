from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse

from . import __version__
from .codex_auth import _auth_path, _read_auth
from .codex_http import (
    UpstreamHTTPError,
    debug_logging_enabled,
    payload_logging_enabled,
)
from .logging_utils import configure_application_logging

API_KEY_ENV_VAR = "OPENAI_COMPAT_API_KEY"
LOGGER = logging.getLogger("codex_openai_server.server")


class Backend(Protocol):
    async def list_models(self) -> list[str]: ...

    async def create_response(self, **kwargs: Any): ...

    def stream_response(
        self,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]: ...


@dataclass(slots=True)
class Settings:
    api_key: str

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(find_dotenv(usecwd=True))
        api_key = os.environ.get(API_KEY_ENV_VAR)
        if not api_key:
            raise RuntimeError(
                f"Set {API_KEY_ENV_VAR} in .env before starting the server.",
            )
        return cls(api_key=api_key)


def validate_startup_configuration() -> Settings:
    settings = Settings.from_env()
    _read_auth(_auth_path())
    return settings


def create_app(*, settings: Settings, backend: Backend) -> FastAPI:
    configure_application_logging()
    app = FastAPI(title="Codex OpenAI Compatibility Server", version=__version__)

    def openai_error_response(status_code: int, **detail: Any) -> JSONResponse:
        error = {
            "message": detail.get("message", "Request failed."),
            "type": detail.get("type", "invalid_request_error"),
            "param": detail.get("param"),
            "code": detail.get("code"),
        }
        return JSONResponse(status_code=status_code, content={"error": error})

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException):
        detail = (
            exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        )
        return openai_error_response(exc.status_code, **detail)

    @app.exception_handler(UpstreamHTTPError)
    async def upstream_http_exception_handler(_: Request, exc: UpstreamHTTPError):
        return openai_error_response(
            400 if 400 <= exc.status_code < 500 else 502,
            message=exc.message,
            type="invalid_request_error" if exc.status_code < 500 else "server_error",
            code=f"upstream_http_{exc.status_code}",
        )

    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return openai_error_response(
                401,
                message="Missing bearer API key.",
                type="invalid_request_error",
                code="invalid_api_key",
            )

        provided_key = auth_header.split(" ", 1)[1]
        if provided_key != settings.api_key:
            return openai_error_response(
                401,
                message="Invalid API key provided.",
                type="invalid_request_error",
                code="invalid_api_key",
            )

        return await call_next(request)

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        models = await backend.list_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "openai",
                }
                for model in models
            ],
        }

    @app.post("/v1/responses")
    async def create_response(request: Request):
        body = await request.json()
        response_kwargs = responses_request_from_body(body)
        log_proxy_request("responses", body, response_kwargs)
        if body.get("stream"):
            stream = await prime_stream(backend.stream_response(**response_kwargs))
            return EventSourceResponse(
                responses_event_source(stream),
                ping=15,
                headers={"Cache-Control": "no-cache"},
            )

        parsed = await backend.create_response(**response_kwargs)
        response_payload = dict(parsed.response)
        response_payload["output"] = parsed.output_items
        response_payload["output_text"] = parsed.text
        return response_payload

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: Request):
        body = await request.json()
        response_kwargs = chat_to_responses_request(body)
        log_proxy_request("chat.completions", body, response_kwargs)
        if body.get("stream"):
            stream = await prime_stream(backend.stream_response(**response_kwargs))
            return EventSourceResponse(
                chat_completions_event_source(
                    stream,
                    requested_model=body["model"],
                ),
                ping=15,
                headers={"Cache-Control": "no-cache"},
            )

        parsed = await backend.create_response(**response_kwargs)
        return parsed_response_to_chat_completion(parsed)

    return app


def create_default_app() -> FastAPI:
    from .codex_http import create_response, list_models

    class CodexBackend:
        async def list_models(self) -> list[str]:
            return await list_models()

        async def create_response(self, **kwargs: Any):
            return await create_response(**kwargs)

        def stream_response(
            self,
            **kwargs: Any,
        ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
            from .codex_http import stream_response

            return stream_response(**kwargs)

    return create_app(settings=validate_startup_configuration(), backend=CodexBackend())


async def prime_stream(
    events: AsyncIterator[tuple[str, dict[str, Any]]],
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    first_event = await anext(events)

    async def replay() -> AsyncIterator[tuple[str, dict[str, Any]]]:
        yield first_event
        async for event in events:
            yield event

    return replay()


if __name__ == "__main__":
    from .__main__ import main

    main()


def responses_request_from_body(body: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "model",
        "input",
        "instructions",
        "store",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "parallel_tool_calls",
        "previous_response_id",
        "reasoning",
        "text",
        "truncation",
    }
    return {
        key: value
        for key, value in body.items()
        if key in allowed_keys and value is not None
    }


def chat_to_responses_request(body: dict[str, Any]) -> dict[str, Any]:
    instructions_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for message in body.get("messages", []):
        role = message["role"]
        content = normalize_chat_content(message.get("content"))

        if role == "system":
            if isinstance(content, str) and content:
                instructions_parts.append(content)
            continue

        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": content,
                },
            )
            continue

        if role == "assistant" and message.get("tool_calls"):
            if content:
                input_items.append({"role": "assistant", "content": content})
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function", {})
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.get("id"),
                        "name": function.get("name"),
                        "arguments": function.get("arguments", "{}"),
                    },
                )
            continue

        input_items.append({"role": role, "content": content})

    payload: dict[str, Any] = {
        "model": body["model"],
        "input": input_items,
    }
    if instructions_parts:
        payload["instructions"] = "\n\n".join(instructions_parts)
    if body.get("tools") is not None:
        payload["tools"] = chat_tools_to_responses_tools(body["tools"])
    if body.get("tool_choice") is not None:
        payload["tool_choice"] = chat_tool_choice_to_responses_tool_choice(
            body["tool_choice"],
        )
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        payload["top_p"] = body["top_p"]

    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    if max_tokens is not None:
        payload["max_output_tokens"] = max_tokens

    return payload


def chat_tools_to_responses_tools(tools: Any) -> Any:
    if not isinstance(tools, list):
        return tools
    return [chat_tool_to_responses_tool(tool) for tool in tools]


def chat_tool_to_responses_tool(tool: Any) -> Any:
    if not isinstance(tool, dict):
        return tool
    if tool.get("type") != "function":
        return tool
    function = tool.get("function")
    if not isinstance(function, dict):
        return tool

    responses_tool = {key: value for key, value in tool.items() if key != "function"}
    for key in ("name", "description", "parameters", "strict"):
        if key in function:
            responses_tool[key] = function[key]
    return responses_tool


def chat_tool_choice_to_responses_tool_choice(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice.get("type") != "function":
        return tool_choice
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        return tool_choice

    responses_tool_choice = {
        key: value for key, value in tool_choice.items() if key != "function"
    }
    if "name" in function:
        responses_tool_choice["name"] = function["name"]
    return responses_tool_choice


def normalize_chat_content(content: Any) -> Any:
    if isinstance(content, list):
        normalized_parts: list[dict[str, Any]] = []
        for item in content:
            if item.get("type") == "text":
                normalized_parts.append(
                    {"type": "input_text", "text": item.get("text", "")},
                )
            elif item.get("type") == "image_url":
                image = item.get("image_url") or {}
                if isinstance(image, dict):
                    normalized_parts.append(
                        {
                            "type": "input_image",
                            "image_url": image.get("url"),
                            "detail": image.get("detail", "auto"),
                        },
                    )
                else:
                    normalized_parts.append(
                        {"type": "input_image", "image_url": image, "detail": "auto"},
                    )
            else:
                normalized_parts.append(item)
        return normalized_parts
    return content or ""


def parsed_response_to_chat_completion(parsed: Any) -> dict[str, Any]:
    tool_calls = output_items_to_tool_calls(parsed.output_items)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": parsed.text or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = None
    if parsed.usage:
        usage = {
            "prompt_tokens": parsed.usage.get("input_tokens", 0),
            "completion_tokens": parsed.usage.get("output_tokens", 0),
            "total_tokens": parsed.usage.get("total_tokens", 0),
        }

    return {
        "id": parsed.response.get("id", "chatcmpl_codex"),
        "object": "chat.completion",
        "created": parsed.response.get("created_at", 0),
        "model": parsed.response.get("model"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "logprobs": None,
            },
        ],
        "usage": usage,
    }


def output_items_to_tool_calls(
    output_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for item in output_items:
        if item.get("type") != "function_call":
            continue
        tool_calls.append(
            {
                "id": item.get("call_id") or item.get("id"),
                "type": "function",
                "function": {
                    "name": item.get("name"),
                    "arguments": _stringify_arguments(item.get("arguments")),
                },
            },
        )
    return tool_calls


def _stringify_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments or {})


def log_proxy_request(
    route_name: str,
    original_body: dict[str, Any],
    translated_body: dict[str, Any],
) -> None:
    if not debug_logging_enabled():
        return

    payload = {
        "route": route_name,
        "stream": bool(original_body.get("stream")),
        "model": original_body.get("model"),
        "translated_keys": sorted(translated_body.keys()),
        "message_count": len(original_body.get("messages") or []),
        "tool_count": len(original_body.get("tools") or []),
    }
    if payload_logging_enabled():
        payload["request_body"] = original_body
        payload["translated_body"] = translated_body

    LOGGER.info("Received compatibility request", extra=payload)


async def responses_event_source(events: Any):
    output_items: list[dict[str, Any]] = []
    delta_chunks: list[str] = []

    async for event_name, payload in events:
        if event_name == "response.output_text.delta":
            delta = payload.get("delta", "")
            if delta:
                delta_chunks.append(delta)
        elif event_name == "response.output_item.done":
            item = payload.get("item")
            if isinstance(item, dict):
                output_items.append(item)
        elif event_name == "response.completed":
            payload = augment_completed_response_payload(
                payload,
                output_items=output_items,
                delta_text="".join(delta_chunks),
            )

        yield {
            "event": event_name,
            "data": json.dumps(payload),
        }


def augment_completed_response_payload(
    payload: dict[str, Any],
    *,
    output_items: list[dict[str, Any]],
    delta_text: str,
) -> dict[str, Any]:
    response = payload.get("response")
    if not isinstance(response, dict):
        return payload

    augmented_response = dict(response)
    if not augmented_response.get("output"):
        augmented_response["output"] = output_items or synthetic_output_items(
            delta_text,
        )
    if delta_text and not augmented_response.get("output_text"):
        augmented_response["output_text"] = delta_text

    augmented_payload = dict(payload)
    augmented_payload["response"] = augmented_response
    return augmented_payload


def synthetic_output_items(delta_text: str) -> list[dict[str, Any]]:
    if not delta_text:
        return []

    return [
        {
            "id": "msg_stream_synthetic",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": delta_text,
                    "annotations": [],
                },
            ],
        },
    ]


async def chat_completions_event_source(events: Any, *, requested_model: str):
    response_id = "chatcmpl_codex"
    model = requested_model
    created = 0
    finish_reason = "stop"

    async for event_name, payload in events:
        if event_name == "response.created":
            response = payload.get("response", {})
            response_id = response.get("id", response_id)
            model = response.get("model", model)
            created = response.get("created_at", created)
            yield {
                "data": json.dumps(
                    chat_completion_chunk(
                        response_id,
                        model,
                        created,
                        {"role": "assistant"},
                        None,
                    ),
                ),
            }
            continue

        if event_name == "response.output_text.delta":
            delta = payload.get("delta", "")
            if delta:
                yield {
                    "data": json.dumps(
                        chat_completion_chunk(
                            response_id,
                            model,
                            created,
                            {"content": delta},
                            None,
                        ),
                    ),
                }
            continue

        if event_name == "response.output_item.done":
            item = payload.get("item", {})
            if item.get("type") == "function_call":
                finish_reason = "tool_calls"
                yield {
                    "data": json.dumps(
                        chat_completion_chunk(
                            response_id,
                            model,
                            created,
                            {"tool_calls": [stream_tool_call(item)]},
                            None,
                        ),
                    ),
                }
            continue

        if event_name == "response.completed":
            response = payload.get("response", {})
            response_id = response.get("id", response_id)
            model = response.get("model", model)
            created = response.get("created_at", created)
            yield {
                "data": json.dumps(
                    chat_completion_chunk(
                        response_id,
                        model,
                        created,
                        {},
                        finish_reason,
                    ),
                ),
            }
            yield {"data": "[DONE]"}
            break


def chat_completion_chunk(
    response_id: str,
    model: str,
    created: int,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": None,
            },
        ],
    }


def stream_tool_call(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": 0,
        "id": item.get("call_id") or item.get("id"),
        "type": "function",
        "function": {
            "name": item.get("name"),
            "arguments": _stringify_arguments(item.get("arguments")),
        },
    }
