import base64
import json
import logging

import httpx
import pytest
from codex_openai_server import codex_auth, codex_http
from codex_openai_server.codex_http import (
    DEFAULT_INSTRUCTIONS,
    UpstreamHTTPError,
    build_responses_request,
    parse_response_stream,
)


def _sse_lines(*entries):
    for event_type, payload in entries:
        yield f"event: {event_type}\n".encode()
        yield f"data: {json.dumps(payload)}\n".encode()
        yield b"\n"


def test_build_responses_request_normalizes_string_input():
    payload = build_responses_request(model="gpt-5.5", input="Say hello")

    assert payload == {
        "model": "gpt-5.5",
        "input": [{"role": "user", "content": "Say hello"}],
        "instructions": DEFAULT_INSTRUCTIONS,
        "store": False,
        "stream": True,
    }


def test_build_responses_request_preserves_list_input_and_extra_options():
    messages = [{"role": "user", "content": "Ping"}]

    payload = build_responses_request(
        model="gpt-5.5",
        input=messages,
        instructions="Be terse.",
        stream=False,
        store=True,
        temperature=0.25,
    )

    assert payload["model"] == "gpt-5.5"
    assert payload["input"] == messages
    assert payload["instructions"] == "Be terse."
    assert payload["stream"] is True
    assert payload["store"] is True
    assert payload["temperature"] == 0.25


def test_parse_response_stream_reconstructs_text_from_output_items():
    result = parse_response_stream(
        _sse_lines(
            (
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "item": {
                        "id": "msg_1",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "raw-http-ok",
                                "annotations": [],
                            },
                        ],
                    },
                },
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "output": [],
                        "output_text": None,
                        "usage": {
                            "input_tokens": 23,
                            "output_tokens": 7,
                            "total_tokens": 30,
                        },
                    },
                },
            ),
        ),
    )

    assert result.text == "raw-http-ok"
    assert result.response["status"] == "completed"
    assert result.usage == {
        "input_tokens": 23,
        "output_tokens": 7,
        "total_tokens": 30,
    }


def test_parse_response_stream_falls_back_to_delta_text():
    result = parse_response_stream(
        _sse_lines(
            (
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "raw-"},
            ),
            (
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "http-ok"},
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_2",
                        "status": "completed",
                        "output": [],
                        "output_text": None,
                        "usage": None,
                    },
                },
            ),
        ),
    )

    assert result.text == "raw-http-ok"
    assert result.delta_text == "raw-http-ok"


def test_parse_response_stream_raises_explicit_error_for_malformed_event_json():
    with pytest.raises(
        ValueError,
        match="Malformed SSE JSON for event 'response.completed'",
    ):
        parse_response_stream(
            [
                b"event: response.completed\n",
                b"data: {not-json}\n",
                b"\n",
            ],
        )


def test_parse_response_stream_raises_explicit_error_for_missing_item_key():
    with pytest.raises(
        ValueError,
        match="response.output_item.done'.*'item'",
    ):
        parse_response_stream(
            _sse_lines(
                (
                    "response.output_item.done",
                    {"type": "response.output_item.done"},
                ),
            ),
        )


def test_jwt_exp_returns_none_for_malformed_payload_object():
    payload = base64.urlsafe_b64encode(json.dumps(["not-an-object"]).encode()).decode()
    token = f"header.{payload}.signature"

    assert codex_auth._jwt_exp(token) is None


@pytest.mark.anyio
async def test_refresh_handles_non_json_error_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, request=request, text="upstream failure")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(codex_auth.BorrowKeyError, match="HTTP 400"):
            await codex_auth._refresh("refresh-token", client=client)


@pytest.mark.anyio
async def test_create_response_uses_async_http_client(monkeypatch):
    async def fake_borrow_codex_key():
        return "token-123", "account-456"

    monkeypatch.setattr(codex_http, "borrow_codex_key", fake_borrow_codex_key)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer token-123"
        assert request.headers["chatgpt-account-id"] == "account-456"
        body = json.loads(request.content)
        assert body["model"] == "gpt-5.5"
        assert body["input"] == [{"role": "user", "content": "Say hello"}]
        return httpx.Response(
            200,
            text=(
                "event: response.output_text.delta\n"
                'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
                "event: response.completed\n"
                'data: {"type":"response.completed","response":{"id":"resp_123","status":"completed","output":[],"output_text":null,"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
            ),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://example.test",
    ) as client:
        response = await codex_http.create_response(
            model="gpt-5.5",
            input="Say hello",
            base_url="https://example.test",
            client=client,
        )

    assert response.text == "hello"
    assert response.response["id"] == "resp_123"


@pytest.mark.anyio
async def test_create_response_raises_explicit_error_for_missing_completed_response(
    monkeypatch,
):
    async def fake_borrow_codex_key():
        return "token-123", None

    monkeypatch.setattr(codex_http, "borrow_codex_key", fake_borrow_codex_key)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                "event: response.completed\n" 'data: {"type":"response.completed"}\n\n'
            ),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://example.test",
    ) as client:
        with pytest.raises(
            ValueError,
            match="response.completed'.*'response'",
        ):
            await codex_http.create_response(
                model="gpt-5.5",
                input="Say hello",
                base_url="https://example.test",
                client=client,
            )


@pytest.mark.anyio
async def test_stream_response_yields_async_events(monkeypatch):
    async def fake_borrow_codex_key():
        return "token-123", None

    monkeypatch.setattr(codex_http, "borrow_codex_key", fake_borrow_codex_key)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                "event: response.created\n"
                'data: {"type":"response.created","response":{"id":"resp_123"}}\n\n'
                "event: response.output_text.delta\n"
                'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
                "event: response.completed\n"
                'data: {"type":"response.completed","response":{"id":"resp_123","status":"completed"}}\n\n'
            ),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://example.test",
    ) as client:
        events = [
            event
            async for event in codex_http.stream_response(
                model="gpt-5.5",
                input="Say hello",
                base_url="https://example.test",
                client=client,
            )
        ]

    assert [event[0] for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]


@pytest.mark.anyio
async def test_list_models_uses_async_http_client(monkeypatch):
    async def fake_borrow_codex_key():
        return "token-123", None

    monkeypatch.setattr(codex_http, "borrow_codex_key", fake_borrow_codex_key)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["client_version"] == codex_http.MODELS_CLIENT_VERSION
        return httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "gpt-5.5", "supported_in_api": True, "visibility": "list"},
                    {
                        "slug": "hidden",
                        "supported_in_api": True,
                        "visibility": "hidden",
                    },
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://example.test",
    ) as client:
        models = await codex_http.list_models(
            base_url="https://example.test",
            client=client,
        )

    assert models == ["gpt-5.5"]


@pytest.mark.anyio
async def test_create_response_retries_once_after_upstream_404(monkeypatch, caplog):
    borrow_calls = []
    caplog.set_level(logging.INFO, logger="codex_openai_server.upstream")

    async def fake_borrow_codex_key(*, client=None, force_refresh=False):
        del client
        borrow_calls.append(force_refresh)
        token = "fresh-token" if force_refresh else "stale-token"
        return token, "account-456"

    monkeypatch.setattr(codex_http, "borrow_codex_key", fake_borrow_codex_key)

    request_tokens = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_tokens.append(request.headers["authorization"])
        if len(request_tokens) == 1:
            return httpx.Response(
                404,
                request=request,
                text='{"error":"not found"}',
            )
        return httpx.Response(
            200,
            request=request,
            text=(
                "event: response.output_text.delta\n"
                'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
                "event: response.completed\n"
                'data: {"type":"response.completed","response":{"id":"resp_123","status":"completed","output":[],"output_text":null,"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
            ),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://example.test",
    ) as client:
        response = await codex_http.create_response(
            model="gpt-5.5",
            input="Say hello",
            base_url="https://example.test",
            client=client,
        )

    assert response.text == "hello"
    assert borrow_calls == [False, True]
    assert request_tokens == ["Bearer stale-token", "Bearer fresh-token"]
    assert any(
        record.msg == "Retrying upstream Codex request with refreshed auth"
        and getattr(record, "status_code", None) == 404
        for record in caplog.records
    )
    assert any(
        record.msg == "Upstream Codex request succeeded after auth refresh retry"
        and getattr(record, "recovered_from_status_code", None) == 404
        and getattr(record, "retried_with_fresh_auth", None) is True
        for record in caplog.records
    )


@pytest.mark.anyio
async def test_create_response_raises_502_style_error_after_persistent_upstream_404(
    monkeypatch,
):
    borrow_calls = []

    async def fake_borrow_codex_key(*, client=None, force_refresh=False):
        del client
        borrow_calls.append(force_refresh)
        token = "fresh-token" if force_refresh else "stale-token"
        return token, None

    monkeypatch.setattr(codex_http, "borrow_codex_key", fake_borrow_codex_key)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            request=request,
            text='{"error":"still not found"}',
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://example.test",
    ) as client:
        with pytest.raises(UpstreamHTTPError) as exc_info:
            await codex_http.create_response(
                model="gpt-5.5",
                input="Say hello",
                base_url="https://example.test",
                client=client,
            )

    assert exc_info.value.status_code == 502
    assert exc_info.value.request_url == "https://example.test/responses"
    assert borrow_calls == [False, True]
