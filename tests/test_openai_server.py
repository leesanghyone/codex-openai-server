import httpx
import openai
import pytest
from codex_openai_server import __main__ as main_module
from codex_openai_server.codex_auth import BorrowKeyError
from codex_openai_server.codex_http import ParsedResponse, UpstreamHTTPError
from codex_openai_server.openai_server import (
    API_KEY_ENV_VAR,
    Settings,
    create_app,
    validate_startup_configuration,
)


class FakeBackend:
    def __init__(self):
        self.list_models_calls = 0
        self.create_response_calls = []
        self.stream_response_calls = []
        self.next_response = None
        self.next_stream_events = None
        self.raise_create_error = None
        self.raise_stream_error = None

    async def list_models(self):
        self.list_models_calls += 1
        return ["gpt-5.5", "gpt-5.4-mini"]

    async def create_response(self, **kwargs):
        self.create_response_calls.append(kwargs)
        if self.raise_create_error is not None:
            raise self.raise_create_error
        if self.next_response is not None:
            return self.next_response
        return ParsedResponse(
            text="server-ok",
            delta_text="server-ok",
            response={
                "id": "resp_test_123",
                "object": "response",
                "created_at": 1777000000,
                "status": "completed",
                "model": kwargs["model"],
                "output": [],
                "output_text": None,
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                },
            },
            output_items=[
                {
                    "id": "msg_123",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "server-ok",
                            "annotations": [],
                        },
                    ],
                },
            ],
            usage={
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
            },
        )

    async def stream_response(self, **kwargs):
        self.stream_response_calls.append(kwargs)
        if self.raise_stream_error is not None:
            raise self.raise_stream_error
        events = self.next_stream_events or [
            (
                "response.created",
                {
                    "type": "response.created",
                    "response": {
                        "id": "resp_stream_123",
                        "created_at": 1777000002,
                        "model": kwargs["model"],
                    },
                },
            ),
            (
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "stream-"},
            ),
            (
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "ok"},
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_stream_123",
                        "created_at": 1777000002,
                        "model": kwargs["model"],
                        "status": "completed",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 8,
                            "total_tokens": 18,
                        },
                    },
                },
            ),
        ]
        for event in events:
            yield event


@pytest.mark.anyio
async def test_models_requires_bearer_api_key():
    app = create_app(settings=Settings(api_key="test-key"), backend=FakeBackend())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


@pytest.mark.anyio
async def test_models_list_works_with_openai_client():
    backend = FakeBackend()
    app = create_app(settings=Settings(api_key="test-key"), backend=backend)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as http_client:
        client = openai.AsyncOpenAI(
            api_key="test-key",
            base_url="http://testserver/v1",
            http_client=http_client,
        )
        models = await client.models.list()

    assert [model.id for model in models.data] == ["gpt-5.5", "gpt-5.4-mini"]
    assert backend.list_models_calls == 1


@pytest.mark.anyio
async def test_responses_create_works_with_openai_client():
    backend = FakeBackend()
    app = create_app(settings=Settings(api_key="test-key"), backend=backend)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as http_client:
        client = openai.AsyncOpenAI(
            api_key="test-key",
            base_url="http://testserver/v1",
            http_client=http_client,
        )
        response = await client.responses.create(model="gpt-5.5", input="Say hello")

    assert response.output_text == "server-ok"
    assert response.model == "gpt-5.5"
    assert backend.create_response_calls == [
        {
            "model": "gpt-5.5",
            "input": "Say hello",
        },
    ]


@pytest.mark.anyio
async def test_chat_completions_create_works_with_openai_client():
    backend = FakeBackend()
    app = create_app(settings=Settings(api_key="test-key"), backend=backend)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as http_client:
        client = openai.AsyncOpenAI(
            api_key="test-key",
            base_url="http://testserver/v1",
            http_client=http_client,
        )
        response = await client.chat.completions.create(
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Say hello"},
            ],
        )

    assert response.choices[0].message.content == "server-ok"
    assert response.choices[0].finish_reason == "stop"
    assert backend.create_response_calls == [
        {
            "model": "gpt-5.5",
            "input": [{"role": "user", "content": "Say hello"}],
            "instructions": "Be terse.",
        },
    ]


@pytest.mark.anyio
async def test_chat_completions_tool_calls_look_like_openai():
    backend = FakeBackend()
    backend.next_response = ParsedResponse(
        text="",
        delta_text="",
        response={
            "id": "resp_tool_123",
            "object": "response",
            "created_at": 1777000001,
            "status": "completed",
            "model": "gpt-5.5",
            "output": [],
            "output_text": None,
            "usage": {
                "input_tokens": 20,
                "output_tokens": 12,
                "total_tokens": 32,
            },
        },
        output_items=[
            {
                "id": "fc_123",
                "type": "function_call",
                "call_id": "call_123",
                "name": "lookup_weather",
                "arguments": '{"city":"Paris"}',
            },
        ],
        usage={
            "input_tokens": 20,
            "output_tokens": 12,
            "total_tokens": 32,
        },
    )
    app = create_app(settings=Settings(api_key="test-key"), backend=backend)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as http_client:
        client = openai.AsyncOpenAI(
            api_key="test-key",
            base_url="http://testserver/v1",
            http_client=http_client,
        )
        response = await client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "What is the weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "description": "Get weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                            },
                            "required": ["city"],
                        },
                    },
                },
            ],
        )

    tool_call = response.choices[0].message.tool_calls[0]
    assert response.choices[0].finish_reason == "tool_calls"
    assert tool_call.id == "call_123"
    assert tool_call.function.name == "lookup_weather"
    assert tool_call.function.arguments == '{"city":"Paris"}'


@pytest.mark.anyio
async def test_responses_stream_returns_openai_style_sse():
    backend = FakeBackend()
    app = create_app(settings=Settings(api_key="test-key"), backend=backend)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        async with client.stream(
            "POST",
            "/v1/responses",
            headers={"Authorization": "Bearer test-key"},
            json={"model": "gpt-5.5", "input": "Say hello", "stream": True},
        ) as response:
            body = "".join([chunk async for chunk in response.aiter_text()])

    assert response.status_code == 200
    assert "event: response.output_text.delta" in body
    assert '"delta": "stream-"' in body
    assert '"delta": "ok"' in body
    assert '"output_text": "stream-ok"' in body
    assert '"type": "message"' in body
    assert backend.stream_response_calls == [{"model": "gpt-5.5", "input": "Say hello"}]


@pytest.mark.anyio
async def test_chat_completions_stream_works_with_openai_client():
    backend = FakeBackend()
    app = create_app(settings=Settings(api_key="test-key"), backend=backend)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as http_client:
        client = openai.AsyncOpenAI(
            api_key="test-key",
            base_url="http://testserver/v1",
            http_client=http_client,
        )
        stream = await client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
        )

        chunks = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

    assert "".join(chunks) == "stream-ok"
    assert backend.stream_response_calls == [
        {"model": "gpt-5.5", "input": [{"role": "user", "content": "Say hello"}]},
    ]


@pytest.mark.anyio
async def test_responses_create_returns_structured_upstream_error():
    backend = FakeBackend()
    backend.raise_create_error = UpstreamHTTPError(
        status_code=400,
        message="Upstream Codex /responses request failed with status 400.",
        response_text='{"error":"bad request"}',
        request_url="https://chatgpt.com/backend-api/codex/responses",
    )
    app = create_app(settings=Settings(api_key="test-key"), backend=backend)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer test-key"},
            json={"model": "gpt-5.5", "input": "Say hello"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "upstream_http_400"


@pytest.mark.anyio
async def test_responses_stream_returns_structured_upstream_error_before_sse():
    backend = FakeBackend()
    backend.raise_stream_error = UpstreamHTTPError(
        status_code=400,
        message="Upstream Codex /responses request failed with status 400.",
        response_text='{"error":"bad request"}',
        request_url="https://chatgpt.com/backend-api/codex/responses",
    )
    app = create_app(settings=Settings(api_key="test-key"), backend=backend)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer test-key"},
            json={"model": "gpt-5.5", "input": "Say hello", "stream": True},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "upstream_http_400"


@pytest.mark.anyio
async def test_responses_create_returns_gateway_error_for_upstream_502():
    backend = FakeBackend()
    backend.raise_create_error = UpstreamHTTPError(
        status_code=502,
        message=(
            "Upstream Codex /responses returned 404 after retrying with a refreshed auth token."
        ),
        response_text='{"error":"not found"}',
        request_url="https://chatgpt.com/backend-api/codex/responses",
    )
    app = create_app(settings=Settings(api_key="test-key"), backend=backend)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer test-key"},
            json={"model": "gpt-5.5", "input": "Say hello"},
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "server_error"
    assert response.json()["error"]["code"] == "upstream_http_502"


def test_settings_from_env_reads_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    (tmp_path / ".env").write_text(
        f"{API_KEY_ENV_VAR}=dotenv-test-key\n",
        encoding="utf-8",
    )

    settings = Settings.from_env()

    assert settings.api_key == "dotenv-test-key"


def test_validate_startup_configuration_requires_codex_auth_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))

    with pytest.raises(BorrowKeyError, match="Codex auth file not found"):
        validate_startup_configuration()


def test_validate_startup_configuration_rejects_invalid_auth_mode(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(API_KEY_ENV_VAR, "test-key")
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        '{"auth_mode": "api_key", "tokens": {"access_token": "token"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    with pytest.raises(BorrowKeyError, match="Expected auth_mode 'chatgpt'"):
        validate_startup_configuration()


def test_main_runs_preflight_before_starting_uvicorn(monkeypatch):
    calls: list[str] = []

    def fake_validate() -> None:
        calls.append("validate")

    def fake_run(*args, **kwargs) -> None:
        calls.append("run")

    monkeypatch.setattr(main_module, "validate_startup_configuration", fake_validate)
    monkeypatch.setattr(main_module.uvicorn, "run", fake_run)

    main_module.main()

    assert calls == ["validate", "run"]


def test_main_does_not_start_uvicorn_when_preflight_fails(monkeypatch):
    def fake_validate() -> None:
        raise BorrowKeyError("bad auth")

    def fail_run(*args, **kwargs) -> None:
        raise AssertionError("uvicorn.run should not be called")

    monkeypatch.setattr(main_module, "validate_startup_configuration", fake_validate)
    monkeypatch.setattr(main_module.uvicorn, "run", fail_run)

    with pytest.raises(BorrowKeyError, match="bad auth"):
        main_module.main()
