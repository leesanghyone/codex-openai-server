from types import SimpleNamespace

from codex_openai_server.openai_server import (
    chat_to_responses_request,
    parsed_response_to_chat_completion,
)


def test_chat_function_tool_schema_becomes_responses_function_tool_schema():
    payload = chat_to_responses_request(
        {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Use the tool."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "description": "Get weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                        "strict": True,
                    },
                },
            ],
        },
    )

    assert payload["tools"] == [
        {
            "type": "function",
            "name": "lookup_weather",
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "strict": True,
        },
    ]
    assert "function" not in payload["tools"][0]


def test_responses_style_function_tool_is_preserved():
    tool = {
        "type": "function",
        "name": "lookup_weather",
        "description": "Get weather.",
        "parameters": {"type": "object"},
        "strict": False,
    }

    payload = chat_to_responses_request(
        {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Use the tool."}],
            "tools": [tool],
        },
    )

    assert payload["tools"] == [tool]
    assert "function" not in payload["tools"][0]


def test_builtin_or_unknown_tool_shape_is_preserved():
    tool = {"type": "web_search"}

    payload = chat_to_responses_request(
        {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Search."}],
            "tools": [tool],
        },
    )

    assert payload["tools"] == [tool]


def test_forced_chat_function_tool_choice_becomes_responses_tool_choice():
    payload = chat_to_responses_request(
        {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Use lookup_weather."}],
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup_weather"},
            },
        },
    )

    assert payload["tool_choice"] == {
        "type": "function",
        "name": "lookup_weather",
    }
    assert "function" not in payload["tool_choice"]


def test_chat_tool_result_message_translates_to_function_call_output():
    payload = chat_to_responses_request(
        {
            "model": "gpt-5.5",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": '{"temperature":22}',
                },
            ],
        },
    )

    assert payload["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": '{"temperature":22}',
        },
    ]


def test_responses_function_call_translates_back_to_chat_tool_calls():
    chat_completion = parsed_response_to_chat_completion(
        SimpleNamespace(
            text="",
            response={
                "id": "resp_123",
                "created_at": 1777000000,
                "model": "gpt-5.5",
            },
            output_items=[
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "lookup_weather",
                    "arguments": {"city": "Paris"},
                },
            ],
            usage=None,
        ),
    )

    assert chat_completion["choices"][0]["message"]["tool_calls"] == [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city": "Paris"}',
            },
        },
    ]
    assert chat_completion["choices"][0]["finish_reason"] == "tool_calls"


def test_regression_translated_payload_has_top_level_tool_name():
    payload = chat_to_responses_request(
        {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Use the tool."}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup_weather"},
                },
            ],
        },
    )

    assert payload["tools"][0]["name"] == "lookup_weather"
    assert "function" not in payload["tools"][0]
