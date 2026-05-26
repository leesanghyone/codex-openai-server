import json
import logging

from codex_openai_server.logging_utils import JsonFormatter, serialize_value


def test_json_formatter_serializes_message_and_extras():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="codex_openai_server.upstream",
        level=logging.ERROR,
        pathname=__file__,
        lineno=10,
        msg="Upstream Codex request failed",
        args=(),
        exc_info=None,
    )
    record.status_code = 400
    record.payload_summary = {
        "model": "gpt-5.5",
        "input": {"kind": "string", "length": 12},
    }

    payload = json.loads(formatter.format(record))

    assert payload["logger"] == "codex_openai_server.upstream"
    assert payload["level"] == "ERROR"
    assert payload["message"] == "Upstream Codex request failed"
    assert payload["status_code"] == 400
    assert payload["payload_summary"]["model"] == "gpt-5.5"


def test_serialize_value_handles_nested_values():
    value = {"items": [1, {"nested": object()}], "flag": True}

    serialized = serialize_value(value)

    assert serialized["items"][0] == 1
    assert isinstance(serialized["items"][1]["nested"], str)
    assert serialized["flag"] is True
