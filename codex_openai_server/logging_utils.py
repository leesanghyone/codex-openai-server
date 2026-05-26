from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

LOG_LEVEL_ENV_VAR = "OPENAI_COMPAT_LOG_LEVEL"
LOG_FORMAT_ENV_VAR = "OPENAI_COMPAT_LOG_FORMAT"

STANDARD_LOG_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "taskName",
    "thread",
    "threadName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in STANDARD_LOG_RECORD_FIELDS
        }
        if extras:
            payload.update(serialize_value(extras))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str)


def configure_root_logging() -> str:
    log_level = os.environ.get(LOG_LEVEL_ENV_VAR, "INFO").upper()
    formatter: logging.Formatter
    if json_logging_enabled():
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        handlers=[handler],
        force=True,
    )
    return log_level.lower()


def configure_application_logging() -> None:
    if not json_logging_enabled():
        return

    logger = logging.getLogger("codex_openai_server")
    if any(isinstance(handler.formatter, JsonFormatter) for handler in logger.handlers):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(
        getattr(
            logging,
            os.environ.get(LOG_LEVEL_ENV_VAR, "INFO").upper(),
            logging.INFO,
        ),
    )


def json_logging_enabled() -> bool:
    return os.environ.get(LOG_FORMAT_ENV_VAR, "text").strip().lower() == "json"


def serialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return [serialize_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
