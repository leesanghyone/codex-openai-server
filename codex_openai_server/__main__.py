from __future__ import annotations

import os

import uvicorn

from .logging_utils import configure_root_logging
from .openai_server import validate_startup_configuration


def main() -> None:
    uvicorn_log_level = configure_root_logging()
    validate_startup_configuration()
    uvicorn.run(
        "codex_openai_server.openai_server:create_default_app",
        factory=True,
        host=os.environ.get("OPENAI_COMPAT_HOST", "127.0.0.1"),
        port=int(os.environ.get("OPENAI_COMPAT_PORT", "8000")),
        log_level=uvicorn_log_level,
    )


if __name__ == "__main__":
    main()
