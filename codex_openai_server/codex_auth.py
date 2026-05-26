from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import time
from typing import Any

import httpx

REFRESH_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REFRESH_SKEW_SECONDS = 30
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


class BorrowKeyError(Exception):
    pass


async def borrow_codex_key(
    client: httpx.AsyncClient | None = None,
    *,
    force_refresh: bool = False,
) -> tuple[str, str | None]:
    auth_path = _auth_path()
    data = await asyncio.to_thread(_read_auth, auth_path)

    tokens = data.get("tokens")
    if not tokens or not tokens.get("access_token"):
        raise BorrowKeyError(
            "No ChatGPT tokens found in auth.json. Run `codex login` first.",
        )

    access_token = tokens["access_token"]
    account_id = tokens.get("account_id")
    exp = _jwt_exp(access_token)

    if (
        not force_refresh
        and exp is not None
        and time.time() < (exp - REFRESH_SKEW_SECONDS)
    ):
        return access_token, account_id

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise BorrowKeyError(
            "No refresh token available. Run `codex login` to re-authenticate.",
        )

    new_tokens = await _refresh(refresh_token, client=client)

    if new_tokens.get("access_token"):
        tokens["access_token"] = new_tokens["access_token"]
    if new_tokens.get("id_token"):
        tokens["id_token"] = new_tokens["id_token"]
    if new_tokens.get("refresh_token"):
        tokens["refresh_token"] = new_tokens["refresh_token"]
    if new_tokens.get("account_id"):
        tokens["account_id"] = new_tokens["account_id"]
        account_id = new_tokens["account_id"]

    data["tokens"] = tokens
    data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    await asyncio.to_thread(_write_auth, auth_path, data)

    return tokens["access_token"], account_id


def _auth_path() -> str:
    codex_home = os.environ.get("CODEX_HOME", os.path.expanduser("~/.codex"))
    path = os.path.join(codex_home, "auth.json")
    if not os.path.exists(path):
        raise BorrowKeyError(
            f"Codex auth file not found at {path}. Run `codex login` first.",
        )
    return path


def _read_auth(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    if data.get("auth_mode") != "chatgpt":
        raise BorrowKeyError(
            f"Expected auth_mode 'chatgpt', got '{data.get('auth_mode')}'. This library only supports ChatGPT OAuth tokens.",
        )
    return data


def _write_auth(path: str, data: dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2)
    os.replace(tmp_path, path)
    os.chmod(path, 0o600)


def _jwt_exp(token: str) -> int | None:
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except (IndexError, binascii.Error, json.JSONDecodeError, TypeError):
        return None

    if not isinstance(payload, dict):
        return None

    exp = payload.get("exp")
    return exp if isinstance(exp, int) else None


async def _refresh(
    refresh_token: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    body = {
        "client_id": CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }

    owns_client = client is None
    async_client = client if client is not None else httpx.AsyncClient()

    try:
        response = await async_client.post(
            REFRESH_URL,
            json=body,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        error_body = exc.response.text
        error_code = None
        try:
            error_payload = exc.response.json()
        except json.JSONDecodeError:
            error_payload = None

        if isinstance(error_payload, dict):
            error_code = error_payload.get("error")

        if error_code in {
            "refresh_token_expired",
            "refresh_token_reused",
            "refresh_token_invalidated",
        }:
            raise BorrowKeyError(
                f"Refresh token is no longer valid ({error_code}). Run `codex login` to re-authenticate.",
            ) from None

        raise BorrowKeyError(
            f"Token refresh failed (HTTP {exc.response.status_code}): {error_body}",
        ) from None
    except httpx.RequestError as exc:
        raise BorrowKeyError(f"Token refresh failed (network error): {exc}") from None
    finally:
        if owns_client:
            await async_client.aclose()
