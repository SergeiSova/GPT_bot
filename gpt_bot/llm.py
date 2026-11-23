"""Клиент официального GigaChat API."""
from __future__ import annotations

import asyncio
import ssl
import time
import uuid

import httpx

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CHAT_URL = "https://api.giga.chat/v1/chat/completions"
TOKEN_LIFETIME_SECONDS = 30 * 60
TOKEN_REFRESH_MARGIN_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 60.0


class LLMError(RuntimeError):
    """GigaChat не смог вернуть текст ответа."""


def _build_tls_context(ca_bundle_file: str | None) -> ssl.SSLContext:
    context = ssl.create_default_context()
    if ca_bundle_file:
        try:
            context.load_verify_locations(cafile=ca_bundle_file)
        except (OSError, ssl.SSLError) as exc:
            raise LLMError(f"Не удалось загрузить сертификат: {ca_bundle_file}") from exc
    return context


def _token_deadline(expires_at: object) -> float:
    """API может вернуть время в секундах или миллисекундах."""
    latest = time.time() + TOKEN_LIFETIME_SECONDS
    try:
        value = float(expires_at)
    except (TypeError, ValueError):
        return latest - TOKEN_REFRESH_MARGIN_SECONDS
    if value > 1e12:
        value /= 1000
    return min(value, latest) - TOKEN_REFRESH_MARGIN_SECONDS


class GigaChatClient:
    def __init__(
        self,
        auth_key: str,
        scope: str = "GIGACHAT_API_PERS",
        model: str = "GigaChat",
        ca_bundle_file: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._auth_key = auth_key.strip().removeprefix("Basic ").strip()
        self._scope = scope
        self._model = model
        self._http = httpx.AsyncClient(
            verify=_build_tls_context(ca_bundle_file),
            timeout=REQUEST_TIMEOUT_SECONDS,
            transport=transport,
        )
        self._token_lock = asyncio.Lock()
        self._access_token: str | None = None
        self._expires_at = 0.0

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _get_token(self, stale_token: str | None = None) -> str:
        async with self._token_lock:
            token_is_fresh = self._access_token and time.time() < self._expires_at
            if token_is_fresh and self._access_token != stale_token:
                return self._access_token

            try:
                response = await self._http.post(
                    OAUTH_URL,
                    headers={
                        "Authorization": f"Basic {self._auth_key}",
                        "RqUID": str(uuid.uuid4()),
                        "Accept": "application/json",
                    },
                    data={"scope": self._scope},
                )
            except httpx.HTTPError as exc:
                raise LLMError(f"Нет соединения с GigaChat: {type(exc).__name__}") from None

            if response.status_code != 200:
                raise LLMError(f"GigaChat отказал в авторизации: HTTP {response.status_code}")
            try:
                payload = response.json()
                self._access_token = payload["access_token"]
            except (KeyError, TypeError, ValueError):
                raise LLMError("GigaChat вернул некорректный токен") from None
            self._expires_at = _token_deadline(payload.get("expires_at"))
            return self._access_token

    async def _post_chat(self, payload: dict, token: str) -> httpx.Response:
        try:
            return await self._http.post(
                CHAT_URL,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"Нет соединения с GigaChat: {type(exc).__name__}") from None

    async def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "stream": False,
        }
        token = await self._get_token()
        response = await self._post_chat(payload, token)
        if response.status_code == 401:
            token = await self._get_token(stale_token=token)
            response = await self._post_chat(payload, token)
        if response.status_code != 200:
            raise LLMError(f"GigaChat вернул ошибку: HTTP {response.status_code}")

        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            raise LLMError("GigaChat вернул ответ неизвестного формата") from None
        text = (content or "").strip()
        if not text:
            raise LLMError("GigaChat вернул пустой ответ")
        return text
