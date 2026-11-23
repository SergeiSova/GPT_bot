"""Клиент официального API Z.ai для бесплатной модели GLM-4.7-Flash."""
from __future__ import annotations

import asyncio

import httpx

CHAT_URL = "https://api.z.ai/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4.7-flash"
REQUEST_TIMEOUT_SECONDS = 90.0
RETRY_STATUSES = {429, 500, 502, 503, 504}
RETRY_DELAYS_SECONDS = (2, 5)
FAILED_FINISH_REASONS = {"sensitive", "model_context_window_exceeded", "network_error"}


class LLMError(RuntimeError):
    """Модель не смогла вернуть текст ответа."""


def _error_code(response: httpx.Response) -> object:
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    return error.get("code") if isinstance(error, dict) else payload.get("code")


class ZaiClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key.strip()}", "Accept-Language": "en-US,en"},
            timeout=REQUEST_TIMEOUT_SECONDS,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(self, payload: dict) -> httpx.Response:
        for delay in (*RETRY_DELAYS_SECONDS, None):
            try:
                response = await self._http.post(CHAT_URL, json=payload)
            except httpx.HTTPError as exc:
                if delay is None:
                    raise LLMError(f"Нет соединения с Z.ai: {type(exc).__name__}") from None
            else:
                if response.status_code not in RETRY_STATUSES or delay is None:
                    return response
            await asyncio.sleep(delay)
        raise LLMError("Z.ai не ответил")

    async def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        response = await self._post({
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "stream": False,
            "thinking": {"type": "disabled"},
        })
        if response.status_code != 200:
            raise LLMError(f"Z.ai вернул ошибку: HTTP {response.status_code}, код {_error_code(response)}")

        try:
            choice = response.json()["choices"][0]
            content = choice["message"].get("content")
            finish_reason = choice.get("finish_reason")
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            raise LLMError("Z.ai вернул ответ неизвестного формата") from None
        if finish_reason in FAILED_FINISH_REASONS:
            raise LLMError(f"Z.ai не завершил ответ: {finish_reason}")
        text = (content or "").strip()
        if not text:
            raise LLMError("Z.ai вернул пустой ответ")
        return text
