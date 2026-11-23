"""Telegram bot that uses GigaChat API to build 1-minute videos from prompts."""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from uuid import uuid4

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import BotCommand, FSInputFile, Message
from dotenv import load_dotenv

from .llm import GigaChatClient
from .video_generator import async_build_video_from_storyboard, VideoGenerationError

logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 1000


@dataclass
class Settings:
    telegram_token: str
    gigachat_auth_key: str
    output_dir: Path = Path("videos")
    video_duration: int = 60
    gigachat_model: str = "GigaChat"
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_ca_bundle_file: str | None = None
    max_parallel_renders: int = 1


@dataclass
class RenderLimiter:
    semaphore: asyncio.Semaphore
    active_users: set[int] = field(default_factory=set)


def load_settings() -> Settings:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Environment variable TELEGRAM_BOT_TOKEN is required")
    auth_key = os.getenv("GIGACHAT_AUTH_KEY")
    if not auth_key:
        raise RuntimeError("Environment variable GIGACHAT_AUTH_KEY is required")
    return Settings(
        telegram_token=token,
        gigachat_auth_key=auth_key,
        output_dir=Path(os.getenv("VIDEO_OUTPUT_DIR", "videos")),
        video_duration=int(os.getenv("VIDEO_DURATION", "60")),
        gigachat_model=os.getenv("GIGACHAT_MODEL", "GigaChat"),
        gigachat_scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        gigachat_ca_bundle_file=os.getenv("GIGACHAT_CA_BUNDLE_FILE") or None,
        max_parallel_renders=max(1, int(os.getenv("MAX_PARALLEL_RENDERS", "1"))),
    )


async def generate_storyboard(prompt: str, llm: GigaChatClient) -> str:
    return await llm.complete(
        "You create concise video storyboards. Return 8-12 short scenes "
        "(bulleted list) that together form a coherent one-minute video. "
        "Write in the language of the request and return only the list.",
        prompt,
    )


async def handle_video_command(
    message: Message, settings: Settings, limiter: RenderLimiter, llm: GigaChatClient
) -> None:
    if not message.text or message.from_user is None:
        return
    parts = message.text.split(maxsplit=1)
    prompt = parts[1].strip() if len(parts) > 1 else ""
    if not prompt:
        await message.answer(
            "Пожалуйста, добавьте описание после команды. Например:\n"
            "<code>/video Создай видео о путешествии кота на Марс</code>",
            parse_mode=ParseMode.HTML,
        )
        return
    if len(prompt) > MAX_PROMPT_LENGTH:
        await message.answer(f"Описание слишком длинное. Максимум {MAX_PROMPT_LENGTH} символов.")
        return

    user_id = message.from_user.id
    if user_id in limiter.active_users:
        await message.answer("Ваше видео уже создаётся. Дождитесь результата.")
        return

    limiter.active_users.add(user_id)
    final_path = None
    status = await message.answer("Генерирую раскадровку...")
    try:
        storyboard = await generate_storyboard(prompt, llm)
        output_path = settings.output_dir / f"video_{uuid4().hex}.mp4"
        await status.edit_text("Создаю видео из раскадровки...")
        async with limiter.semaphore:
            final_path = await async_build_video_from_storyboard(
                storyboard, output_path, settings.video_duration
            )
        await status.edit_text("Видео готово! Отправляю файл...")
        await message.answer_video(FSInputFile(final_path))
    except VideoGenerationError:
        logger.warning("Storyboard generation returned no scenes")
        await status.edit_text(
            "Не удалось создать видео: модель не вернула ни одной сцены. Попробуйте другой запрос."
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to generate video")
        await status.edit_text("Произошла ошибка при создании видео. Попробуйте позже.")
    finally:
        limiter.active_users.discard(user_id)
        if final_path is not None:
            try:
                os.remove(final_path)
            except OSError:
                logger.warning("Could not remove temporary file %s", final_path)


async def _register_commands(bot: Bot) -> None:
    await bot.set_my_commands([BotCommand(command="video", description="Сгенерировать бесплатное минутное видео")])


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = load_settings()

    llm = GigaChatClient(
        settings.gigachat_auth_key,
        scope=settings.gigachat_scope,
        model=settings.gigachat_model,
        ca_bundle_file=settings.gigachat_ca_bundle_file,
    )
    bot = Bot(token=settings.telegram_token, parse_mode=ParseMode.HTML)
    dp = Dispatcher()
    limiter = RenderLimiter(asyncio.Semaphore(settings.max_parallel_renders))
    dp.message.register(
        partial(handle_video_command, settings=settings, limiter=limiter, llm=llm),
        Command("video"),
    )

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await _register_commands(bot)
        await dp.start_polling(bot)
    finally:
        await llm.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
