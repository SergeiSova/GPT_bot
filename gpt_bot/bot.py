"""Telegram bot that uses g4f to build 1-minute videos from prompts."""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message
from dotenv import load_dotenv
from g4f.client import Client

from .video_generator import async_build_video_from_storyboard, VideoGenerationError

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    telegram_token: str
    output_dir: Path = Path("videos")
    video_duration: int = 60
    model: str = "gpt-4o-mini"


def load_settings() -> Settings:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Environment variable TELEGRAM_BOT_TOKEN is required")
    output_dir = Path(os.getenv("VIDEO_OUTPUT_DIR", "videos"))
    duration = int(os.getenv("VIDEO_DURATION", "60"))
    model = os.getenv("G4F_MODEL", "gpt-4o-mini")
    return Settings(token, output_dir, duration, model)


async def generate_storyboard(prompt: str, model: str) -> str:
    client = Client()
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You create concise video storyboards. Return 8-12 short scenes "
                    "(bulleted list) that together form a coherent one-minute video."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    choices = getattr(response, "choices", [])
    if not choices:
        raise RuntimeError("g4f не вернул ни одного варианта ответа")
    content = choices[0].message.content
    if not content:
        raise RuntimeError("g4f вернул пустой ответ")
    return content


async def handle_video_command(message: Message, settings: Settings) -> None:
    if not message.text:
        return
    parts = message.text.split(maxsplit=1)
    prompt = parts[1] if len(parts) > 1 else ""
    if not prompt:
        await message.answer(
            "Пожалуйста, добавьте описание после команды. Например:\n"
            "<code>/video Создай видео о путешествии кота на Марс</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    status = await message.answer("Генерирую раскадровку с помощью g4f...")
    try:
        storyboard = await generate_storyboard(prompt, settings.model)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"video_{timestamp}.mp4"
        output_path = settings.output_dir / filename
        await status.edit_text("Создаю видео из раскадровки...")
        final_path = await async_build_video_from_storyboard(
            storyboard, output_path, settings.video_duration
        )
    except VideoGenerationError:
        logger.warning("Storyboard generation returned no scenes")
        await status.edit_text(
            "Не удалось создать видео: модель не вернула ни одной сцены. Попробуйте другой запрос."
        )
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to generate video")
        await status.edit_text(f"Произошла ошибка при создании видео: {exc}")
        return

    await status.edit_text("Видео готово! Отправляю файл...")
    await message.answer_video(FSInputFile(final_path))

    try:
        os.remove(final_path)
    except OSError:
        logger.warning("Could not remove temporary file %s", final_path)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()

    bot = Bot(token=settings.telegram_token)
    dp = Dispatcher()

    dp.message.register(partial(handle_video_command, settings=settings), Command("video"))

    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
