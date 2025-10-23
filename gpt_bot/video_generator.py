"""Utility helpers to build short videos from storyboard text."""
from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, concatenate_videoclips

DEFAULT_RESOLUTION = (1280, 720)


class VideoGenerationError(RuntimeError):
    """Raised when the storyboard does not contain any usable content."""


def _pick_background_color(seed: int | None = None) -> tuple[int, int, int]:
    """Return a deterministic but varied background color."""
    rng = random.Random(seed)
    hue = rng.randint(0, 360)
    saturation = 0.4 + rng.random() * 0.4
    value = 0.6 + rng.random() * 0.3

    c = value * saturation
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = value - c

    if hue < 60:
        r, g, b = c, x, 0
    elif hue < 120:
        r, g, b = x, c, 0
    elif hue < 180:
        r, g, b = 0, c, x
    elif hue < 240:
        r, g, b = 0, x, c
    elif hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return tuple(int((channel + m) * 255) for channel in (r, g, b))


def parse_storyboard(storyboard: str) -> List[str]:
    """Extract individual scene descriptions from a language-model answer."""
    lines = []
    for raw_line in storyboard.splitlines():
        line = raw_line.strip(" -*\t")
        if not line:
            continue
        lines.append(line)
    return lines


def _render_scene(text: str, resolution: tuple[int, int]) -> np.ndarray:
    """Render the text to an RGB frame using Pillow."""
    image = Image.new("RGB", resolution, _pick_background_color(hash(text) % 10_000))
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Try to load a truetype font. Fall back to default if unavailable.
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=42)
    except OSError:
        font = ImageFont.load_default()

    max_text_width = width - 80
    words = text.split()
    lines: List[str] = []
    current_line: List[str] = []

    for word in words:
        tentative = " ".join(current_line + [word]) if current_line else word
        if draw.textlength(tentative, font=font) <= max_text_width:
            current_line.append(word)
        else:
            if not current_line:
                # Very long word, force break
                lines.append(tentative)
                current_line = []
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    total_text_height = sum(
        (bbox[3] - bbox[1]) for bbox in (draw.textbbox((0, 0), line, font=font) for line in lines)
    )
    y = (height - total_text_height) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        draw.text((x, y), line, fill="white", font=font)
        y += text_height + 5

    return np.array(image)


def build_video_from_storyboard(
    storyboard: str,
    output_path: Path,
    duration: int = 60,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
) -> Path:
    """Create a video file from a storyboard description."""
    scenes = parse_storyboard(storyboard)
    if not scenes:
        raise VideoGenerationError("The language model did not return any scenes")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_scene_duration = duration / len(scenes)
    clips = []
    for scene in scenes:
        frame = _render_scene(scene, resolution)
        clip = ImageClip(frame).set_duration(per_scene_duration)
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")
    video.write_videofile(
        str(output_path),
        fps=24,
        codec="libx264",
        audio=False,
        verbose=False,
        logger=None,
    )
    video.close()
    for clip in clips:
        clip.close()

    return output_path


async def async_build_video_from_storyboard(
    storyboard: str,
    output_path: Path,
    duration: int = 60,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
) -> Path:
    """Async wrapper that runs :func:`build_video_from_storyboard` in a thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        build_video_from_storyboard,
        storyboard,
        output_path,
        duration,
        resolution,
    )
