from __future__ import annotations

import base64
import json
import math
import os
import random
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .models import EpisodePlan


SHOT_SECONDS = 5.0
FINAL_SECONDS = 30.0
WIDTH = 1080
HEIGHT = 1920
FPS = 30


def log(message: str) -> None:
    print(f"[factory] {message}", flush=True)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    pretty = " ".join(command[:4])
    log(f"Running: {pretty}{' …' if len(command) > 4 else ''}")
    if command and Path(command[0]).name == "ffmpeg" and "-hide_banner" not in command:
        command = [command[0], "-hide_banner", "-loglevel", "error", *command[1:]]
    subprocess.run(command, cwd=cwd, check=True)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary is not installed: {name}")


def decode_reference(encoded_path: Path, output_path: Path, size: tuple[int, int]) -> Path:
    raw = base64.b64decode(encoded_path.read_text(encoding="utf-8").strip())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    with Image.open(output_path) as image:
        image = image.convert("RGB")
        image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=90, threshold=3))
        image = ImageEnhance.Contrast(image).enhance(1.03)
        image.save(output_path, "JPEG", quality=92, optimize=True)
    return output_path


def decode_references(repo_root: Path, target_dir: Path) -> dict[str, Path]:
    source = repo_root / "assets" / "characters"
    return {
        "pip": decode_reference(
            source / "pip-front.jpg.b64", target_dir / "pip-front.jpg", (768, 1152)
        ),
        "lumi": decode_reference(
            source / "lumi-front.jpg.b64", target_dir / "lumi-front.jpg", (768, 1152)
        ),
        "pip_angle": decode_reference(
            source / "pip-angle.jpg.b64",
            target_dir / "pip-angle.jpg",
            (768, 1152),
        ),
        "lumi_angle": decode_reference(
            source / "lumi-angle.jpg.b64",
            target_dir / "lumi-angle.jpg",
            (768, 1152),
        ),
        "style": decode_reference(
            source / "style-reference.jpg.b64",
            target_dir / "style-reference.jpg",
            (720, 1280),
        ),
    }


def download(url: str, output_path: Path, *, attempts: int = 3) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with output_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if output_path.stat().st_size < 1024:
                raise RuntimeError(f"Downloaded file is unexpectedly small: {output_path}")
            return output_path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < attempts:
                time.sleep(attempt * 2)
    raise RuntimeError(f"Failed to download {url}") from last_error


def ffprobe_json(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def probe_duration(path: Path) -> float:
    return float(ffprobe_json(path)["format"]["duration"])


def normalize_clip(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    video_filter = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},setsar=1,fps={FPS},"
        f"tpad=stop_mode=clone:stop_duration={SHOT_SECONDS},"
        f"trim=duration={SHOT_SECONDS},setpts=PTS-STARTPTS,format=yuv420p"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-an",
            "-vf",
            video_filter,
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )
    return target


def concat_clips(clips: list[Path], target: Path) -> Path:
    command = ["ffmpeg", "-y"]
    for clip in clips:
        command.extend(["-i", str(clip)])
    filters = [
        f"[{index}:v]setsar=1,setpts=PTS-STARTPTS[v{index}]"
        for index in range(len(clips))
    ]
    filters.append(
        "".join(f"[v{index}]" for index in range(len(clips)))
        + f"concat=n={len(clips)}:v=1:a=0[outv]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-t",
            str(FINAL_SECONDS),
            str(target),
        ]
    )
    run(command)
    return target


def atempo_chain(speed: float) -> str:
    factors: list[float] = []
    while speed > 2.0:
        factors.append(2.0)
        speed /= 2.0
    while speed < 0.5:
        factors.append(0.5)
        speed /= 0.5
    factors.append(speed)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def fit_voice(source: Path, target: Path) -> Path:
    duration = probe_duration(source)
    filters: list[str] = []
    maximum_voice_time = 4.62
    if duration > maximum_voice_time:
        filters.append(atempo_chain(duration / maximum_voice_time))
    filters.extend(
        [
            "adelay=160:all=1",
            f"apad=pad_dur={SHOT_SECONDS}",
            f"atrim=0:{SHOT_SECONDS}",
            "aresample=44100",
        ]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-af",
            ",".join(filters),
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(target),
        ]
    )
    return target


def concat_audio(parts: list[Path], target: Path) -> Path:
    list_path = target.with_suffix(".txt")
    list_path.write_text(
        "\n".join(f"file '{part.resolve()}'" for part in parts) + "\n",
        encoding="utf-8",
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:a",
            "pcm_s16le",
            str(target),
        ]
    )
    return target


def caption_chunks(text: str, *, maximum_words: int = 4) -> list[str]:
    words = text.replace("—", " ").split()
    if not words:
        return []
    count = max(1, math.ceil(len(words) / maximum_words))
    base, extra = divmod(len(words), count)
    chunks = []
    cursor = 0
    for index in range(count):
        take = base + (1 if index < extra else 0)
        chunks.append(" ".join(words[cursor : cursor + take]))
        cursor += take
    return chunks


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    centiseconds = int(round((seconds - int(seconds)) * 100))
    if centiseconds == 100:
        whole_seconds += 1
        centiseconds = 0
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def ass_escape(text: str) -> str:
    return (
        text.replace("\\", "")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", " ")
    )


def write_ass(plan: EpisodePlan, target: Path) -> Path:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {HEIGHT}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,DejaVu Sans,70,&H00FFFFFF,&H0000FFFF,&H00101018,&H60000000,-1,0,0,0,100,100,0,0,1,8,2,2,72,72,330,1
Style: Hook,DejaVu Sans,78,&H0000FFFF,&H00FFFFFF,&H00101018,&H70000000,-1,0,0,0,100,100,0,0,1,9,2,2,64,64,350,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for shot_index, shot in enumerate(plan.shots):
        chunks = caption_chunks(shot.narration)
        available = 4.66
        each = available / max(1, len(chunks))
        for chunk_index, chunk in enumerate(chunks):
            start = shot_index * SHOT_SECONDS + 0.08 + chunk_index * each
            end = min((shot_index + 1) * SHOT_SECONDS - 0.10, start + each + 0.06)
            style = "Hook" if shot_index == 0 and chunk_index == 0 else "Main"
            color = (
                r"{\1c&H0000FFFF&}"
                if (shot_index + chunk_index) % 4 == 0 and style != "Hook"
                else ""
            )
            text = ass_escape(chunk.upper())
            lines.append(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},{style},,0,0,0,,"
                f"{color}{text}\n"
            )
    target.write_text("".join(lines), encoding="utf-8")
    return target


def create_sfx(target: Path, *, seed: int = 1234) -> Path:
    sample_rate = 44100
    total_samples = int(FINAL_SECONDS * sample_rate)
    rng = random.Random(seed)
    events = [
        (0.04, "spark"),
        (4.92, "whoosh"),
        (9.92, "whoosh"),
        (14.92, "spark"),
        (19.92, "whoosh"),
        (24.92, "spark"),
        (29.25, "spark"),
    ]
    frames = bytearray()
    for index in range(total_samples):
        now = index / sample_rate
        value = 0.0
        for event_time, kind in events:
            age = now - event_time
            length = 0.55 if kind == "whoosh" else 0.80
            if 0 <= age < length:
                envelope = math.sin(math.pi * age / length) ** 2
                if kind == "whoosh":
                    noise = rng.uniform(-1.0, 1.0)
                    value += 0.22 * envelope * noise * (1.0 - age / length)
                else:
                    tone = math.sin(2 * math.pi * (950 + 480 * age) * age)
                    overtone = math.sin(2 * math.pi * 1510 * age)
                    value += envelope * (0.17 * tone + 0.08 * overtone)
        sample = int(max(-1.0, min(1.0, value)) * 32767)
        packed = sample.to_bytes(2, byteorder="little", signed=True)
        frames.extend(packed)
        frames.extend(packed)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))
    return target


def create_fallback_music(target: Path, *, seed: int = 2026) -> Path:
    """Create an original, light adventure bed if the music API is unavailable."""
    sample_rate = 44100
    total_samples = int(FINAL_SECONDS * sample_rate)
    rng = random.Random(seed)
    progressions = [
        (130.81, 164.81, 196.00),
        (98.00, 123.47, 146.83),
        (110.00, 130.81, 164.81),
        (87.31, 110.00, 130.81),
    ]
    melody = [523.25, 659.25, 783.99, 659.25, 587.33, 698.46, 880.00, 783.99]
    frames = bytearray()
    for index in range(total_samples):
        now = index / sample_rate
        chord = progressions[int(now / 4.0) % len(progressions)]
        chord_phase = now % 4.0
        pad_envelope = min(1.0, chord_phase / 0.18) * min(1.0, (4.0 - chord_phase) / 0.25)
        pad = sum(math.sin(2 * math.pi * frequency * now) for frequency in chord)
        pad *= 0.035 * pad_envelope

        note_phase = now % 0.5
        note = melody[int(now / 0.5) % len(melody)]
        pluck_envelope = math.exp(-6.0 * note_phase)
        pluck = 0.075 * pluck_envelope * math.sin(2 * math.pi * note * now)

        beat_phase = now % 0.5
        kick = 0.07 * math.exp(-18 * beat_phase) * math.sin(
            2 * math.pi * (72 - 30 * beat_phase) * beat_phase
        )
        shaker_phase = (now + 0.25) % 0.5
        shaker = 0.018 * math.exp(-35 * shaker_phase) * rng.uniform(-1.0, 1.0)

        master_fade = min(1.0, now / 0.7, (FINAL_SECONDS - now) / 1.2)
        left = master_fade * (pad + pluck * 0.88 + kick + shaker)
        right = master_fade * (pad + pluck * 1.08 + kick + shaker * 0.8)
        for value in (left, right):
            sample = int(max(-1.0, min(1.0, value)) * 32767)
            frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))
    return target


def assemble_final(
    silent_video: Path,
    narration: Path,
    music: Path,
    sfx: Path,
    captions: Path,
    target: Path,
) -> Path:
    escaped_captions = str(captions.resolve()).replace("\\", "\\\\").replace(":", "\\:")
    subtitle_filter = (
        f"subtitles=filename='{escaped_captions}':"
        "fontsdir='/usr/share/fonts/truetype/dejavu'"
    )
    filter_complex = (
        f"[0:v]{subtitle_filter},format=yuv420p[v];"
        "[1:a]volume=1.18,highpass=f=85[voice];"
        "[2:a]volume=0.105[bed];"
        "[3:a]volume=0.24[fx];"
        "[voice][bed][fx]amix=inputs=3:duration=first:normalize=0,"
        "alimiter=limit=0.95,loudnorm=I=-14:TP=-1.5:LRA=7[a]"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(silent_video),
            "-i",
            str(narration),
            "-stream_loop",
            "-1",
            "-i",
            str(music),
            "-i",
            str(sfx),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-t",
            str(FINAL_SECONDS),
            "-movflags",
            "+faststart",
            str(target),
        ]
    )
    return target


def extract_thumbnail(video: Path, target: Path) -> Path:
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "0.65",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            "scale=1080:1920",
            "-q:v",
            "2",
            str(target),
        ]
    )
    return target


def validate_video(path: Path) -> dict:
    info = ffprobe_json(path)
    duration = float(info["format"]["duration"])
    if not 29.5 <= duration <= 30.5:
        raise RuntimeError(f"Final duration must be approximately 30 seconds, got {duration}")
    streams = info.get("streams", [])
    video = next((stream for stream in streams if stream["codec_type"] == "video"), None)
    audio = next((stream for stream in streams if stream["codec_type"] == "audio"), None)
    if video is None or audio is None:
        raise RuntimeError("Final file must contain video and audio")
    if (video.get("width"), video.get("height")) != (WIDTH, HEIGHT):
        raise RuntimeError(
            f"Final dimensions must be {WIDTH}x{HEIGHT}, "
            f"got {video.get('width')}x{video.get('height')}"
        )
    if video.get("codec_name") != "h264" or audio.get("codec_name") != "aac":
        raise RuntimeError("Final codecs must be H.264 video and AAC audio")
    run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"])
    return info


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def require_environment(names: Iterable[str]) -> None:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise RuntimeError("Missing required secrets: " + ", ".join(missing))
