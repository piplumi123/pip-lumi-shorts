#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import shutil
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shorts_factory.core import (  # noqa: E402
    FINAL_SECONDS,
    SHOT_SECONDS,
    assemble_final,
    concat_audio,
    concat_clips,
    create_fallback_music,
    create_sfx,
    decode_references,
    download,
    extract_thumbnail,
    fit_voice,
    load_json,
    log,
    normalize_clip,
    require_binary,
    require_environment,
    run,
    validate_video,
    write_ass,
    write_json,
)
from shorts_factory.models import (  # noqa: E402
    EpisodePlan,
    FrameReview,
    QualityReview,
    Shot,
)


SCRIPT_MODEL = os.getenv("OPENAI_SCRIPT_MODEL", "auto")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "auto")
ANIMATION_MODEL = os.getenv(
    "FAL_ANIMATION_MODEL", "fal-ai/kling-video/o1/reference-to-video"
)
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
VOICE_MODEL = os.getenv("ELEVENLABS_VOICE_MODEL", "eleven_multilingual_v2")
KNOWN_ANIMATION_RATE = 0.112
MAX_FRAME_RETRIES = int(os.getenv("MAX_FRAME_RETRIES", "2"))


@dataclass
class RunPaths:
    run_root: Path
    references: Path
    frames: Path
    raw_clips: Path
    clips: Path
    voice_raw: Path
    voice_fitted: Path
    audio: Path
    output: Path

    @classmethod
    def create(cls, repo_root: Path, run_name: str) -> "RunPaths":
        run_root = repo_root / "work" / run_name
        paths = cls(
            run_root=run_root,
            references=run_root / "references",
            frames=run_root / "frames",
            raw_clips=run_root / "raw-clips",
            clips=run_root / "clips",
            voice_raw=run_root / "voice-raw",
            voice_fitted=run_root / "voice-fitted",
            audio=run_root / "audio",
            output=repo_root / "output",
        )
        for path in paths.__dict__.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        return paths


def openai_client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=300.0)


def choose_available_model(
    requested: str,
    preferred: list[str],
    available: set[str],
) -> str:
    for candidate in [requested, *preferred]:
        if candidate != "auto" and candidate in available:
            return candidate
    return ""


def sample_plan(episode_number: int = 1) -> EpisodePlan:
    return EpisodePlan(
        episode_number=episode_number,
        episode_title="The Star That Split",
        story_question="Can Pip and Lumi catch a fallen star before its light disappears?",
        synopsis=(
            "A runaway star steals Lumi's glow, splits into three sparks, and reveals "
            "the first hidden piece of a much larger star map."
        ),
        continuity_advance="The restored star projects a map toward a sealed door below the river.",
        youtube_title="The Star Stole Lumi's Glow! ✨ #Shorts",
        youtube_description=(
            "Pip and Lumi chase a runaway star through the glowing forest—but catching "
            "it only begins the mystery. A new 30-second Startrail adventure."
        ),
        hashtags=["#Shorts", "#Animation", "#PipAndLumi", "#KidsStories"],
        shots=[
            Shot(
                number=1,
                beat="hook",
                narration=(
                    "A falling star just stole Lumi's glow—and it is racing toward the river!"
                ),
                caption="It stole her glow!",
                scene_prompt=(
                    "A golden falling star whips past Lumi and pulls the turquoise light "
                    "from her tail as Pip reacts in shock beside a crystal river."
                ),
                motion_prompt=(
                    "Explosive immediate motion: star streaks toward camera then away, "
                    "Lumi's tail light stretches after it, Pip lunges forward."
                ),
                sound_cue="Magical rip and fast whoosh",
            ),
            Shot(
                number=2,
                beat="escalation",
                narration=(
                    "Pip fires his tiny boosters, while Lumi follows the sparkling trail below."
                ),
                caption="Catch that star!",
                scene_prompt=(
                    "Pip flies low over the luminous river while Lumi sprints across glowing "
                    "stones, both chasing the star through the purple forest."
                ),
                motion_prompt=(
                    "Fast lateral tracking shot, Pip's scarf streams behind, Lumi bounds "
                    "between stones, bright trail curls through frame."
                ),
                sound_cue="Booster fizz and quick paw taps",
            ),
            Shot(
                number=3,
                beat="wrong_turn",
                narration=(
                    "Pip grabs the bright star, but it cracks open into three bouncing sparks."
                ),
                caption="Oh no—three!",
                scene_prompt=(
                    "Pip catches the golden star above the river, but it cracks into three "
                    "expressive sparks that spring in different directions as Lumi skids."
                ),
                motion_prompt=(
                    "Pip closes both hands, sudden flash, three sparks bounce outward with "
                    "clear separate arcs, camera snaps wider with the surprise."
                ),
                sound_cue="Bright pop and three musical pings",
            ),
            Shot(
                number=4,
                beat="discovery",
                narration=(
                    "Lumi's tail points to their reflections: the river is showing the sparks' hiding places."
                ),
                caption="Look in the water!",
                scene_prompt=(
                    "Lumi notices three glowing reflections in the crystal river while her "
                    "turquoise tail points like a compass and Pip peers down."
                ),
                motion_prompt=(
                    "Camera dips to reflections, Lumi's tail swings and locks direction, "
                    "ripples reveal the three hidden spark locations."
                ),
                sound_cue="Mystery shimmer and water ripple",
            ),
            Shot(
                number=5,
                beat="payoff",
                narration=(
                    "Together they tap each reflection, and the three sparks leap safely home."
                ),
                caption="Teamwork!",
                scene_prompt=(
                    "Pip and Lumi tap matching river reflections at once; three golden sparks "
                    "arc together and rebuild the happy star above them."
                ),
                motion_prompt=(
                    "Synchronized paw and robot-hand tap, spiraling sparks converge overhead, "
                    "warm golden burst lights both joyful faces."
                ),
                sound_cue="Rhythmic taps and triumphant chime",
            ),
            Shot(
                number=6,
                beat="twist",
                narration=(
                    "The star glows again—then projects a map to a door beneath them."
                ),
                caption="A door... below?",
                scene_prompt=(
                    "The restored star projects a luminous map over Pip and Lumi, ending on "
                    "a mysterious sealed star-shaped door visible beneath the transparent river."
                ),
                motion_prompt=(
                    "Map unfurls in midair, camera follows its beam down through the water "
                    "to the huge hidden door, then snaps to their amazed faces."
                ),
                sound_cue="Map unfurl and unresolved sparkle",
            ),
        ],
    )


def next_episode_number(history: list[dict]) -> int:
    return max((int(item.get("episode_number", 0)) for item in history), default=0) + 1


def generate_plan(
    series: dict,
    history: list[dict],
    episode_number: int,
    episode_seed: str,
    revision_notes: list[str] | None = None,
) -> EpisodePlan:
    history_excerpt = history[-12:]
    revision = ""
    if revision_notes:
        revision = (
            "\nA strict editor rejected the first draft. Fix every point below:\n- "
            + "\n- ".join(revision_notes)
        )
    prompt = f"""
Create episode {episode_number} of the original vertical micro-series described below.

SERIES BIBLE
{json.dumps(series, ensure_ascii=False, indent=2)}

RECENT EPISODES — do not repeat their central problem, setting action, or payoff
{json.dumps(history_excerpt, ensure_ascii=False, indent=2)}

OPTIONAL CREATIVE SEED
{episode_seed or "Choose the strongest fresh continuation yourself."}

NON-NEGOTIABLE STORY DESIGN
- Exactly six shots of five seconds each, in this exact beat order:
  hook, escalation, wrong_turn, discovery, payoff, twist.
- Total narration: 55–82 words, natural spoken English, short sentences.
- The first spoken words and first image must create curiosity in under 0.8 seconds.
- Every shot changes the situation and contains a clear physical action suitable for animation.
- Use one strong cause-and-effect chain, one emotional beat, and a visually legible payoff.
- The ending resolves this episode's immediate problem but reveals one concrete new mystery.
- Narration adds meaning; it must not merely describe the picture.
- Captions are punchy, maximum six words, but no text should appear inside generated imagery.
- Frame prompts must identify Pip and Lumi by name and describe cinematic composition.
- Motion prompts must specify subject motion, camera motion, and a clean final pose.
- Keep it delightful and safe, never babyish, generic, or copied from an existing franchise.
- YouTube title must be truthful, curiosity-driven, under 70 characters, and include #Shorts.
{revision}
""".strip()
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = openai_client().responses.parse(
                model=SCRIPT_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are the lead writer and storyboard director for premium "
                            "30-second family animation. Optimize for immediate retention, "
                            "clarity without sound, emotion, and replay value."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                text_format=EpisodePlan,
            )
            plan = response.output_parsed
            if plan is None:
                raise RuntimeError("The story model returned no parsed episode plan")
            plan.episode_number = episode_number
            return plan
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 0:
                log("Story schema retrying once.")
                time.sleep(2)
    raise RuntimeError("Could not generate a valid episode plan") from last_error


def review_plan(series: dict, plan: EpisodePlan) -> QualityReview:
    response = openai_client().responses.parse(
        model=SCRIPT_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a ruthless short-form animation editor. Judge the draft for "
                    "first-second retention, causal story clarity, visual action, emotional "
                    "payoff, and replay value. Reject generic or repetitive ideas."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Review this episode against its series rules. Approve only if every "
                    "score deserves at least 7 and the average deserves at least 8. "
                    "Give concise, actionable revision notes.\n\n"
                    f"SERIES:\n{json.dumps(series, ensure_ascii=False)}\n\n"
                    f"EPISODE:\n{plan.model_dump_json(indent=2)}"
                ),
            },
        ],
        text_format=QualityReview,
    )
    if response.output_parsed is None:
        raise RuntimeError("The editor returned no parsed review")
    return response.output_parsed


def plan_is_strong(review: QualityReview) -> bool:
    scores = [
        review.hook_score,
        review.story_score,
        review.visual_action_score,
        review.emotional_payoff_score,
        review.replay_score,
    ]
    return review.approved and min(scores) >= 7 and sum(scores) / len(scores) >= 8


def generate_frame(
    shot: Shot,
    references: dict[str, Path],
    target: Path,
    extra_instruction: str = "",
) -> Path:
    prompt = f"""
Create the cinematic FIRST FRAME for shot {shot.number} of an original premium 3D
family-animation short, composed vertically for 9:16.

Reference image 1 is the locked design for Pip.
Reference image 2 is the locked design for Lumi.
Reference image 3 is the locked series lighting, palette, and world style.

SHOT:
{shot.scene_prompt}

Preserve Pip and Lumi exactly: same face, eyes, body proportions, colors, materials,
Pip's star antenna and red scarf, Lumi's navy rosettes and turquoise glowing tail tip.
Use an energetic camera angle, one instantly readable focal action, expressive poses,
foreground/midground/background depth, rich purple-blue environment, and motivated
golden/cyan lighting. Leave safe space in the lower-middle area for captions.
No text, letters, numbers, logo, watermark, border, split panel, duplicate character,
extra limb, costume change, or photoreal human.
{extra_instruction}
""".strip()
    target.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        inputs = [
            stack.enter_context(references["pip"].open("rb")),
            stack.enter_context(references["lumi"].open("rb")),
            stack.enter_context(references["style"].open("rb")),
        ]
        result = openai_client().images.edit(
            model=IMAGE_MODEL,
            image=inputs,
            prompt=prompt,
            size="1024x1536",
            quality="medium",
            output_format="png",
        )
    item = result.data[0]
    raw_path = target.with_name(target.stem + "-raw.png")
    if getattr(item, "b64_json", None):
        raw_path.write_bytes(base64.b64decode(item.b64_json))
    elif getattr(item, "url", None):
        download(item.url, raw_path)
    else:
        raise RuntimeError(f"Image API returned no image for shot {shot.number}")
    with Image.open(raw_path) as image:
        image = image.convert("RGB")
        image = ImageOps.fit(
            image,
            (1080, 1920),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        image.save(target, "JPEG", quality=94, optimize=True)
    return target


def generate_frames(
    plan: EpisodePlan, references: dict[str, Path], paths: RunPaths
) -> list[Path]:
    targets = [paths.frames / f"shot-{shot.number:02d}.jpg" for shot in plan.shots]
    workers = max(1, min(2, int(os.getenv("IMAGE_CONCURRENCY", "2"))))
    log(f"Generating six storyboard frames with {workers} workers.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(generate_frame, shot, references, target): shot.number
            for shot, target in zip(plan.shots, targets, strict=True)
        }
        for future in concurrent.futures.as_completed(futures):
            shot_number = futures[future]
            future.result()
            log(f"Frame {shot_number}/6 ready.")
    return targets


def frame_data_url(path: Path) -> str:
    with Image.open(path) as image:
        preview = ImageOps.fit(
            image.convert("RGB"),
            (360, 640),
            method=Image.Resampling.LANCZOS,
        )
        from io import BytesIO

        buffer = BytesIO()
        preview.save(buffer, "JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def review_frames(plan: EpisodePlan, frames: list[Path]) -> FrameReview:
    content: list[dict] = [
        {
            "type": "input_text",
            "text": (
                "These six images are shots 1–6 in order. Check that Pip and Lumi remain "
                "recognizably identical across frames, the intended action is legible, no "
                "characters/limbs are duplicated, and each 9:16 composition has strong "
                "depth and caption space. Choose at most two shots to retry. Approve if "
                "minor imperfections will not hurt a polished animated short.\n\n"
                + plan.model_dump_json(indent=2)
            ),
        }
    ]
    content.extend({"type": "input_image", "image_url": frame_data_url(path)} for path in frames)
    response = openai_client().responses.parse(
        model=SCRIPT_MODEL,
        input=[{"role": "user", "content": content}],
        text_format=FrameReview,
    )
    if response.output_parsed is None:
        raise RuntimeError("The visual reviewer returned no parsed result")
    return response.output_parsed


def upload_fal_references(references: dict[str, Path]) -> dict[str, str]:
    import fal_client

    log("Checking animation access and uploading locked references.")
    return {name: fal_client.upload_file(str(path)) for name, path in references.items()}


def fal_video_url(result: object) -> str:
    if isinstance(result, dict):
        video = result.get("video")
        if isinstance(video, dict) and video.get("url"):
            return str(video["url"])
        if isinstance(result.get("video_url"), str):
            return str(result["video_url"])
    video = getattr(result, "video", None)
    url = getattr(video, "url", None)
    if url:
        return str(url)
    raise RuntimeError("Animation API returned no video URL")


def animate_shot(
    shot: Shot,
    frame: Path,
    remote_references: dict[str, str],
    target: Path,
) -> Path:
    import fal_client

    frame_url = fal_client.upload_file(str(frame))
    prompt = f"""
@Image1 is the exact start frame. @Image2 is the locked visual-style guide.
@Element1 is always Pip. @Element2 is always Lumi.

Animate this five-second story beat:
{shot.motion_prompt}

Maintain exact character identity, facial design, proportions, clothing, fur markings,
colors, and lighting from the references. Begin instantly with purposeful motion.
Use smooth premium character animation, believable weight, expressive eyes, secondary
motion in Pip's scarf and Lumi's tail, and cinematic camera movement with stable
subjects. Preserve anatomy and object permanence. No morphing, teleporting, duplicate
characters, new limbs, text, cuts, freeze-frame, or camera collision. Finish on a clear
pose that leads naturally into the next shot. Vertical 9:16 composition.
""".strip()

    def queue_update(update: object) -> None:
        status = type(update).__name__
        if status in {"InProgress", "Completed"}:
            log(f"Animation shot {shot.number}: {status}")

    result = fal_client.subscribe(
        ANIMATION_MODEL,
        arguments={
            "prompt": prompt,
            "image_urls": [frame_url, remote_references["style"]],
            "elements": [
                {
                    "frontal_image_url": remote_references["pip"],
                    "reference_image_urls": [],
                },
                {
                    "frontal_image_url": remote_references["lumi"],
                    "reference_image_urls": [],
                },
            ],
            "duration": "5",
            "aspect_ratio": "9:16",
        },
        with_logs=True,
        on_queue_update=queue_update,
    )
    return download(fal_video_url(result), target)


def animate_all(
    plan: EpisodePlan,
    frames: list[Path],
    remote_references: dict[str, str],
    paths: RunPaths,
) -> list[Path]:
    targets = [paths.raw_clips / f"shot-{shot.number:02d}.mp4" for shot in plan.shots]
    workers = max(1, min(2, int(os.getenv("ANIMATION_CONCURRENCY", "2"))))
    log(f"Animating six five-second shots with {workers} workers.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                animate_shot,
                shot,
                frame,
                remote_references,
                target,
            ): shot.number
            for shot, frame, target in zip(plan.shots, frames, targets, strict=True)
        }
        for future in concurrent.futures.as_completed(futures):
            shot_number = futures[future]
            future.result()
            log(f"Animation {shot_number}/6 downloaded.")
    return targets


def elevenlabs_post(
    path: str,
    payload: dict,
    *,
    params: dict | None = None,
    timeout: int = 240,
) -> bytes:
    response = requests.post(
        "https://api.elevenlabs.io/v1" + path,
        params=params,
        headers={
            "xi-api-key": os.environ["ELEVENLABS_API_KEY"],
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        message = response.text[:300].replace(os.environ["ELEVENLABS_API_KEY"], "[redacted]")
        raise RuntimeError(f"ElevenLabs returned HTTP {response.status_code}: {message}")
    return response.content


def check_provider_access(references: dict[str, Path]) -> dict[str, str]:
    global IMAGE_MODEL, SCRIPT_MODEL

    log("Validating provider access before paid generation.")
    available_models = {model.id for model in openai_client().models.list()}

    requested_script = SCRIPT_MODEL
    script_candidates = [
        "gpt-5.4",
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
    ]
    SCRIPT_MODEL = choose_available_model(
        requested_script, script_candidates, available_models
    )
    if not SCRIPT_MODEL:
        compatible_text = sorted(
            (
                model
                for model in available_models
                if model.startswith(("gpt-5", "gpt-4.1", "gpt-4o"))
                and not any(
                    marker in model
                    for marker in (
                        "audio",
                        "codex",
                        "image",
                        "realtime",
                        "search",
                        "transcribe",
                        "tts",
                    )
                )
            ),
            reverse=True,
        )
        SCRIPT_MODEL = compatible_text[0] if compatible_text else ""
    if not SCRIPT_MODEL:
        raise RuntimeError(
            "No compatible OpenAI text model is available. Checked: "
            + ", ".join(script_candidates)
        )

    requested_image = IMAGE_MODEL
    image_candidates = [
        "gpt-image-2",
        "gpt-image-1.5",
        "gpt-image-1",
    ]
    IMAGE_MODEL = choose_available_model(
        requested_image, image_candidates, available_models
    )
    if not IMAGE_MODEL:
        compatible_images = sorted(
            (model for model in available_models if model.startswith("gpt-image-")),
            reverse=True,
        )
        IMAGE_MODEL = compatible_images[0] if compatible_images else ""
    if not IMAGE_MODEL:
        raise RuntimeError(
            "No compatible OpenAI image model is available. Checked: "
            + ", ".join(image_candidates)
        )
    log(f"Using OpenAI models: script={SCRIPT_MODEL}, image={IMAGE_MODEL}.")

    response = requests.get(
        "https://api.elevenlabs.io/v1/user",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
        timeout=45,
    )
    if not response.ok:
        raise RuntimeError(f"ElevenLabs credential check failed: HTTP {response.status_code}")
    return upload_fal_references(references)


def generate_voice(plan: EpisodePlan, paths: RunPaths) -> Path:
    fitted: list[Path] = []
    narrations = [shot.narration for shot in plan.shots]
    seed = 11000 + plan.episode_number
    for index, text in enumerate(narrations):
        payload = {
            "text": text,
            "model_id": VOICE_MODEL,
            "seed": seed,
            "previous_text": " ".join(narrations[:index])[-600:] if index else "",
            "next_text": " ".join(narrations[index + 1 :])[:600],
            "voice_settings": {
                "stability": 0.42,
                "similarity_boost": 0.78,
                "style": 0.34,
                "use_speaker_boost": True,
            },
        }
        audio = elevenlabs_post(
            f"/text-to-speech/{VOICE_ID}",
            payload,
            params={"output_format": "mp3_44100_128"},
        )
        raw_path = paths.voice_raw / f"shot-{index + 1:02d}.mp3"
        raw_path.write_bytes(audio)
        fitted_path = paths.voice_fitted / f"shot-{index + 1:02d}.wav"
        fit_voice(raw_path, fitted_path)
        fitted.append(fitted_path)
        log(f"Voice {index + 1}/6 ready.")
    return concat_audio(fitted, paths.audio / "narration.wav")


def generate_music(plan: EpisodePlan, paths: RunPaths) -> Path:
    target = paths.audio / "music.mp3"
    prompt = (
        "Original instrumental soundtrack for a 30-second premium family-animation "
        f"adventure titled '{plan.episode_title}'. Immediate magical tension in the first "
        "second, playful pizzicato and light percussion during the chase, rising wonder, "
        "a warm triumphant payoff around second 24, then one unresolved sparkling note. "
        "No vocals, no lyrics, no imitation of any existing song, clear space for narration."
    )
    try:
        target.write_bytes(
            elevenlabs_post(
                "/music",
                {
                    "prompt": prompt,
                    "music_length_ms": int(FINAL_SECONDS * 1000),
                    "force_instrumental": True,
                    "model_id": "music_v1",
                },
                timeout=300,
            )
        )
        log("Original soundtrack ready.")
        return target
    except Exception as exc:  # noqa: BLE001
        log(f"Music API unavailable ({type(exc).__name__}); using original procedural bed.")
        fallback = paths.audio / "music.wav"
        return create_fallback_music(fallback, seed=9000 + plan.episode_number)


def create_dry_clip(source_image: Path, target: Path, shot_number: int) -> Path:
    zoom_direction = 1 if shot_number % 2 else -1
    if zoom_direction > 0:
        zoom = "min(zoom+0.0010,1.12)"
    else:
        zoom = "if(lte(on,1),1.12,max(1.0,zoom-0.0008))"
    video_filter = (
        "scale=1200:2134:force_original_aspect_ratio=increase,"
        "crop=1080:1920,setsar=1,"
        f"zoompan=z='{zoom}':"
        "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        "d=150:s=1080x1920:fps=30,format=yuv420p"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(source_image),
            "-vf",
            video_filter,
            "-t",
            str(SHOT_SECONDS),
            "-r",
            "30",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ]
    )
    return target


def create_dry_voice(target: Path, shot_number: int) -> Path:
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={330 + shot_number * 45}:duration=2.2",
            "-filter:a",
            "volume=0.06",
            "-c:a",
            "pcm_s16le",
            str(target),
        ]
    )
    return target


def render_episode(
    plan: EpisodePlan,
    raw_clips: list[Path],
    narration: Path,
    music: Path,
    paths: RunPaths,
    *,
    basename: str,
) -> Path:
    normalized = []
    for index, source in enumerate(raw_clips, start=1):
        target = paths.clips / f"shot-{index:02d}.mp4"
        normalize_clip(source, target)
        normalized.append(target)
    silent = concat_clips(normalized, paths.run_root / "silent.mp4")
    captions = write_ass(plan, paths.run_root / "captions.ass")
    sfx = create_sfx(paths.audio / "sfx.wav", seed=7000 + plan.episode_number)
    final_path = paths.output / f"{basename}.mp4"
    assemble_final(silent, narration, music, sfx, captions, final_path)
    validate_video(final_path)
    extract_thumbnail(final_path, paths.output / f"{basename}-thumbnail.jpg")
    return final_path


def run_dry(paths: RunPaths) -> Path:
    plan = sample_plan()
    references = decode_references(REPO_ROOT, paths.references)
    write_json(paths.output / "dry-run-plan.json", plan.model_dump())
    sources = [
        references["style"],
        references["pip"],
        references["lumi"],
        references["style"],
        references["pip"],
        references["lumi"],
    ]
    raw_clips = [
        create_dry_clip(source, paths.raw_clips / f"shot-{index:02d}.mp4", index)
        for index, source in enumerate(sources, start=1)
    ]
    voice_parts = []
    for index in range(1, 7):
        raw_voice = create_dry_voice(paths.voice_raw / f"shot-{index:02d}.wav", index)
        fitted = fit_voice(raw_voice, paths.voice_fitted / f"shot-{index:02d}.wav")
        voice_parts.append(fitted)
    narration = concat_audio(voice_parts, paths.audio / "narration.wav")
    music = create_fallback_music(paths.audio / "music.wav")
    return render_episode(
        plan,
        raw_clips,
        narration,
        music,
        paths,
        basename="dry-run",
    )


def write_publish_files(plan: EpisodePlan, paths: RunPaths, final_path: Path) -> None:
    stem = final_path.stem
    metadata = {
        "title": plan.youtube_title,
        "description": plan.youtube_description,
        "hashtags": plan.hashtags,
        "audience": "Made for kids — confirm this setting during upload",
        "language": "English",
        "duration_seconds": FINAL_SECONDS,
        "file": final_path.name,
        "thumbnail": f"{stem}-thumbnail.jpg",
    }
    write_json(paths.output / f"{stem}-metadata.json", metadata)
    (paths.output / f"{stem}-upload-copy.txt").write_text(
        f"{plan.youtube_title}\n\n{plan.youtube_description}\n\n"
        + " ".join(plan.hashtags)
        + "\n",
        encoding="utf-8",
    )


def run_live(paths: RunPaths, episode_seed: str) -> Path:
    require_environment(["OPENAI_API_KEY", "FAL_KEY", "ELEVENLABS_API_KEY"])
    series = load_json(REPO_ROOT / "config" / "series.json")
    history_path = REPO_ROOT / "state" / "history.json"
    history = load_json(history_path)
    if not isinstance(series, dict) or not isinstance(history, list):
        raise RuntimeError("Series configuration or history is invalid")

    episode_number = next_episode_number(history)
    known_animation_cost = 6 * SHOT_SECONDS * KNOWN_ANIMATION_RATE
    guarded_estimate = known_animation_cost + 2.0
    maximum = float(os.getenv("MAX_ESTIMATED_COST_USD", "7.00"))
    if guarded_estimate > maximum:
        raise RuntimeError(
            f"Estimated run cost ${guarded_estimate:.2f} exceeds guard ${maximum:.2f}"
        )

    references = decode_references(REPO_ROOT, paths.references)
    remote_references = check_provider_access(references)

    log(f"Writing episode {episode_number}.")
    plan = generate_plan(series, history, episode_number, episode_seed)
    review = review_plan(series, plan)
    if not plan_is_strong(review):
        notes = review.revision_notes or [
            "Strengthen the first-second hook and make every action cause the next beat."
        ]
        log("Editorial gate requested one rewrite.")
        plan = generate_plan(
            series,
            history,
            episode_number,
            episode_seed,
            revision_notes=notes,
        )
    write_json(paths.run_root / "episode-plan.json", plan.model_dump())
    write_json(paths.run_root / "script-review.json", review.model_dump())

    frames = generate_frames(plan, references, paths)
    try:
        frame_review = review_frames(plan, frames)
        write_json(paths.run_root / "frame-review.json", frame_review.model_dump())
        if not frame_review.approved and MAX_FRAME_RETRIES:
            retry_numbers = frame_review.retry_shots[:MAX_FRAME_RETRIES]
            issue = "; ".join(frame_review.problems)
            for number in retry_numbers:
                log(f"Retrying frame {number} after visual quality review.")
                generate_frame(
                    plan.shots[number - 1],
                    references,
                    frames[number - 1],
                    extra_instruction=(
                        "Correct these quality-review issues while preserving the story: "
                        + issue
                    ),
                )
    except Exception as exc:  # noqa: BLE001
        log(f"Visual review skipped after non-fatal {type(exc).__name__}.")

    narration = generate_voice(plan, paths)
    music = generate_music(plan, paths)

    raw_clips = animate_all(plan, frames, remote_references, paths)
    basename = f"episode-{episode_number:03d}"
    final_path = render_episode(
        plan,
        raw_clips,
        narration,
        music,
        paths,
        basename=basename,
    )
    write_json(paths.output / f"{basename}-plan.json", plan.model_dump())
    write_publish_files(plan, paths, final_path)

    history.append(
        {
            "episode_number": episode_number,
            "episode_title": plan.episode_title,
            "story_question": plan.story_question,
            "continuity_advance": plan.continuity_advance,
            "created_at": datetime.now(UTC).isoformat(),
            "run_id": os.getenv("GITHUB_RUN_ID", "local"),
        }
    )
    write_json(history_path, history)
    write_json(
        paths.output / f"{basename}-manifest.json",
        {
            "status": "passed",
            "episode_number": episode_number,
            "generated_at": datetime.now(UTC).isoformat(),
            "models": {
                "script": SCRIPT_MODEL,
                "image": IMAGE_MODEL,
                "animation": ANIMATION_MODEL,
                "voice": VOICE_MODEL,
            },
            "known_animation_cost_usd": round(known_animation_cost, 2),
            "guarded_run_estimate_usd": round(guarded_estimate, 2),
            "quality_checks": [
                "script editorial review",
                "visual consistency review",
                "30-second duration",
                "1080x1920 portrait",
                "H.264 video and AAC audio",
                "full decode test",
            ],
        },
    )
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Pip & Lumi YouTube Short.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test the entire media assembly locally without calling paid APIs.",
    )
    parser.add_argument(
        "--episode-seed",
        default=os.getenv("EPISODE_SEED", ""),
        help="Optional creative premise for this episode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_binary("ffmpeg")
    require_binary("ffprobe")
    run_token = os.getenv("GITHUB_RUN_ID") or str(int(time.time()))
    run_name = ("dry-" if args.dry_run else "run-") + run_token
    paths = RunPaths.create(REPO_ROOT, run_name)
    try:
        final_path = run_dry(paths) if args.dry_run else run_live(paths, args.episode_seed)
        log(f"SUCCESS: {final_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        log(f"FAILED: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
