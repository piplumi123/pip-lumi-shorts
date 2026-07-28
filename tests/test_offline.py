from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_short import (  # noqa: E402
    RunPaths,
    choose_available_model,
    sample_plan,
    write_publish_files,
)
from shorts_factory.core import (  # noqa: E402
    ass_time,
    atempo_chain,
    caption_chunks,
    decode_references,
    write_ass,
)
from shorts_factory.models import EpisodePlan  # noqa: E402


def test_sample_plan_has_exact_story_shape() -> None:
    plan = sample_plan()
    assert len(plan.shots) == 6
    assert [shot.beat for shot in plan.shots] == [
        "hook",
        "escalation",
        "wrong_turn",
        "discovery",
        "payoff",
        "twist",
    ]
    assert 55 <= sum(len(shot.narration.split()) for shot in plan.shots) <= 82


def test_episode_plan_round_trip() -> None:
    plan = sample_plan(7)
    restored = EpisodePlan.model_validate_json(plan.model_dump_json())
    assert restored.episode_number == 7
    assert restored.youtube_title.endswith("#Shorts")


def test_model_resolution_prefers_requested_then_falls_back() -> None:
    available = {"gpt-4.1", "gpt-image-1"}
    assert choose_available_model("gpt-4.1", ["gpt-5", "gpt-4.1"], available) == "gpt-4.1"
    assert choose_available_model("auto", ["gpt-5", "gpt-4.1"], available) == "gpt-4.1"
    assert choose_available_model("auto", ["gpt-5"], available) == ""


def test_locked_multi_angle_references_decode(tmp_path: Path) -> None:
    references = decode_references(REPO_ROOT, tmp_path)
    assert set(references) == {"pip", "lumi", "pip_angle", "lumi_angle", "style"}
    assert all(path.stat().st_size > 10_000 for path in references.values())


def test_upload_copy_does_not_duplicate_hashtags(tmp_path: Path) -> None:
    plan = sample_plan()
    plan.youtube_description += "\n\n#Shorts #Animation"
    paths = RunPaths.create(tmp_path, "test")
    fake_video = paths.output / "episode-001.mp4"
    fake_video.touch()
    write_publish_files(plan, paths, fake_video)
    copy = (paths.output / "episode-001-upload-copy.txt").read_text()
    assert copy.count("#Shorts") == 2  # once in the title, once in the description
    assert copy.count("#Animation") == 1
    assert "#PipAndLumi" in copy


def test_caption_chunks_are_short_and_complete() -> None:
    text = "One two three four five six seven eight nine ten eleven"
    chunks = caption_chunks(text, maximum_words=4)
    assert all(1 <= len(chunk.split()) <= 4 for chunk in chunks)
    assert " ".join(chunks) == text


def test_ass_timing_and_render(tmp_path: Path) -> None:
    assert ass_time(5.25) == "0:00:05.25"
    target = write_ass(sample_plan(), tmp_path / "captions.ass")
    rendered = target.read_text()
    assert "[Events]" in rendered
    assert "A FALLING STAR" in rendered
    assert "0:00:29" in rendered


@pytest.mark.parametrize(
    ("speed", "expected_parts"),
    [(1.2, 1), (4.5, 3), (0.2, 3)],
)
def test_atempo_chain(speed: float, expected_parts: int) -> None:
    chain = atempo_chain(speed)
    assert len(chain.split(",")) == expected_parts
