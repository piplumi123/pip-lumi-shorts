#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ASSETS = [
    "pip-front.jpg.b64",
    "lumi-front.jpg.b64",
    "style-reference.jpg.b64",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise RuntimeError(f"Missing required binary: {binary}")

    series = json.loads((REPO_ROOT / "config" / "series.json").read_text())
    if series.get("shot_count") != 6 or series.get("episode_seconds") != 30:
        raise RuntimeError("Series timing configuration is invalid")

    asset_root = REPO_ROOT / "assets" / "characters"
    for name in REQUIRED_ASSETS:
        path = asset_root / name
        if not path.exists():
            raise RuntimeError(f"Missing locked character asset: {name}")
        if len(base64.b64decode(path.read_text().strip())) < 2000:
            raise RuntimeError(f"Locked character asset is invalid: {name}")

    if args.online:
        required = ["OPENAI_API_KEY", "FAL_KEY", "ELEVENLABS_API_KEY"]
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError("Missing required secrets: " + ", ".join(missing))

    print("Setup validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

