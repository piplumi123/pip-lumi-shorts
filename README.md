# Pip & Lumi Shorts Factory

This repository creates one finished 30-second vertical episode at a time:

- an original hook-driven six-shot story;
- locked Pip and Lumi character designs;
- six AI-generated storyboard frames;
- six real five-second animated shots;
- consistent voice-over, original music, sound effects, and burned-in captions;
- a validated 1080×1920 H.264/AAC MP4 plus thumbnail and upload copy.

## Generate an episode

Open **Actions → Generate Pip & Lumi Short → Run workflow**. An optional seed can be
as simple as `a bridge made of moonlight`. When the run finishes, download the
`pip-lumi-episode-*` artifact. It contains the MP4, thumbnail, title, description,
hashtags, plan, and validation manifest.

The first workflow run starts automatically when the workflow is installed. Later
runs are manual so no API money is spent accidentally.

## Required GitHub Actions secrets

- `OPENAI_API_KEY`
- `FAL_KEY`
- `ELEVENLABS_API_KEY`

Optional repository variables:

- `OPENAI_SCRIPT_MODEL` (default `gpt-5.6`)
- `OPENAI_IMAGE_MODEL` (default `gpt-image-2`)
- `ELEVENLABS_VOICE_ID` (default `JBFqnCBsd6RMkjVDRZzb`)
- `MAX_ESTIMATED_COST_USD` (default `7.00`)

## Local no-cost test

```bash
python -m pip install -r requirements.txt
python scripts/validate_setup.py
pytest -q
python scripts/generate_short.py --dry-run
```

The dry run tests the entire 30-second media assembly without calling paid APIs.

## Publishing

For the first three episodes, inspect the MP4 and upload it manually. Set the YouTube
audience accurately; this series is designed as made-for-kids content. Automatic
publishing should only be enabled after the visual and story quality is proven.
