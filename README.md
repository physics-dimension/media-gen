# Media-Gen

AI media generation skill for [Droid](https://claude.ai/code) with 4 atomic capabilities:

| Capability | Input | Output | Flag |
|---|---|---|---|
| **text2img** | Text prompt | Image | _(default)_ |
| **img2img** | Text + reference image | Image | `--ref` |
| **text2video** | Text prompt | Video | `--video` |
| **img2video** | Text + reference image | Video | `--ref --video` |

Pure Python stdlib -- zero dependencies, no `pip install` required.

## Features

- **4 atomic functions** -- each independently callable via CLI or Python API
- **Auto model selection** -- picks the right model based on aspect ratio (images) or quality/orientation (videos)
- **LLM prompt enhancement** -- optional rewrite of short prompts into detailed image prompts
- **Batch generation** -- concurrent execution with configurable parallelism (default: 5 workers)
- **Separate API channels** -- image and video generation can use different API endpoints
- **File validation** -- checks JPEG/PNG/WebP/MP4 magic bytes before saving
- **OpenAI-compatible API** -- works with any provider exposing `/v1/chat/completions`

## Quick Start

```bash
# Set up your API credentials
cp .env.example .env
# Edit .env with your API base URL and key

# Generate an image
PYTHONUTF8=1 python scripts/media_gen.py -p "a red rose in morning light"

# Generate with prompt enhancement
PYTHONUTF8=1 python scripts/media_gen.py -p "sunset" --enhance

# Image-to-image transformation
PYTHONUTF8=1 python scripts/media_gen.py -p "convert to watercolor style" --ref photo.jpg

# Generate a video
PYTHONUTF8=1 python scripts/media_gen.py -p "cat playing in garden" --video

# Animate an image
PYTHONUTF8=1 python scripts/media_gen.py -p "add gentle motion" --ref photo.jpg --video
```

## Python API

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
from _core import text2img, img2img, text2video, img2video, batch_generate

# Text to image
result = text2img("sunset over mountains", enhance=True, aspect_ratio="16:9")
print(result["saved_paths"])  # ['output/images/sunset-over-mountains.jpg']

# Image to image
result = img2img("convert to oil painting", "photo.jpg")
print(result["saved_paths"])

# Text to video
result = text2video("cinematic landscape", quality="ultra", orientation="landscape")
print(result["saved_path"])   # 'output/videos/cinematic-landscape.mp4'

# Image to video
result = img2video("add gentle camera motion", "photo.jpg")
print(result["saved_path"])

# Batch generation (concurrent)
results = batch_generate([
    {"fn": "text2img", "prompt": "sunset", "aspect_ratio": "16:9"},
    {"fn": "text2img", "prompt": "mountain", "enhance": True},
    {"fn": "text2video", "prompt": "ocean waves", "quality": "fast"},
], concurrency=5)
```

## Supported Models

### Image Models (auto-selected by aspect ratio)

| Aspect Ratio | Model |
|---|---|
| 16:9 | `gemini-3.0-pro-image-landscape` |
| 9:16 | `gemini-3.0-pro-image-portrait` |
| 1:1 | `gemini-3.0-pro-image-square` |
| 4:3 | `gemini-3.0-pro-image-four-three` |
| 3:4 | `gemini-3.0-pro-image-three-four` |

### Video Models -- Text-to-Video

| Quality | Landscape | Portrait |
|---|---|---|
| fast | `veo_3_1_t2v_fast_landscape` | `veo_3_1_t2v_fast_portrait` |
| standard | `veo_3_1_t2v_landscape` | `veo_3_1_t2v_portrait` |
| lite | `veo_3_1_t2v_lite_landscape` | `veo_3_1_t2v_lite_portrait` |
| ultra | `veo_3_1_t2v_fast_ultra` | `veo_3_1_t2v_fast_portrait_ultra` |
| 4k | `veo_3_1_t2v_fast_4k` | `veo_3_1_t2v_fast_portrait_4k` |
| 1080p | `veo_3_1_t2v_fast_1080p` | `veo_3_1_t2v_fast_portrait_1080p` |

### Video Models -- Image-to-Video

| Quality | Landscape | Portrait |
|---|---|---|
| fast | `veo_3_1_i2v_s_fast_fl` | `veo_3_1_i2v_s_fast_portrait_fl` |
| ultra | `veo_3_1_i2v_s_fast_ultra_fl` | `veo_3_1_i2v_s_fast_portrait_ultra_fl` |
| 4k | `veo_3_1_i2v_s_fast_ultra_fl_4k` | `veo_3_1_i2v_s_fast_portrait_ultra_fl_4k` |
| lite | `veo_3_1_i2v_lite_landscape` | `veo_3_1_i2v_lite_portrait` |

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```env
# Image generation API
IMG_BASE_URL=https://your-api-provider.com
IMG_API_KEY=your-api-key

# Video generation API (can be same or different endpoint)
VIDEO_BASE_URL=https://your-api-provider.com
VIDEO_API_KEY=your-api-key

# Fallback (used when IMG_*/VIDEO_* not set)
HUOSHAN2_BASE_URL=https://your-api-provider.com
HUOSHAN2_API_KEY=your-api-key
```

Image and video generation can use separate API endpoints. If `IMG_*` / `VIDEO_*` are not set, the `HUOSHAN2_*` values are used as fallback.

## CLI Reference

```
usage: media_gen [-h] -p PROMPT [--ref IMAGE] [--video] [-m MODEL]
                 [-a {16:9,9:16,1:1,4:3,3:4}]
                 [--orientation {landscape,portrait}]
                 [--quality {fast,standard,lite,ultra,4k,1080p}]
                 [-e] [-o OUTPUT_DIR] [-s STEM] [-t TIMEOUT]

Options:
  -p, --prompt         Text prompt (required)
  --ref IMAGE          Reference image path or URL
  --video              Enable video generation mode
  -m, --model          Model override (default: auto)
  -a, --aspect-ratio   Image aspect ratio
  --orientation        Video orientation (default: landscape)
  --quality            Video quality preset (default: fast)
  -e, --enhance        Enable LLM prompt enhancement (images only)
  -o, --output-dir     Output directory
  -s, --stem           Output filename stem
  -t, --timeout        Request timeout in seconds
```

**Routing logic:** The `--video` flag selects video mode; `--ref` selects reference-image mode. Their combination determines which atomic function runs.

## Architecture

```
media-gen/
  .env.example     # Configuration template
  .env             # Your API credentials (git-ignored)
  SKILL.md         # Droid skill manifest
  README.md        # This file
  scripts/
    _core.py       # 4 atomic functions + shared infrastructure
    media_gen.py   # Unified CLI entry point with auto-routing
```

All API calls use the OpenAI Chat Completions format (`POST /v1/chat/completions`), making the skill compatible with any provider that exposes this interface. Image URLs are extracted from markdown in the response; video URLs from `<video>` tags.

## Requirements

- Python 3.10+
- No third-party packages (pure stdlib)
- An API provider compatible with OpenAI Chat Completions format

## As a Claude Skill

This repo is designed to be used as a [Claude skill](https://docs.claude.ai/code). Place it at `~/.claude/skills/media-gen/` and Droid will automatically detect it via `SKILL.md`. Droid can then generate images and videos on demand during conversations.

## License

MIT
