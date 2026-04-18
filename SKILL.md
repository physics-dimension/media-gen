---
name: media-gen
description: >
  AI media generation with 4 atomic capabilities: text-to-image, image-to-image,
  text-to-video, image-to-video. Supports Gemini/Imagen image models and Veo 3.1
  video models. Auto-selects model by aspect ratio/orientation, LLM prompt enhancement,
  batch generation with concurrency=5. Image and video can use separate API channels.
  Use when user asks to generate images, create videos, draw, paint, animate,
  or produce any visual media content.
---

# Media Generation

4 atomic capabilities in a single skill. Pure Python stdlib, no pip install.

## Quick Start

```bash
SKILL=~/.claude/skills/media-gen/scripts/media_gen.py

# text-to-image
PYTHONUTF8=1 python $SKILL -p "a red rose in morning light" --enhance

# image-to-image
PYTHONUTF8=1 python $SKILL -p "convert to watercolor style" --ref photo.jpg

# text-to-video
PYTHONUTF8=1 python $SKILL -p "cat playing in garden" --video

# image-to-video
PYTHONUTF8=1 python $SKILL -p "animate this scene" --ref photo.jpg --video

# image-to-video (first + last frame interpolation)
PYTHONUTF8=1 python $SKILL -p "smooth transition" --ref start.jpg --ref-last end.jpg --video
```

## Python API

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("~/.claude/skills/media-gen/scripts").expanduser()))
from _core import text2img, img2img, text2video, img2video, batch_generate

# text2img(prompt, *, model="auto", aspect_ratio=None, enhance=False,
#          output_dir=None, stem=None, timeout=240) -> dict

# img2img(prompt, reference_image, *, model="auto",
#         output_dir=None, stem=None, timeout=240) -> dict

# text2video(prompt, *, model="auto", orientation="landscape", quality="fast",
#            output_dir=None, stem=None, timeout=600) -> dict

# img2video(prompt, reference_image, *, last_frame=None, model="auto",
#           orientation="landscape", quality="fast",
#           output_dir=None, stem=None, timeout=600) -> dict
# When last_frame is provided, runs first-last-frame interpolation.

result = text2img("sunset over mountains", enhance=True, aspect_ratio="16:9")
# result["saved_paths"], result["urls"], result["elapsed_ms"]

result = text2video("cinematic landscape", quality="ultra", orientation="landscape")
# result["saved_path"], result["url"]
```

## Image Models (auto-selected by aspect ratio)

| Aspect | Model |
|--------|-------|
| 16:9   | gemini-3.0-pro-image-landscape |
| 9:16   | gemini-3.0-pro-image-portrait |
| 1:1    | gemini-3.0-pro-image-square |
| 4:3    | gemini-3.0-pro-image-four-three |
| 3:4    | gemini-3.0-pro-image-three-four |
| default | gemini-3.0-pro-image-landscape |

## Video Models

### Text-to-Video (quality x orientation)

| Quality | Landscape | Portrait |
|---------|-----------|----------|
| fast | veo_3_1_t2v_fast_landscape | veo_3_1_t2v_fast_portrait |
| ultra | veo_3_1_t2v_fast_ultra | veo_3_1_t2v_fast_portrait_ultra |
| 4k | veo_3_1_t2v_fast_4k | veo_3_1_t2v_fast_portrait_4k |
| 1080p | veo_3_1_t2v_fast_1080p | veo_3_1_t2v_fast_portrait_1080p |
| lite | veo_3_1_t2v_lite_landscape | veo_3_1_t2v_lite_portrait |
| standard | veo_3_1_t2v_landscape | veo_3_1_t2v_portrait |

### Image-to-Video (quality x orientation)

| Quality | Landscape | Portrait |
|---------|-----------|----------|
| fast | veo_3_1_i2v_s_fast_fl | veo_3_1_i2v_s_fast_portrait_fl |
| ultra | veo_3_1_i2v_s_fast_ultra_fl | veo_3_1_i2v_s_fast_portrait_ultra_fl |
| 4k | veo_3_1_i2v_s_fast_ultra_fl_4k | veo_3_1_i2v_s_fast_portrait_ultra_fl_4k |
| lite | veo_3_1_i2v_lite_landscape | veo_3_1_i2v_lite_portrait |

## Configuration

Create `.env` in the skill directory (`~/.claude/skills/media-gen/.env`):

```
IMG_BASE_URL=https://your-api-provider.com
IMG_API_KEY=your-api-key
VIDEO_BASE_URL=https://your-api-provider.com
VIDEO_API_KEY=your-api-key
HUOSHAN2_BASE_URL=https://your-api-provider.com    # fallback
HUOSHAN2_API_KEY=your-api-key                      # fallback
```

Image and video can use separate API endpoints. If `IMG_*` / `VIDEO_*` are not
set, the `HUOSHAN2_*` values are used as fallback.

## Batch Generation

```python
results = batch_generate([
    {"fn": "text2img", "prompt": "sunset", "aspect_ratio": "16:9"},
    {"fn": "text2img", "prompt": "mountain", "enhance": True},
    {"fn": "text2video", "prompt": "waves", "quality": "fast"},
], concurrency=5)
```

## CLI Parameters

```
-p, --prompt       Description / instruction (required)
--ref              Reference image path/URL (img2img / img2video first frame)
--ref-last         Last-frame image for img2video (enables first-last-frame mode)
--video            Video mode flag (text2video / img2video)
-m, --model        Model override (default: auto)
-a, --aspect-ratio Image aspect ratio: 16:9, 9:16, 1:1, 4:3, 3:4
--orientation      Video orientation: landscape, portrait (default: landscape)
--quality          Video quality: fast, standard, lite, ultra, 4k, 1080p
-e, --enhance      LLM prompt enhancement (images only)
-o, --output-dir   Output directory
-s, --stem         Filename stem
-t, --timeout      Timeout in seconds
```

Routing: `--video` flag selects video mode; `--ref` selects reference-image mode.
Combination determines which atomic function runs.
