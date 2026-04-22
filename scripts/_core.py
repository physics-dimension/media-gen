"""Media generation core: 4 atomic functions + shared infrastructure.

Pure Python stdlib only -- no pip install needed.

Atomic functions:
    text2img   -- Generate image from text prompt
    img2img    -- Transform image with text guidance
    text2video -- Generate video from text prompt
    img2video  -- Animate image with text guidance

Batch:
    batch_generate -- Run multiple atomic functions concurrently
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent

DEFAULT_IMAGE_DIR = pathlib.Path.cwd() / "output" / "images"
DEFAULT_VIDEO_DIR = pathlib.Path.cwd() / "output" / "videos"

DEFAULT_CONCURRENCY = 5

ENHANCE_MODEL = "gemini-3-flash-preview"

ENHANCE_SYSTEM_PROMPT = (
    "You are an expert image prompt engineer. Rewrite the user's short "
    "description into a vivid, detailed English prompt for AI image generation.\n"
    "\n"
    "RULES:\n"
    "1. Output ONLY the rewritten prompt, nothing else.\n"
    "2. Write in natural English sentences (not keyword lists).\n"
    "3. Preserve the user's core intent -- never add unrelated subjects.\n"
    "4. Add these dimensions naturally: subject details, environment, "
    "lighting, composition, style, mood, quality.\n"
    "5. Use positive descriptions only.\n"
    "6. Keep it 2-4 sentences. Concise but rich.\n"
    "7. If the prompt is already detailed, return it as-is "
    "(translated to English if needed)."
)

# -- Image model auto-selection by aspect ratio -----------------------------

_IMG_ASPECT_MAP: dict[str, str] = {
    "16:9": "gemini-3.0-pro-image-landscape",
    "9:16": "gemini-3.0-pro-image-portrait",
    "1:1":  "gemini-3.0-pro-image-square",
    "4:3":  "gemini-3.0-pro-image-four-three",
    "3:4":  "gemini-3.0-pro-image-three-four",
}

_IMG_DEFAULT_MODEL = "gemini-3.0-pro-image-landscape"

# -- GPT image model auto-selection by aspect ratio (via CPA) ----------------

_GPT_IMG_ASPECT_MAP: dict[str, str] = {
    "16:9": "gpt-draw-1536x1024",
    "9:16": "gpt-draw-1024x1536",
    "1:1":  "gpt-draw-1024x1024",
    "4:3":  "gpt-draw-1536x1024",
    "3:4":  "gpt-draw-1024x1536",
}

_GPT_IMG_DEFAULT_MODEL = "gpt-draw-1024x1024"

# -- Text-to-video model catalog (quality x orientation) --------------------

_T2V_MODELS: dict[str, dict[str, str]] = {
    "fast":     {"landscape": "veo_3_1_t2v_fast_landscape",     "portrait": "veo_3_1_t2v_fast_portrait"},
    "ultra":    {"landscape": "veo_3_1_t2v_fast_ultra",         "portrait": "veo_3_1_t2v_fast_portrait_ultra"},
    "4k":       {"landscape": "veo_3_1_t2v_fast_4k",           "portrait": "veo_3_1_t2v_fast_portrait_4k"},
    "1080p":    {"landscape": "veo_3_1_t2v_fast_1080p",        "portrait": "veo_3_1_t2v_fast_portrait_1080p"},
    "lite":     {"landscape": "veo_3_1_t2v_lite_landscape",     "portrait": "veo_3_1_t2v_lite_portrait"},
    "standard": {"landscape": "veo_3_1_t2v_landscape",          "portrait": "veo_3_1_t2v_portrait"},
}

# -- Image-to-video model catalog (quality x orientation) -------------------

_I2V_MODELS: dict[str, dict[str, str]] = {
    "fast":  {"landscape": "veo_3_1_i2v_s_fast_fl",           "portrait": "veo_3_1_i2v_s_fast_portrait_fl"},
    "ultra": {"landscape": "veo_3_1_i2v_s_fast_ultra_fl",     "portrait": "veo_3_1_i2v_s_fast_portrait_ultra_fl"},
    "4k":    {"landscape": "veo_3_1_i2v_s_fast_ultra_fl_4k",  "portrait": "veo_3_1_i2v_s_fast_portrait_ultra_fl_4k"},
    "lite":  {"landscape": "veo_3_1_i2v_lite_landscape",       "portrait": "veo_3_1_i2v_lite_portrait"},
}

# Dedicated first-last-frame lite models (flow2api requires 2 images exactly).
_I2V_INTERP_MODELS: dict[str, dict[str, str]] = {
    "lite":  {"landscape": "veo_3_1_interpolation_lite_landscape", "portrait": "veo_3_1_interpolation_lite_portrait"},
}


# ===========================================================================
# Exception
# ===========================================================================

class MediaGenError(RuntimeError):
    """Raised when any media generation operation fails."""


# ===========================================================================
# Shared infrastructure
# ===========================================================================

def _load_dotenv() -> None:
    """Load .env from SKILL_DIR into ``os.environ``.

    Parses KEY=VALUE lines, skips comments and blank lines.
    Does **not** override variables already present in the environment.
    """
    env_path = SKILL_DIR / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


# Auto-load on import
_load_dotenv()


def _resolve_config(prefix: str) -> tuple[str, str]:
    """Return ``(base_url, api_key)`` for the given channel prefix.

    Checks ``{PREFIX}_BASE_URL`` / ``{PREFIX}_API_KEY`` first, then falls
    back to ``HUOSHAN2_BASE_URL`` / ``HUOSHAN2_API_KEY``.
    """
    base_url = (
        os.environ.get(f"{prefix}_BASE_URL", "").strip()
        or os.environ.get("HUOSHAN2_BASE_URL", "").strip()
    )
    api_key = (
        os.environ.get(f"{prefix}_API_KEY", "").strip()
        or os.environ.get("HUOSHAN2_API_KEY", "").strip()
    )
    if not base_url:
        raise MediaGenError(
            f"No API base URL configured.  "
            f"Set {prefix}_BASE_URL or HUOSHAN2_BASE_URL in .env"
        )
    if not api_key:
        raise MediaGenError(
            f"No API key configured.  "
            f"Set {prefix}_API_KEY or HUOSHAN2_API_KEY in .env"
        )
    return base_url.rstrip("/"), api_key


def _api_call(
    base_url: str,
    api_key: str,
    payload: dict,
    timeout: int = 240,
) -> dict:
    """POST to ``/v1/chat/completions`` and return parsed JSON."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise MediaGenError(f"API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise MediaGenError(f"API request failed: {exc.reason}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MediaGenError(f"Non-JSON API response: {raw[:300]}") from exc


def _download(url: str, timeout: int = 120) -> bytes:
    """Download file bytes from *url*."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "media-gen-skill/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        raise MediaGenError(f"Download failed ({url}): {exc.reason}") from exc


def _extract_image_urls(content: str) -> list[str]:
    """Extract image URLs from markdown ``![](url)`` or bare URLs."""
    md_urls = re.findall(r"!\[[^\]]*\]\((https?://[^\s)]+)\)", content)
    if md_urls:
        return md_urls
    return re.findall(
        r"(https?://[^\s\])<>\"']+\.(?:jpg|jpeg|png|webp|gif)(?:\?[^\s\])<>\"']*)?)",
        content,
    )


def _extract_video_url(content: str) -> str | None:
    """Extract a video URL from ``<video src='URL'>`` or bare URLs."""
    match = re.search(r"""<video\s+src=['"]([^'"]+)['"]""", content)
    if match:
        return match.group(1)
    match = re.search(
        r"(https?://[^\s\])<>\"']+\.(?:mp4|webm|mov|avi)(?:\?[^\s\])<>\"']*)?)",
        content,
    )
    if match:
        return match.group(1)
    # Last resort: any URL in the content
    match = re.search(r"(https?://[^\s<>'\"]+)", content)
    return match.group(1) if match else None


def _encode_image(path_or_url: str) -> str:
    """Read image from local file path or URL and return a base64 string."""
    p = pathlib.Path(path_or_url)
    if p.is_file():
        return base64.b64encode(p.read_bytes()).decode()
    if path_or_url.startswith(("http://", "https://")):
        req = urllib.request.Request(
            path_or_url, headers={"User-Agent": "media-gen-skill/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return base64.b64encode(resp.read()).decode()
    raise MediaGenError(f"Cannot read image: {path_or_url}")


def _safe_slug(text: str, default: str = "media") -> str:
    """Turn *text* into a filesystem-safe slug (lowercase, max 60 chars)."""
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text)
    cleaned = cleaned.strip("-").lower()
    cleaned = "-".join(filter(None, cleaned.split("-")))
    return cleaned[:60] or default


def _log(msg: str) -> None:
    """Print diagnostic message to stderr."""
    print(msg, file=sys.stderr, flush=True)


# -- Quality checks ---------------------------------------------------------

def _check_image_bytes(data: bytes) -> None:
    """Lightweight validation that *data* looks like a real image file."""
    if len(data) == 0:
        raise MediaGenError("Downloaded image file is empty (0 bytes)")
    hdr = data[:8]
    if (
        hdr[:2] != b"\xff\xd8"            # JPEG
        and hdr[:4] != b"\x89PNG"          # PNG
        and hdr[:4] != b"RIFF"             # WEBP
    ):
        raise MediaGenError(
            "Downloaded file is not a valid image "
            f"(header bytes: {hdr[:8].hex()})"
        )


def _check_video_bytes(data: bytes) -> None:
    """Lightweight validation that *data* looks like an MP4 container."""
    if len(data) == 0:
        raise MediaGenError("Downloaded video file is empty (0 bytes)")
    if b"ftyp" not in data[:12]:
        raise MediaGenError(
            "Downloaded file is not a valid MP4 "
            f"(first 12 bytes: {data[:12].hex()})"
        )


# ===========================================================================
# Atomic function 1: text2img
# ===========================================================================

def text2img(
    prompt: str,
    *,
    provider: str = "auto",
    model: str = "auto",
    aspect_ratio: str | None = None,
    enhance: bool = False,
    output_dir: str | pathlib.Path | None = None,
    stem: str | None = None,
    timeout: int = 240,
) -> dict:
    """Generate image(s) from a text prompt.

    Args:
        provider: "auto" (default Gemini), "gemini", or "gpt" (via CPA).

    Returns a dict with keys: type, provider, model, prompt, enhanced_prompt,
    elapsed_ms, saved_paths, urls.
    """
    if provider == "auto":
        provider = "gemini"

    if provider == "gpt":
        return _text2img_gpt(
            prompt, model=model, aspect_ratio=aspect_ratio, enhance=enhance,
            output_dir=output_dir, stem=stem, timeout=timeout,
        )

    base_url, api_key = _resolve_config("IMG")
    out = pathlib.Path(output_dir) if output_dir else DEFAULT_IMAGE_DIR

    # Model selection
    if model == "auto":
        model = _IMG_ASPECT_MAP.get(aspect_ratio or "", _IMG_DEFAULT_MODEL)

    # Prompt enhancement (uses new enhance module with example library)
    enhanced_prompt: str | None = None
    if enhance:
        try:
            from enhance import enhance_image as _enh_img
            enhanced_prompt = _enh_img(prompt)
            _log(f"[enhance] {prompt}")
            _log(f"[enhance] -> {enhanced_prompt}")
        except Exception as exc:
            _log(f"[enhance] failed ({exc}), using original prompt")
            enhanced_prompt = None

    effective_prompt = enhanced_prompt or prompt

    # API call
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"Generate an image: {effective_prompt}"}
        ],
        "max_tokens": 4096,
    }

    t0 = time.time()
    data = _api_call(base_url, api_key, payload, timeout=timeout)
    elapsed_ms = int((time.time() - t0) * 1000)

    # Extract URLs
    content = _chat_content(data)
    urls = _extract_image_urls(content)
    if not urls:
        raise MediaGenError(f"No image URLs in response. Content: {content[:300]}")
    _log(f"[text2img:gemini] {len(urls)} image(s) in {elapsed_ms}ms")

    # Download & save
    saved_paths = _save_images(urls, out, stem or _safe_slug(prompt))

    return {
        "type": "text2img",
        "provider": "gemini",
        "model": model,
        "prompt": prompt,
        "enhanced_prompt": enhanced_prompt,
        "elapsed_ms": elapsed_ms,
        "saved_paths": saved_paths,
        "urls": urls,
    }


# ===========================================================================
# GPT provider: text2img via CPA
# ===========================================================================

def _text2img_gpt(
    prompt: str,
    *,
    model: str = "auto",
    aspect_ratio: str | None = None,
    enhance: bool = False,
    output_dir: str | pathlib.Path | None = None,
    stem: str | None = None,
    timeout: int = 240,
) -> dict:
    """Generate image via GPT (CPA cli-proxy-api). Returns base64 in message.images."""
    base_url, api_key = _resolve_config("CPA")
    out = pathlib.Path(output_dir) if output_dir else DEFAULT_IMAGE_DIR

    if model == "auto":
        model = _GPT_IMG_ASPECT_MAP.get(aspect_ratio or "", _GPT_IMG_DEFAULT_MODEL)

    enhanced_prompt: str | None = None
    if enhance:
        try:
            from enhance import enhance_image as _enh_img
            enhanced_prompt = _enh_img(prompt)
            _log(f"[enhance] {prompt}")
            _log(f"[enhance] -> {enhanced_prompt}")
        except Exception as exc:
            _log(f"[enhance] failed ({exc}), using original prompt")

    effective_prompt = enhanced_prompt or prompt

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": effective_prompt}],
        "max_tokens": 4096,
    }

    _log(f"[text2img:gpt] model={model}")
    t0 = time.time()
    data = _api_call(base_url, api_key, payload, timeout=timeout)
    elapsed_ms = int((time.time() - t0) * 1000)

    # GPT/CPA returns images in message.images[] as data URIs
    images = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("images", [])
    )
    if not images:
        raise MediaGenError(
            f"No images in GPT response. Keys: "
            f"{list(data.get('choices', [{}])[0].get('message', {}).keys())}"
        )

    saved_paths = _save_images_from_base64(images, out, stem or _safe_slug(prompt))
    _log(f"[text2img:gpt] {len(saved_paths)} image(s) in {elapsed_ms}ms")

    return {
        "type": "text2img",
        "provider": "gpt",
        "model": model,
        "prompt": prompt,
        "enhanced_prompt": enhanced_prompt,
        "elapsed_ms": elapsed_ms,
        "saved_paths": saved_paths,
        "urls": [],
    }


def _save_images_from_base64(
    images: list[dict],
    output_dir: pathlib.Path,
    slug: str,
) -> list[str]:
    """Decode base64 data-URI images and save to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for i, img in enumerate(images):
        url = img.get("image_url", {}).get("url", "")
        if not url.startswith("data:"):
            continue
        # data:image/png;base64,AAAA...
        header, _, b64 = url.partition(",")
        raw = base64.b64decode(b64)
        _check_image_bytes(raw)
        ext = "png"
        if "jpeg" in header or "jpg" in header:
            ext = "jpg"
        elif "webp" in header:
            ext = "webp"
        suffix = f"_{i + 1}" if len(images) > 1 else ""
        path = output_dir / f"{slug}{suffix}.{ext}"
        path.write_bytes(raw)
        saved.append(str(path))
        _log(f"  saved {path} ({len(raw)} bytes)")
    return saved


# ===========================================================================
# Atomic function 2: img2img
# ===========================================================================

def img2img(
    prompt: str,
    reference_image: str,
    *,
    model: str = "auto",
    output_dir: str | pathlib.Path | None = None,
    stem: str | None = None,
    timeout: int = 240,
) -> dict:
    """Transform an existing image guided by a text prompt.

    Returns a dict with keys: type, model, prompt, reference_image,
    elapsed_ms, saved_paths, urls.
    """
    base_url, api_key = _resolve_config("IMG")
    out = pathlib.Path(output_dir) if output_dir else DEFAULT_IMAGE_DIR

    if model == "auto":
        model = _IMG_DEFAULT_MODEL

    img_b64 = _encode_image(reference_image)
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        },
        {"type": "text", "text": prompt},
    ]

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": 4096,
    }

    t0 = time.time()
    data = _api_call(base_url, api_key, payload, timeout=timeout)
    elapsed_ms = int((time.time() - t0) * 1000)

    content = _chat_content(data)
    urls = _extract_image_urls(content)
    if not urls:
        raise MediaGenError(f"No image URLs in response. Content: {content[:300]}")
    _log(f"[img2img] {len(urls)} image(s) in {elapsed_ms}ms")

    saved_paths = _save_images(urls, out, stem or _safe_slug(prompt))

    return {
        "type": "img2img",
        "model": model,
        "prompt": prompt,
        "reference_image": reference_image,
        "elapsed_ms": elapsed_ms,
        "saved_paths": saved_paths,
        "urls": urls,
    }


# ===========================================================================
# Atomic function 3: text2video
# ===========================================================================

def text2video(
    prompt: str,
    *,
    model: str = "auto",
    orientation: str = "landscape",
    quality: str = "fast",
    output_dir: str | pathlib.Path | None = None,
    stem: str | None = None,
    timeout: int = 600,
) -> dict:
    """Generate a video from a text prompt.

    Returns a dict with keys: type, model, prompt, orientation, quality,
    elapsed_ms, saved_path, url.
    """
    base_url, api_key = _resolve_config("VIDEO")
    out = pathlib.Path(output_dir) if output_dir else DEFAULT_VIDEO_DIR

    if model == "auto":
        model = _resolve_video_model(_T2V_MODELS, quality, orientation)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
    }

    _log(f"[text2video] model={model} orientation={orientation} quality={quality}")
    _log(f"[text2video] prompt: {prompt[:120]}...")

    t0 = time.time()
    data = _api_call(base_url, api_key, payload, timeout=timeout)
    elapsed_ms = int((time.time() - t0) * 1000)

    content = _chat_content(data)
    url = _extract_video_url(content)
    if not url:
        raise MediaGenError(
            f"No video URL in response. Content: {content[:300]}"
        )
    _log(f"[text2video] got URL in {elapsed_ms}ms")

    saved_path = _save_video(url, out, stem or _safe_slug(prompt))

    return {
        "type": "text2video",
        "model": model,
        "prompt": prompt,
        "orientation": orientation,
        "quality": quality,
        "elapsed_ms": elapsed_ms,
        "saved_path": saved_path,
        "url": url,
    }


# ===========================================================================
# Atomic function 4: img2video
# ===========================================================================

def img2video(
    prompt: str,
    reference_image: str,
    *,
    last_frame: str | None = None,
    model: str = "auto",
    orientation: str = "landscape",
    quality: str = "fast",
    output_dir: str | pathlib.Path | None = None,
    stem: str | None = None,
    timeout: int = 600,
) -> dict:
    """Animate an image guided by a text prompt.

    When *last_frame* is provided, flow2api runs first-last-frame
    interpolation (both images guide the video); otherwise only the
    first frame drives generation.

    Returns a dict with keys: type, model, prompt, reference_image,
    last_frame, orientation, quality, elapsed_ms, saved_path, url.
    """
    base_url, api_key = _resolve_config("VIDEO")
    out = pathlib.Path(output_dir) if output_dir else DEFAULT_VIDEO_DIR

    if model == "auto":
        if last_frame and quality in _I2V_INTERP_MODELS:
            model = _resolve_video_model(_I2V_INTERP_MODELS, quality, orientation)
        else:
            model = _resolve_video_model(_I2V_MODELS, quality, orientation)

    img_b64 = _encode_image(reference_image)
    user_content: list[dict] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        },
    ]
    if last_frame:
        last_b64 = _encode_image(last_frame)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{last_b64}"},
        })
    user_content.append({"type": "text", "text": prompt})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": 4096,
    }

    _log(f"[img2video] model={model} orientation={orientation} quality={quality}")
    _log(f"[img2video] prompt: {prompt[:120]}...")

    t0 = time.time()
    data = _api_call(base_url, api_key, payload, timeout=timeout)
    elapsed_ms = int((time.time() - t0) * 1000)

    content = _chat_content(data)
    url = _extract_video_url(content)
    if not url:
        raise MediaGenError(
            f"No video URL in response. Content: {content[:300]}"
        )
    _log(f"[img2video] got URL in {elapsed_ms}ms")

    saved_path = _save_video(url, out, stem or _safe_slug(prompt))

    return {
        "type": "img2video",
        "model": model,
        "prompt": prompt,
        "reference_image": reference_image,
        "last_frame": last_frame,
        "orientation": orientation,
        "quality": quality,
        "elapsed_ms": elapsed_ms,
        "saved_path": saved_path,
        "url": url,
    }


# ===========================================================================
# Batch generation
# ===========================================================================

def batch_generate(
    items: list[dict],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict]:
    """Run multiple atomic functions concurrently.

    Each item is a dict with ``"fn"`` (one of ``"text2img"``, ``"img2img"``,
    ``"text2video"``, ``"img2video"``) and remaining keys passed as kwargs.

    Returns a list of result dicts in the same order as *items*.
    """
    fn_map = {
        "text2img": text2img,
        "img2img": img2img,
        "text2video": text2video,
        "img2video": img2video,
    }

    def _run(item: dict, idx: int) -> dict:
        fn_name = item.get("fn", "")
        fn = fn_map.get(fn_name)
        if fn is None:
            return {"index": idx, "status": "error",
                    "error": f"Unknown function: {fn_name}"}
        kwargs = {k: v for k, v in item.items() if k != "fn"}
        try:
            result = fn(**kwargs)
            return {"index": idx, "status": "ok", **result}
        except Exception as exc:
            return {"index": idx, "status": "error", "error": str(exc)}

    results: list[dict | None] = [None] * len(items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_run, item, i): i for i, item in enumerate(items)}
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            results[r["index"]] = r  # type: ignore[index]
            idx_display = r["index"] + 1
            if r["status"] == "ok":
                _log(f"[batch] {idx_display}/{len(items)} ok")
            else:
                _log(f"[batch] {idx_display}/{len(items)} FAILED: "
                     f"{r.get('error', '')[:100]}")

    ok = sum(1 for r in results if r and r["status"] == "ok")
    _log(f"[batch] Done: {ok}/{len(items)} succeeded (concurrency={concurrency})")
    return results  # type: ignore[return-value]


# ===========================================================================
# Internal helpers
# ===========================================================================

def _chat_content(data: dict) -> str:
    """Extract the assistant message content from a Chat Completions response."""
    choices = data.get("choices", [])
    if not choices:
        raise MediaGenError(
            f"No choices in API response: {json.dumps(data)[:300]}"
        )
    return choices[0].get("message", {}).get("content", "")


def _enhance_prompt(
    prompt: str,
    base_url: str,
    api_key: str,
) -> str:
    """Use a lightweight LLM to rewrite *prompt* into a richer image prompt."""
    payload = {
        "model": ENHANCE_MODEL,
        "messages": [
            {"role": "system", "content": ENHANCE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 400,
        "temperature": 0.7,
    }
    data = _api_call(base_url, api_key, payload, timeout=30)
    text = _chat_content(data).strip()
    # Strip wrapping quotes if the model added them
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text or prompt


def _resolve_video_model(
    catalog: dict[str, dict[str, str]],
    quality: str,
    orientation: str,
) -> str:
    """Pick a video model from *catalog* by quality and orientation."""
    quality_map = catalog.get(quality)
    if not quality_map:
        # Fall back to "fast"
        quality_map = catalog.get("fast")
        if not quality_map:
            raise MediaGenError(
                f"Quality '{quality}' not available. "
                f"Options: {list(catalog.keys())}"
            )
    model = quality_map.get(orientation)
    if not model:
        raise MediaGenError(
            f"Orientation '{orientation}' not available for quality "
            f"'{quality}'. Options: {list(quality_map.keys())}"
        )
    return model


def _save_images(
    urls: list[str],
    output_dir: pathlib.Path,
    slug: str,
) -> list[str]:
    """Download images, validate, and save to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for i, url in enumerate(urls):
        data = _download(url)
        _check_image_bytes(data)
        # Determine extension from URL
        parsed = urllib.parse.urlparse(url)
        ext = pathlib.Path(parsed.path).suffix.lower().lstrip(".")
        if ext not in ("jpg", "jpeg", "png", "webp", "gif"):
            ext = "jpg"
        suffix = f"_{i + 1}" if len(urls) > 1 else ""
        path = output_dir / f"{slug}{suffix}.{ext}"
        path.write_bytes(data)
        saved.append(str(path))
        _log(f"  saved {path} ({len(data)} bytes)")
    return saved


def _save_video(
    url: str,
    output_dir: pathlib.Path,
    slug: str,
) -> str:
    """Download a video, validate, and save to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _download(url)
    _check_video_bytes(data)
    # Determine extension from URL
    parsed = urllib.parse.urlparse(url)
    ext = pathlib.Path(parsed.path).suffix.lower().lstrip(".")
    if ext not in ("mp4", "webm", "mov", "avi"):
        ext = "mp4"
    path = output_dir / f"{slug}.{ext}"
    path.write_bytes(data)
    _log(f"  saved {path} ({len(data)} bytes)")
    return str(path)
