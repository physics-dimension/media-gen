#!/usr/bin/env python3
"""Prompt enhancement for image and video generation.

Loads system prompts and example libraries from prompts/ directory,
selects relevant examples by category matching, and rewrites user
input into professional-grade prompts via LLM.

Usage:
    # CLI
    python enhance.py -p "一只猫在花园里" --type image
    python enhance.py -p "城市延时摄影" --type video

    # Python API
    from enhance import enhance_image, enhance_video
    result = enhance_image("一只猫在花园里")
    result = enhance_video("城市延时摄影")
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_PROMPTS_DIR = _HERE.parent / "prompts"

# Ensure _core is importable
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _core import _api_call, _resolve_config, _log, _load_dotenv  # noqa: E402

_load_dotenv()

# ── LLM model for enhancement ──
ENHANCE_MODEL = "deepseek-chat"
MAX_EXAMPLES = 3


# ======================================================================
# Asset loading
# ======================================================================

def _load_system_prompt(media_type: str) -> str:
    """Load the system prompt markdown for *media_type* ('image' or 'video')."""
    path = _PROMPTS_DIR / f"{media_type}_system.md"
    if not path.is_file():
        raise FileNotFoundError(f"System prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _load_examples(media_type: str) -> list[dict]:
    """Load the example library JSON for *media_type*."""
    path = _PROMPTS_DIR / f"{media_type}_examples.json"
    if not path.is_file():
        raise FileNotFoundError(f"Examples file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ======================================================================
# Category matching
# ======================================================================

def _select_examples(
    user_input: str,
    examples: list[dict],
    max_count: int = MAX_EXAMPLES,
) -> list[dict]:
    """Select up to *max_count* examples matching the user input by category.

    Strategy:
    1. Scan each example's category keywords against user_input (case-insensitive).
    2. Score = number of matched keywords.
    3. Return top-N by score.
    4. Fallback: if no matches, return random sample.
    """
    input_lower = user_input.lower()
    scored: list[tuple[int, int, dict]] = []

    for i, ex in enumerate(examples):
        cats = ex.get("category", [])
        score = sum(1 for kw in cats if kw.lower() in input_lower)
        scored.append((score, i, ex))

    # Sort by score descending, then by original order
    scored.sort(key=lambda t: (-t[0], t[1]))

    # Take examples with score > 0
    matched = [ex for score, _, ex in scored if score > 0]

    if matched:
        return matched[:max_count]

    # Fallback: random sample
    return random.sample(examples, min(max_count, len(examples)))


def _format_examples(selected: list[dict]) -> str:
    """Format selected examples into a readable block for the system prompt."""
    lines: list[str] = []
    for i, ex in enumerate(selected, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Input: {ex['input']}")
        lines.append(f"  Output: {ex['output']}")
        lines.append("")
    return "\n".join(lines)


# ======================================================================
# Core enhance functions
# ======================================================================

def _enhance(
    user_input: str,
    media_type: str,
    model: str = ENHANCE_MODEL,
    timeout: int = 30,
) -> str:
    """Enhance a prompt for the given media type ('image' or 'video').

    Returns the enhanced prompt string.
    """
    # Load assets
    system_template = _load_system_prompt(media_type)
    examples = _load_examples(media_type)

    # Select relevant examples
    selected = _select_examples(user_input, examples)
    examples_text = _format_examples(selected)

    # Inject examples into system prompt
    system_prompt = system_template.replace("{examples}", examples_text)

    # Resolve API config — uses ENHANCE channel first, then IMG as fallback
    base_url, api_key = _resolve_config("ENHANCE")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        "max_tokens": 600,
        "temperature": 0.7,
    }

    data = _api_call(base_url, api_key, payload, timeout=timeout)
    choices = data.get("choices", [])
    if not choices:
        _log("[enhance] No response from LLM, returning original")
        return user_input

    text = choices[0].get("message", {}).get("content", "").strip()

    # Strip wrapping quotes if the model added them
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    return text or user_input


def enhance_image(prompt: str, **kwargs) -> str:
    """Enhance a prompt for image generation."""
    return _enhance(prompt, "image", **kwargs)


def enhance_video(prompt: str, **kwargs) -> str:
    """Enhance a prompt for video generation."""
    return _enhance(prompt, "video", **kwargs)


# ======================================================================
# CLI
# ======================================================================

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="enhance",
        description="Enhance prompts for image or video generation",
    )
    parser.add_argument(
        "-p", "--prompt", required=True, help="User prompt to enhance",
    )
    parser.add_argument(
        "--type", choices=["image", "video"], default="image",
        help="Media type (default: image)",
    )
    parser.add_argument(
        "-m", "--model", default=ENHANCE_MODEL,
        help=f"LLM model for enhancement (default: {ENHANCE_MODEL})",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Output raw text only (no JSON wrapper)",
    )
    args = parser.parse_args(argv)

    _log(f"[enhance] type={args.type} model={args.model}")
    _log(f"[enhance] input: {args.prompt}")

    enhanced = _enhance(args.prompt, args.type, model=args.model)

    _log(f"[enhance] output: {enhanced}")

    if args.raw:
        print(enhanced)
    else:
        result = {
            "type": args.type,
            "input": args.prompt,
            "enhanced": enhanced,
            "model": args.model,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
