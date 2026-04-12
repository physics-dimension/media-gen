#!/usr/bin/env python3
"""Unified CLI entry point for AI media generation.

Routes to 4 atomic functions based on flags:

    python media_gen.py -p "a cute cat"                           # text2img
    python media_gen.py -p "convert to oil painting" --ref img.jpg  # img2img
    python media_gen.py -p "cat running in park" --video            # text2video
    python media_gen.py -p "animate this" --ref img.jpg --video     # img2video

Output: JSON result to stdout, logs to stderr.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Ensure the scripts/ directory is on the path so _core can be imported
# regardless of the working directory.
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _core import (  # noqa: E402
    text2img,
    img2img,
    text2video,
    img2video,
    MediaGenError,
    _log,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="media_gen",
        description=(
            "AI media generation -- text-to-image, image-to-image, "
            "text-to-video, image-to-video"
        ),
    )

    parser.add_argument(
        "-p", "--prompt",
        required=True,
        help="Description / instruction (required)",
    )
    parser.add_argument(
        "--ref",
        default=None,
        metavar="IMAGE",
        help="Reference image path or URL (triggers img2img / img2video)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Video mode (triggers text2video / img2video)",
    )
    parser.add_argument(
        "-m", "--model",
        default="auto",
        help="Model override (default: auto)",
    )
    parser.add_argument(
        "-a", "--aspect-ratio",
        default=None,
        choices=["16:9", "9:16", "1:1", "4:3", "3:4"],
        help="Image aspect ratio (images only)",
    )
    parser.add_argument(
        "--orientation",
        default="landscape",
        choices=["landscape", "portrait"],
        help="Video orientation (default: landscape)",
    )
    parser.add_argument(
        "--quality",
        default="fast",
        choices=["fast", "standard", "lite", "ultra", "4k", "1080p"],
        help="Video quality preset (default: fast)",
    )
    parser.add_argument(
        "-e", "--enhance",
        action="store_true",
        help="LLM prompt enhancement (images only)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "-s", "--stem",
        default=None,
        help="Filename stem",
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── Route to the right atomic function ──
    is_video = args.video
    has_ref = args.ref is not None

    try:
        if is_video and has_ref:
            # img2video
            kwargs: dict = {
                "prompt": args.prompt,
                "reference_image": args.ref,
                "model": args.model,
                "orientation": args.orientation,
                "quality": args.quality,
            }
            if args.output_dir:
                kwargs["output_dir"] = pathlib.Path(args.output_dir)
            if args.stem:
                kwargs["stem"] = args.stem
            if args.timeout is not None:
                kwargs["timeout"] = args.timeout
            result = img2video(**kwargs)

        elif is_video:
            # text2video
            kwargs = {
                "prompt": args.prompt,
                "model": args.model,
                "orientation": args.orientation,
                "quality": args.quality,
            }
            if args.output_dir:
                kwargs["output_dir"] = pathlib.Path(args.output_dir)
            if args.stem:
                kwargs["stem"] = args.stem
            if args.timeout is not None:
                kwargs["timeout"] = args.timeout
            result = text2video(**kwargs)

        elif has_ref:
            # img2img
            kwargs = {
                "prompt": args.prompt,
                "reference_image": args.ref,
                "model": args.model,
            }
            if args.output_dir:
                kwargs["output_dir"] = pathlib.Path(args.output_dir)
            if args.stem:
                kwargs["stem"] = args.stem
            if args.timeout is not None:
                kwargs["timeout"] = args.timeout
            result = img2img(**kwargs)

        else:
            # text2img (default)
            kwargs = {
                "prompt": args.prompt,
                "model": args.model,
                "aspect_ratio": args.aspect_ratio,
                "enhance": args.enhance,
            }
            if args.output_dir:
                kwargs["output_dir"] = pathlib.Path(args.output_dir)
            if args.stem:
                kwargs["stem"] = args.stem
            if args.timeout is not None:
                kwargs["timeout"] = args.timeout
            result = text2img(**kwargs)

        # Output JSON to stdout
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except MediaGenError as exc:
        _log(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
