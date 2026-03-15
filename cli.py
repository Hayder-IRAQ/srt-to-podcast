#!/usr/bin/env python3
"""
SRT-to-Podcast — CLI (Chatterbox Multilingual GPU)
"""

import argparse
import logging
import sys
from pathlib import Path

from engine import (
    PodcastConfig,
    TextSegment,
    VoiceConfig,
    check_ffmpeg,
    generate_podcast,
)

logger = logging.getLogger("srt_podcast")


def main():
    p = argparse.ArgumentParser(
        description="Convert multilingual SRT to podcast using Chatterbox (GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py lesson.srt -o podcast.mp3
  python cli.py lesson.srt -o podcast.mp3 --voice voices/my_voice.wav
  python cli.py lesson.srt -o podcast.mp3 --voice-ar voices/arabic.wav --voice-ru voices/russian.wav
  python cli.py lesson.srt -o podcast.mp3 --exaggeration 0.7 --device cuda
        """,
    )
    p.add_argument("input", help="Input SRT file path")
    p.add_argument("-o", "--output", default="podcast.mp3", help="Output audio file path")
    p.add_argument("--format", choices=["mp3", "wav", "ogg", "flac"], default=None)
    p.add_argument("--device", default=None, help="Device: cuda, cpu, mps (auto-detect if omitted)")

    voice = p.add_argument_group("Voice cloning")
    voice.add_argument("--voice", help="Global voice reference .wav (5-10s, clean)")
    voice.add_argument("--voice-ar", help="Arabic voice reference .wav")
    voice.add_argument("--voice-ru", help="Russian voice reference .wav")
    voice.add_argument("--voice-en", help="English voice reference .wav")
    voice.add_argument("--exaggeration", type=float, default=0.5,
                       help="Emotion intensity 0.0-1.0 (default: 0.5)")
    voice.add_argument("--cfg-weight", type=float, default=0.5,
                       help="Voice cloning fidelity 0.0-1.0 (default: 0.5)")

    timing = p.add_argument_group("Timing")
    timing.add_argument("--pause-lines", type=int, default=600)
    timing.add_argument("--pause-blocks", type=int, default=1200)
    timing.add_argument("--pause-reps", type=int, default=400)

    p.add_argument("--bitrate", default="192k")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")

    args = p.parse_args()

    log_level = logging.ERROR if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=log_level, format="%(message)s", stream=sys.stderr)

    if not check_ffmpeg():
        logger.error("ffmpeg not found in PATH!")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    voice_config = VoiceConfig(
        voice_prompt_path=args.voice,
        voice_prompt_ar=args.voice_ar,
        voice_prompt_ru=args.voice_ru,
        voice_prompt_en=args.voice_en,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
    )

    podcast_config = PodcastConfig(
        pause_between_lines_ms=args.pause_lines,
        pause_between_blocks_ms=args.pause_blocks,
        pause_between_repetitions_ms=args.pause_reps,
        output_bitrate=args.bitrate,
    )
    if args.format:
        podcast_config.output_format = args.format

    # Optionally pre-load on specific device
    if args.device:
        from engine import get_engine
        get_engine().load(args.device)

    def on_progress(done: int, total: int, seg: TextSegment):
        emoji = {"ar": "🟣", "ru": "🔵", "en": "🟢"}.get(seg.language.value, "⚪")
        logger.info("  [%d/%d] %s %s: %s", done, total, emoji, seg.language.value.upper(), seg.text[:60])

    result = generate_podcast(
        srt_path=input_path,
        output_path=args.output,
        voice_config=voice_config,
        podcast_config=podcast_config,
        on_progress=on_progress,
    )

    if not args.quiet:
        print(f"\nDone! {result.total_segments} segments -> {result.output_path}", file=sys.stderr)
        print(f"Duration: {result.duration_seconds:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
