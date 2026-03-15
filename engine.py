"""
SRT-to-Podcast Engine — Chatterbox Multilingual (GPU)
=====================================================
Local TTS using Chatterbox Multilingual on CUDA GPU.
Supports Arabic, Russian, English with voice cloning.

Enterprise-grade: handles unlimited text lengths via smart chunking,
GPU memory management, hierarchical WAV assembly, and progressive retry.

Author  : Hayder Odhafa (حيدر عذافة)
GitHub  : https://github.com/Hayder-IRAQ
Version : 1.0.0
License : MIT
"""

from __future__ import annotations

import gc
import logging
import re
import shutil
import subprocess
import tempfile
import unicodedata
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger("srt_podcast")

# Lazy imports for torch/torchaudio — allows GUI to start without them installed
torch = None
ta = None


def _ensure_torch():
    """Lazy-load torch and torchaudio on first actual use."""
    global torch, ta
    if torch is None:
        try:
            import torch as _torch
            import torchaudio as _ta
            torch = _torch
            ta = _ta
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for TTS generation.\n"
                "Install with: pip install torch torchaudio\n"
                "See https://pytorch.org/get-started/locally/"
            ) from e

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Max characters per TTS call. Most neural TTS models degrade above ~300 chars.
# 250 is a safe ceiling for Chatterbox across Arabic/Russian/English.
MAX_CHUNK_CHARS = 250

# After this many generate() calls, flush GPU cache to prevent VRAM exhaustion.
GPU_FLUSH_INTERVAL = 50

# Max WAV files per single ffmpeg concat call. Avoids OS arg limits.
MAX_CONCAT_BATCH = 500

# ─────────────────────────────────────────────
# Language detection
# ─────────────────────────────────────────────

class Language(Enum):
    ARABIC = "ar"
    RUSSIAN = "ru"
    ENGLISH = "en"
    UNKNOWN = "unknown"


_ARABIC_RANGE = set(range(0x0600, 0x06FF + 1)) | set(range(0x0750, 0x077F + 1)) | \
                set(range(0xFB50, 0xFDFF + 1)) | set(range(0xFE70, 0xFEFF + 1)) | \
                set(range(0x0620, 0x064A + 1))
_CYRILLIC_RANGE = set(range(0x0400, 0x04FF + 1)) | set(range(0x0500, 0x052F + 1))


def _script_counts(text: str) -> dict[str, int]:
    counts = {"arabic": 0, "cyrillic": 0, "latin": 0}
    for ch in text:
        cp = ord(ch)
        if cp in _ARABIC_RANGE:
            counts["arabic"] += 1
        elif cp in _CYRILLIC_RANGE:
            counts["cyrillic"] += 1
        elif unicodedata.category(ch).startswith("L") and cp < 0x0250:
            counts["latin"] += 1
    return counts


def detect_language(text: str) -> Language:
    cleaned = re.sub(r"[0-9\s\-—–:;.,!?\"'()\[\]{}]", "", text)
    if not cleaned:
        return Language.UNKNOWN
    counts = _script_counts(cleaned)
    total = sum(counts.values())
    if total == 0:
        return Language.UNKNOWN
    if counts["arabic"] / total > 0.3:
        return Language.ARABIC
    if counts["cyrillic"] / total > 0.3:
        return Language.RUSSIAN
    if counts["latin"] / total > 0.3:
        return Language.ENGLISH
    return Language.UNKNOWN


# ─────────────────────────────────────────────
# Voice configuration
# ─────────────────────────────────────────────

LANG_ID_MAP = {
    Language.ARABIC: "ar",
    Language.RUSSIAN: "ru",
    Language.ENGLISH: "en",
}


@dataclass
class VoiceConfig:
    voice_prompt_path: str | None = None
    voice_prompt_ar: str | None = None
    voice_prompt_ru: str | None = None
    voice_prompt_en: str | None = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.5

    def get_voice_prompt(self, lang: Language) -> str | None:
        per_lang = {
            Language.ARABIC: self.voice_prompt_ar,
            Language.RUSSIAN: self.voice_prompt_ru,
            Language.ENGLISH: self.voice_prompt_en,
        }
        return per_lang.get(lang) or self.voice_prompt_path


# ─────────────────────────────────────────────
# SRT parsing
# ─────────────────────────────────────────────

@dataclass
class SRTEntry:
    index: int
    start_ms: int
    end_ms: int
    lines: list[str]


@dataclass
class TextSegment:
    text: str
    language: Language
    srt_index: int
    start_ms: int
    end_ms: int


def parse_timestamp(ts: str) -> int:
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp: {ts}")
    h, m = int(parts[0]), int(parts[1])
    s_parts = parts[2].split(".")
    s = int(s_parts[0])
    ms_str = s_parts[1] if len(s_parts) > 1 else "0"
    ms = int(ms_str.ljust(3, "0")[:3])
    return h * 3600000 + m * 60000 + s * 1000 + ms


def parse_srt(content: str) -> list[SRTEntry]:
    entries: list[SRTEntry] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        block_lines = block.split("\n")
        if len(block_lines) < 2:
            continue
        idx_line = block_lines[0].strip()
        if not idx_line.isdigit():
            continue
        ts_line = block_lines[1].strip()
        ts_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{1,3})",
            ts_line,
        )
        if not ts_match:
            continue
        start_ms = parse_timestamp(ts_match.group(1))
        end_ms = parse_timestamp(ts_match.group(2))
        text_lines = [
            line.strip()
            for line in block_lines[2:]
            if line.strip() and not re.match(r"^\d+$", line.strip())
        ]
        if text_lines:
            entries.append(SRTEntry(
                index=int(idx_line), start_ms=start_ms,
                end_ms=end_ms, lines=text_lines,
            ))
    return entries


def extract_segments(entries: list[SRTEntry]) -> list[TextSegment]:
    segments: list[TextSegment] = []
    for entry in entries:
        for line in entry.lines:
            parts = re.split(r"\s+—\s+", line)
            if len(parts) > 1:
                for part in parts:
                    part = part.strip()
                    if part:
                        lang = detect_language(part)
                        if lang != Language.UNKNOWN:
                            segments.append(TextSegment(
                                text=part, language=lang,
                                srt_index=entry.index,
                                start_ms=entry.start_ms, end_ms=entry.end_ms,
                            ))
            else:
                lang = detect_language(line)
                if lang != Language.UNKNOWN:
                    segments.append(TextSegment(
                        text=line, language=lang,
                        srt_index=entry.index,
                        start_ms=entry.start_ms, end_ms=entry.end_ms,
                    ))
    return segments


# ─────────────────────────────────────────────
# Smart text chunking for unlimited lengths
# ─────────────────────────────────────────────

# Ranked by boundary strength: strongest first
_SPLIT_PATTERNS = [
    re.compile(r'(?<=[.!?؟。])\s+'),       # Sentence endings
    re.compile(r'(?<=[;:؛])\s+'),           # Semicolons, colons
    re.compile(r'(?<=[,،])\s+'),            # Commas
    re.compile(r'\s+'),                      # Any whitespace (word boundary)
]


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into TTS-safe chunks ≤ max_chars at natural boundaries.

    Priority cascade: sentence end > semicolon > comma > word boundary > hard split.
    Handles Arabic, Russian, English. Never drops text.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    return _split_recursive(text, max_chars)


def _split_recursive(text: str, max_len: int) -> list[str]:
    """Recursively split text at the best natural boundary."""
    if len(text) <= max_len:
        return [text] if text.strip() else []

    for pattern in _SPLIT_PATTERNS:
        result = _greedy_pack(text, pattern, max_len)
        if result is not None:
            return result

    # Nuclear fallback: hard-split (no natural boundary found — e.g. one giant word)
    chunks = []
    for i in range(0, len(text), max_len):
        chunk = text[i:i + max_len].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _greedy_pack(text: str, pattern: re.Pattern, max_len: int) -> list[str] | None:
    """Greedily pack text into chunks ≤ max_len using the given split pattern.

    Returns None if this pattern can't produce a valid first split
    (meaning we should fall through to the next pattern level).
    """
    splits = list(pattern.finditer(text))
    if not splits:
        return None

    chunks: list[str] = []
    start = 0
    last_good = -1  # index into splits[] of last boundary that keeps chunk ≤ max_len

    for idx, match in enumerate(splits):
        candidate_end = match.end()
        candidate = text[start:candidate_end]
        if len(candidate.strip()) <= max_len:
            last_good = idx
        else:
            # This boundary would exceed max_len. Flush at last_good.
            if last_good >= 0:
                flush_end = splits[last_good].end()
                chunk = text[start:flush_end].strip()
                if chunk:
                    chunks.append(chunk)
                start = flush_end
                last_good = -1
                # Re-check current match from new start
                candidate = text[start:candidate_end]
                if len(candidate.strip()) <= max_len:
                    last_good = idx
            else:
                # No good boundary found at this pattern level for the current span.
                # The text from `start` to the first split point is already > max_len.
                # Return None to fall through to the next (finer) pattern.
                if not chunks:
                    return None
                # We have some chunks already; recurse for the remainder
                remainder = text[start:].strip()
                if remainder:
                    chunks.extend(_split_recursive(remainder, max_len))
                return chunks

    # Flush remaining
    if last_good >= 0:
        flush_end = splits[last_good].end()
        chunk = text[start:flush_end].strip()
        if chunk:
            chunks.append(chunk)
        remainder = text[flush_end:].strip()
        if remainder:
            chunks.extend(_split_recursive(remainder, max_len))
    elif start < len(text):
        remainder = text[start:].strip()
        if remainder:
            if chunks:
                chunks.extend(_split_recursive(remainder, max_len))
            else:
                return None  # Fall through to finer pattern

    return chunks if chunks else None


# ─────────────────────────────────────────────
# Audio utilities
# ─────────────────────────────────────────────

def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def generate_silence_wav(duration_ms: int, output_path: Path, sample_rate: int = 24000):
    n_samples = int(sample_rate * duration_ms / 1000)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)


def concatenate_wavs(wav_paths: list[Path], output_path: Path):
    """Concatenate WAVs using hierarchical batching to handle any file count."""
    if not wav_paths:
        return
    if len(wav_paths) == 1:
        shutil.copy2(wav_paths[0], output_path)
        return

    if len(wav_paths) <= MAX_CONCAT_BATCH:
        _concat_batch(wav_paths, output_path)
    else:
        # Hierarchical: batch → merge batches → recurse if needed
        tmp_dir = output_path.parent
        batch_outputs: list[Path] = []
        for batch_idx in range(0, len(wav_paths), MAX_CONCAT_BATCH):
            batch = wav_paths[batch_idx:batch_idx + MAX_CONCAT_BATCH]
            batch_out = tmp_dir / f"_batch_{batch_idx:06d}.wav"
            _concat_batch(batch, batch_out)
            batch_outputs.append(batch_out)

        if len(batch_outputs) == 1:
            shutil.move(str(batch_outputs[0]), str(output_path))
        else:
            concatenate_wavs(batch_outputs, output_path)

        for p in batch_outputs:
            p.unlink(missing_ok=True)


def _concat_batch(wav_paths: list[Path], output_path: Path):
    """Concatenate a single batch of WAVs via ffmpeg concat demuxer."""
    list_file = output_path.parent / f"_cl_{id(wav_paths)}.txt"
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for p in wav_paths:
                safe = str(p.resolve()).replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(list_file), "-c", "copy", str(output_path)],
            capture_output=True, check=True,
        )
    finally:
        list_file.unlink(missing_ok=True)


def export_final(wav_path: Path, output_path: Path, fmt: str, bitrate: str = "192k"):
    if fmt == "wav":
        shutil.copy2(wav_path, output_path)
        return
    cmd = ["ffmpeg", "-y", "-i", str(wav_path)]
    if fmt == "mp3":
        cmd += ["-codec:a", "libmp3lame", "-b:a", bitrate]
    elif fmt == "ogg":
        cmd += ["-codec:a", "libvorbis", "-b:a", bitrate]
    elif fmt == "flac":
        cmd += ["-codec:a", "flac"]
    cmd.append(str(output_path))
    subprocess.run(cmd, capture_output=True, check=True)


def get_wav_duration_ms(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return (wf.getnframes() / wf.getframerate()) * 1000


def resample_wav(input_path: Path, output_path: Path, target_sr: int = 24000):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path),
         "-ar", str(target_sr), "-ac", "1", "-sample_fmt", "s16",
         str(output_path)],
        capture_output=True, check=True,
    )


# ─────────────────────────────────────────────
# GPU memory management
# ─────────────────────────────────────────────

def flush_gpu_cache():
    """Release cached GPU memory to prevent VRAM exhaustion on long jobs."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


# ─────────────────────────────────────────────
# Chatterbox TTS Model Wrapper
# ─────────────────────────────────────────────

class TTSEngine:
    """Wraps Chatterbox Multilingual with chunked generation and memory management."""

    def __init__(self):
        self._model = None
        self._device = None
        self._sample_rate = None
        self._gen_counter = 0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate or 24000

    @property
    def device_name(self) -> str:
        if self._device:
            return str(self._device)
        return "not loaded"

    def load(self, device: str | None = None):
        _ensure_torch()
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        logger.info("Loading Chatterbox Multilingual on %s...", device)
        self._device = device
        self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        self._sample_rate = self._model.sr
        logger.info("Model loaded. Sample rate: %d Hz", self._sample_rate)

        if device == "cuda":
            mem = torch.cuda.memory_allocated() / 1024**2
            logger.info("GPU memory used: %.0f MB", mem)

    def _generate_single(
        self,
        text: str,
        language: Language,
        voice_config: VoiceConfig,
    ) -> torch.Tensor:
        """Generate audio tensor for a single short text chunk."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        lang_id = LANG_ID_MAP.get(language, "en")
        voice_prompt = voice_config.get_voice_prompt(language)

        kwargs = {
            "text": text,
            "language_id": lang_id,
            "exaggeration": voice_config.exaggeration,
            "cfg_weight": voice_config.cfg_weight,
        }
        if voice_prompt:
            kwargs["audio_prompt_path"] = voice_prompt

        wav = self._model.generate(**kwargs)

        self._gen_counter += 1
        if self._gen_counter % GPU_FLUSH_INTERVAL == 0:
            flush_gpu_cache()

        return wav

    def generate(
        self,
        text: str,
        language: Language,
        voice_config: VoiceConfig,
    ) -> torch.Tensor:
        """Generate audio, auto-chunking long text and concatenating results."""
        chunks = chunk_text(text, MAX_CHUNK_CHARS)
        if not chunks:
            raise ValueError("Empty text")

        if len(chunks) == 1:
            return self._generate_single(chunks[0], language, voice_config)

        logger.info("Text chunked into %d parts (%d chars total)",
                     len(chunks), len(text))

        wavs: list[torch.Tensor] = []
        for i, chunk in enumerate(chunks):
            try:
                wav = self._generate_single(chunk, language, voice_config)
                wavs.append(wav)
            except Exception as e:
                logger.warning("Chunk %d/%d failed (%d chars): %s — retrying halved",
                               i + 1, len(chunks), len(chunk), e)
                flush_gpu_cache()
                # Retry: split this chunk further
                sub_chunks = chunk_text(chunk, max(50, MAX_CHUNK_CHARS // 2))
                for j, sc in enumerate(sub_chunks):
                    try:
                        wav = self._generate_single(sc, language, voice_config)
                        wavs.append(wav)
                    except Exception as e2:
                        logger.error("Sub-chunk %d.%d irrecoverably failed: %s", i, j, e2)

        if not wavs:
            raise RuntimeError(f"All chunks failed for text: {text[:80]}...")

        normalized = []
        for w in wavs:
            if w.dim() == 1:
                w = w.unsqueeze(0)
            normalized.append(w)

        return torch.cat(normalized, dim=-1)

    def generate_to_file(
        self,
        text: str,
        language: Language,
        voice_config: VoiceConfig,
        output_path: Path,
    ) -> Path:
        wav = self.generate(text, language, voice_config)
        ta.save(str(output_path), wav, self.sample_rate)
        return output_path


# Global engine instance
_engine = TTSEngine()


def get_engine() -> TTSEngine:
    return _engine


# ─────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────

@dataclass
class PodcastConfig:
    pause_between_lines_ms: int = 600
    pause_between_blocks_ms: int = 1200
    pause_between_repetitions_ms: int = 400
    output_format: str = "mp3"
    output_bitrate: str = "192k"


def assemble_podcast(
    segment_wav_pairs: list[tuple[TextSegment, Path]],
    config: PodcastConfig,
    tmp_dir: Path,
    sample_rate: int = 24000,
) -> Path:
    wav_parts: list[Path] = []
    silence_cache: dict[int, Path] = {}

    def get_silence(ms: int) -> Path:
        if ms not in silence_cache:
            p = tmp_dir / f"silence_{ms}ms.wav"
            generate_silence_wav(ms, p, sample_rate)
            silence_cache[ms] = p
        return silence_cache[ms]

    wav_parts.append(get_silence(500))
    prev_srt_index: int | None = None
    prev_language: Language | None = None

    for seg, wav_path in segment_wav_pairs:
        normalized = wav_path.with_suffix(".norm.wav")
        try:
            resample_wav(wav_path, normalized, sample_rate)
        except subprocess.CalledProcessError:
            normalized = wav_path

        if prev_srt_index is not None:
            if seg.srt_index != prev_srt_index:
                pause_ms = config.pause_between_blocks_ms
            elif seg.language == prev_language:
                pause_ms = config.pause_between_repetitions_ms
            else:
                pause_ms = config.pause_between_lines_ms
            if pause_ms > 0:
                wav_parts.append(get_silence(pause_ms))

        wav_parts.append(normalized)
        prev_srt_index = seg.srt_index
        prev_language = seg.language

    wav_parts.append(get_silence(500))
    combined = tmp_dir / "combined.wav"
    concatenate_wavs(wav_parts, combined)
    return combined


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

@dataclass
class PipelineResult:
    output_path: Path
    total_segments: int
    duration_seconds: float
    languages_used: set[Language]
    entries_count: int


def generate_podcast(
    srt_path: str | Path,
    output_path: str | Path,
    voice_config: VoiceConfig | None = None,
    podcast_config: PodcastConfig | None = None,
    on_progress: Callable[[int, int, TextSegment], None] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> PipelineResult:
    """
    Main pipeline: SRT → multilingual podcast audio (Chatterbox GPU).

    Handles unlimited text lengths via automatic chunking, GPU memory
    management, and hierarchical WAV assembly.
    """
    srt_path = Path(srt_path)
    output_path = Path(output_path)
    voice_config = voice_config or VoiceConfig()
    podcast_config = podcast_config or PodcastConfig()

    if not check_ffmpeg():
        raise RuntimeError("ffmpeg not found in PATH")

    ext = output_path.suffix.lower().lstrip(".")
    if ext in ("mp3", "wav", "ogg", "flac"):
        podcast_config.output_format = ext

    engine = get_engine()

    def status(msg: str):
        logger.info(msg)
        if on_status:
            on_status(msg)

    if not engine.is_loaded:
        status("Loading Chatterbox model (first time takes ~30s)...")
        engine.load()

    status("Parsing SRT...")
    content = srt_path.read_text(encoding="utf-8")
    entries = parse_srt(content)
    if not entries:
        raise ValueError(f"No valid entries in {srt_path}")

    status("Detecting languages...")
    segments = extract_segments(entries)
    if not segments:
        raise ValueError("No valid text segments found")
    langs_found = {s.language for s in segments}

    total_chars = sum(len(s.text) for s in segments)
    status(f"Generating speech for {len(segments)} segments "
           f"({total_chars:,} chars) on {engine.device_name}...")

    with tempfile.TemporaryDirectory(prefix="srt_podcast_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        pairs: list[tuple[TextSegment, Path]] = []
        failed_count = 0

        for i, seg in enumerate(segments):
            wav_path = tmp_path / f"seg_{i:05d}.wav"
            try:
                engine.generate_to_file(seg.text, seg.language, voice_config, wav_path)
            except Exception as e:
                failed_count += 1
                logger.warning("Failed segment %d/%d (%s, %d chars): %s",
                               i + 1, len(segments), seg.text[:30], len(seg.text), e)
                # Aggressive GPU cleanup then retry once
                flush_gpu_cache()
                try:
                    engine.generate_to_file(seg.text, seg.language, voice_config, wav_path)
                    logger.info("Retry succeeded for segment %d", i + 1)
                except Exception as e2:
                    logger.error("Segment %d permanently failed: %s", i + 1, e2)
                    continue

            pairs.append((seg, wav_path))

            if on_progress:
                on_progress(i + 1, len(segments), seg)

            # Periodic GPU flush for very long jobs
            if (i + 1) % GPU_FLUSH_INTERVAL == 0:
                flush_gpu_cache()

        if not pairs:
            raise RuntimeError("All segments failed to generate")

        if failed_count > 0:
            status(f"Assembling podcast... ({failed_count} segments skipped)")
        else:
            status("Assembling podcast...")

        combined_wav = assemble_podcast(pairs, podcast_config, tmp_path, engine.sample_rate)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_final(combined_wav, output_path, podcast_config.output_format, podcast_config.output_bitrate)

        duration_sec = get_wav_duration_ms(combined_wav) / 1000.0
        status(f"Done! Duration: {duration_sec:.1f}s")

        flush_gpu_cache()

        return PipelineResult(
            output_path=output_path,
            total_segments=len(pairs),
            duration_seconds=duration_sec,
            languages_used=langs_found,
            entries_count=len(entries),
        )
