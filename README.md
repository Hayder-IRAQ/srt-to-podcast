<div align="center">

# 🎙️ SRT → Podcast

**Convert multilingual SRT subtitles into podcast audio using local GPU TTS**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Chatterbox](https://img.shields.io/badge/TTS-Chatterbox%20Multilingual-purple)](https://github.com/resemble-ai/chatterbox)
[![CUDA](https://img.shields.io/badge/GPU-CUDA-76b900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-green)]()

</div>

---

## What It Does

Takes a multilingual SRT file like this:

```srt
1
00:00:01,000 --> 00:00:05,000
информатика — علم المعلومات — informatics
```

And generates a **podcast audio file** where:
- `информатика` is read in a **Russian voice**
- `علم المعلومات` is read in an **Arabic voice**
- `informatics` is read in an **English voice**

All **automatically**, from a single SRT file, **running locally on your GPU**.

---

## Use Case

Perfect for **language learning content** — convert vocabulary SRT lessons into listenable podcasts. One SRT file → one MP3 with natural multilingual narration.

---

## Features

| Feature | Details |
|---|---|
| 🌍 **Auto language detection** | Arabic (script), Russian (Cyrillic), English (Latin) — per line |
| 🎭 **Voice cloning** | Use your own 5-10s `.wav` reference per language |
| ⚡ **GPU-accelerated** | Chatterbox Multilingual on CUDA |
| 📏 **Unlimited text length** | Smart chunking — sentence → comma → word boundary |
| 🔄 **Retry on GPU OOM** | Progressive chunk halving + GPU cache flush |
| 🎚️ **Emotion control** | Adjustable exaggeration + CFG weight |
| ⏱️ **Custom pauses** | Between lines, blocks, and repetitions |
| 📤 **Export formats** | MP3, WAV, OGG, FLAC |
| 🖥️ **GUI + CLI** | Both interfaces included |
| 🌐 **Trilingual UI** | Arabic, English, Russian interface |

---

## Requirements

| Requirement | Details |
|---|---|
| **Python** | 3.11 only (Chatterbox incompatible with 3.12+) |
| **GPU** | NVIDIA with ~4GB VRAM (RTX 3060+) |
| **CUDA** | Drivers installed |
| **ffmpeg** | Must be in PATH |

---

## Setup

### Windows (automated)
```batch
setup.bat
```

### Manual (all platforms)
```bash
# 1. Create Python 3.11 venv
python3.11 -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 2. Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install ffmpeg
# Windows: winget install ffmpeg
# Linux:   sudo apt install ffmpeg
# macOS:   brew install ffmpeg
```

---

## Usage

### GUI
```bash
python gui.py
```

### CLI

```bash
# Basic
python cli.py sample.srt -o podcast.mp3

# With global voice cloning (5-10s clean .wav)
python cli.py sample.srt -o podcast.mp3 --voice my_voice.wav

# Per-language voice cloning
python cli.py sample.srt -o podcast.mp3 \
    --voice-ar voices/arabic_speaker.wav \
    --voice-ru voices/russian_speaker.wav \
    --voice-en voices/english_speaker.wav

# Adjust emotion + timing
python cli.py sample.srt -o podcast.mp3 \
    --exaggeration 0.7 \
    --pause-blocks 1500 \
    --format wav
```

---

## SRT Format

Standard SRT with multilingual content. Two supported styles:

**Inline separator** (em-dash):
```srt
1
00:00:01,000 --> 00:00:05,000
информатика — علم المعلومات — informatics
```

**Multi-line** (one language per line):
```srt
2
00:00:07,000 --> 00:00:12,000
Я изучаю информатику в университете.
أدرس علم المعلومات في الجامعة.
I study informatics at the university.
```

---

## Project Structure

```
srt-to-podcast/
├── engine.py        # Core pipeline: SRT parsing, TTS, chunking, assembly
├── gui.py           # CustomTkinter GUI (trilingual: AR/EN/RU)
├── cli.py           # Command-line interface
├── i18n.py          # UI translations (Arabic, English, Russian)
├── requirements.txt
├── setup.bat        # Windows one-click setup
├── sample.srt       # Example multilingual SRT
└── README.md
```

---

## Architecture

```
SRT file
  │
  ▼
parse_srt()          — Extract entries with timestamps
  │
  ▼
extract_segments()   — Split by language (inline — or multi-line)
  │
  ▼
detect_language()    — Arabic (Unicode range) / Russian (Cyrillic) / English
  │
  ▼
chunk_text()         — Smart split: sentence → semicolon → comma → word → hard
  │
  ▼
TTSEngine.generate() — Chatterbox Multilingual (CUDA)
  │                    + GPU cache flush every 50 calls
  │                    + Retry with halved chunk on failure
  ▼
assemble_podcast()   — WAV concat with silence padding (ffmpeg)
  │
  ▼
export_final()       — MP3 / WAV / OGG / FLAC
```

---

## Voice Cloning Tips

- Use **5-10 seconds** of clean audio (no background noise)
- Match the language of the reference to the target language
- Higher `exaggeration` (0.7-0.9) = more expressive
- Lower `cfg_weight` (0.3-0.4) = closer to reference voice

---

## 🤝 Contributing

Ideas for future versions:
- Support for `.ass` / `.vtt` subtitle formats
- More languages (French, German, Chinese...)
- Batch folder processing
- Speaker diarization from audio

Fork → feature branch → PR welcome.

---

## 📄 License

MIT — see [LICENSE](LICENSE)

Third-party: Chatterbox TTS ([Apache 2.0](https://github.com/resemble-ai/chatterbox))

---

## 👤 Author

**Hayder Odhafa / حيدر عذافة**
GitHub: [@Hayder-IRAQ](https://github.com/Hayder-IRAQ)

---

<div align="center">

**🎙️ SRT → Podcast — Local GPU TTS for Language Learning**

*© 2025 Hayder Odhafa — MIT License*

</div>
