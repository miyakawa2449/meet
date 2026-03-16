"""Audio extraction from video/audio files."""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .models import AudioInfo

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".wav", ".m4a", ".mp3", ".flac"}


def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return 0.0


def extract_audio(
    input_file: str,
    temp_dir: str,
    keep_audio: bool,
) -> AudioInfo:
    """
    Extract audio from input file using ffmpeg.
    Output: 16kHz, mono, pcm_s16le WAV
    """
    # Check ffmpeg availability
    if not shutil.which("ffmpeg"):
        print(
            "Error: ffmpeg is required but not found in PATH",
            file=sys.stderr,
        )
        print(
            "Install: https://ffmpeg.org/download.html",
            file=sys.stderr,
        )
        sys.exit(2)

    # Check supported format
    ext = Path(input_file).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(
            f"Error: Unsupported file format: {ext}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            file=sys.stderr,
        )
        sys.exit(2)

    os.makedirs(temp_dir, exist_ok=True)
    basename = Path(input_file).stem
    output_path = os.path.join(temp_dir, f"{basename}.wav")

    logger.info("Extracting audio: %s -> %s", input_file, output_path)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"Error: ffmpeg failed to extract audio from {input_file}",
                file=sys.stderr,
            )
            print(f"ffmpeg stderr: {result.stderr}", file=sys.stderr)
            sys.exit(3)
    except Exception as e:
        print(
            f"Error: Failed to run ffmpeg: {e}",
            file=sys.stderr,
        )
        sys.exit(3)

    # Get duration using ffprobe
    duration_sec = _get_audio_duration(output_path)

    logger.info("Audio extracted: duration=%.1fs", duration_sec)

    return AudioInfo(
        path=output_path,
        sample_rate=16000,
        channels=1,
        duration_sec=duration_sec,
    )
