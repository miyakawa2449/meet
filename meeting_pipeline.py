#!/usr/bin/env python3
"""
Meeting Speaker Diarization Pipeline

Processes audio/video files to generate speaker-labeled meeting transcripts.
Combines speaker diarization (pyannote-audio) with ASR (faster-whisper/whisper)
to produce structured meeting logs in JSON and Markdown formats.

Usage:
    python meeting_pipeline.py input.mp4 --enable-diarization --format both
"""

import logging

from src.meeting_pipeline import parse_args, run_pipeline

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def main() -> None:
    config = parse_args()
    run_pipeline(config)


if __name__ == "__main__":
    main()
