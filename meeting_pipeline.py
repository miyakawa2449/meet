#!/usr/bin/env python3
"""
Meeting Speaker Diarization Pipeline

Processes audio/video files to generate speaker-labeled meeting transcripts.
Combines speaker diarization (pyannote-audio) with ASR (faster-whisper/whisper)
to produce structured meeting logs in JSON and Markdown formats.

Usage:
    python meeting_pipeline.py input.mp4 --enable-diarization --format both
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task 1: Core type definitions
# ---------------------------------------------------------------------------


@dataclass
class AudioInfo:
    path: str
    sample_rate: int
    channels: int
    duration_sec: float = 0.0


@dataclass
class InputInfo:
    path: str
    audio: AudioInfo
    duration_sec: float


@dataclass
class DeviceInfo:
    requested: str
    resolved: str


@dataclass
class DiarizationConfig:
    enabled: bool
    engine: str
    model: str
    hf_token_used: bool


@dataclass
class ASRConfigInfo:
    engine: str
    model: str
    device: str
    compute_type: str
    language: str
    beam_size: int
    best_of: int
    vad_filter: bool


@dataclass
class AlignConfig:
    method: str
    unit: str


@dataclass
class PipelineInfo:
    device: DeviceInfo
    diarization: DiarizationConfig
    asr: ASRConfigInfo
    align: AlignConfig


@dataclass
class Speaker:
    id: str
    label: str


@dataclass
class SegmentSource:
    asr_segment_id: str
    diarization_turn_id: Optional[str]
    overlap_sec: float


@dataclass
class AlignedSegment:
    id: str
    start: float
    end: float
    speaker_id: str
    speaker_label: str
    text: str
    confidence: Optional[float]
    source: SegmentSource


@dataclass
class SpeakerTurn:
    id: str
    speaker_id: str
    start: float
    end: float


@dataclass
class ASRSegment:
    id: str
    start: float
    end: float
    text: str
    words: Optional[List[Dict[str, Any]]] = None


@dataclass
class DiarizationResult:
    turns: List[SpeakerTurn]
    speakers: List[str]
    model: str
    engine: str
    hf_token_used: bool


@dataclass
class ASRResult:
    segments: List[ASRSegment]
    model: str
    engine: str
    device: str
    compute_type: str
    language: str
    beam_size: int
    best_of: int
    vad_filter: bool
    asr_load_sec: float = 0.0


@dataclass
class Artifacts:
    diarization_turns: List[dict]
    asr_segments: List[dict]


@dataclass
class Timing:
    extract_sec: float = 0.0
    diarization_sec: float = 0.0
    asr_load_sec: float = 0.0
    asr_sec: float = 0.0
    align_sec: float = 0.0
    summary_sec: float = 0.0
    total_sec: float = 0.0


@dataclass
class MeetingJSON:
    schema_version: str
    created_at: str
    title: str
    input: InputInfo
    pipeline: PipelineInfo
    speakers: List[Speaker]
    segments: List[AlignedSegment]
    artifacts: Artifacts
    timing: Timing
    notes: str


@dataclass
class PipelineConfig:
    input_file: str
    device: str = "auto"
    enable_diarization: bool = False
    diar_model: str = "pyannote/speaker-diarization-3.1"
    asr_engine: str = "faster-whisper"
    asr_model: str = "medium"
    language: str = "ja"
    beam_size: int = 1
    best_of: int = 1
    vad_filter: bool = False
    output_dir: str = "output"
    temp_dir: str = "temp"
    keep_audio: bool = False
    format: str = "both"
    bench_jsonl: Optional[str] = None
    run_id: Optional[str] = None
    note: Optional[str] = None
    align_unit: str = "segment"


# ---------------------------------------------------------------------------
# Task 2: CLI Parser
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> PipelineConfig:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(
        description="Meeting Speaker Diarization Pipeline: "
        "Generate speaker-labeled meeting transcripts from audio/video files."
    )

    parser.add_argument(
        "input_file",
        help="Input audio or video file (mp4, avi, mkv, wav, m4a, mp3, flac)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--enable-diarization",
        action="store_true",
        help="Enable speaker diarization using pyannote-audio",
    )
    parser.add_argument(
        "--diar-model",
        default="pyannote/speaker-diarization-3.1",
        help="Diarization model name (default: pyannote/speaker-diarization-3.1)",
    )
    parser.add_argument(
        "--asr-engine",
        default="faster-whisper",
        choices=["faster-whisper", "whisper"],
        help="ASR engine (default: faster-whisper)",
    )
    parser.add_argument(
        "--asr-model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="ASR model size (default: medium)",
    )
    parser.add_argument(
        "--language",
        default="ja",
        help="Language code for ASR (default: ja)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Beam size for ASR (default: 1)",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Best-of for ASR (default: 1)",
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="Enable VAD filter for ASR",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--temp-dir",
        default="temp",
        help="Temporary directory (default: temp)",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep extracted audio file after processing",
    )
    parser.add_argument(
        "--format",
        default="both",
        choices=["json", "md", "both"],
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--bench-jsonl",
        default=None,
        help="Path to benchmark JSONL file for logging",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID for benchmark logging",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Note for benchmark logging",
    )
    parser.add_argument(
        "--align-unit",
        default="segment",
        choices=["segment", "word"],
        help="Alignment unit: segment (default) or word-level",
    )

    args = parser.parse_args(argv)

    config = PipelineConfig(
        input_file=args.input_file,
        device=args.device,
        enable_diarization=args.enable_diarization,
        diar_model=args.diar_model,
        asr_engine=args.asr_engine,
        asr_model=args.asr_model,
        language=args.language,
        beam_size=args.beam_size,
        best_of=args.best_of,
        vad_filter=args.vad_filter,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        keep_audio=args.keep_audio,
        format=args.format,
        bench_jsonl=args.bench_jsonl,
        run_id=args.run_id,
        note=args.note,
        align_unit=args.align_unit,
    )

    validate_config(config)
    return config


def validate_config(config: PipelineConfig) -> None:
    """Validate pipeline configuration."""
    # Check input file exists
    if not os.path.exists(config.input_file):
        print(
            f"Error: Input file not found: {config.input_file}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Check output directory can be created
    try:
        os.makedirs(config.output_dir, exist_ok=True)
    except OSError as e:
        print(
            f"Error: Cannot create output directory: {config.output_dir} ({e})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check temp directory can be created
    try:
        os.makedirs(config.temp_dir, exist_ok=True)
    except OSError as e:
        print(
            f"Error: Cannot create temp directory: {config.temp_dir} ({e})",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Task 3: Device Resolver
# ---------------------------------------------------------------------------


def resolve_device(requested: str) -> DeviceInfo:
    """
    Resolve device based on availability.
    Priority: CUDA > MPS > CPU (when requested='auto')
    """
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
    elif requested == "cuda":
        if not torch.cuda.is_available():
            print(
                "Error: CUDA device requested but CUDA is not available",
                file=sys.stderr,
            )
            sys.exit(2)
        resolved = "cuda"
    elif requested == "mps":
        if not torch.backends.mps.is_available():
            print(
                "Error: MPS device requested but MPS is not available",
                file=sys.stderr,
            )
            sys.exit(2)
        resolved = "mps"
    else:
        resolved = "cpu"

    device_info = DeviceInfo(requested=requested, resolved=resolved)
    logger.info("Device: requested=%s, resolved=%s", requested, resolved)
    return device_info


# ---------------------------------------------------------------------------
# Task 4: Audio Extractor
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".wav", ".m4a", ".mp3", ".flac"}


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


# ---------------------------------------------------------------------------
# Task 5: Diarization Engine
# ---------------------------------------------------------------------------


def run_diarization(
    audio_path: str,
    device: str,
    model_name: str,
) -> DiarizationResult:
    """
    Run speaker diarization using pyannote-audio.
    Requires HF_TOKEN in environment.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print(
            "Error: HF_TOKEN environment variable is required for diarization",
            file=sys.stderr,
        )
        print(
            "Get token: https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        sys.exit(2)

    logger.info("Loading diarization model: %s", model_name)

    from pyannote.audio import Pipeline as PyannotePipeline

    pipeline = PyannotePipeline.from_pretrained(
        model_name,
        token=hf_token,
    )

    # Move pipeline to device
    import torch

    if device == "cuda" and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    elif device == "mps" and torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))

    logger.info("Running diarization...")
    diarization = pipeline(audio_path)

    # Build speaker turns with sequential IDs
    turns: List[SpeakerTurn] = []
    speaker_order: List[str] = []  # Track order of appearance
    speaker_map: Dict[str, str] = {}  # original label -> SPEAKER_XX

    turn_counter = 0
    # pyannote-audio 4.0+ API: iterate over output.speaker_diarization
    for turn, speaker_label in diarization.speaker_diarization:
        # Map speaker labels to sequential SPEAKER_XX IDs by order of appearance
        if speaker_label not in speaker_map:
            idx = len(speaker_map)
            mapped_id = f"SPEAKER_{idx:02d}"
            speaker_map[speaker_label] = mapped_id
            speaker_order.append(mapped_id)

        turn_counter += 1
        turns.append(
            SpeakerTurn(
                id=f"turn_{turn_counter:06d}",
                speaker_id=speaker_map[speaker_label],
                start=turn.start,
                end=turn.end,
            )
        )

    logger.info(
        "Diarization complete: %d turns, %d speakers",
        len(turns),
        len(speaker_order),
    )

    # Release model
    del pipeline
    gc.collect()
    import torch as _torch

    if device == "cuda" and _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    logger.info("Diarization model released")

    return DiarizationResult(
        turns=turns,
        speakers=speaker_order,
        model=model_name,
        engine="pyannote-audio",
        hf_token_used=True,
    )


# ---------------------------------------------------------------------------
# Task 6: ASR Engine
# ---------------------------------------------------------------------------


def _determine_compute_type(device: str) -> str:
    """Determine compute type based on device."""
    if device in ("cuda", "mps"):
        return "float16"
    return "int8"


def run_asr(
    audio_path: str,
    device: str,
    config: PipelineConfig,
) -> ASRResult:
    """Run ASR using faster-whisper or whisper."""
    # faster-whisper does not support MPS, fallback to CPU
    asr_device = device
    if config.asr_engine == "faster-whisper" and device == "mps":
        logger.warning("faster-whisper does not support MPS, using CPU instead")
        asr_device = "cpu"
    
    compute_type = _determine_compute_type(asr_device)

    if config.asr_engine == "faster-whisper":
        return _run_faster_whisper(audio_path, asr_device, compute_type, config)
    elif config.asr_engine == "whisper":
        return _run_whisper(audio_path, asr_device, compute_type, config)
    else:
        print(
            f"Error: Unknown ASR engine: {config.asr_engine}",
            file=sys.stderr,
        )
        sys.exit(1)


def _run_faster_whisper(
    audio_path: str,
    device: str,
    compute_type: str,
    config: PipelineConfig,
) -> ASRResult:
    """Run ASR using faster-whisper."""
    from faster_whisper import WhisperModel

    logger.info(
        "Loading faster-whisper model: %s (device=%s, compute_type=%s)",
        config.asr_model,
        device,
        compute_type,
    )

    asr_load_start = time.time()
    model = WhisperModel(
        model_size_or_path=config.asr_model,
        device=device,
        compute_type=compute_type,
    )
    asr_load_sec = time.time() - asr_load_start

    logger.info("Running faster-whisper transcription...")
    segments_iter, info = model.transcribe(
        audio_path,
        language=config.language,
        beam_size=config.beam_size,
        best_of=config.best_of,
        vad_filter=config.vad_filter,
        word_timestamps=True,
    )

    asr_segments: List[ASRSegment] = []
    seg_counter = 0
    for seg in segments_iter:
        seg_counter += 1
        words = None
        if hasattr(seg, "words") and seg.words:
            words = [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                }
                for w in seg.words
            ]
        asr_segments.append(
            ASRSegment(
                id=f"asr_{seg_counter:06d}",
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                words=words,
            )
        )

    logger.info("ASR complete: %d segments", len(asr_segments))

    return ASRResult(
        segments=asr_segments,
        model=config.asr_model,
        engine="faster-whisper",
        device=device,
        compute_type=compute_type,
        language=config.language,
        beam_size=config.beam_size,
        best_of=config.best_of,
        vad_filter=config.vad_filter,
        asr_load_sec=asr_load_sec,
    )


def _run_whisper(
    audio_path: str,
    device: str,
    compute_type: str,
    config: PipelineConfig,
) -> ASRResult:
    """Run ASR using OpenAI whisper."""
    import whisper

    logger.info(
        "Loading whisper model: %s (device=%s)",
        config.asr_model,
        device,
    )

    asr_load_start = time.time()
    model = whisper.load_model(config.asr_model, device=device)
    asr_load_sec = time.time() - asr_load_start

    logger.info("Running whisper transcription...")
    result = model.transcribe(
        audio_path,
        language=config.language,
        beam_size=config.beam_size,
        best_of=config.best_of,
    )

    asr_segments: List[ASRSegment] = []
    seg_counter = 0
    for seg in result.get("segments", []):
        seg_counter += 1
        asr_segments.append(
            ASRSegment(
                id=f"asr_{seg_counter:06d}",
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
            )
        )

    logger.info("ASR complete: %d segments", len(asr_segments))

    return ASRResult(
        segments=asr_segments,
        model=config.asr_model,
        engine="whisper",
        device=device,
        compute_type=compute_type,
        language=config.language,
        beam_size=config.beam_size,
        best_of=config.best_of,
        vad_filter=config.vad_filter,
        asr_load_sec=asr_load_sec,
    )


# ---------------------------------------------------------------------------
# Task 7: Alignment Module
# ---------------------------------------------------------------------------


def _calculate_overlap(
    seg_start: float,
    seg_end: float,
    turn_start: float,
    turn_end: float,
) -> float:
    """Calculate temporal overlap in seconds."""
    overlap_start = max(seg_start, turn_start)
    overlap_end = min(seg_end, turn_end)
    return max(0.0, overlap_end - overlap_start)


def align_segments(
    asr_segments: List[ASRSegment],
    speaker_turns: List[SpeakerTurn],
    speakers: List[str],
    method: str = "max_overlap",
) -> List[AlignedSegment]:
    """
    Align ASR segments with speaker turns using max_overlap method.
    Assigns UNKNOWN when no overlap is found.
    """
    # Build speaker label mapping: SPEAKER_00 -> "Speaker 1", etc.
    speaker_label_map: Dict[str, str] = {}
    for i, spk_id in enumerate(speakers):
        speaker_label_map[spk_id] = f"Speaker {i + 1}"
    speaker_label_map["UNKNOWN"] = "Unknown"

    aligned: List[AlignedSegment] = []
    seg_counter = 0

    for asr_seg in asr_segments:
        max_overlap = 0.0
        best_turn: Optional[SpeakerTurn] = None

        for turn in speaker_turns:
            overlap = _calculate_overlap(
                asr_seg.start,
                asr_seg.end,
                turn.start,
                turn.end,
            )
            if overlap > max_overlap:
                max_overlap = overlap
                best_turn = turn

        if max_overlap > 0 and best_turn is not None:
            speaker_id = best_turn.speaker_id
            turn_id = best_turn.id
        else:
            speaker_id = "UNKNOWN"
            turn_id = None
            logger.warning(
                "No overlap found for segment %s, assigning UNKNOWN",
                asr_seg.id,
            )

        seg_counter += 1
        aligned.append(
            AlignedSegment(
                id=f"seg_{seg_counter:06d}",
                start=asr_seg.start,
                end=asr_seg.end,
                speaker_id=speaker_id,
                speaker_label=speaker_label_map.get(speaker_id, "Unknown"),
                text=asr_seg.text,
                confidence=None,
                source=SegmentSource(
                    asr_segment_id=asr_seg.id,
                    diarization_turn_id=turn_id,
                    overlap_sec=max_overlap,
                ),
            )
        )

    # Sort by start time
    aligned.sort(key=lambda s: s.start)

    logger.info("Alignment complete: %d segments", len(aligned))
    return aligned


# ---------------------------------------------------------------------------
# Phase 3: Word-level Alignment
# ---------------------------------------------------------------------------


def _get_speaker_label(speaker_id: str) -> str:
    """
    Convert speaker_id to human-readable label.

    SPEAKER_00 -> Speaker 1, SPEAKER_01 -> Speaker 2, ..., UNKNOWN -> Unknown
    """
    if speaker_id == "UNKNOWN":
        return "Unknown"
    if speaker_id.startswith("SPEAKER_"):
        try:
            num = int(speaker_id.split("_")[1])
            return f"Speaker {num + 1}"
        except (IndexError, ValueError):
            return speaker_id
    return speaker_id


def _align_single_segment(
    asr_seg: ASRSegment,
    speaker_turns: List[SpeakerTurn],
    seg_counter: int,
) -> AlignedSegment:
    """Align a single ASR segment using segment-level overlap (fallback)."""
    max_overlap = 0.0
    best_turn: Optional[SpeakerTurn] = None

    for turn in speaker_turns:
        overlap = _calculate_overlap(
            asr_seg.start, asr_seg.end, turn.start, turn.end
        )
        if overlap > max_overlap:
            max_overlap = overlap
            best_turn = turn

    if max_overlap > 0 and best_turn is not None:
        speaker_id = best_turn.speaker_id
        turn_id = best_turn.id
    else:
        speaker_id = "UNKNOWN"
        turn_id = None

    return AlignedSegment(
        id=f"seg_{seg_counter:06d}",
        start=asr_seg.start,
        end=asr_seg.end,
        speaker_id=speaker_id,
        speaker_label=_get_speaker_label(speaker_id),
        text=asr_seg.text,
        confidence=None,
        source=SegmentSource(
            asr_segment_id=asr_seg.id,
            diarization_turn_id=turn_id,
            overlap_sec=max_overlap,
        ),
    )


def _merge_consecutive_words(
    word_alignments: List[Dict[str, Any]],
    asr_segment_id: str,
) -> List[Dict[str, Any]]:
    """Merge consecutive words with the same speaker into segments."""
    if not word_alignments:
        return []

    merged: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {
        "speaker_id": word_alignments[0]["speaker_id"],
        "turn_id": word_alignments[0]["turn_id"],
        "start": word_alignments[0]["start"],
        "end": word_alignments[0]["end"],
        "words": [word_alignments[0]["word"]],
        "overlap": word_alignments[0]["overlap"],
    }

    for wa in word_alignments[1:]:
        if wa["speaker_id"] == current["speaker_id"]:
            current["end"] = wa["end"]
            current["words"].append(wa["word"])
            current["overlap"] += wa["overlap"]
        else:
            current["text"] = "".join(current["words"]).strip()
            merged.append(current)
            current = {
                "speaker_id": wa["speaker_id"],
                "turn_id": wa["turn_id"],
                "start": wa["start"],
                "end": wa["end"],
                "words": [wa["word"]],
                "overlap": wa["overlap"],
            }

    current["text"] = "".join(current["words"]).strip()
    merged.append(current)

    return merged


def align_segments_word_level(
    asr_segments: List[ASRSegment],
    speaker_turns: List[SpeakerTurn],
    speakers: List[str],
) -> List[AlignedSegment]:
    """
    Align ASR segments with speaker turns at word level.

    For each word in an ASR segment, calculates overlap with all speaker turns
    and assigns the speaker with maximum overlap. Consecutive words with the
    same speaker are merged into a single AlignedSegment.

    Falls back to segment-level alignment when word timestamps are unavailable.
    """
    # Build speaker label mapping for consistent labels
    speaker_label_map: Dict[str, str] = {}
    for i, spk_id in enumerate(speakers):
        speaker_label_map[spk_id] = f"Speaker {i + 1}"
    speaker_label_map["UNKNOWN"] = "Unknown"

    aligned_segments: List[AlignedSegment] = []
    seg_counter = 0

    for asr_seg in asr_segments:
        # Fallback to segment-level if no word timestamps
        if not asr_seg.words:
            seg_counter += 1
            aligned_segments.append(
                _align_single_segment(asr_seg, speaker_turns, seg_counter)
            )
            continue

        # Word-level alignment
        word_alignments: List[Dict[str, Any]] = []
        for word_info in asr_seg.words:
            word_start = word_info["start"]
            word_end = word_info["end"]
            word_text = word_info["word"]

            max_overlap = 0.0
            best_turn: Optional[SpeakerTurn] = None

            for turn in speaker_turns:
                overlap = _calculate_overlap(
                    word_start, word_end, turn.start, turn.end
                )
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_turn = turn

            if max_overlap > 0 and best_turn is not None:
                speaker_id = best_turn.speaker_id
                turn_id = best_turn.id
            else:
                speaker_id = "UNKNOWN"
                turn_id = None

            word_alignments.append(
                {
                    "word": word_text,
                    "start": word_start,
                    "end": word_end,
                    "speaker_id": speaker_id,
                    "turn_id": turn_id,
                    "overlap": max_overlap,
                }
            )

        # Merge consecutive words with the same speaker
        merged_segments = _merge_consecutive_words(word_alignments, asr_seg.id)

        for merged in merged_segments:
            seg_counter += 1
            aligned_segments.append(
                AlignedSegment(
                    id=f"seg_{seg_counter:06d}",
                    start=merged["start"],
                    end=merged["end"],
                    speaker_id=merged["speaker_id"],
                    speaker_label=speaker_label_map.get(
                        merged["speaker_id"], _get_speaker_label(merged["speaker_id"])
                    ),
                    text=merged["text"],
                    confidence=None,
                    source=SegmentSource(
                        asr_segment_id=asr_seg.id,
                        diarization_turn_id=merged["turn_id"],
                        overlap_sec=merged["overlap"],
                    ),
                )
            )

    # Sort by start time
    aligned_segments.sort(key=lambda s: s.start)

    logger.info(
        "Word-level alignment complete: %d segments", len(aligned_segments)
    )
    return aligned_segments


# ---------------------------------------------------------------------------
# Task 8: JSON Generator
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj: object) -> object:
    """Recursively convert dataclass instances to dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for f_name in obj.__dataclass_fields__:
            value = getattr(obj, f_name)
            result[f_name] = _dataclass_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def generate_meeting_json(
    config: PipelineConfig,
    device_info: DeviceInfo,
    audio_info: AudioInfo,
    diarization_result: Optional[DiarizationResult],
    asr_result: ASRResult,
    aligned_segments: List[AlignedSegment],
    timing: Timing,
) -> MeetingJSON:
    """Generate Meeting JSON from pipeline results."""

    # Build speakers list
    speakers: List[Speaker] = []
    if diarization_result:
        for i, spk_id in enumerate(diarization_result.speakers):
            speakers.append(Speaker(id=spk_id, label=f"Speaker {i + 1}"))
    # Always add UNKNOWN speaker
    speakers.append(Speaker(id="UNKNOWN", label="Unknown"))

    # Build diarization config
    if diarization_result:
        diar_config = DiarizationConfig(
            enabled=True,
            engine=diarization_result.engine,
            model=diarization_result.model,
            hf_token_used=diarization_result.hf_token_used,
        )
    else:
        diar_config = DiarizationConfig(
            enabled=False,
            engine="",
            model="",
            hf_token_used=False,
        )

    # Build ASR config info
    asr_config_info = ASRConfigInfo(
        engine=asr_result.engine,
        model=asr_result.model,
        device=asr_result.device,
        compute_type=asr_result.compute_type,
        language=asr_result.language,
        beam_size=asr_result.beam_size,
        best_of=asr_result.best_of,
        vad_filter=asr_result.vad_filter,
    )

    # Build artifacts
    diar_turns_dicts = []
    if diarization_result:
        for t in diarization_result.turns:
            diar_turns_dicts.append(
                {
                    "id": t.id,
                    "speaker_id": t.speaker_id,
                    "start": t.start,
                    "end": t.end,
                }
            )

    asr_segs_dicts = []
    for s in asr_result.segments:
        asr_segs_dicts.append(
            {
                "id": s.id,
                "start": s.start,
                "end": s.end,
                "text": s.text,
            }
        )

    # Build input info
    input_info = InputInfo(
        path=config.input_file,
        audio=AudioInfo(
            path=audio_info.path,
            sample_rate=audio_info.sample_rate,
            channels=audio_info.channels,
            duration_sec=audio_info.duration_sec,
        ),
        duration_sec=audio_info.duration_sec,
    )

    # Build pipeline info
    pipeline_info = PipelineInfo(
        device=device_info,
        diarization=diar_config,
        asr=asr_config_info,
        align=AlignConfig(method="max_overlap", unit=config.align_unit),
    )

    # ISO 8601 timestamp with timezone
    created_at = datetime.now(timezone.utc).astimezone().isoformat()

    meeting = MeetingJSON(
        schema_version="1.0",
        created_at=created_at,
        title="",
        input=input_info,
        pipeline=pipeline_info,
        speakers=speakers,
        segments=aligned_segments,
        artifacts=Artifacts(
            diarization_turns=diar_turns_dicts,
            asr_segments=asr_segs_dicts,
        ),
        timing=timing,
        notes=config.note or "",
    )

    return meeting


def save_meeting_json(meeting: MeetingJSON, output_path: str) -> None:
    """Serialize and save Meeting JSON to file."""
    meeting_dict = _dataclass_to_dict(meeting)

    try:
        json_str = json.dumps(meeting_dict, ensure_ascii=False, indent=2)
        # Validate round-trip
        json.loads(json_str)
    except (TypeError, ValueError) as e:
        print(
            f"Error: Failed to serialize Meeting JSON: {e}",
            file=sys.stderr,
        )
        sys.exit(4)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_str)

    logger.info("Meeting JSON saved: %s", output_path)


# ---------------------------------------------------------------------------
# Task 9: Main Pipeline Integration
# ---------------------------------------------------------------------------


def run_pipeline(config: PipelineConfig) -> None:
    """Execute the full meeting pipeline."""
    total_start = time.time()
    timing = Timing()
    audio_info: Optional[AudioInfo] = None
    temp_audio_path: Optional[str] = None

    try:
        # --- Stage 1: Device Resolution ---
        device_info = resolve_device(config.device)

        # --- Stage 2: Audio Extraction ---
        logger.info("Stage: Audio extraction")
        t0 = time.time()
        audio_info = extract_audio(
            config.input_file,
            config.temp_dir,
            config.keep_audio,
        )
        timing.extract_sec = round(time.time() - t0, 1)
        temp_audio_path = audio_info.path

        # --- Stage 3: Diarization ---
        diarization_result: Optional[DiarizationResult] = None
        if config.enable_diarization:
            logger.info("Stage: Diarization")
            t0 = time.time()
            try:
                diarization_result = run_diarization(
                    audio_info.path,
                    device_info.resolved,
                    config.diar_model,
                )
            except SystemExit:
                raise
            except Exception as e:
                print(
                    "Error: Diarization failed at stage 'diarization'",
                    file=sys.stderr,
                )
                print(f"Reason: {e}", file=sys.stderr)
                print(
                    f"Audio file preserved at: {audio_info.path}",
                    file=sys.stderr,
                )
                traceback.print_exc()
                sys.exit(3)
            timing.diarization_sec = round(time.time() - t0, 1)
        else:
            logger.info("Diarization disabled, skipping")

        # --- Stage 4: ASR ---
        logger.info("Stage: ASR")
        t0 = time.time()
        try:
            asr_result = run_asr(
                audio_info.path,
                device_info.resolved,
                config,
            )
        except SystemExit:
            raise
        except Exception as e:
            print(
                "Error: ASR failed at stage 'asr'",
                file=sys.stderr,
            )
            print(f"Reason: {e}", file=sys.stderr)
            if audio_info:
                print(
                    f"Audio file preserved at: {audio_info.path}",
                    file=sys.stderr,
                )
            traceback.print_exc()
            sys.exit(3)
        timing.asr_sec = round(time.time() - t0, 1)
        timing.asr_load_sec = round(asr_result.asr_load_sec, 1)

        # --- Stage 5: Alignment ---
        logger.info("Stage: Alignment")
        t0 = time.time()

        speakers_list = diarization_result.speakers if diarization_result else []
        turns_list = diarization_result.turns if diarization_result else []

        if config.align_unit == "word":
            # Check if word timestamps are available
            has_words = any(seg.words for seg in asr_result.segments)
            if not has_words:
                logger.warning(
                    "Word-level alignment requested but no word timestamps available. "
                    "Falling back to segment-level alignment."
                )
                aligned_segments = align_segments(
                    asr_result.segments,
                    turns_list,
                    speakers_list,
                )
            else:
                aligned_segments = align_segments_word_level(
                    asr_result.segments,
                    turns_list,
                    speakers_list,
                )
        else:
            aligned_segments = align_segments(
                asr_result.segments,
                turns_list,
                speakers_list,
            )

        timing.align_sec = round(time.time() - t0, 1)

        # --- Stage 6: Total timing ---
        timing.total_sec = round(time.time() - total_start, 1)

        # --- Stage 7: Generate and save outputs ---
        basename = Path(config.input_file).stem

        meeting = generate_meeting_json(
            config=config,
            device_info=device_info,
            audio_info=audio_info,
            diarization_result=diarization_result,
            asr_result=asr_result,
            aligned_segments=aligned_segments,
            timing=timing,
        )

        os.makedirs(config.output_dir, exist_ok=True)

        if config.format in ("json", "both"):
            json_path = os.path.join(
                config.output_dir,
                f"{basename}_meeting.json",
            )
            save_meeting_json(meeting, json_path)

        if config.format in ("md", "both"):
            md_path = os.path.join(
                config.output_dir,
                f"{basename}_transcript.md",
            )
            md_content = generate_transcript_markdown(meeting)
            save_transcript_markdown(md_content, md_path)

        # --- Benchmark logging (optional) ---
        if config.bench_jsonl:
            log_benchmark(config, device_info, asr_result, timing)

        logger.info(
            "Pipeline complete: total=%.1fs (extract=%.1fs, diar=%.1fs, asr=%.1fs, align=%.1fs)",
            timing.total_sec,
            timing.extract_sec,
            timing.diarization_sec,
            timing.asr_sec,
            timing.align_sec,
        )

    finally:
        # Cleanup temp audio if not keeping
        if (
            temp_audio_path
            and not config.keep_audio
            and os.path.exists(temp_audio_path)
        ):
            os.remove(temp_audio_path)
            logger.info("Temporary audio file removed: %s", temp_audio_path)


# ---------------------------------------------------------------------------
# Markdown Generator (Phase 2 preview - minimal for format='md'/'both')
# ---------------------------------------------------------------------------


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format with zero-padding."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def generate_transcript_markdown(meeting: MeetingJSON) -> str:
    """Generate Markdown transcript from Meeting JSON."""
    lines: List[str] = []
    lines.append("# 会議ログ（話者付き）")
    lines.append("")
    lines.append("## Transcript")
    lines.append("")

    current_speaker: Optional[str] = None

    for seg in meeting.segments:
        # Skip empty text
        if not seg.text or not seg.text.strip():
            continue

        # Insert speaker heading when speaker changes
        if seg.speaker_label != current_speaker:
            current_speaker = seg.speaker_label
            lines.append(f"### {current_speaker}")

        start_ts = _format_timestamp(seg.start)
        end_ts = _format_timestamp(seg.end)
        lines.append(f"- [{start_ts} - {end_ts}] {seg.text}")

    lines.append("")
    return "\n".join(lines)


def save_transcript_markdown(content: str, output_path: str) -> None:
    """Save Markdown content to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Transcript Markdown saved: %s", output_path)


# ---------------------------------------------------------------------------
# Benchmark Logger (optional)
# ---------------------------------------------------------------------------


def log_benchmark(
    config: PipelineConfig,
    device_info: DeviceInfo,
    asr_result: ASRResult,
    timing: Timing,
) -> None:
    """Append benchmark record to JSONL file."""
    run_id = config.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "input_file": config.input_file,
        "device": {
            "requested": device_info.requested,
            "resolved": device_info.resolved,
        },
        "models": {
            "diarization": config.diar_model if config.enable_diarization else None,
            "asr": asr_result.model,
        },
        "timing": _dataclass_to_dict(timing),
        "note": config.note,
    }

    json_line = json.dumps(record, ensure_ascii=False)

    with open(config.bench_jsonl, "a", encoding="utf-8") as f:
        f.write(json_line + "\n")

    logger.info("Benchmark record appended to: %s", config.bench_jsonl)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    config = parse_args()
    run_pipeline(config)


if __name__ == "__main__":
    main()
