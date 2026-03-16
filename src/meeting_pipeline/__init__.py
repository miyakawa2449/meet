"""Meeting Speaker Diarization Pipeline."""

import shutil
import subprocess

from .models import (
    AlignConfig,
    AlignedSegment,
    Artifacts,
    ASRConfigInfo,
    ASRResult,
    ASRSegment,
    AudioInfo,
    DeviceInfo,
    DiarizationConfig,
    DiarizationResult,
    InputInfo,
    MeetingJSON,
    PipelineConfig,
    PipelineInfo,
    SegmentSource,
    Speaker,
    SpeakerTurn,
    Timing,
)
from .cli import parse_args, validate_config
from .device import resolve_device
from .audio import extract_audio, _get_audio_duration
from .diarization import run_diarization
from .asr import run_asr, _run_faster_whisper, _run_whisper
from .alignment import (
    align_segments,
    align_segments_word_level,
    _calculate_overlap,
    _get_speaker_label,
    _merge_consecutive_words,
)
from .output import (
    generate_meeting_json,
    save_meeting_json,
    generate_transcript_markdown,
    save_transcript_markdown,
    log_benchmark,
    _dataclass_to_dict,
    _format_timestamp,
)
from .pipeline import run_pipeline

__all__ = [
    # Models
    "AlignConfig",
    "AlignedSegment",
    "Artifacts",
    "ASRConfigInfo",
    "ASRResult",
    "ASRSegment",
    "AudioInfo",
    "DeviceInfo",
    "DiarizationConfig",
    "DiarizationResult",
    "InputInfo",
    "MeetingJSON",
    "PipelineConfig",
    "PipelineInfo",
    "SegmentSource",
    "Speaker",
    "SpeakerTurn",
    "Timing",
    # Functions
    "parse_args",
    "validate_config",
    "resolve_device",
    "extract_audio",
    "run_diarization",
    "run_asr",
    "align_segments",
    "align_segments_word_level",
    "generate_meeting_json",
    "save_meeting_json",
    "generate_transcript_markdown",
    "save_transcript_markdown",
    "log_benchmark",
    "run_pipeline",
    # Internal functions (for testing)
    "_dataclass_to_dict",
    "_format_timestamp",
    "_calculate_overlap",
    "_get_speaker_label",
    "_merge_consecutive_words",
    "_get_audio_duration",
    "_run_faster_whisper",
    "_run_whisper",
    # Modules
    "shutil",
    "subprocess",
]
