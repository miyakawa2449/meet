"""Data models for meeting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
