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
    minutes_sec: float = 0.0
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
class MinutesConfig:
    """議事録生成の設定"""

    enabled: bool
    model: str  # "gpt-4" または "gpt-3.5-turbo"
    language: str  # "auto", "ja", "en", など
    temperature: float = 0.3
    max_tokens: int = 4000


@dataclass
class ActionItem:
    """会議から抽出されたアクションアイテム"""

    task: str
    assignee: str  # 話者ラベルまたは名前
    deadline: Optional[str] = None  # ISO日付または自然言語
    timestamp: float = 0.0  # おおよそのタイムスタンプ（秒）


@dataclass
class Decision:
    """会議中に行われた決定"""

    text: str
    speaker: str  # 話者ラベル
    timestamp: float = 0.0  # おおよそのタイムスタンプ（秒）


@dataclass
class Topic:
    """会議で議論されたトピック"""

    title: str
    summary: str
    start: float = 0.0  # 開始タイムスタンプ（秒）
    end: float = 0.0  # 終了タイムスタンプ（秒）


@dataclass
class MeetingMinutes:
    """構造化された議事録"""

    schema_version: str  # "1.0"
    created_at: str  # ISO 8601タイムスタンプ
    meeting_title: str
    meeting_date: str  # ISO日付
    duration_sec: float
    participants: List[str]  # 話者ラベル
    summary: str  # 会議全体の要約
    decisions: List[Decision]
    action_items: List[ActionItem]
    topics: List[Topic]
    model_info: MinutesConfig
    generation_time_sec: float


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
    generate_minutes: bool = False
    minutes_model: str = "gpt-3.5-turbo"
    minutes_language: str = "auto"
