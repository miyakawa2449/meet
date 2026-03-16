from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import settings

from meeting_pipeline import (
    ASRResult,
    ASRSegment,
    AudioInfo,
    DeviceInfo,
    DiarizationResult,
    PipelineConfig,
    SpeakerTurn,
    Timing,
)

settings.register_profile("ci", max_examples=100)
settings.load_profile("ci")


@pytest.fixture
def sample_input_file(tmp_path: Path) -> str:
    input_path = tmp_path / "sample.mp4"
    input_path.write_bytes(b"fake-media")
    return str(input_path)


@pytest.fixture
def sample_config(sample_input_file: str, tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        input_file=sample_input_file,
        output_dir=str(tmp_path / "output"),
        temp_dir=str(tmp_path / "temp"),
    )


@pytest.fixture
def sample_device_info() -> DeviceInfo:
    return DeviceInfo(requested="auto", resolved="cpu")


@pytest.fixture
def sample_audio_info(tmp_path: Path) -> AudioInfo:
    audio_path = tmp_path / "temp" / "sample.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"wav")
    return AudioInfo(path=str(audio_path), sample_rate=16000, channels=1, duration_sec=12.5)


@pytest.fixture
def sample_diarization_result() -> DiarizationResult:
    return DiarizationResult(
        turns=[
            SpeakerTurn(id="turn_000001", speaker_id="SPEAKER_00", start=0.0, end=4.0),
            SpeakerTurn(id="turn_000002", speaker_id="SPEAKER_01", start=4.0, end=8.0),
        ],
        speakers=["SPEAKER_00", "SPEAKER_01"],
        model="pyannote/speaker-diarization-3.1",
        engine="pyannote-audio",
        hf_token_used=True,
    )


@pytest.fixture
def sample_asr_result() -> ASRResult:
    return ASRResult(
        segments=[
            ASRSegment(id="asr_000001", start=0.5, end=1.5, text="hello"),
            ASRSegment(id="asr_000002", start=4.5, end=5.5, text="world"),
        ],
        model="medium",
        engine="faster-whisper",
        device="cpu",
        compute_type="int8",
        language="ja",
        beam_size=1,
        best_of=1,
        vad_filter=False,
    )


@pytest.fixture
def sample_timing() -> Timing:
    return Timing(
        extract_sec=0.1,
        diarization_sec=0.2,
        asr_load_sec=0.3,
        asr_sec=0.4,
        align_sec=0.1,
        total_sec=0.9,
    )
