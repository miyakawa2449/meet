from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from hypothesis import given, strategies as st

from src import meeting_pipeline as mp
from src.meeting_pipeline import (
    ASRResult,
    ASRSegment,
    AlignedSegment,
    AudioInfo,
    DeviceInfo,
    DiarizationResult,
    PipelineConfig,
    SegmentSource,
    SpeakerTurn,
    Timing,
)
from src.meeting_pipeline import cli as cli_module
from src.meeting_pipeline import audio as audio_module
from src.meeting_pipeline import minutes as minutes_module
from src.meeting_pipeline import output as output_module
from src.meeting_pipeline import pipeline as pipeline_module
from src.meeting_pipeline.models import ActionItem, Decision, MeetingMinutes, MinutesConfig, Topic


# ---------------------------------------------------------------------------
# Task 1.1 - Core type property test
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    seg_start=st.floats(min_value=0, max_value=3600, allow_nan=False, allow_infinity=False),
    seg_len=st.floats(min_value=0, max_value=30, allow_nan=False, allow_infinity=False),
    text=st.text(min_size=0, max_size=50),
)
def test_json_serialization_round_trip(seg_start: float, seg_len: float, text: str) -> None:
    """Property 28: JSON Serialization Round-Trip"""
    seg_end = seg_start + seg_len
    meeting = mp.MeetingJSON(
        schema_version="1.0",
        created_at="2026-03-16T00:00:00+00:00",
        title="",
        input=mp.InputInfo(
            path="input.mp4",
            audio=AudioInfo(path="temp/input.wav", sample_rate=16000, channels=1, duration_sec=seg_end),
            duration_sec=seg_end,
        ),
        pipeline=mp.PipelineInfo(
            device=DeviceInfo(requested="auto", resolved="cpu"),
            diarization=mp.DiarizationConfig(enabled=False, engine="", model="", hf_token_used=False),
            asr=mp.ASRConfigInfo(
                engine="faster-whisper",
                model="medium",
                device="cpu",
                compute_type="int8",
                language="ja",
                beam_size=1,
                best_of=1,
                vad_filter=False,
            ),
            align=mp.AlignConfig(method="max_overlap", unit="asr_segment"),
        ),
        speakers=[mp.Speaker(id="UNKNOWN", label="Unknown")],
        segments=[
            AlignedSegment(
                id="seg_000001",
                start=seg_start,
                end=seg_end,
                speaker_id="UNKNOWN",
                speaker_label="Unknown",
                text=text,
                confidence=None,
                source=SegmentSource(asr_segment_id="asr_000001", diarization_turn_id=None, overlap_sec=0.0),
            )
        ],
        artifacts=mp.Artifacts(
            diarization_turns=[],
            asr_segments=[{"id": "asr_000001", "start": seg_start, "end": seg_end, "text": text}],
        ),
        timing=Timing(total_sec=1.0),
        notes="",
    )

    encoded = json.dumps(mp._dataclass_to_dict(meeting), ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded == asdict(meeting)


# ---------------------------------------------------------------------------
# Task 2.3 - CLI parser tests
# ---------------------------------------------------------------------------


def test_cli_parser_missing_input_file() -> None:
    with pytest.raises(SystemExit) as exc_info:
        mp.parse_args([])
    assert exc_info.value.code != 0


def test_cli_parser_invalid_device(sample_input_file: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        mp.parse_args([sample_input_file, "--device", "tpu"])
    assert exc_info.value.code != 0


def test_cli_parser_defaults(sample_input_file: str) -> None:
    config = mp.parse_args([sample_input_file])
    assert config.device == "auto"
    assert config.asr_engine == "faster-whisper"
    assert config.asr_model == "medium"
    assert config.language == "ja"
    assert config.format == "both"


# ---------------------------------------------------------------------------
# Task 3.3, 3.4 - Device resolver tests
# ---------------------------------------------------------------------------


@pytest.mark.property
@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_device_selection_priority(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected: str,
) -> None:
    """Property 4: Device Selection Priority"""
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    resolved = mp.resolve_device("auto")
    assert resolved.resolved == expected


def test_device_resolver_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(SystemExit) as exc_info:
        mp.resolve_device("cuda")
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Task 4.4, 4.5 - Audio extractor tests
# ---------------------------------------------------------------------------


@pytest.mark.property
def test_audio_extraction_format_consistency(
    monkeypatch: pytest.MonkeyPatch,
    sample_input_file: str,
    tmp_path: Path,
) -> None:
    """Property 1: Audio Extraction Format Consistency"""

    def fake_run(cmd: list[str], capture_output: bool, text: bool):
        if cmd[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout="12.34\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(subprocess, "run", fake_run)

    audio = mp.extract_audio(sample_input_file, str(tmp_path / "temp"), keep_audio=False)

    assert audio.sample_rate == 16000
    assert audio.channels == 1
    assert audio.path.endswith(".wav")
    assert audio.duration_sec == pytest.approx(12.34)


def test_audio_extractor_unsupported_format(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    unsupported = tmp_path / "sample.txt"
    unsupported.write_text("x", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")

    with pytest.raises(SystemExit) as exc_info:
        mp.extract_audio(str(unsupported), str(tmp_path), keep_audio=False)
    assert exc_info.value.code == 2


def test_audio_extractor_keep_audio_flag(monkeypatch: pytest.MonkeyPatch, sample_config: PipelineConfig, tmp_path: Path) -> None:
    extracted = tmp_path / "temp" / "sample.wav"
    extracted.parent.mkdir(parents=True, exist_ok=True)

    def fake_extract(input_file: str, temp_dir: str, keep_audio: bool) -> AudioInfo:
        extracted.write_bytes(b"wav")
        return AudioInfo(path=str(extracted), sample_rate=16000, channels=1, duration_sec=1.0)

    monkeypatch.setattr(pipeline_module, "resolve_device", lambda requested: DeviceInfo(requested=requested, resolved="cpu"))
    monkeypatch.setattr(pipeline_module, "extract_audio", fake_extract)
    monkeypatch.setattr(
        pipeline_module,
        "run_asr",
        lambda audio_path, device, config: ASRResult(
            segments=[],
            model="medium",
            engine="faster-whisper",
            device=device,
            compute_type="int8",
            language="ja",
            beam_size=1,
            best_of=1,
            vad_filter=False,
        ),
    )
    monkeypatch.setattr(pipeline_module, "align_segments", lambda asr_segments, speaker_turns, speakers: [])
    monkeypatch.setattr(
        pipeline_module,
        "generate_meeting_json",
        lambda **kwargs: mp.MeetingJSON(
            schema_version="1.0",
            created_at="2026-03-16T00:00:00+00:00",
            title="",
            input=mp.InputInfo(path=sample_config.input_file, audio=kwargs["audio_info"], duration_sec=1.0),
            pipeline=mp.PipelineInfo(
                device=kwargs["device_info"],
                diarization=mp.DiarizationConfig(enabled=False, engine="", model="", hf_token_used=False),
                asr=mp.ASRConfigInfo(
                    engine="faster-whisper",
                    model="medium",
                    device="cpu",
                    compute_type="int8",
                    language="ja",
                    beam_size=1,
                    best_of=1,
                    vad_filter=False,
                ),
                align=mp.AlignConfig(method="max_overlap", unit="asr_segment"),
            ),
            speakers=[mp.Speaker(id="UNKNOWN", label="Unknown")],
            segments=[],
            artifacts=mp.Artifacts(diarization_turns=[], asr_segments=[]),
            timing=kwargs["timing"],
            notes="",
        ),
    )
    monkeypatch.setattr(pipeline_module, "save_meeting_json", lambda meeting, output_path: None)

    keep_cfg = PipelineConfig(**{**sample_config.__dict__, "format": "json", "keep_audio": True})
    mp.run_pipeline(keep_cfg)
    assert extracted.exists()

    delete_cfg = PipelineConfig(**{**sample_config.__dict__, "format": "json", "keep_audio": False})
    mp.run_pipeline(delete_cfg)
    assert not extracted.exists()


# ---------------------------------------------------------------------------
# Task 5.5 - Diarization engine tests
# ---------------------------------------------------------------------------


def test_diarization_missing_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        mp.run_diarization("dummy.wav", "cpu", "pyannote/speaker-diarization-3.1")
    assert exc_info.value.code == 2


def test_diarization_speaker_id_assignment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "token")

    class FakeTurn:
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end

    class FakeDiarization:
        def __init__(self):
            self.speaker_diarization = [
                (FakeTurn(0.0, 1.0), "A"),
                (FakeTurn(1.0, 2.0), "B"),
                (FakeTurn(2.0, 3.0), "A"),
            ]

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_name: str, token: str):
            return cls()

        def to(self, device):
            return None

        def __call__(self, audio_path: str):
            return FakeDiarization()

    pyannote_mod = ModuleType("pyannote")
    pyannote_audio = ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = FakePipeline
    pyannote_mod.audio = pyannote_audio

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        device=lambda name: name,
    )

    monkeypatch.setitem(sys.modules, "pyannote", pyannote_mod)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    result = mp.run_diarization("dummy.wav", "cpu", "model")

    assert result.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert [t.speaker_id for t in result.turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert [t.id for t in result.turns] == ["turn_000001", "turn_000002", "turn_000003"]


# ---------------------------------------------------------------------------
# Task 6.4 - ASR engine tests
# ---------------------------------------------------------------------------


def test_asr_segment_id_generation(monkeypatch: pytest.MonkeyPatch, sample_config: PipelineConfig) -> None:
    class FakeSegment:
        def __init__(self, start: float, end: float, text: str):
            self.start = start
            self.end = end
            self.text = text

    captured: dict[str, object] = {}

    class FakeWhisperModel:
        def __init__(self, model_size_or_path: str, device: str, compute_type: str):
            captured["init"] = {
                "model_size_or_path": model_size_or_path,
                "device": device,
                "compute_type": compute_type,
            }

        def transcribe(self, audio_path: str, language: str, beam_size: int, best_of: int, vad_filter: bool, word_timestamps: bool = False):
            captured["transcribe"] = {
                "audio_path": audio_path,
                "language": language,
                "beam_size": beam_size,
                "best_of": best_of,
                "vad_filter": vad_filter,
            }
            return [FakeSegment(0.0, 1.0, " a "), FakeSegment(1.0, 2.0, " b ")], {"language": "ja"}

    fw_mod = ModuleType("faster_whisper")
    fw_mod.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fw_mod)

    cfg = PipelineConfig(**{**sample_config.__dict__, "asr_engine": "faster-whisper", "beam_size": 3, "best_of": 5, "vad_filter": True})
    result = mp.run_asr("audio.wav", "cpu", cfg)

    assert [s.id for s in result.segments] == ["asr_000001", "asr_000002"]
    assert [s.text for s in result.segments] == ["a", "b"]
    assert captured["init"] == {"model_size_or_path": cfg.asr_model, "device": "cpu", "compute_type": "int8"}
    assert captured["transcribe"] == {
        "audio_path": "audio.wav",
        "language": cfg.language,
        "beam_size": 3,
        "best_of": 5,
        "vad_filter": True,
    }


def test_asr_parameter_application_whisper(monkeypatch: pytest.MonkeyPatch, sample_config: PipelineConfig) -> None:
    captured: dict[str, object] = {}

    class FakeWhisperModel:
        def transcribe(self, audio_path: str, language: str, beam_size: int, best_of: int):
            captured["transcribe"] = {
                "audio_path": audio_path,
                "language": language,
                "beam_size": beam_size,
                "best_of": best_of,
            }
            return {"segments": [{"start": 0.0, "end": 1.0, "text": " hi "}]}

    whisper_mod = ModuleType("whisper")
    whisper_mod.load_model = lambda model, device: FakeWhisperModel()
    monkeypatch.setitem(sys.modules, "whisper", whisper_mod)

    cfg = PipelineConfig(**{**sample_config.__dict__, "asr_engine": "whisper", "beam_size": 2, "best_of": 4})
    result = mp.run_asr("audio.wav", "cpu", cfg)

    assert result.engine == "whisper"
    assert [s.id for s in result.segments] == ["asr_000001"]
    assert captured["transcribe"] == {
        "audio_path": "audio.wav",
        "language": cfg.language,
        "beam_size": 2,
        "best_of": 4,
    }


def test_compute_type_selection_cuda() -> None:
    """Test compute_type is float16 for CUDA."""
    from src.meeting_pipeline.asr import _determine_compute_type

    assert _determine_compute_type("cuda") == "float16"


def test_compute_type_selection_mps() -> None:
    """Test compute_type is float16 for MPS."""
    from src.meeting_pipeline.asr import _determine_compute_type

    assert _determine_compute_type("mps") == "float16"


def test_compute_type_selection_cpu() -> None:
    """Test compute_type is int8 for CPU."""
    from src.meeting_pipeline.asr import _determine_compute_type

    assert _determine_compute_type("cpu") == "int8"


def test_faster_whisper_mps_fallback_to_cpu(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Test that faster-whisper falls back to CPU when MPS is requested."""
    import logging

    from src.meeting_pipeline.asr import run_asr
    from src.meeting_pipeline.models import PipelineConfig

    caplog.set_level(logging.WARNING)

    def mock_run_faster_whisper(audio_path: str, device: str, compute_type: str, config: PipelineConfig) -> ASRResult:
        return ASRResult(
            segments=[],
            model=config.asr_model,
            engine="faster-whisper",
            device=device,
            compute_type=compute_type,
            language=config.language,
            beam_size=config.beam_size,
            best_of=config.best_of,
            vad_filter=config.vad_filter,
            asr_load_sec=0.0,
        )

    monkeypatch.setattr("src.meeting_pipeline.asr._run_faster_whisper", mock_run_faster_whisper)

    config = PipelineConfig(
        input_file="test.mp4",
        device="mps",
        enable_diarization=False,
        diar_model="pyannote/speaker-diarization",
        asr_engine="faster-whisper",
        asr_model="tiny",
        language="ja",
        beam_size=1,
        best_of=1,
        vad_filter=False,
        output_dir="output",
        temp_dir="temp",
        keep_audio=False,
        format="both",
        bench_jsonl=None,
        run_id=None,
        note=None,
        align_unit="asr_segment",
    )

    result = run_asr("dummy_audio.wav", "mps", config)

    assert result.device == "cpu"
    assert result.compute_type == "int8"
    assert "faster-whisper does not support MPS" in caplog.text


# ---------------------------------------------------------------------------
# Task 7.4, 7.5, 7.6 - Alignment tests
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    seg_start=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    seg_len=st.floats(min_value=0, max_value=20, allow_nan=False, allow_infinity=False),
    turn_start=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    turn_len=st.floats(min_value=0, max_value=20, allow_nan=False, allow_infinity=False),
)
def test_alignment_overlap_calculation(seg_start: float, seg_len: float, turn_start: float, turn_len: float) -> None:
    """Property 11: Alignment Overlap Calculation"""
    seg_end = seg_start + seg_len
    turn_end = turn_start + turn_len

    overlap = mp._calculate_overlap(seg_start, seg_end, turn_start, turn_end)
    expected = max(0.0, min(seg_end, turn_end) - max(seg_start, turn_start))
    assert overlap == pytest.approx(expected)


@pytest.mark.property
def test_alignment_unknown_assignment_zero_overlap() -> None:
    """Property 12: UNKNOWN Speaker Assignment"""
    asr_segments = [ASRSegment(id="asr_000001", start=10.0, end=12.0, text="x")]
    turns = [SpeakerTurn(id="turn_000001", speaker_id="SPEAKER_00", start=0.0, end=5.0)]

    aligned = mp.align_segments(asr_segments, turns, ["SPEAKER_00"])

    assert aligned[0].speaker_id == "UNKNOWN"
    assert aligned[0].speaker_label == "Unknown"
    assert aligned[0].source.overlap_sec == 0.0
    assert aligned[0].source.diarization_turn_id is None


def test_alignment_max_overlap_selection() -> None:
    asr_segments = [ASRSegment(id="asr_000001", start=1.0, end=5.0, text="x")]
    turns = [
        SpeakerTurn(id="turn_000001", speaker_id="SPEAKER_00", start=0.0, end=2.0),
        SpeakerTurn(id="turn_000002", speaker_id="SPEAKER_01", start=1.0, end=5.0),
    ]

    aligned = mp.align_segments(asr_segments, turns, ["SPEAKER_00", "SPEAKER_01"])

    assert aligned[0].speaker_id == "SPEAKER_01"
    assert aligned[0].source.diarization_turn_id == "turn_000002"
    assert aligned[0].source.overlap_sec == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Task 8.6, 8.7, 8.8 - JSON generator tests
# ---------------------------------------------------------------------------


@pytest.mark.property
def test_speaker_registry_completeness(
    sample_config: PipelineConfig,
    sample_device_info: DeviceInfo,
    sample_audio_info: AudioInfo,
    sample_diarization_result: DiarizationResult,
    sample_asr_result: ASRResult,
    sample_timing: Timing,
) -> None:
    """Property 14: Speaker Registry Completeness"""
    aligned = mp.align_segments(sample_asr_result.segments, sample_diarization_result.turns, sample_diarization_result.speakers)
    meeting = mp.generate_meeting_json(
        config=sample_config,
        device_info=sample_device_info,
        audio_info=sample_audio_info,
        diarization_result=sample_diarization_result,
        asr_result=sample_asr_result,
        aligned_segments=aligned,
        timing=sample_timing,
    )

    speaker_ids = [s.id for s in meeting.speakers]
    assert "SPEAKER_00" in speaker_ids
    assert "SPEAKER_01" in speaker_ids
    assert "UNKNOWN" in speaker_ids
    assert any(s.label == "Speaker 1" for s in meeting.speakers)


@pytest.mark.property
def test_meeting_json_schema_conformance(
    sample_config: PipelineConfig,
    sample_device_info: DeviceInfo,
    sample_audio_info: AudioInfo,
    sample_diarization_result: DiarizationResult,
    sample_asr_result: ASRResult,
    sample_timing: Timing,
) -> None:
    """Property 15: Meeting JSON Schema Conformance"""
    meeting = mp.generate_meeting_json(
        config=sample_config,
        device_info=sample_device_info,
        audio_info=sample_audio_info,
        diarization_result=sample_diarization_result,
        asr_result=sample_asr_result,
        aligned_segments=mp.align_segments(
            sample_asr_result.segments,
            sample_diarization_result.turns,
            sample_diarization_result.speakers,
        ),
        timing=sample_timing,
    )
    data = mp._dataclass_to_dict(meeting)

    required_top = {
        "schema_version",
        "created_at",
        "title",
        "input",
        "pipeline",
        "speakers",
        "segments",
        "artifacts",
        "timing",
        "notes",
    }
    assert required_top.issubset(data.keys())
    assert data["schema_version"] == "1.0"
    assert isinstance(data["input"], dict)
    assert isinstance(data["pipeline"], dict)
    assert isinstance(data["speakers"], list)
    assert isinstance(data["segments"], list)
    assert isinstance(data["artifacts"], dict)
    assert isinstance(data["timing"], dict)


def test_iso8601_timestamp_format(
    sample_config: PipelineConfig,
    sample_device_info: DeviceInfo,
    sample_audio_info: AudioInfo,
    sample_asr_result: ASRResult,
    sample_timing: Timing,
) -> None:
    meeting = mp.generate_meeting_json(
        config=sample_config,
        device_info=sample_device_info,
        audio_info=sample_audio_info,
        diarization_result=None,
        asr_result=sample_asr_result,
        aligned_segments=[],
        timing=sample_timing,
    )
    # e.g. 2026-03-16T10:22:33.123456+09:00
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?[+-]\d{2}:\d{2}$", meeting.created_at)


def test_json_filename_generation(monkeypatch: pytest.MonkeyPatch, sample_config: PipelineConfig, tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    monkeypatch.setattr(pipeline_module, "resolve_device", lambda requested: DeviceInfo(requested=requested, resolved="cpu"))
    monkeypatch.setattr(
        pipeline_module,
        "extract_audio",
        lambda input_file, temp_dir, keep_audio: AudioInfo(
            path=str(tmp_path / "temp" / "sample.wav"),
            sample_rate=16000,
            channels=1,
            duration_sec=1.0,
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_asr",
        lambda audio_path, device, config: ASRResult(
            segments=[],
            model="medium",
            engine="faster-whisper",
            device=device,
            compute_type="int8",
            language="ja",
            beam_size=1,
            best_of=1,
            vad_filter=False,
        ),
    )
    monkeypatch.setattr(pipeline_module, "align_segments", lambda asr_segments, speaker_turns, speakers: [])
    monkeypatch.setattr(
        pipeline_module,
        "generate_meeting_json",
        lambda **kwargs: mp.MeetingJSON(
            schema_version="1.0",
            created_at="2026-03-16T00:00:00+00:00",
            title="",
            input=mp.InputInfo(path=sample_config.input_file, audio=kwargs["audio_info"], duration_sec=1.0),
            pipeline=mp.PipelineInfo(
                device=kwargs["device_info"],
                diarization=mp.DiarizationConfig(enabled=False, engine="", model="", hf_token_used=False),
                asr=mp.ASRConfigInfo(
                    engine="faster-whisper",
                    model="medium",
                    device="cpu",
                    compute_type="int8",
                    language="ja",
                    beam_size=1,
                    best_of=1,
                    vad_filter=False,
                ),
                align=mp.AlignConfig(method="max_overlap", unit="asr_segment"),
            ),
            speakers=[mp.Speaker(id="UNKNOWN", label="Unknown")],
            segments=[],
            artifacts=mp.Artifacts(diarization_turns=[], asr_segments=[]),
            timing=kwargs["timing"],
            notes="",
        ),
    )

    def fake_save_json(meeting, output_path: str) -> None:
        captured["json_path"] = output_path

    monkeypatch.setattr(pipeline_module, "save_meeting_json", fake_save_json)

    cfg = PipelineConfig(**{**sample_config.__dict__, "format": "json", "output_dir": str(tmp_path / "out")})
    mp.run_pipeline(cfg)

    assert captured["json_path"].endswith("sample_meeting.json")


def test_save_meeting_json_validation_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Bad:
        pass

    monkeypatch.setattr(output_module, "_dataclass_to_dict", lambda obj: {"x": Bad()})

    with pytest.raises(SystemExit) as exc_info:
        dummy = mp.MeetingJSON(
            schema_version="1.0",
            created_at="2026-03-16T00:00:00+00:00",
            title="",
            input=mp.InputInfo(path="in.mp4", audio=AudioInfo(path="a.wav", sample_rate=16000, channels=1), duration_sec=0.0),
            pipeline=mp.PipelineInfo(
                device=DeviceInfo(requested="cpu", resolved="cpu"),
                diarization=mp.DiarizationConfig(enabled=False, engine="", model="", hf_token_used=False),
                asr=mp.ASRConfigInfo(
                    engine="faster-whisper",
                    model="tiny",
                    device="cpu",
                    compute_type="int8",
                    language="ja",
                    beam_size=1,
                    best_of=1,
                    vad_filter=False,
                ),
                align=mp.AlignConfig(method="max_overlap", unit="segment"),
            ),
            speakers=[mp.Speaker(id="UNKNOWN", label="Unknown")],
            segments=[],
            artifacts=mp.Artifacts(diarization_turns=[], asr_segments=[]),
            timing=Timing(),
            notes="",
        )
        mp.save_meeting_json(dummy, str(tmp_path / "x.json"))
    assert exc_info.value.code == 4


# ---------------------------------------------------------------------------
# Task 9.4 - Main pipeline tests
# ---------------------------------------------------------------------------


def test_pipeline_error_handling(monkeypatch: pytest.MonkeyPatch, sample_config: PipelineConfig, tmp_path: Path) -> None:
    extracted = tmp_path / "temp" / "sample.wav"
    extracted.parent.mkdir(parents=True, exist_ok=True)
    extracted.write_bytes(b"wav")

    monkeypatch.setattr(pipeline_module, "resolve_device", lambda requested: DeviceInfo(requested=requested, resolved="cpu"))
    monkeypatch.setattr(
        pipeline_module,
        "extract_audio",
        lambda input_file, temp_dir, keep_audio: AudioInfo(
            path=str(extracted), sample_rate=16000, channels=1, duration_sec=1.0
        ),
    )
    monkeypatch.setattr(pipeline_module, "run_asr", lambda audio_path, device, config: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(SystemExit) as exc_info:
        mp.run_pipeline(PipelineConfig(**{**sample_config.__dict__, "format": "json"}))

    assert exc_info.value.code == 3
    # finally block should clean temp audio when keep_audio=False
    assert not extracted.exists()


def test_pipeline_resource_cleanup(monkeypatch: pytest.MonkeyPatch, sample_config: PipelineConfig, tmp_path: Path) -> None:
    calls: list[str] = []
    extracted = tmp_path / "temp" / "sample.wav"
    extracted.parent.mkdir(parents=True, exist_ok=True)

    def fake_extract(input_file: str, temp_dir: str, keep_audio: bool) -> AudioInfo:
        extracted.write_bytes(b"wav")
        calls.append("extract")
        return AudioInfo(path=str(extracted), sample_rate=16000, channels=1, duration_sec=1.0)

    def fake_diar(audio_path: str, device: str, model_name: str) -> DiarizationResult:
        calls.append("diar")
        return DiarizationResult(
            turns=[SpeakerTurn(id="turn_000001", speaker_id="SPEAKER_00", start=0.0, end=1.0)],
            speakers=["SPEAKER_00"],
            model=model_name,
            engine="pyannote-audio",
            hf_token_used=True,
        )

    def fake_asr(audio_path: str, device: str, config: PipelineConfig) -> ASRResult:
        calls.append("asr")
        return ASRResult(
            segments=[ASRSegment(id="asr_000001", start=0.0, end=1.0, text="x")],
            model="medium",
            engine="faster-whisper",
            device=device,
            compute_type="int8",
            language="ja",
            beam_size=1,
            best_of=1,
            vad_filter=False,
        )

    monkeypatch.setattr(pipeline_module, "resolve_device", lambda requested: DeviceInfo(requested=requested, resolved="cpu"))
    monkeypatch.setattr(pipeline_module, "extract_audio", fake_extract)
    monkeypatch.setattr(pipeline_module, "run_diarization", fake_diar)
    monkeypatch.setattr(pipeline_module, "run_asr", fake_asr)
    monkeypatch.setattr(pipeline_module, "save_meeting_json", lambda meeting, output_path: None)

    cfg = PipelineConfig(**{**sample_config.__dict__, "enable_diarization": True, "format": "json"})
    mp.run_pipeline(cfg)

    assert calls.index("diar") < calls.index("asr")
    assert not extracted.exists()


# ---------------------------------------------------------------------------
# Task 11.5, 11.6, 11.7 - Markdown tests
# ---------------------------------------------------------------------------


@pytest.mark.property
def test_markdown_speaker_grouping(sample_config: PipelineConfig) -> None:
    """Property 20: Markdown Speaker Grouping"""
    meeting = mp.MeetingJSON(
        schema_version="1.0",
        created_at="2026-03-16T00:00:00+00:00",
        title="",
        input=mp.InputInfo(path=sample_config.input_file, audio=AudioInfo(path="a.wav", sample_rate=16000, channels=1), duration_sec=10.0),
        pipeline=mp.PipelineInfo(
            device=DeviceInfo(requested="auto", resolved="cpu"),
            diarization=mp.DiarizationConfig(enabled=False, engine="", model="", hf_token_used=False),
            asr=mp.ASRConfigInfo(engine="faster-whisper", model="medium", device="cpu", compute_type="int8", language="ja", beam_size=1, best_of=1, vad_filter=False),
            align=mp.AlignConfig(method="max_overlap", unit="asr_segment"),
        ),
        speakers=[mp.Speaker(id="SPEAKER_00", label="Speaker 1"), mp.Speaker(id="SPEAKER_01", label="Speaker 2"), mp.Speaker(id="UNKNOWN", label="Unknown")],
        segments=[
            AlignedSegment(
                id="seg_000001",
                start=0.0,
                end=1.0,
                speaker_id="SPEAKER_00",
                speaker_label="Speaker 1",
                text="a",
                confidence=None,
                source=SegmentSource(asr_segment_id="asr_000001", diarization_turn_id="turn_000001", overlap_sec=1.0),
            ),
            AlignedSegment(
                id="seg_000002",
                start=1.0,
                end=2.0,
                speaker_id="SPEAKER_00",
                speaker_label="Speaker 1",
                text="b",
                confidence=None,
                source=SegmentSource(asr_segment_id="asr_000002", diarization_turn_id="turn_000001", overlap_sec=1.0),
            ),
            AlignedSegment(
                id="seg_000003",
                start=2.0,
                end=3.0,
                speaker_id="SPEAKER_01",
                speaker_label="Speaker 2",
                text="c",
                confidence=None,
                source=SegmentSource(asr_segment_id="asr_000003", diarization_turn_id="turn_000002", overlap_sec=1.0),
            ),
        ],
        artifacts=mp.Artifacts(diarization_turns=[], asr_segments=[]),
        timing=Timing(),
        notes="",
    )

    md = mp.generate_transcript_markdown(meeting)
    assert md.count("### Speaker 1") == 1
    assert md.count("### Speaker 2") == 1


@pytest.mark.property
def test_markdown_segment_formatting(sample_config: PipelineConfig) -> None:
    """Property 21: Markdown Segment Formatting"""
    meeting = mp.MeetingJSON(
        schema_version="1.0",
        created_at="2026-03-16T00:00:00+00:00",
        title="",
        input=mp.InputInfo(path=sample_config.input_file, audio=AudioInfo(path="a.wav", sample_rate=16000, channels=1), duration_sec=10.0),
        pipeline=mp.PipelineInfo(
            device=DeviceInfo(requested="auto", resolved="cpu"),
            diarization=mp.DiarizationConfig(enabled=False, engine="", model="", hf_token_used=False),
            asr=mp.ASRConfigInfo(engine="faster-whisper", model="medium", device="cpu", compute_type="int8", language="ja", beam_size=1, best_of=1, vad_filter=False),
            align=mp.AlignConfig(method="max_overlap", unit="asr_segment"),
        ),
        speakers=[mp.Speaker(id="UNKNOWN", label="Unknown")],
        segments=[
            AlignedSegment(
                id="seg_000001",
                start=1.0,
                end=2.0,
                speaker_id="UNKNOWN",
                speaker_label="Unknown",
                text="hello",
                confidence=None,
                source=SegmentSource(asr_segment_id="asr_000001", diarization_turn_id=None, overlap_sec=0.0),
            ),
            AlignedSegment(
                id="seg_000002",
                start=3.0,
                end=4.0,
                speaker_id="UNKNOWN",
                speaker_label="Unknown",
                text="   ",
                confidence=None,
                source=SegmentSource(asr_segment_id="asr_000002", diarization_turn_id=None, overlap_sec=0.0),
            ),
        ],
        artifacts=mp.Artifacts(diarization_turns=[], asr_segments=[]),
        timing=Timing(),
        notes="",
    )

    md = mp.generate_transcript_markdown(meeting)
    assert "- [00:00:01 - 00:00:02] hello" in md
    assert "00:00:03" not in md


def test_markdown_timestamp_format() -> None:
    assert mp._format_timestamp(0) == "00:00:00"
    assert mp._format_timestamp(61) == "00:01:01"
    assert mp._format_timestamp(3661) == "01:01:01"


def test_markdown_empty_text_skip(sample_config: PipelineConfig) -> None:
    meeting = mp.MeetingJSON(
        schema_version="1.0",
        created_at="2026-03-16T00:00:00+00:00",
        title="",
        input=mp.InputInfo(path=sample_config.input_file, audio=AudioInfo(path="a.wav", sample_rate=16000, channels=1), duration_sec=10.0),
        pipeline=mp.PipelineInfo(
            device=DeviceInfo(requested="auto", resolved="cpu"),
            diarization=mp.DiarizationConfig(enabled=False, engine="", model="", hf_token_used=False),
            asr=mp.ASRConfigInfo(engine="faster-whisper", model="medium", device="cpu", compute_type="int8", language="ja", beam_size=1, best_of=1, vad_filter=False),
            align=mp.AlignConfig(method="max_overlap", unit="asr_segment"),
        ),
        speakers=[mp.Speaker(id="UNKNOWN", label="Unknown")],
        segments=[
            AlignedSegment(
                id="seg_000001",
                start=0.0,
                end=1.0,
                speaker_id="UNKNOWN",
                speaker_label="Unknown",
                text=" ",
                confidence=None,
                source=SegmentSource(asr_segment_id="asr_000001", diarization_turn_id=None, overlap_sec=0.0),
            )
        ],
        artifacts=mp.Artifacts(diarization_turns=[], asr_segments=[]),
        timing=Timing(),
        notes="",
    )

    md = mp.generate_transcript_markdown(meeting)
    assert "- [" not in md


def test_meeting_json_schema_structure() -> None:
    """Test Meeting JSON has all required fields regardless of device."""
    from datetime import datetime

    from src.meeting_pipeline.models import (
        ASRConfigInfo,
        AlignConfig,
        Artifacts,
        AudioInfo,
        DeviceInfo,
        DiarizationConfig,
        InputInfo,
        MeetingJSON,
        PipelineInfo,
        Speaker,
        Timing,
    )
    from src.meeting_pipeline.output import _dataclass_to_dict

    meeting = MeetingJSON(
        schema_version="1.0",
        created_at=datetime.now().isoformat(),
        title="",
        input=InputInfo(
            path="test.mp4",
            audio=AudioInfo(path="test.wav", sample_rate=16000, channels=1),
            duration_sec=100.0,
        ),
        pipeline=PipelineInfo(
            device=DeviceInfo(requested="auto", resolved="cpu"),
            diarization=DiarizationConfig(
                enabled=False,
                engine="",
                model="",
                hf_token_used=False,
            ),
            asr=ASRConfigInfo(
                engine="faster-whisper",
                model="tiny",
                device="cpu",
                compute_type="int8",
                language="ja",
                beam_size=1,
                best_of=1,
                vad_filter=False,
            ),
            align=AlignConfig(method="max_overlap", unit="asr_segment"),
        ),
        speakers=[Speaker(id="UNKNOWN", label="Unknown")],
        segments=[],
        artifacts=Artifacts(diarization_turns=[], asr_segments=[]),
        timing=Timing(
            extract_sec=0.0,
            diarization_sec=0.0,
            asr_load_sec=0.0,
            asr_sec=0.0,
            align_sec=0.0,
            summary_sec=0.0,
            total_sec=0.0,
        ),
        notes="",
    )

    meeting_dict = _dataclass_to_dict(meeting)

    required_fields = [
        "schema_version",
        "created_at",
        "title",
        "input",
        "pipeline",
        "speakers",
        "segments",
        "artifacts",
        "timing",
        "notes",
    ]
    for field in required_fields:
        assert field in meeting_dict, f"Missing required field: {field}"

    assert meeting_dict["schema_version"] == "1.0"
    assert "device" in meeting_dict["pipeline"]
    assert "diarization" in meeting_dict["pipeline"]
    assert "asr" in meeting_dict["pipeline"]
    assert "align" in meeting_dict["pipeline"]
    assert "requested" in meeting_dict["pipeline"]["device"]
    assert "resolved" in meeting_dict["pipeline"]["device"]


# ---------------------------------------------------------------------------
# Task 12.2 - Output format control tests
# ---------------------------------------------------------------------------


def _run_pipeline_with_format(monkeypatch: pytest.MonkeyPatch, sample_config: PipelineConfig, tmp_path: Path, fmt: str) -> tuple[int, int]:
    json_calls = {"n": 0}
    md_calls = {"n": 0}
    extracted = tmp_path / "temp" / "sample.wav"

    def fake_extract(input_file: str, temp_dir: str, keep_audio: bool) -> AudioInfo:
        extracted.parent.mkdir(parents=True, exist_ok=True)
        extracted.write_bytes(b"wav")
        return AudioInfo(path=str(extracted), sample_rate=16000, channels=1, duration_sec=1.0)

    monkeypatch.setattr(pipeline_module, "resolve_device", lambda requested: DeviceInfo(requested=requested, resolved="cpu"))
    monkeypatch.setattr(pipeline_module, "extract_audio", fake_extract)
    monkeypatch.setattr(
        pipeline_module,
        "run_asr",
        lambda audio_path, device, config: ASRResult(
            segments=[],
            model="medium",
            engine="faster-whisper",
            device=device,
            compute_type="int8",
            language="ja",
            beam_size=1,
            best_of=1,
            vad_filter=False,
        ),
    )
    monkeypatch.setattr(pipeline_module, "align_segments", lambda asr_segments, speaker_turns, speakers: [])
    monkeypatch.setattr(
        pipeline_module,
        "generate_meeting_json",
        lambda **kwargs: mp.MeetingJSON(
            schema_version="1.0",
            created_at="2026-03-16T00:00:00+00:00",
            title="",
            input=mp.InputInfo(path=sample_config.input_file, audio=kwargs["audio_info"], duration_sec=1.0),
            pipeline=mp.PipelineInfo(
                device=kwargs["device_info"],
                diarization=mp.DiarizationConfig(enabled=False, engine="", model="", hf_token_used=False),
                asr=mp.ASRConfigInfo(
                    engine="faster-whisper",
                    model="medium",
                    device="cpu",
                    compute_type="int8",
                    language="ja",
                    beam_size=1,
                    best_of=1,
                    vad_filter=False,
                ),
                align=mp.AlignConfig(method="max_overlap", unit="asr_segment"),
            ),
            speakers=[mp.Speaker(id="UNKNOWN", label="Unknown")],
            segments=[],
            artifacts=mp.Artifacts(diarization_turns=[], asr_segments=[]),
            timing=kwargs["timing"],
            notes="",
        ),
    )
    monkeypatch.setattr(pipeline_module, "save_meeting_json", lambda meeting, output_path: json_calls.__setitem__("n", json_calls["n"] + 1))
    monkeypatch.setattr(pipeline_module, "save_transcript_markdown", lambda content, output_path: md_calls.__setitem__("n", md_calls["n"] + 1))

    cfg = PipelineConfig(**{**sample_config.__dict__, "format": fmt, "output_dir": str(tmp_path / "out")})
    mp.run_pipeline(cfg)
    return json_calls["n"], md_calls["n"]


def test_output_format_json_only(monkeypatch: pytest.MonkeyPatch, sample_config: PipelineConfig, tmp_path: Path) -> None:
    json_n, md_n = _run_pipeline_with_format(monkeypatch, sample_config, tmp_path, "json")
    assert json_n == 1
    assert md_n == 0


def test_output_format_md_only(monkeypatch: pytest.MonkeyPatch, sample_config: PipelineConfig, tmp_path: Path) -> None:
    json_n, md_n = _run_pipeline_with_format(monkeypatch, sample_config, tmp_path, "md")
    assert json_n == 0
    assert md_n == 1


def test_output_format_both(monkeypatch: pytest.MonkeyPatch, sample_config: PipelineConfig, tmp_path: Path) -> None:
    json_n, md_n = _run_pipeline_with_format(monkeypatch, sample_config, tmp_path, "both")
    assert json_n == 1
    assert md_n == 1


# ---------------------------------------------------------------------------
# Task 13.3 - Benchmark logger tests
# ---------------------------------------------------------------------------


def test_benchmark_run_id_generation(sample_config: PipelineConfig, sample_device_info: DeviceInfo, sample_asr_result: ASRResult, tmp_path: Path) -> None:
    bench = tmp_path / "bench.jsonl"
    cfg = PipelineConfig(**{**sample_config.__dict__, "bench_jsonl": str(bench), "run_id": None})

    mp.log_benchmark(cfg, sample_device_info, sample_asr_result, Timing(total_sec=1.0))

    line = bench.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert re.match(r"^\d{8}_\d{6}$", record["run_id"])


def test_benchmark_jsonl_append(sample_config: PipelineConfig, sample_device_info: DeviceInfo, sample_asr_result: ASRResult, tmp_path: Path) -> None:
    bench = tmp_path / "bench.jsonl"
    cfg = PipelineConfig(**{**sample_config.__dict__, "bench_jsonl": str(bench), "run_id": "run-a", "note": "memo"})

    mp.log_benchmark(cfg, sample_device_info, sample_asr_result, Timing(total_sec=1.0))
    cfg2 = PipelineConfig(**{**sample_config.__dict__, "bench_jsonl": str(bench), "run_id": "run-b"})
    mp.log_benchmark(cfg2, sample_device_info, sample_asr_result, Timing(total_sec=2.0))

    lines = [json.loads(l) for l in bench.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0]["run_id"] == "run-a"
    assert lines[0]["note"] == "memo"
    assert lines[1]["run_id"] == "run-b"


# ---------------------------------------------------------------------------
# Additional verification: required issues
# ---------------------------------------------------------------------------


def test_type_hints_consistency_for_target_functions() -> None:
    target = [
        mp._dataclass_to_dict,
        mp._get_audio_duration,
        mp._calculate_overlap,
        mp._format_timestamp,
    ]
    for fn in target:
        assert "return" in fn.__annotations__
        for name in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
            assert name in fn.__annotations__


def test_asr_load_sec_should_be_recorded_in_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: PipelineConfig,
    tmp_path: Path,
) -> None:
    captured_timing: dict[str, Timing] = {}

    extracted = tmp_path / "temp" / "sample.wav"

    def fake_extract(input_file: str, temp_dir: str, keep_audio: bool) -> AudioInfo:
        extracted.parent.mkdir(parents=True, exist_ok=True)
        extracted.write_bytes(b"wav")
        return AudioInfo(path=str(extracted), sample_rate=16000, channels=1, duration_sec=1.0)

    monkeypatch.setattr(pipeline_module, "resolve_device", lambda requested: DeviceInfo(requested=requested, resolved="cpu"))
    monkeypatch.setattr(pipeline_module, "extract_audio", fake_extract)
    monkeypatch.setattr(
        pipeline_module,
        "run_asr",
        lambda audio_path, device, config: ASRResult(
            segments=[],
            model="medium",
            engine="faster-whisper",
            device=device,
            compute_type="int8",
            language="ja",
            beam_size=1,
            best_of=1,
            vad_filter=False,
            asr_load_sec=1.26,
        ),
    )
    monkeypatch.setattr(pipeline_module, "align_segments", lambda asr_segments, speaker_turns, speakers: [])

    def fake_generate(**kwargs):
        captured_timing["timing"] = kwargs["timing"]
        return mp.MeetingJSON(
            schema_version="1.0",
            created_at="2026-03-16T00:00:00+00:00",
            title="",
            input=mp.InputInfo(path=sample_config.input_file, audio=kwargs["audio_info"], duration_sec=1.0),
            pipeline=mp.PipelineInfo(
                device=kwargs["device_info"],
                diarization=mp.DiarizationConfig(enabled=False, engine="", model="", hf_token_used=False),
                asr=mp.ASRConfigInfo(
                    engine="faster-whisper",
                    model="medium",
                    device="cpu",
                    compute_type="int8",
                    language="ja",
                    beam_size=1,
                    best_of=1,
                    vad_filter=False,
                ),
                align=mp.AlignConfig(method="max_overlap", unit="asr_segment"),
            ),
            speakers=[mp.Speaker(id="UNKNOWN", label="Unknown")],
            segments=[],
            artifacts=mp.Artifacts(diarization_turns=[], asr_segments=[]),
            timing=kwargs["timing"],
            notes="",
        )

    monkeypatch.setattr(pipeline_module, "generate_meeting_json", fake_generate)
    monkeypatch.setattr(pipeline_module, "save_meeting_json", lambda meeting, output_path: None)

    cfg = PipelineConfig(**{**sample_config.__dict__, "format": "json"})
    mp.run_pipeline(cfg)

    assert captured_timing["timing"].asr_load_sec == 1.3


def test_edge_case_no_diarization_assigns_unknown(sample_asr_result: ASRResult) -> None:
    aligned = mp.align_segments(sample_asr_result.segments, speaker_turns=[], speakers=[])
    assert all(seg.speaker_id == "UNKNOWN" for seg in aligned)


def test_edge_case_empty_asr_segments() -> None:
    aligned = mp.align_segments([], speaker_turns=[], speakers=[])
    assert aligned == []


# ---------------------------------------------------------------------------
# Phase 3 tests - Word-level alignment
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_word_timestamps_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that word timestamps are extracted from faster-whisper.
    Validates: Phase 3, Task 15.1
    """

    class FakeWord:
        def __init__(self, word: str, start: float, end: float, probability: float) -> None:
            self.word = word
            self.start = start
            self.end = end
            self.probability = probability

    class FakeSegment:
        def __init__(self) -> None:
            self.start = 0.0
            self.end = 2.0
            self.text = " hello world "
            self.words = [
                FakeWord("hello", 0.1, 0.7, 0.91),
                FakeWord("world", 1.1, 1.8, 0.89),
            ]

    captured: dict[str, object] = {}

    class FakeWhisperModel:
        def __init__(self, model_size_or_path: str, device: str, compute_type: str) -> None:
            captured["init"] = {
                "model_size_or_path": model_size_or_path,
                "device": device,
                "compute_type": compute_type,
            }

        def transcribe(
            self,
            audio_path: str,
            language: str,
            beam_size: int,
            best_of: int,
            vad_filter: bool,
            word_timestamps: bool,
        ):
            captured["transcribe"] = {
                "audio_path": audio_path,
                "language": language,
                "beam_size": beam_size,
                "best_of": best_of,
                "vad_filter": vad_filter,
                "word_timestamps": word_timestamps,
            }
            return [FakeSegment()], {}

    fw_mod = ModuleType("faster_whisper")
    fw_mod.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fw_mod)

    cfg = PipelineConfig(input_file="dummy.mp4", asr_model="medium")
    result = mp._run_faster_whisper("audio.wav", "cpu", "int8", cfg)

    assert captured["transcribe"]["word_timestamps"] is True
    assert len(result.segments) == 1
    assert result.segments[0].words == [
        {"word": "hello", "start": 0.1, "end": 0.7, "probability": 0.91},
        {"word": "world", "start": 1.1, "end": 1.8, "probability": 0.89},
    ]


@pytest.mark.unit
def test_word_level_overlap_calculation() -> None:
    """
    Test overlap calculation at word level.
    Validates: Phase 3, Task 15.2
    """
    asr_segments = [
        ASRSegment(
            id="asr_000001",
            start=0.0,
            end=1.0,
            text="a",
            words=[{"word": "a", "start": 0.0, "end": 1.0, "probability": 0.9}],
        )
    ]
    speaker_turns = [
        SpeakerTurn(id="turn_001", speaker_id="SPEAKER_00", start=0.0, end=0.5),
        SpeakerTurn(id="turn_002", speaker_id="SPEAKER_01", start=0.5, end=1.5),
    ]

    aligned = mp.align_segments_word_level(asr_segments, speaker_turns, ["SPEAKER_00", "SPEAKER_01"])
    assert len(aligned) == 1
    # Tie (0.5 vs 0.5): first encountered speaker is kept.
    assert aligned[0].speaker_id == "SPEAKER_00"
    assert aligned[0].source.overlap_sec == pytest.approx(0.5)


@pytest.mark.unit
def test_consecutive_word_merging() -> None:
    """
    Test merging of consecutive words with same speaker.
    Validates: Phase 3, Task 15.2
    """
    word_alignments = [
        {"word": "こんにちは", "start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "turn_id": "turn_001", "overlap": 1.0},
        {"word": "です", "start": 1.0, "end": 1.5, "speaker_id": "SPEAKER_00", "turn_id": "turn_001", "overlap": 0.5},
        {"word": "ありがとう", "start": 2.0, "end": 3.0, "speaker_id": "SPEAKER_01", "turn_id": "turn_002", "overlap": 1.0},
    ]

    merged = mp._merge_consecutive_words(word_alignments, "asr_000001")

    assert len(merged) == 2
    assert merged[0]["speaker_id"] == "SPEAKER_00"
    assert merged[0]["text"] == "こんにちはです"
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 1.5
    assert merged[1]["speaker_id"] == "SPEAKER_01"
    assert merged[1]["text"] == "ありがとう"


@pytest.mark.unit
def test_word_level_fallback_to_segment() -> None:
    """
    Test fallback to segment-level when word timestamps unavailable.
    Validates: Phase 3, backward compatibility
    """
    asr_seg = ASRSegment(
        id="asr_000001",
        start=0.0,
        end=5.0,
        text="test text",
        words=None,
    )
    speaker_turns = [SpeakerTurn(id="turn_001", speaker_id="SPEAKER_00", start=0.0, end=10.0)]

    aligned = mp.align_segments_word_level([asr_seg], speaker_turns, ["SPEAKER_00"])

    assert len(aligned) == 1
    assert aligned[0].speaker_id == "SPEAKER_00"
    assert aligned[0].text == "test text"
    assert aligned[0].start == 0.0
    assert aligned[0].end == 5.0


@pytest.mark.unit
def test_get_speaker_label() -> None:
    """
    Test speaker ID to label conversion.
    Validates: Phase 3, speaker label generation
    """
    assert mp._get_speaker_label("SPEAKER_00") == "Speaker 1"
    assert mp._get_speaker_label("SPEAKER_01") == "Speaker 2"
    assert mp._get_speaker_label("SPEAKER_09") == "Speaker 10"
    assert mp._get_speaker_label("UNKNOWN") == "Unknown"
    assert mp._get_speaker_label("INVALID") == "INVALID"


@pytest.mark.integration
def test_word_level_alignment_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_input_file: str) -> None:
    """
    Test complete pipeline with word-level alignment.
    Validates: Phase 3, end-to-end integration
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    def fake_extract(input_file: str, temp_dir: str, keep_audio: bool) -> AudioInfo:
        temp_audio = tmp_path / "temp" / "sample.wav"
        temp_audio.parent.mkdir(parents=True, exist_ok=True)
        temp_audio.write_bytes(b"wav")
        return AudioInfo(path=str(temp_audio), sample_rate=16000, channels=1, duration_sec=4.0)

    monkeypatch.setattr(pipeline_module, "resolve_device", lambda requested: DeviceInfo(requested=requested, resolved="cpu"))
    monkeypatch.setattr(pipeline_module, "extract_audio", fake_extract)
    monkeypatch.setattr(
        pipeline_module,
        "run_diarization",
        lambda audio_path, device, model: DiarizationResult(
            turns=[
                SpeakerTurn(id="turn_001", speaker_id="SPEAKER_00", start=0.0, end=2.0),
                SpeakerTurn(id="turn_002", speaker_id="SPEAKER_01", start=2.0, end=4.0),
            ],
            speakers=["SPEAKER_00", "SPEAKER_01"],
            model="pyannote/speaker-diarization-3.1",
            engine="pyannote-audio",
            hf_token_used=True,
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_asr",
        lambda audio_path, device, config: ASRResult(
            segments=[
                ASRSegment(
                    id="asr_000001",
                    start=0.0,
                    end=4.0,
                    text="こんにちはありがとう",
                    words=[
                        {"word": "こんにちは", "start": 0.5, "end": 1.5, "probability": 0.9},
                        {"word": "ありがとう", "start": 2.5, "end": 3.5, "probability": 0.9},
                    ],
                )
            ],
            model="medium",
            engine="faster-whisper",
            device=device,
            compute_type="int8",
            language="ja",
            beam_size=1,
            best_of=1,
            vad_filter=False,
            asr_load_sec=0.5,
        ),
    )

    cfg = PipelineConfig(
        input_file=sample_input_file,
        device="cpu",
        enable_diarization=True,
        align_unit="word",
        format="json",
        output_dir=str(output_dir),
        temp_dir=str(tmp_path / "temp"),
    )
    mp.run_pipeline(cfg)

    json_path = output_dir / f"{Path(sample_input_file).stem}_meeting.json"
    assert json_path.exists()

    with json_path.open(encoding="utf-8") as f:
        meeting = json.load(f)

    assert meeting["pipeline"]["align"]["unit"] == "word"
    assert len(meeting["segments"]) == 2
    assert meeting["segments"][0]["speaker_id"] == "SPEAKER_00"
    assert meeting["segments"][1]["speaker_id"] == "SPEAKER_01"


def test_device_resolution_macos_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test device resolution on macOS without CUDA."""
    from src.meeting_pipeline.device import resolve_device

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    device_info = resolve_device("auto")
    assert device_info.resolved == "mps"


def test_device_resolution_macos_no_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test device resolution on macOS without MPS."""
    from src.meeting_pipeline.device import resolve_device

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    device_info = resolve_device("auto")
    assert device_info.resolved == "cpu"


@pytest.mark.unit
def test_word_level_unknown_assignment() -> None:
    """
    Test UNKNOWN assignment at word level when no overlap.
    Validates: Phase 3, UNKNOWN preservation
    """
    asr_seg = ASRSegment(
        id="asr_000001",
        start=10.0,
        end=12.0,
        text="テストテキスト",
        words=[
            {"word": "テスト", "start": 10.5, "end": 11.0, "probability": 0.9},
            {"word": "テキスト", "start": 11.0, "end": 11.5, "probability": 0.9},
        ],
    )
    speaker_turns = [
        SpeakerTurn(id="turn_001", speaker_id="SPEAKER_00", start=0.0, end=5.0),
        SpeakerTurn(id="turn_002", speaker_id="SPEAKER_01", start=15.0, end=20.0),
    ]

    aligned = mp.align_segments_word_level([asr_seg], speaker_turns, ["SPEAKER_00", "SPEAKER_01"])

    assert len(aligned) == 1
    assert aligned[0].speaker_id == "UNKNOWN"
    assert aligned[0].speaker_label == "Unknown"
    assert aligned[0].text == "テストテキスト"
    assert aligned[0].source.diarization_turn_id is None
    assert aligned[0].source.overlap_sec == 0.0


@pytest.mark.unit
def test_word_level_speaker_transition() -> None:
    """
    Test detection of speaker transitions within ASR segment.
    Validates: Phase 3, speaker transition accuracy
    """
    asr_seg = ASRSegment(
        id="asr_000001",
        start=0.0,
        end=6.0,
        text="こんにちはありがとう",
        words=[
            {"word": "こんにちは", "start": 0.5, "end": 1.5, "probability": 0.9},
            {"word": "ありがとう", "start": 4.5, "end": 5.5, "probability": 0.9},
        ],
    )
    speaker_turns = [
        SpeakerTurn(id="turn_001", speaker_id="SPEAKER_00", start=0.0, end=2.0),
        SpeakerTurn(id="turn_002", speaker_id="SPEAKER_01", start=4.0, end=6.0),
    ]

    aligned = mp.align_segments_word_level([asr_seg], speaker_turns, ["SPEAKER_00", "SPEAKER_01"])

    assert len(aligned) == 2
    assert aligned[0].speaker_id == "SPEAKER_00"
    assert aligned[0].text == "こんにちは"
    assert aligned[1].speaker_id == "SPEAKER_01"
    assert aligned[1].text == "ありがとう"


@pytest.mark.unit
def test_cli_align_unit_parameter(sample_input_file: str) -> None:
    """
    Test --align-unit CLI parameter parsing.
    Validates: Phase 3, CLI integration
    """
    args = mp.parse_args([sample_input_file, "--align-unit", "segment"])
    assert args.align_unit == "segment"

    args = mp.parse_args([sample_input_file, "--align-unit", "word"])
    assert args.align_unit == "word"

    args = mp.parse_args([sample_input_file])
    assert args.align_unit == "segment"

    with pytest.raises(SystemExit):
        mp.parse_args([sample_input_file, "--align-unit", "invalid"])


@pytest.mark.unit
def test_align_config_unit_metadata(
    sample_config: PipelineConfig,
    sample_device_info: DeviceInfo,
    sample_audio_info: AudioInfo,
    sample_asr_result: ASRResult,
    sample_timing: Timing,
) -> None:
    """
    Test that align.unit is correctly recorded in Meeting JSON.
    Validates: Phase 3, Task 15.3
    """
    seg_cfg = PipelineConfig(**{**sample_config.__dict__, "align_unit": "segment"})
    meeting_segment = mp.generate_meeting_json(
        config=seg_cfg,
        device_info=sample_device_info,
        audio_info=sample_audio_info,
        diarization_result=None,
        asr_result=sample_asr_result,
        aligned_segments=[],
        timing=sample_timing,
    )
    assert meeting_segment.pipeline.align.unit == "segment"

    word_cfg = PipelineConfig(**{**sample_config.__dict__, "align_unit": "word"})
    meeting_word = mp.generate_meeting_json(
        config=word_cfg,
        device_info=sample_device_info,
        audio_info=sample_audio_info,
        diarization_result=None,
        asr_result=sample_asr_result,
        aligned_segments=[],
        timing=sample_timing,
    )
    assert meeting_word.pipeline.align.unit == "word"


@pytest.mark.unit
def test_phase3_backward_compatibility() -> None:
    """
    Test that phase 3 keeps default behavior compatible.
    Validates: Phase 3, backward compatibility
    """
    cfg = PipelineConfig(input_file="dummy.mp4")
    assert cfg.align_unit == "segment"


# ---------------------------------------------------------------------------
# Phase 6 tests - Minutes output, pipeline integration, CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_meeting_minutes() -> MeetingMinutes:
    return MeetingMinutes(
        schema_version="1.0",
        created_at="2026-03-29T12:00:00+09:00",
        meeting_title="週次定例",
        meeting_date="2026-03-29",
        duration_sec=3661.0,
        participants=["Speaker 1", "Speaker 2"],
        summary="会議全体の要約です。",
        decisions=[Decision(text="仕様を確定する", speaker="Speaker 1", timestamp=75.0)],
        action_items=[
            ActionItem(
                task="資料を更新する",
                assignee="Speaker 2",
                deadline="2026-04-01",
                timestamp=120.0,
            )
        ],
        topics=[Topic(title="進捗確認", summary="進捗の共有を行った。", start=0.0, end=180.0)],
        model_info=MinutesConfig(
            enabled=True,
            model="gpt-4",
            language="ja",
            temperature=0.3,
            max_tokens=4000,
        ),
        generation_time_sec=2.4,
    )


def test_generate_minutes_markdown_structure(sample_meeting_minutes: MeetingMinutes) -> None:
    md = output_module.generate_minutes_markdown(sample_meeting_minutes)

    assert md.startswith("# 議事録: 週次定例")
    assert "## 要約" in md
    assert "## 決定事項" in md
    assert "## アクションアイテム" in md
    assert "## トピック" in md


def test_generate_minutes_markdown_summary_section(sample_meeting_minutes: MeetingMinutes) -> None:
    md = output_module.generate_minutes_markdown(sample_meeting_minutes)

    assert "## 要約" in md
    assert "会議全体の要約です。" in md
    assert "**時間**: 01:01:01" in md


def test_generate_minutes_markdown_decisions_section(sample_meeting_minutes: MeetingMinutes) -> None:
    md = output_module.generate_minutes_markdown(sample_meeting_minutes)

    assert "1. [00:01:15] 仕様を確定する (Speaker 1による)" in md


def test_generate_minutes_markdown_action_items_section(sample_meeting_minutes: MeetingMinutes) -> None:
    md = output_module.generate_minutes_markdown(sample_meeting_minutes)

    assert "| タスク | 担当者 | 期限 | タイムスタンプ |" in md
    assert "| 資料を更新する | Speaker 2 | 2026-04-01 | 00:02:00 |" in md


def test_save_minutes_json(sample_meeting_minutes: MeetingMinutes, tmp_path: Path) -> None:
    output_path = tmp_path / "minutes.json"

    output_module.save_minutes_json(sample_meeting_minutes, str(output_path))

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["meeting_title"] == "週次定例"
    assert saved["action_items"][0]["task"] == "資料を更新する"


def test_save_minutes_json_schema_version_validation(
    sample_meeting_minutes: MeetingMinutes,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "minutes.json"

    output_module.save_minutes_json(sample_meeting_minutes, str(output_path))

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == "1.0"


def test_save_minutes_markdown(sample_meeting_minutes: MeetingMinutes, tmp_path: Path) -> None:
    output_path = tmp_path / "minutes.md"
    content = output_module.generate_minutes_markdown(sample_meeting_minutes)

    output_module.save_minutes_markdown(content, str(output_path))

    assert output_path.read_text(encoding="utf-8") == content


def _patch_pipeline_for_minutes(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: PipelineConfig,
    tmp_path: Path,
) -> Path:
    extracted = tmp_path / "temp" / "sample.wav"

    def fake_extract(input_file: str, temp_dir: str, keep_audio: bool) -> AudioInfo:
        extracted.parent.mkdir(parents=True, exist_ok=True)
        extracted.write_bytes(b"wav")
        return AudioInfo(path=str(extracted), sample_rate=16000, channels=1, duration_sec=30.0)

    monkeypatch.setattr(pipeline_module, "resolve_device", lambda requested: DeviceInfo(requested=requested, resolved="cpu"))
    monkeypatch.setattr(pipeline_module, "extract_audio", fake_extract)
    monkeypatch.setattr(
        pipeline_module,
        "run_asr",
        lambda audio_path, device, config: ASRResult(
            segments=[ASRSegment(id="asr_000001", start=0.0, end=1.0, text="hello")],
            model="medium",
            engine="faster-whisper",
            device=device,
            compute_type="int8",
            language="ja",
            beam_size=1,
            best_of=1,
            vad_filter=False,
            asr_load_sec=0.1,
        ),
    )
    monkeypatch.setattr(pipeline_module, "align_segments", lambda asr_segments, speaker_turns, speakers: [])
    monkeypatch.setattr(
        pipeline_module,
        "generate_meeting_json",
        lambda **kwargs: mp.MeetingJSON(
            schema_version="1.0",
            created_at="2026-03-29T12:00:00+09:00",
            title="定例会議",
            input=mp.InputInfo(path=sample_config.input_file, audio=kwargs["audio_info"], duration_sec=30.0),
            pipeline=mp.PipelineInfo(
                device=kwargs["device_info"],
                diarization=mp.DiarizationConfig(enabled=False, engine="", model="", hf_token_used=False),
                asr=mp.ASRConfigInfo(
                    engine="faster-whisper",
                    model="medium",
                    device="cpu",
                    compute_type="int8",
                    language="ja",
                    beam_size=1,
                    best_of=1,
                    vad_filter=False,
                ),
                align=mp.AlignConfig(method="max_overlap", unit="segment"),
            ),
            speakers=[mp.Speaker(id="UNKNOWN", label="Unknown")],
            segments=[],
            artifacts=mp.Artifacts(diarization_turns=[], asr_segments=[]),
            timing=kwargs["timing"],
            notes="",
        ),
    )
    return extracted


def test_pipeline_runs_with_minutes_generation(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: PipelineConfig,
    sample_meeting_minutes: MeetingMinutes,
    tmp_path: Path,
) -> None:
    _patch_pipeline_for_minutes(monkeypatch, sample_config, tmp_path)
    json_path = tmp_path / "out" / "sample_meeting.json"
    saved: dict[str, object] = {}

    def fake_save_meeting_json(meeting, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("{}", encoding="utf-8")
        saved["meeting_json_path"] = output_path

    monkeypatch.setattr(pipeline_module, "save_meeting_json", fake_save_meeting_json)
    monkeypatch.setattr(pipeline_module, "save_transcript_markdown", lambda content, output_path: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(minutes_module, "generate_minutes", lambda path, config, api_key: sample_meeting_minutes)

    def fake_save_minutes_markdown(content: str, output_path: str) -> None:
        saved["minutes_md_path"] = output_path

    def fake_save_minutes_json(minutes: MeetingMinutes, output_path: str) -> None:
        saved["minutes_json_path"] = output_path

    monkeypatch.setattr(pipeline_module, "save_minutes_markdown", fake_save_minutes_markdown)
    monkeypatch.setattr(pipeline_module, "save_minutes_json", fake_save_minutes_json)

    cfg = PipelineConfig(
        **{
            **sample_config.__dict__,
            "format": "both",
            "output_dir": str(tmp_path / "out"),
            "generate_minutes": True,
            "minutes_model": "gpt-4",
            "minutes_language": "en",
        }
    )
    mp.run_pipeline(cfg)

    assert saved["meeting_json_path"] == str(json_path)
    assert str(saved["minutes_md_path"]).endswith("sample_minutes.md")
    assert str(saved["minutes_json_path"]).endswith("sample_minutes.json")


def test_pipeline_continues_when_minutes_generation_fails(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: PipelineConfig,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_pipeline_for_minutes(monkeypatch, sample_config, tmp_path)
    saved = {"meeting_json": 0}
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def fake_save_meeting_json(meeting, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("{}", encoding="utf-8")
        saved["meeting_json"] += 1

    monkeypatch.setattr(pipeline_module, "save_meeting_json", fake_save_meeting_json)
    monkeypatch.setattr(pipeline_module, "save_transcript_markdown", lambda content, output_path: None)
    monkeypatch.setattr(
        minutes_module,
        "generate_minutes",
        lambda path, config, api_key: (_ for _ in ()).throw(RuntimeError("minutes failed")),
    )
    monkeypatch.setattr(pipeline_module, "save_minutes_markdown", lambda content, output_path: (_ for _ in ()).throw(AssertionError("should not save")))
    monkeypatch.setattr(pipeline_module, "save_minutes_json", lambda minutes, output_path: (_ for _ in ()).throw(AssertionError("should not save")))

    cfg = PipelineConfig(
        **{
            **sample_config.__dict__,
            "format": "both",
            "output_dir": str(tmp_path / "out"),
            "generate_minutes": True,
        }
    )
    mp.run_pipeline(cfg)

    stderr = capsys.readouterr().err
    assert saved["meeting_json"] == 1
    assert "エラー: 議事録生成失敗: minutes failed" in stderr
    assert "Meeting JSONと文字起こしは正常に保存されました。" in stderr


def test_warning_when_openai_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
    sample_input_file: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = cli_module.parse_args([sample_input_file, "--generate-minutes"])

    stderr = capsys.readouterr().err
    assert config.generate_minutes is True
    assert "警告: OPENAI_API_KEYが設定されていません。議事録生成はスキップされます。" in stderr


def test_cli_generate_minutes_option(sample_input_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    args = mp.parse_args([sample_input_file, "--generate-minutes"])

    assert args.generate_minutes is True


def test_cli_minutes_model_option(sample_input_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    args = mp.parse_args([sample_input_file, "--minutes-model", "gpt-4"])

    assert args.minutes_model == "gpt-4"


def test_cli_minutes_language_option(sample_input_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    args = mp.parse_args([sample_input_file, "--minutes-language", "en"])

    assert args.minutes_language == "en"
