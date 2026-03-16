"""Output generation: JSON and Markdown."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import List, Optional

from .models import (
    AlignConfig,
    AlignedSegment,
    Artifacts,
    ASRConfigInfo,
    ASRResult,
    AudioInfo,
    DeviceInfo,
    DiarizationConfig,
    DiarizationResult,
    InputInfo,
    MeetingJSON,
    PipelineConfig,
    PipelineInfo,
    Speaker,
    Timing,
)

logger = logging.getLogger(__name__)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format with zero-padding."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
