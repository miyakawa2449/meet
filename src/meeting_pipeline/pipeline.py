"""Main pipeline orchestration."""

import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from .alignment import align_segments, align_segments_word_level
from .asr import run_asr
from .audio import extract_audio
from .device import resolve_device
from .diarization import run_diarization
from .models import (
    AudioInfo,
    DiarizationResult,
    MinutesConfig,
    PipelineConfig,
    Timing,
)
from .output import (
    generate_meeting_json,
    generate_minutes_markdown,
    generate_transcript_markdown,
    log_benchmark,
    save_meeting_json,
    save_minutes_json,
    save_minutes_markdown,
    save_transcript_markdown,
)

logger = logging.getLogger(__name__)


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

        # --- Stage 8: 議事録生成（オプション） ---
        if config.generate_minutes:
            logger.info("ステージ: 議事録生成")
            t0 = time.time()

            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error("環境にOPENAI_API_KEYが見つかりません")
                print("エラー: OPENAI_API_KEYが設定されていません。議事録生成をスキップします。", file=sys.stderr)
            else:
                try:
                    meeting_json_path = os.path.join(config.output_dir, f"{basename}_meeting.json")

                    minutes_config = MinutesConfig(
                        enabled=True,
                        model=config.minutes_model,
                        language=config.minutes_language if config.minutes_language != "auto" else config.language,
                        temperature=0.3,
                        max_tokens=4000,
                    )

                    from .minutes import generate_minutes

                    minutes = generate_minutes(
                        meeting_json_path,
                        minutes_config,
                        openai_api_key,
                    )

                    minutes_md_path = os.path.join(config.output_dir, f"{basename}_minutes.md")
                    minutes_json_path = os.path.join(config.output_dir, f"{basename}_minutes.json")

                    md_content = generate_minutes_markdown(minutes)
                    save_minutes_markdown(md_content, minutes_md_path)
                    save_minutes_json(minutes, minutes_json_path)

                    timing.minutes_sec = round(time.time() - t0, 1)
                    logger.info("議事録生成完了: %.1f秒", timing.minutes_sec)

                except Exception as e:
                    logger.error("議事録生成失敗: %s", e)
                    print(f"エラー: 議事録生成失敗: {e}", file=sys.stderr)
                    print("Meeting JSONと文字起こしは正常に保存されました。", file=sys.stderr)
        else:
            logger.info("議事録生成無効、スキップ")

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
