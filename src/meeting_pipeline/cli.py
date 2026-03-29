"""CLI argument parsing and validation."""

import argparse
import os
import sys
from typing import List, Optional

from .models import PipelineConfig


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
    parser.add_argument(
        "--generate-minutes",
        action="store_true",
        help="OpenAI APIを使用して議事録を生成",
    )
    parser.add_argument(
        "--minutes-model",
        default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        help="議事録生成用のOpenAIモデル（デフォルト: gpt-3.5-turbo）",
    )
    parser.add_argument(
        "--minutes-language",
        default="auto",
        help="議事録出力の言語（デフォルト: 入力から自動検出）",
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
        generate_minutes=args.generate_minutes,
        minutes_model=args.minutes_model,
        minutes_language=args.minutes_language,
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

    # 議事録生成が有効な場合、APIキーをチェック
    if config.generate_minutes:
        if not os.getenv("OPENAI_API_KEY"):
            print(
                "警告: OPENAI_API_KEYが設定されていません。議事録生成はスキップされます。",
                file=sys.stderr,
            )
