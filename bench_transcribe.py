#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import platform
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ffmpeg

try:
    import config
except Exception:
    config = None


# -----------------------------
# Timing
# -----------------------------


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


@dataclass
class StageTimes:
    extract_sec: float = 0.0
    load_sec: float = 0.0
    asr_sec: float = 0.0
    summary_sec: float = 0.0
    total_sec: float = 0.0


@dataclass
class BenchRecord:
    ts: str
    host: str
    os: str
    python: str

    script: str
    input_path: str
    audio_path: str

    engine: str
    requested_device: str
    backend: str
    model: str
    compute_type: str

    language: str
    beam_size: int
    best_of: int
    vad_filter: bool
    no_summary: bool

    t_extract_sec: float
    t_model_load_sec: float
    t_asr_sec: float
    t_summary_sec: float
    t_total_sec: float

    transcript_path: str
    summary_path: str
    note: str

    extra: dict[str, Any]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def now_compact() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_round(v: float) -> float:
    return round(float(v), 3)


# -----------------------------
# I/O
# -----------------------------


def write_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_markdown(path: Path, rec: BenchRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "| ts | engine | device | model | beam/best_of | vad | "
        "extract | load | asr | summary | total | note |\n"
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|\n"
    )

    row = (
        f"| {rec.ts} | {rec.engine} | {rec.backend} | {rec.model} | "
        f"{rec.beam_size}/{rec.best_of} | {int(rec.vad_filter)} | "
        f"{rec.t_extract_sec:.3f} | {rec.t_model_load_sec:.3f} | "
        f"{rec.t_asr_sec:.3f} | {rec.t_summary_sec:.3f} | {rec.t_total_sec:.3f} | "
        f"{rec.note.replace('|', '/')} |\n"
    )

    if not path.exists() or path.stat().st_size == 0:
        path.write_text(header + row, encoding="utf-8")
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(row)


def extract_audio(video_path: Path, audio_path: Path) -> None:
    (
        ffmpeg.input(str(video_path))
        .output(str(audio_path), acodec="pcm_s16le", ac=1, ar="16k")
        .overwrite_output()
        .run(quiet=True)
    )


# -----------------------------
# Device
# -----------------------------


def resolve_backend(engine: str, requested_device: str) -> tuple[str, str]:
    """
    Returns: (backend, fallback_reason)
    """
    reason = ""

    if engine == "whisper":
        import torch

        if requested_device == "cpu":
            return "cpu", reason
        if requested_device == "cuda":
            return ("cuda", reason) if torch.cuda.is_available() else ("cpu", "cuda unavailable")
        if requested_device == "mps":
            return ("mps", reason) if torch.backends.mps.is_available() else ("cpu", "mps unavailable")

        if torch.cuda.is_available():
            return "cuda", reason
        if torch.backends.mps.is_available():
            return "mps", reason
        return "cpu", reason

    # faster-whisper
    try:
        import ctranslate2

        cuda_available = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        cuda_available = False

    if requested_device == "cpu":
        return "cpu", reason
    if requested_device == "mps":
        return "cpu", "mps unsupported for faster-whisper"
    if requested_device == "cuda":
        return ("cuda", reason) if cuda_available else ("cpu", "cuda unavailable")

    if cuda_available:
        return "cuda", reason
    return "cpu", reason


# -----------------------------
# ASR
# -----------------------------


def run_whisper(
    audio_path: Path,
    model_name: str,
    backend: str,
    language: str,
    beam_size: int,
    best_of: int,
    whisper_fp16_mode: str,
) -> tuple[str, str, float, float, dict[str, Any]]:
    import whisper

    def _resolve_fp16(device: str) -> bool:
        if whisper_fp16_mode == "on":
            return device in ("cuda", "mps")
        if whisper_fp16_mode == "off":
            return False
        return device == "cuda"

    def _run_with(device: str) -> tuple[str, float, float]:
        load_timer = Timer()
        model = whisper.load_model(model_name, device=device)
        t_load_inner = load_timer.elapsed()

        kwargs: dict[str, Any] = {
            "language": language,
            "beam_size": beam_size,
            "best_of": best_of,
            "temperature": 0.0,
            "fp16": _resolve_fp16(device),
            "verbose": False,
        }

        asr_timer = Timer()
        result = model.transcribe(str(audio_path), **kwargs)
        t_asr_inner = asr_timer.elapsed()
        text_inner = (result.get("text") or "").strip()
        return text_inner, t_load_inner, t_asr_inner

    tried_backend = backend
    try:
        text, t_load, t_asr = _run_with(backend)
    except ValueError as exc:
        # MPS環境でまれにNaN logitsが出るため、CPUへフォールバックして再実行する。
        if backend == "mps" and "nan" in str(exc).lower():
            print("[asr:whisper] mpsでNaN検出。cpuへフォールバックして再試行します。")
            tried_backend = "cpu"
            text, t_load, t_asr = _run_with("cpu")
        else:
            raise

    compute_type = "float16" if tried_backend == "cuda" else "float32"
    extra = {
        "engine": "whisper",
        "backend": tried_backend,
        "requested_backend": backend,
        "whisper_fp16_mode": whisper_fp16_mode,
        "whisper_fp16_effective": _resolve_fp16(tried_backend),
    }
    return text, compute_type, t_load, t_asr, extra


def run_faster_whisper(
    audio_path: Path,
    model_name: str,
    backend: str,
    language: str,
    beam_size: int,
    best_of: int,
    vad_filter: bool,
    cpu_threads: int,
    num_workers: int,
) -> tuple[str, str, float, float, dict[str, Any]]:
    from faster_whisper import WhisperModel

    compute_type = "float16" if backend == "cuda" else "int8"

    model_kwargs: dict[str, Any] = {
        "device": backend,
        "compute_type": compute_type,
    }
    if backend == "cpu":
        model_kwargs["cpu_threads"] = cpu_threads
        model_kwargs["num_workers"] = num_workers

    load_timer = Timer()
    model = WhisperModel(model_name, **model_kwargs)
    t_load = load_timer.elapsed()

    transcribe_kwargs: dict[str, Any] = {
        "language": language,
        "beam_size": beam_size,
        "best_of": best_of,
        "vad_filter": vad_filter,
        "temperature": 0.0,
    }

    # Guard for version differences
    supported = set(inspect.signature(model.transcribe).parameters)
    filtered_kwargs = {k: v for k, v in transcribe_kwargs.items() if k in supported}

    asr_timer = Timer()
    segments, info = model.transcribe(str(audio_path), **filtered_kwargs)
    text = "".join(seg.text for seg in segments).strip()
    t_asr = asr_timer.elapsed()

    extra = {
        "engine": "faster-whisper",
        "backend": backend,
        "detected_language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
    }
    return text, compute_type, t_load, t_asr, extra


# -----------------------------
# Summary
# -----------------------------


def summarize_with_llm(text: str, summary_model: str) -> str:
    if config is None or not getattr(config, "OPENAI_API_KEY", None):
        return ""

    from openai import OpenAI

    prompt = (
        "以下は会議の文字起こしです。この内容から議事録を作成してください。\n\n"
        "要件:\n"
        "- 会議の主要なトピックを箇条書きで整理\n"
        "- 決定事項を明確に記載\n"
        "- アクションアイテムがあれば抽出\n"
        "- 重要な議論のポイントをまとめる\n\n"
        f"文字起こし:\n{text}\n"
    )

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=summary_model,
        messages=[
            {"role": "system", "content": "あなたは議事録作成の専門家です。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcription benchmark (whisper / faster-whisper)",
    )
    parser.add_argument("input_file", help="Input mp4/wav file")

    parser.add_argument(
        "--engine",
        choices=["whisper", "faster-whisper"],
        required=True,
        help="ASR engine",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Requested device",
    )
    parser.add_argument("--model", default="medium", help="ASR model name")
    parser.add_argument("--language", default="ja", help="Language code")
    parser.add_argument(
        "--whisper-fp16",
        choices=["auto", "on", "off"],
        default="auto",
        help="Whisper fp16 mode (auto: cuda only, on: cuda/mps, off: disable)",
    )

    parser.add_argument("--beam-size", type=int, default=1, help="Beam size")
    parser.add_argument("--best-of", type=int, default=1, help="best_of")
    parser.add_argument("--vad-filter", action="store_true", help="Use VAD (faster-whisper)")

    parser.add_argument("--cpu-threads", type=int, default=8, help="CPU threads for faster-whisper")
    parser.add_argument("--num-workers", type=int, default=2, help="Workers for faster-whisper")

    parser.add_argument("--summary-model", default="gpt-4o", help="LLM model for summary")
    parser.add_argument("--no-summary", action="store_true", help="Skip summary")

    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--temp-dir", default="temp", help="Temporary directory for extracted wav")
    parser.add_argument("--audio-file", default="", help="Use existing wav path and skip extraction")
    parser.add_argument("--keep-audio", action="store_true", help="Keep extracted wav file")

    parser.add_argument("--bench-jsonl", default="bench/bench.jsonl", help="JSONL log path")
    parser.add_argument("--bench-md", default="", help="Optional markdown log path")
    parser.add_argument(
        "--run-id",
        default="",
        help="Output filename suffix. If empty, timestamp is used",
    )
    parser.add_argument("--note", default="", help="Optional note")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise SystemExit(f"ERROR: input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    temp_dir = Path(args.temp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    backend, reason = resolve_backend(args.engine, args.device)
    if reason:
        print(f"[device] fallback: {reason}")
    print(
        f"[config] engine={args.engine}, requested_device={args.device}, backend={backend}, "
        f"model={args.model}, beam={args.beam_size}, best_of={args.best_of}, vad={args.vad_filter}"
    )
    if args.engine == "whisper":
        print(f"[config] whisper_fp16={args.whisper_fp16}")

    base_name = input_path.stem
    run_id = args.run_id.strip() or now_compact()
    if args.audio_file:
        audio_path = Path(args.audio_file)
    elif input_path.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"}:
        audio_path = input_path
    else:
        audio_path = temp_dir / f"{base_name}.wav"

    transcript_path = output_dir / f"{base_name}_{args.engine}_{backend}_{run_id}_transcript.txt"
    summary_path = output_dir / f"{base_name}_{args.engine}_{backend}_{run_id}_summary.txt"

    times = StageTimes()
    total_timer = Timer()

    # 1) extract
    need_extract = not args.audio_file and input_path != audio_path
    if need_extract:
        t = Timer()
        extract_audio(input_path, audio_path)
        times.extract_sec = t.elapsed()
        print(f"[extract] done: {audio_path}")
    else:
        if not audio_path.exists():
            raise SystemExit(f"ERROR: audio file not found: {audio_path}")
        print(f"[extract] skipped: {audio_path}")

    # 2) load 3) asr
    if args.engine == "whisper":
        text, compute_type, times.load_sec, times.asr_sec, extra = run_whisper(
            audio_path=audio_path,
            model_name=args.model,
            backend=backend,
            language=args.language,
            beam_size=args.beam_size,
            best_of=args.best_of,
            whisper_fp16_mode=args.whisper_fp16,
        )
    else:
        text, compute_type, times.load_sec, times.asr_sec, extra = run_faster_whisper(
            audio_path=audio_path,
            model_name=args.model,
            backend=backend,
            language=args.language,
            beam_size=args.beam_size,
            best_of=args.best_of,
            vad_filter=args.vad_filter,
            cpu_threads=args.cpu_threads,
            num_workers=args.num_workers,
        )

    effective_backend = str(extra.get("backend", backend))
    if effective_backend != backend:
        print(f"[device] effective backend changed: {backend} -> {effective_backend}")

    transcript_path.write_text(text, encoding="utf-8")
    print(f"[asr] transcript: {transcript_path}")

    # 4) summary
    summary_text = ""
    if not args.no_summary:
        t = Timer()
        summary_text = summarize_with_llm(text, args.summary_model)
        times.summary_sec = t.elapsed()
        if summary_text:
            summary_path.write_text(summary_text, encoding="utf-8")
            print(f"[summary] written: {summary_path}")
        else:
            print("[summary] skipped (API key missing or empty response)")

    # cleanup
    if need_extract and not args.keep_audio and audio_path.exists():
        audio_path.unlink()
        print(f"[cleanup] removed: {audio_path}")

    times.total_sec = total_timer.elapsed()

    print("\n========== Benchmark ==========")
    print(f"extract : {times.extract_sec:.3f}s")
    print(f"load    : {times.load_sec:.3f}s")
    print(f"asr     : {times.asr_sec:.3f}s")
    print(f"summary : {times.summary_sec:.3f}s")
    print(f"total   : {times.total_sec:.3f}s")
    print("===============================")

    rec = BenchRecord(
        ts=now_iso(),
        host=socket.gethostname(),
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        script=Path(sys.argv[0]).name,
        input_path=str(input_path),
        audio_path=str(audio_path),
        engine=args.engine,
        requested_device=args.device,
        backend=effective_backend,
        model=args.model,
        compute_type=compute_type,
        language=args.language,
        beam_size=args.beam_size,
        best_of=args.best_of,
        vad_filter=bool(args.vad_filter),
        no_summary=bool(args.no_summary),
        t_extract_sec=safe_round(times.extract_sec),
        t_model_load_sec=safe_round(times.load_sec),
        t_asr_sec=safe_round(times.asr_sec),
        t_summary_sec=safe_round(times.summary_sec),
        t_total_sec=safe_round(times.total_sec),
        transcript_path=str(transcript_path),
        summary_path=str(summary_path if summary_text else ""),
        note=args.note,
        extra=extra,
    )

    write_jsonl(Path(args.bench_jsonl), asdict(rec))
    print(f"[bench] jsonl appended: {args.bench_jsonl}")

    if args.bench_md:
        append_markdown(Path(args.bench_md), rec)
        print(f"[bench] markdown appended: {args.bench_md}")


if __name__ == "__main__":
    main()
