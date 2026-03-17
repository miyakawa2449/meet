"""ASR (Automatic Speech Recognition) engines."""

import logging
import sys
import time
from typing import List

from .models import ASRResult, ASRSegment, PipelineConfig

logger = logging.getLogger(__name__)


def _determine_compute_type(device: str) -> str:
    """Determine optimal compute type based on device.

    - CUDA: float16 (best GPU throughput; falls back to int8 on FP16 failure)
    - MPS: float16 (used only by whisper engine; faster-whisper falls back to CPU)
    - CPU: int8 (best speed/accuracy balance on CPU via quantization)
    """
    if device in ("cuda", "mps"):
        return "float16"
    return "int8"


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
    try:
        model = WhisperModel(
            model_size_or_path=config.asr_model,
            device=device,
            compute_type=compute_type,
        )
    except Exception as e:
        if device == "cuda" and compute_type == "float16":
            logger.warning(
                "CUDA float16 failed (%s), retrying with int8", e
            )
            compute_type = "int8"
            model = WhisperModel(
                model_size_or_path=config.asr_model,
                device=device,
                compute_type=compute_type,
            )
        else:
            raise
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


def run_asr(
    audio_path: str,
    device: str,
    config: PipelineConfig,
) -> ASRResult:
    """Run ASR using faster-whisper or whisper."""
    # faster-whisper does not support MPS, fallback to CPU
    asr_device = device
    if config.asr_engine == "faster-whisper" and device == "mps":
        logger.warning(
            "faster-whisper does not support MPS device; "
            "falling back to CPU (compute_type will be int8)"
        )
        asr_device = "cpu"

    compute_type = _determine_compute_type(asr_device)
    logger.info("ASR device=%s, compute_type=%s", asr_device, compute_type)

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
