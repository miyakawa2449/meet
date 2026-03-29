"""Speaker diarization using pyannote-audio."""

import gc
import logging
import os
import sys
from typing import Dict, List

from .models import DiarizationResult, SpeakerTurn

logger = logging.getLogger(__name__)

def _pyannote_audio_input(audio_path: str):
    """
    Return a pyannote-compatible audio input.
    Always preload audio without torchaudio to avoid torchcodec runtime dependency.
    """
    try:
        import soundfile as sf
        import torch
    except Exception as exc:
        raise RuntimeError(
            "soundfile is required for diarization audio loading. "
            "Install it with: python -m pip install soundfile"
        ) from exc

    data, sample_rate = sf.read(audio_path, always_2d=True)
    # soundfile returns (time, channels); pyannote expects (channels, time)
    waveform = torch.from_numpy(data.T).to(torch.float32)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def run_diarization(
    audio_path: str,
    device: str,
    model_name: str,
) -> DiarizationResult:
    """
    Run speaker diarization using pyannote-audio.
    Requires HF_TOKEN in environment.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print(
            "Error: HF_TOKEN environment variable is required for diarization",
            file=sys.stderr,
        )
        print(
            "Get token: https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        sys.exit(2)

    logger.info("Loading diarization model: %s", model_name)

    from pyannote.audio import Pipeline as PyannotePipeline

    pipeline = PyannotePipeline.from_pretrained(
        model_name,
        token=hf_token,
    )

    # Move pipeline to device
    import torch

    if device == "cuda" and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    elif device == "mps" and torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))

    logger.info("Running diarization...")
    diarization = pipeline(_pyannote_audio_input(audio_path))

    # Build speaker turns with sequential IDs
    turns: List[SpeakerTurn] = []
    speaker_order: List[str] = []
    speaker_map: Dict[str, str] = {}

    turn_counter = 0
    for turn, speaker_label in diarization.speaker_diarization:
        if speaker_label not in speaker_map:
            idx = len(speaker_map)
            mapped_id = f"SPEAKER_{idx:02d}"
            speaker_map[speaker_label] = mapped_id
            speaker_order.append(mapped_id)

        turn_counter += 1
        turns.append(
            SpeakerTurn(
                id=f"turn_{turn_counter:06d}",
                speaker_id=speaker_map[speaker_label],
                start=turn.start,
                end=turn.end,
            )
        )

    logger.info(
        "Diarization complete: %d turns, %d speakers",
        len(turns),
        len(speaker_order),
    )

    # Release model
    del pipeline
    gc.collect()
    import torch as _torch

    if device == "cuda" and _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    elif device == "mps" and _torch.backends.mps.is_available():
        _torch.mps.empty_cache()

    logger.info("Diarization model released (device=%s)", device)

    return DiarizationResult(
        turns=turns,
        speakers=speaker_order,
        model=model_name,
        engine="pyannote-audio",
        hf_token_used=True,
    )
