"""Alignment of ASR segments with speaker turns."""

import logging
from typing import Any, Dict, List, Optional

from .models import AlignedSegment, ASRSegment, SegmentSource, SpeakerTurn

logger = logging.getLogger(__name__)


def _calculate_overlap(
    seg_start: float,
    seg_end: float,
    turn_start: float,
    turn_end: float,
) -> float:
    """Calculate temporal overlap in seconds."""
    overlap_start = max(seg_start, turn_start)
    overlap_end = min(seg_end, turn_end)
    return max(0.0, overlap_end - overlap_start)


def _get_speaker_label(speaker_id: str) -> str:
    """
    Convert speaker_id to human-readable label.

    SPEAKER_00 -> Speaker 1, SPEAKER_01 -> Speaker 2, ..., UNKNOWN -> Unknown
    """
    if speaker_id == "UNKNOWN":
        return "Unknown"
    if speaker_id.startswith("SPEAKER_"):
        try:
            num = int(speaker_id.split("_")[1])
            return f"Speaker {num + 1}"
        except (IndexError, ValueError):
            return speaker_id
    return speaker_id


def align_segments(
    asr_segments: List[ASRSegment],
    speaker_turns: List[SpeakerTurn],
    speakers: List[str],
    method: str = "max_overlap",
) -> List[AlignedSegment]:
    """
    Align ASR segments with speaker turns using max_overlap method.
    Assigns UNKNOWN when no overlap is found.
    """
    # Build speaker label mapping: SPEAKER_00 -> "Speaker 1", etc.
    speaker_label_map: Dict[str, str] = {}
    for i, spk_id in enumerate(speakers):
        speaker_label_map[spk_id] = f"Speaker {i + 1}"
    speaker_label_map["UNKNOWN"] = "Unknown"

    aligned: List[AlignedSegment] = []
    seg_counter = 0

    for asr_seg in asr_segments:
        max_overlap = 0.0
        best_turn: Optional[SpeakerTurn] = None

        for turn in speaker_turns:
            overlap = _calculate_overlap(
                asr_seg.start,
                asr_seg.end,
                turn.start,
                turn.end,
            )
            if overlap > max_overlap:
                max_overlap = overlap
                best_turn = turn

        if max_overlap > 0 and best_turn is not None:
            speaker_id = best_turn.speaker_id
            turn_id = best_turn.id
        else:
            speaker_id = "UNKNOWN"
            turn_id = None
            logger.warning(
                "No overlap found for segment %s, assigning UNKNOWN",
                asr_seg.id,
            )

        seg_counter += 1
        aligned.append(
            AlignedSegment(
                id=f"seg_{seg_counter:06d}",
                start=asr_seg.start,
                end=asr_seg.end,
                speaker_id=speaker_id,
                speaker_label=speaker_label_map.get(speaker_id, "Unknown"),
                text=asr_seg.text,
                confidence=None,
                source=SegmentSource(
                    asr_segment_id=asr_seg.id,
                    diarization_turn_id=turn_id,
                    overlap_sec=max_overlap,
                ),
            )
        )

    # Sort by start time
    aligned.sort(key=lambda s: s.start)

    logger.info("Alignment complete: %d segments", len(aligned))
    return aligned


def _align_single_segment(
    asr_seg: ASRSegment,
    speaker_turns: List[SpeakerTurn],
    seg_counter: int,
) -> AlignedSegment:
    """Align a single ASR segment using segment-level overlap (fallback)."""
    max_overlap = 0.0
    best_turn: Optional[SpeakerTurn] = None

    for turn in speaker_turns:
        overlap = _calculate_overlap(
            asr_seg.start, asr_seg.end, turn.start, turn.end
        )
        if overlap > max_overlap:
            max_overlap = overlap
            best_turn = turn

    if max_overlap > 0 and best_turn is not None:
        speaker_id = best_turn.speaker_id
        turn_id = best_turn.id
    else:
        speaker_id = "UNKNOWN"
        turn_id = None

    return AlignedSegment(
        id=f"seg_{seg_counter:06d}",
        start=asr_seg.start,
        end=asr_seg.end,
        speaker_id=speaker_id,
        speaker_label=_get_speaker_label(speaker_id),
        text=asr_seg.text,
        confidence=None,
        source=SegmentSource(
            asr_segment_id=asr_seg.id,
            diarization_turn_id=turn_id,
            overlap_sec=max_overlap,
        ),
    )


def _merge_consecutive_words(
    word_alignments: List[Dict[str, Any]],
    asr_segment_id: str,
) -> List[Dict[str, Any]]:
    """Merge consecutive words with the same speaker into segments."""
    if not word_alignments:
        return []

    merged: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {
        "speaker_id": word_alignments[0]["speaker_id"],
        "turn_id": word_alignments[0]["turn_id"],
        "start": word_alignments[0]["start"],
        "end": word_alignments[0]["end"],
        "words": [word_alignments[0]["word"]],
        "overlap": word_alignments[0]["overlap"],
    }

    for wa in word_alignments[1:]:
        if wa["speaker_id"] == current["speaker_id"]:
            current["end"] = wa["end"]
            current["words"].append(wa["word"])
            current["overlap"] += wa["overlap"]
        else:
            current["text"] = "".join(current["words"]).strip()
            merged.append(current)
            current = {
                "speaker_id": wa["speaker_id"],
                "turn_id": wa["turn_id"],
                "start": wa["start"],
                "end": wa["end"],
                "words": [wa["word"]],
                "overlap": wa["overlap"],
            }

    current["text"] = "".join(current["words"]).strip()
    merged.append(current)

    return merged


def align_segments_word_level(
    asr_segments: List[ASRSegment],
    speaker_turns: List[SpeakerTurn],
    speakers: List[str],
) -> List[AlignedSegment]:
    """
    Align ASR segments with speaker turns at word level.

    For each word in an ASR segment, calculates overlap with all speaker turns
    and assigns the speaker with maximum overlap. Consecutive words with the
    same speaker are merged into a single AlignedSegment.

    Falls back to segment-level alignment when word timestamps are unavailable.
    """
    # Build speaker label mapping for consistent labels
    speaker_label_map: Dict[str, str] = {}
    for i, spk_id in enumerate(speakers):
        speaker_label_map[spk_id] = f"Speaker {i + 1}"
    speaker_label_map["UNKNOWN"] = "Unknown"

    aligned_segments: List[AlignedSegment] = []
    seg_counter = 0

    for asr_seg in asr_segments:
        # Fallback to segment-level if no word timestamps
        if not asr_seg.words:
            seg_counter += 1
            aligned_segments.append(
                _align_single_segment(asr_seg, speaker_turns, seg_counter)
            )
            continue

        # Word-level alignment
        word_alignments: List[Dict[str, Any]] = []
        for word_info in asr_seg.words:
            word_start = word_info["start"]
            word_end = word_info["end"]
            word_text = word_info["word"]

            max_overlap = 0.0
            best_turn: Optional[SpeakerTurn] = None

            for turn in speaker_turns:
                overlap = _calculate_overlap(
                    word_start, word_end, turn.start, turn.end
                )
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_turn = turn

            if max_overlap > 0 and best_turn is not None:
                speaker_id = best_turn.speaker_id
                turn_id = best_turn.id
            else:
                speaker_id = "UNKNOWN"
                turn_id = None

            word_alignments.append(
                {
                    "word": word_text,
                    "start": word_start,
                    "end": word_end,
                    "speaker_id": speaker_id,
                    "turn_id": turn_id,
                    "overlap": max_overlap,
                }
            )

        # Merge consecutive words with the same speaker
        merged_segments = _merge_consecutive_words(word_alignments, asr_seg.id)

        for merged in merged_segments:
            seg_counter += 1
            aligned_segments.append(
                AlignedSegment(
                    id=f"seg_{seg_counter:06d}",
                    start=merged["start"],
                    end=merged["end"],
                    speaker_id=merged["speaker_id"],
                    speaker_label=speaker_label_map.get(
                        merged["speaker_id"], _get_speaker_label(merged["speaker_id"])
                    ),
                    text=merged["text"],
                    confidence=None,
                    source=SegmentSource(
                        asr_segment_id=asr_seg.id,
                        diarization_turn_id=merged["turn_id"],
                        overlap_sec=merged["overlap"],
                    ),
                )
            )

    # Sort by start time
    aligned_segments.sort(key=lambda s: s.start)

    logger.info(
        "Word-level alignment complete: %d segments", len(aligned_segments)
    )
    return aligned_segments
