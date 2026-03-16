# Requirements Document

## Introduction

This document specifies the requirements for a meeting speaker diarization pipeline that processes audio/video files to generate speaker-labeled meeting transcripts. The system combines speaker diarization (identifying who spoke when) with automatic speech recognition (ASR) to produce structured meeting logs in both JSON and Markdown formats. The pipeline is designed for cross-platform operation (Windows with CUDA, macOS with MPS/CPU) and supports phased development from basic functionality to advanced accuracy improvements.

## Glossary

- **Pipeline**: The complete system that processes input media files through diarization, ASR, alignment, and output generation
- **Diarization_Engine**: The component that performs speaker diarization using pyannote-audio
- **ASR_Engine**: The component that performs automatic speech recognition using faster-whisper or whisper
- **Alignment_Module**: The component that matches ASR segments with speaker turns to assign speaker labels
- **Meeting_JSON**: The structured JSON output containing all pipeline metadata, speaker information, and timestamped segments
- **Transcript_Markdown**: The human-readable Markdown output formatted for easy reading
- **Speaker_Turn**: A continuous time interval during which a single speaker is active
- **ASR_Segment**: A continuous time interval containing recognized speech text
- **Aligned_Segment**: A segment with both speech text and assigned speaker label
- **UNKNOWN_Speaker**: A special speaker label assigned when speaker attribution cannot be determined
- **Device_Resolver**: The component that selects the appropriate compute device (CUDA/MPS/CPU)
- **Audio_Extractor**: The component that extracts and converts audio from input files using ffmpeg
- **HF_Token**: Hugging Face authentication token required for accessing pyannote models

## Requirements

### Requirement 1: Audio Extraction from Input Files

**User Story:** As a user, I want to process various audio and video formats, so that I can generate meeting transcripts from any common media file.

#### Acceptance Criteria

1. WHEN a video file (mp4, avi, mkv) is provided, THE Audio_Extractor SHALL extract audio to 16kHz mono PCM WAV format
2. WHEN an audio file (wav, m4a, mp3, flac) is provided, THE Audio_Extractor SHALL convert it to 16kHz mono PCM WAV format if needed
3. IF the input file format is unsupported, THEN THE Pipeline SHALL terminate with a descriptive error message
4. WHERE the user specifies --keep-audio, THE Audio_Extractor SHALL preserve the extracted audio file after processing
5. THE Audio_Extractor SHALL record the extracted audio path, sample rate, and channel count in Meeting_JSON

### Requirement 2: Device Selection and Resolution

**User Story:** As a user, I want the pipeline to automatically select the best available compute device, so that I can run the same command on different platforms without manual configuration.

#### Acceptance Criteria

1. WHEN --device auto is specified, THE Device_Resolver SHALL select devices in priority order: CUDA, MPS, CPU
2. WHEN --device cuda is specified and CUDA is unavailable, THEN THE Pipeline SHALL terminate with a descriptive error message
3. WHEN --device mps is specified and MPS is unavailable, THEN THE Pipeline SHALL terminate with a descriptive error message
4. THE Device_Resolver SHALL record both requested and resolved device names in Meeting_JSON
5. THE Pipeline SHALL use the resolved device for both Diarization_Engine and ASR_Engine

### Requirement 3: Speaker Diarization Execution

**User Story:** As a user, I want to identify who spoke when in a meeting recording, so that I can attribute statements to specific speakers.

#### Acceptance Criteria

1. WHERE --enable-diarization is specified, THE Diarization_Engine SHALL process the audio file to identify speaker turns
2. WHEN HF_Token is not available in environment variables, THEN THE Pipeline SHALL terminate with a clear error message indicating HF_TOKEN is required
3. THE Diarization_Engine SHALL generate Speaker_Turn records with speaker_id, start time, and end time
4. THE Diarization_Engine SHALL assign sequential speaker identifiers (SPEAKER_00, SPEAKER_01, etc.) without limiting the number of speakers
5. THE Diarization_Engine SHALL record the model name, engine name, and HF_Token usage status in Meeting_JSON
6. THE Pipeline SHALL release Diarization_Engine resources before starting ASR_Engine to avoid memory pressure

### Requirement 4: Automatic Speech Recognition

**User Story:** As a user, I want to convert speech to text with timestamps, so that I can read what was said during the meeting.

#### Acceptance Criteria

1. WHEN --asr-engine faster-whisper is specified, THE ASR_Engine SHALL use the faster-whisper implementation
2. WHEN --asr-engine whisper is specified, THE ASR_Engine SHALL use the whisper implementation
3. THE ASR_Engine SHALL generate ASR_Segment records with start time, end time, and recognized text
4. THE ASR_Engine SHALL use the language specified by --language parameter (default: ja)
5. THE ASR_Engine SHALL use the model size specified by --asr-model parameter (default: medium)
6. THE ASR_Engine SHALL record model name, engine name, device, compute type, language, beam_size, best_of, and vad_filter settings in Meeting_JSON
7. THE ASR_Engine SHALL assign sequential segment identifiers (asr_000001, asr_000002, etc.)

### Requirement 5: Speaker-Text Alignment

**User Story:** As a user, I want each speech segment to be labeled with the correct speaker, so that I can understand who said what.

#### Acceptance Criteria

1. FOR EACH ASR_Segment, THE Alignment_Module SHALL calculate temporal overlap with all Speaker_Turn records
2. WHEN maximum overlap is greater than zero, THE Alignment_Module SHALL assign the speaker_id with maximum overlap to the ASR_Segment
3. WHEN maximum overlap is zero for all speakers, THE Alignment_Module SHALL assign speaker_id "UNKNOWN" to the ASR_Segment
4. THE Alignment_Module SHALL generate Aligned_Segment records with segment_id, start, end, speaker_id, speaker_label, and text
5. THE Alignment_Module SHALL record the source ASR_Segment identifier, Speaker_Turn identifier, and overlap duration in each Aligned_Segment
6. THE Alignment_Module SHALL record the alignment method and unit in Meeting_JSON

### Requirement 6: Speaker Registry with Unknown Support

**User Story:** As a user, I want to see all speakers including unidentified ones, so that I have a complete picture of the meeting participants.

#### Acceptance Criteria

1. THE Pipeline SHALL include all identified speakers in the Meeting_JSON speakers array
2. THE Pipeline SHALL include an UNKNOWN_Speaker entry (id: "UNKNOWN", label: "Unknown") in the speakers array
3. FOR EACH speaker entry, THE Pipeline SHALL record both machine-readable id and human-readable label
4. THE Pipeline SHALL generate speaker labels in the format "Speaker N" where N starts from 1

### Requirement 7: Meeting JSON Output Generation

**User Story:** As a developer, I want a structured JSON output with all pipeline data, so that I can reprocess or analyze the results programmatically.

#### Acceptance Criteria

1. THE Pipeline SHALL generate Meeting_JSON conforming to schema version 1.0
2. THE Meeting_JSON SHALL include input metadata (file path, audio properties, duration)
3. THE Meeting_JSON SHALL include pipeline configuration (device, diarization settings, ASR settings, alignment method)
4. THE Meeting_JSON SHALL include the speakers array with all identified speakers and UNKNOWN_Speaker
5. THE Meeting_JSON SHALL include the segments array with all Aligned_Segment records in chronological order
6. THE Meeting_JSON SHALL include artifacts section with raw diarization_turns and asr_segments arrays
7. THE Meeting_JSON SHALL include timing measurements for each pipeline stage (extract, diarization, ASR load, ASR, align, total)
8. THE Meeting_JSON SHALL include creation timestamp in ISO 8601 format with timezone
9. THE Pipeline SHALL save Meeting_JSON to the output directory with filename pattern {input_basename}_meeting.json

### Requirement 8: Transcript Markdown Output Generation

**User Story:** As a user, I want a human-readable meeting transcript, so that I can quickly review what was discussed without parsing JSON.

#### Acceptance Criteria

1. WHERE --format md or --format both is specified, THE Pipeline SHALL generate Transcript_Markdown from Meeting_JSON
2. THE Transcript_Markdown SHALL group segments by speaker_label in chronological order
3. WHEN speaker_label changes, THE Pipeline SHALL insert a heading "### {speaker_label}"
4. FOR EACH segment, THE Pipeline SHALL format the line as "- [HH:MM:SS - HH:MM:SS] {text}"
5. WHEN segment text is empty or whitespace-only, THE Pipeline SHALL skip that segment
6. THE Pipeline SHALL format timestamps with zero-padding (HH:MM:SS format)
7. THE Pipeline SHALL save Transcript_Markdown to the output directory with filename pattern {input_basename}_transcript.md

### Requirement 9: Output Format Control

**User Story:** As a user, I want to control which output formats are generated, so that I can optimize processing time and disk usage.

#### Acceptance Criteria

1. WHEN --format json is specified, THE Pipeline SHALL generate only Meeting_JSON
2. WHEN --format md is specified, THE Pipeline SHALL generate only Transcript_Markdown
3. WHEN --format both is specified, THE Pipeline SHALL generate both Meeting_JSON and Transcript_Markdown
4. THE Pipeline SHALL use "both" as the default format when --format is not specified

### Requirement 10: Benchmark Logging

**User Story:** As a developer, I want to log pipeline runs for performance analysis, so that I can track improvements and regressions over time.

#### Acceptance Criteria

1. WHERE --bench-jsonl is specified, THE Pipeline SHALL append a benchmark record to the specified JSONL file
2. THE benchmark record SHALL include run_id, timestamp, input file, device, model settings, timing data, and optional note
3. WHEN --run-id is not specified, THE Pipeline SHALL generate a run_id from the current timestamp
4. WHERE --note is specified, THE Pipeline SHALL include the note text in the benchmark record

### Requirement 11: Error Handling and Validation

**User Story:** As a user, I want clear error messages when something goes wrong, so that I can quickly identify and fix issues.

#### Acceptance Criteria

1. WHEN a required parameter is missing, THEN THE Pipeline SHALL terminate with a usage message indicating the missing parameter
2. WHEN the input file does not exist, THEN THE Pipeline SHALL terminate with an error message indicating the file path
3. WHEN ffmpeg is not available, THEN THE Pipeline SHALL terminate with an error message indicating ffmpeg is required
4. WHEN a model fails to load, THEN THE Pipeline SHALL terminate with an error message including the model name and failure reason
5. WHEN processing fails mid-pipeline, THEN THE Pipeline SHALL report which stage failed and preserve any intermediate outputs

### Requirement 12: Reproducibility and Metadata

**User Story:** As a researcher, I want complete metadata about how results were generated, so that I can reproduce experiments and validate findings.

#### Acceptance Criteria

1. THE Meeting_JSON SHALL record all model names used (diarization model, ASR model)
2. THE Meeting_JSON SHALL record all significant parameters (language, beam_size, best_of, vad_filter)
3. THE Meeting_JSON SHALL record device information (requested device, resolved device)
4. THE Meeting_JSON SHALL record timing for each pipeline stage with precision of at least 0.1 seconds
5. THE Meeting_JSON SHALL record schema version to enable future format evolution

### Requirement 13: Sequential Processing to Manage Resources

**User Story:** As a user running on limited GPU memory, I want the pipeline to avoid loading multiple models simultaneously, so that processing completes successfully without out-of-memory errors.

#### Acceptance Criteria

1. THE Pipeline SHALL complete Diarization_Engine processing before starting ASR_Engine processing
2. THE Pipeline SHALL release Diarization_Engine model references before loading ASR_Engine models
3. THE Pipeline SHALL NOT execute diarization and ASR in parallel

### Requirement 14: Cross-Platform Compatibility

**User Story:** As a user on different operating systems, I want the same CLI to work on Windows and macOS, so that I can use consistent workflows across platforms.

#### Acceptance Criteria

1. THE Pipeline SHALL execute successfully on Windows with CUDA device
2. THE Pipeline SHALL execute successfully on macOS with MPS device
3. THE Pipeline SHALL execute successfully on macOS with CPU device
4. THE Pipeline SHALL produce Meeting_JSON with identical schema across all platforms
5. THE Pipeline SHALL produce Transcript_Markdown with identical format across all platforms

### Requirement 15: Preservation of Unattributed Segments

**User Story:** As a user, I want to see all recognized speech even when speaker attribution fails, so that I don't lose information from overlapping speech or unclear audio.

#### Acceptance Criteria

1. WHEN an ASR_Segment cannot be attributed to any identified speaker, THE Alignment_Module SHALL assign speaker_id "UNKNOWN"
2. THE Pipeline SHALL include all segments with speaker_id "UNKNOWN" in the Meeting_JSON segments array
3. THE Pipeline SHALL include all segments with speaker_label "Unknown" in the Transcript_Markdown output
4. THE Pipeline SHALL NOT discard or omit segments due to failed speaker attribution

### Requirement 16: Parser and Serializer for Meeting JSON

**User Story:** As a developer, I want to reliably parse and generate Meeting JSON files, so that I can build tools that process pipeline outputs.

#### Acceptance Criteria

1. THE Pipeline SHALL serialize Meeting_JSON conforming to the defined schema structure
2. THE Pipeline SHALL validate that generated Meeting_JSON is valid JSON before writing to disk
3. FOR ALL valid Meeting_JSON objects, parsing the JSON file SHALL produce an equivalent data structure
4. WHEN Meeting_JSON contains invalid UTF-8 or malformed data, THE Pipeline SHALL report a descriptive error

