# Phase 4 Test Implementation Instructions for Codex

## Context

Claude Code has completed Phase 4 implementation (macOS MPS/CPU support and cross-platform optimization). Your task is to create unit tests for Task 17.4 to verify the platform-specific logic.

## Implementation Summary by Claude Code

### Changes Made

1. **device.py**: Enhanced device detection logging
   - Added CUDA/MPS/platform information logging
   - Added auto-selection result logging

2. **asr.py**: Optimized compute_type and fault tolerance
   - Documented device-specific compute_type selection rationale
   - Added automatic retry with int8 when CUDA float16 fails
   - Clearer logging for MPS→CPU fallback (explicitly mentions int8)
   - Added logging after ASR device/compute_type determination

3. **diarization.py**: MPS memory management
   - Added torch.mps.empty_cache() after MPS usage
   - Added device information to release logs

### E2E Verification Results

| Test | Device | Diarization | ASR | Result |
|------|--------|-------------|-----|--------|
| CPU | cpu | CPU (207.9s) | CPU/int8 | ✅ OK |
| Auto | mps | MPS (17.7s) | CPU/int8 (fallback) | ✅ OK |

- MPS accelerated diarization by 12x
- Output JSON structure, schema version, and fields are identical across devices
- All 49 existing tests pass

## Your Task: Task 17.4 - Create Unit Tests

Add the following unit tests to `tests/test_meeting_pipeline.py`:

### Test Group 1: compute_type Selection Logic

Test the `_determine_compute_type()` function in `src/meeting_pipeline/asr.py`:

```python
def test_compute_type_selection_cuda():
    """Test compute_type is float16 for CUDA."""
    from src.meeting_pipeline.asr import _determine_compute_type
    assert _determine_compute_type("cuda") == "float16"

def test_compute_type_selection_mps():
    """Test compute_type is float16 for MPS."""
    from src.meeting_pipeline.asr import _determine_compute_type
    assert _determine_compute_type("mps") == "float16"

def test_compute_type_selection_cpu():
    """Test compute_type is int8 for CPU."""
    from src.meeting_pipeline.asr import _determine_compute_type
    assert _determine_compute_type("cpu") == "int8"
```

### Test Group 2: Device Resolution on macOS

Test the `resolve_device()` function in `src/meeting_pipeline/device.py`:

```python
def test_device_resolution_macos_no_cuda(monkeypatch):
    """Test device resolution on macOS without CUDA."""
    import torch
    from src.meeting_pipeline.device import resolve_device
    
    # Mock torch to simulate macOS with MPS
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    
    device_info = resolve_device("auto")
    assert device_info.resolved == "mps"

def test_device_resolution_macos_no_mps(monkeypatch):
    """Test device resolution on macOS without MPS."""
    import torch
    from src.meeting_pipeline.device import resolve_device
    
    # Mock torch to simulate macOS without CUDA and MPS
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    
    device_info = resolve_device("auto")
    assert device_info.resolved == "cpu"
```

### Test Group 3: faster-whisper MPS Fallback

Test that faster-whisper correctly falls back to CPU when MPS is requested:

```python
def test_faster_whisper_mps_fallback_to_cpu(monkeypatch, caplog):
    """Test that faster-whisper falls back to CPU when MPS is requested."""
    import logging
    from src.meeting_pipeline.asr import run_asr
    from src.meeting_pipeline.models import PipelineConfig
    
    # Setup logging capture
    caplog.set_level(logging.WARNING)
    
    # Create mock config
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
        align_unit="asr_segment"
    )
    
    # Mock the actual ASR execution to avoid loading models
    def mock_run_faster_whisper(audio_path, device, compute_type, config):
        from src.meeting_pipeline.models import ASRResult
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
            asr_load_sec=0.0
        )
    
    monkeypatch.setattr(
        "src.meeting_pipeline.asr._run_faster_whisper",
        mock_run_faster_whisper
    )
    
    # Run ASR with MPS device
    result = run_asr("dummy_audio.wav", "mps", config)
    
    # Verify fallback to CPU
    assert result.device == "cpu"
    assert result.compute_type == "int8"
    
    # Verify warning was logged
    assert "faster-whisper does not support MPS" in caplog.text
```

### Test Group 4: Cross-Platform Schema Consistency

Test that Meeting JSON has all required fields regardless of device (validates Property 27):

```python
def test_meeting_json_schema_structure():
    """Test Meeting JSON has all required fields regardless of device."""
    from src.meeting_pipeline.models import (
        MeetingJSON, InputInfo, AudioInfo, PipelineInfo,
        DeviceInfo, DiarizationConfig, ASRConfig, AlignConfig,
        Speaker, Artifacts, Timing
    )
    from src.meeting_pipeline.output import _dataclass_to_dict
    from datetime import datetime
    
    # Create minimal MeetingJSON structure
    meeting = MeetingJSON(
        schema_version="1.0",
        created_at=datetime.now().isoformat(),
        title="",
        input=InputInfo(
            path="test.mp4",
            audio=AudioInfo(path="test.wav", sample_rate=16000, channels=1),
            duration_sec=100.0
        ),
        pipeline=PipelineInfo(
            device=DeviceInfo(requested="auto", resolved="cpu"),
            diarization=DiarizationConfig(
                enabled=False,
                engine="",
                model="",
                hf_token_used=False
            ),
            asr=ASRConfig(
                engine="faster-whisper",
                model="tiny",
                device="cpu",
                compute_type="int8",
                language="ja",
                beam_size=1,
                best_of=1,
                vad_filter=False
            ),
            align=AlignConfig(method="max_overlap", unit="asr_segment")
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
            total_sec=0.0
        ),
        notes=""
    )
    
    # Convert to dict
    meeting_dict = _dataclass_to_dict(meeting)
    
    # Verify all required top-level fields exist
    required_fields = [
        "schema_version", "created_at", "title", "input",
        "pipeline", "speakers", "segments", "artifacts",
        "timing", "notes"
    ]
    for field in required_fields:
        assert field in meeting_dict, f"Missing required field: {field}"
    
    # Verify schema version
    assert meeting_dict["schema_version"] == "1.0"
    
    # Verify pipeline structure
    assert "device" in meeting_dict["pipeline"]
    assert "diarization" in meeting_dict["pipeline"]
    assert "asr" in meeting_dict["pipeline"]
    assert "align" in meeting_dict["pipeline"]
    
    # Verify device info
    assert "requested" in meeting_dict["pipeline"]["device"]
    assert "resolved" in meeting_dict["pipeline"]["device"]
```

## Implementation Guidelines

### Import Paths

Use the following import pattern:
```python
from src.meeting_pipeline.asr import _determine_compute_type, run_asr
from src.meeting_pipeline.device import resolve_device
from src.meeting_pipeline.models import PipelineConfig, ASRResult
from src.meeting_pipeline.output import _dataclass_to_dict
```

### Mocking Strategy

- Mock `torch.cuda.is_available()` and `torch.backends.mps.is_available()` for device tests
- Mock `_run_faster_whisper()` to avoid loading actual models
- Use `caplog` fixture to capture and verify log messages
- Use `monkeypatch` fixture for all mocking

### Test Naming Convention

- Prefix: `test_`
- Descriptive names that explain what is being tested
- Examples: `test_compute_type_selection_cuda`, `test_device_resolution_macos_no_cuda`

## Testing

### Environment Setup

```bash
source venv/bin/activate
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Phase 4 tests only
pytest tests/ -v -k "compute_type or device_resolution or mps or schema_structure"

# With coverage
pytest tests/ --cov=src.meeting_pipeline --cov-report=term
```

## Success Criteria

- [ ] 4 new test functions added (or more if you split them further)
- [ ] All new tests pass
- [ ] All 49 existing tests still pass
- [ ] Total: 53+ tests passing
- [ ] No import errors
- [ ] No deprecation warnings

## Expected Test Count

- Existing tests: 49
- New tests (minimum): 7
  - 3 compute_type tests
  - 2 device resolution tests
  - 1 MPS fallback test
  - 1 schema structure test
- Total: 56+ tests

## Validation Checklist

Before reporting completion:

- [ ] Run `pytest tests/ -v` and verify all tests pass
- [ ] Check that test execution time is reasonable (< 5 seconds for new tests)
- [ ] Verify no warnings or errors in test output
- [ ] Confirm test coverage includes all Phase 4 changes

## Reporting

After completion, report:

1. Number of tests added
2. Total test count (passing/total)
3. Test execution time
4. Any issues encountered
5. Suggestions for improvement (optional)

## Notes

- The `_determine_compute_type()` function is private (starts with `_`) but can be imported for testing
- Use `caplog.set_level(logging.WARNING)` to capture warning logs
- Mock functions should return appropriate types (e.g., `ASRResult` for ASR mocks)
- Keep tests fast by avoiding actual model loading
