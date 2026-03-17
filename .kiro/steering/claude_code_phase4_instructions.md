# Phase 4 Implementation Instructions for Claude Code

## Context

You are implementing Phase 4 (Cross-platform Optimization) of the meeting speaker diarization pipeline. Phase 1, 2, and 3 are already complete and all 49 tests are passing.

## Current Status

- **Completed**: Phase 1 (Basic Pipeline), Phase 2 (Markdown Generation), Phase 3 (Word-level Alignment)
- **Current Environment**: macOS
- **Working**: CPU device testing already successful
- **Project Structure**: Modular architecture in `src/meeting_pipeline/`

## Phase 4 Tasks Overview

Phase 4 focuses on verifying and optimizing cross-platform compatibility, particularly for macOS MPS (Metal Performance Shaders) and CPU devices.

### Task 17.1: macOS MPS/CPU End-to-End Testing

**Objective**: Verify the pipeline works correctly on macOS with both MPS and CPU devices.

**Implementation Steps**:

1. **Test MPS Device (if available)**:
   - Run the pipeline with `--device mps` on the test video
   - Verify output JSON and Markdown are generated correctly
   - Check that MPS device is properly detected and used
   - Command: `python meeting_pipeline.py --input-file temp/202602017_short_test.mp4 --device mps --output-dir output/test_mps`

2. **Test CPU Device**:
   - Run the pipeline with `--device cpu` on the test video
   - Verify output JSON and Markdown are generated correctly
   - Command: `python meeting_pipeline.py --input-file temp/202602017_short_test.mp4 --device cpu --output-dir output/test_cpu`

3. **Test Auto Device Selection**:
   - Run with `--device auto` and verify correct device priority (CUDA > MPS > CPU)
   - On macOS without CUDA, should select MPS if available, otherwise CPU
   - Command: `python meeting_pipeline.py --input-file temp/202602017_short_test.mp4 --device auto --output-dir output/test_auto`

**Expected Behavior**:
- MPS device should work if available on the system
- CPU device should always work as fallback
- Auto selection should follow priority: CUDA > MPS > CPU
- All outputs should conform to Meeting JSON Schema v1.0

**Files to Review**:
- `src/meeting_pipeline/device.py` - Device resolution logic
- `src/meeting_pipeline/asr.py` - ASR engine with device handling
- `src/meeting_pipeline/diarization.py` - Diarization engine

### Task 17.2: Optimize compute_type Auto-Selection Logic

**Objective**: Ensure compute_type is optimally selected for each device type.

**Current Logic** (in `src/meeting_pipeline/asr.py`):
```python
def _determine_compute_type(device: str) -> str:
    """Determine compute type based on device."""
    if device in ("cuda", "mps"):
        return "float16"
    return "int8"
```

**Optimization Considerations**:

1. **MPS Device**:
   - faster-whisper does NOT support MPS directly
   - Current code already has fallback: when `asr_engine="faster-whisper"` and `device="mps"`, it falls back to CPU
   - Verify this fallback works correctly
   - Consider logging a clear warning message

2. **CPU Device**:
   - Current setting: `int8` for CPU
   - Verify this provides good balance of speed and accuracy
   - Consider testing `float32` vs `int8` performance

3. **CUDA Device**:
   - Current setting: `float16` for CUDA
   - This is optimal for most GPUs

**Implementation Steps**:

1. Review the current `_determine_compute_type()` function
2. Verify the MPS → CPU fallback logic in `run_asr()` is working correctly
3. Add clearer logging for device fallback scenarios
4. Test with different compute_type values if needed
5. Document the rationale for each device's compute_type choice

**Files to Modify**:
- `src/meeting_pipeline/asr.py` - compute_type determination and device fallback

### Task 17.3: Verify Cross-Platform Consistency

**Objective**: Ensure the output format and schema are consistent across different platforms and devices.

**Verification Steps**:

1. **Run on Multiple Devices**:
   - Execute the same input file on CPU and MPS (if available)
   - Compare the generated JSON files

2. **Schema Structure Validation**:
   - Verify all required fields are present in both outputs
   - Check that field types and formats are identical
   - Confirm `schema_version` is "1.0" in all cases

3. **Markdown Format Validation**:
   - Verify Markdown structure is identical
   - Check timestamp formatting (HH:MM:SS)
   - Confirm speaker grouping is consistent

4. **Metadata Recording**:
   - Verify `pipeline.device.resolved` correctly reflects the actual device used
   - Check `pipeline.asr.device` and `pipeline.asr.compute_type` are accurate
   - Confirm timing measurements are present and reasonable

**Comparison Script** (optional helper):
```python
import json

def compare_meeting_jsons(file1, file2):
    """Compare two Meeting JSON files for structural consistency."""
    with open(file1) as f1, open(file2) as f2:
        json1 = json.load(f1)
        json2 = json.load(f2)
    
    # Compare schema structure (not values, as timing/device will differ)
    assert json1.keys() == json2.keys()
    assert json1["schema_version"] == json2["schema_version"]
    assert json1["pipeline"].keys() == json2["pipeline"].keys()
    # ... more structural checks
```

**Files to Review**:
- `src/meeting_pipeline/output.py` - JSON generation logic
- Generated output files in `output/` directory

### Task 17.4: Create Unit Tests for Cross-Platform Verification

**Objective**: Add unit tests that verify platform-specific logic works correctly.

**Test Cases to Add** (in `tests/test_meeting_pipeline.py`):

1. **Test MPS Device Fallback for faster-whisper**:
```python
def test_faster_whisper_mps_fallback_to_cpu(monkeypatch):
    """Test that faster-whisper falls back to CPU when MPS is requested."""
    # Mock torch.backends.mps.is_available() to return True
    # Mock faster-whisper to verify it's called with device="cpu"
    # Verify warning is logged about MPS fallback
```

2. **Test compute_type Selection**:
```python
def test_compute_type_selection_cuda():
    """Test compute_type is float16 for CUDA."""
    assert _determine_compute_type("cuda") == "float16"

def test_compute_type_selection_mps():
    """Test compute_type is float16 for MPS."""
    assert _determine_compute_type("mps") == "float16"

def test_compute_type_selection_cpu():
    """Test compute_type is int8 for CPU."""
    assert _determine_compute_type("cpu") == "int8"
```

3. **Test Device Resolution on macOS**:
```python
def test_device_resolution_macos_no_cuda(monkeypatch):
    """Test device resolution on macOS without CUDA."""
    # Mock torch.cuda.is_available() to return False
    # Mock torch.backends.mps.is_available() to return True
    device_info = resolve_device("auto")
    assert device_info.resolved == "mps"

def test_device_resolution_macos_no_mps(monkeypatch):
    """Test device resolution on macOS without MPS."""
    # Mock both CUDA and MPS as unavailable
    device_info = resolve_device("auto")
    assert device_info.resolved == "cpu"
```

4. **Test Cross-Platform Schema Consistency**:
```python
def test_meeting_json_schema_structure():
    """Test Meeting JSON has all required fields regardless of device."""
    # Generate Meeting JSON with mocked data
    # Verify all required fields exist
    # This validates Property 27: Cross-Platform Schema Consistency
```

**Files to Modify**:
- `tests/test_meeting_pipeline.py` - Add new test functions

## Important Notes

### Environment Setup

Before running any tests or pipeline commands:
```bash
source venv/bin/activate
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)
```

### Test Execution

After implementation:
```bash
# Run all tests
pytest tests/ -v

# Run only Phase 4 related tests (if you add markers)
pytest tests/ -v -k "mps or cpu or compute_type or cross_platform"
```

### MPS Availability Check

To check if MPS is available on the current macOS system:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

### Known Constraints

1. **faster-whisper + MPS**: faster-whisper does NOT support MPS device directly. The code already handles this by falling back to CPU. This is correct behavior.

2. **pyannote-audio + MPS**: pyannote-audio should work with MPS on macOS. Test this during Task 17.1.

3. **Test Video**: Use `temp/202602017_short_test.mp4` (5 minutes, 324 seconds) for faster testing.

## Success Criteria

Phase 4 is complete when:

1. ✅ Pipeline runs successfully on macOS with MPS device (if available)
2. ✅ Pipeline runs successfully on macOS with CPU device
3. ✅ Auto device selection works correctly on macOS
4. ✅ compute_type is optimally selected for each device
5. ✅ Output JSON and Markdown are consistent across devices
6. ✅ All existing tests (49 tests) still pass
7. ✅ New unit tests for cross-platform logic are added and passing
8. ✅ Device fallback scenarios are properly logged

## Task Checklist

Use this checklist to track your progress:

- [ ] Task 17.1: Run end-to-end tests on macOS with MPS (if available) and CPU
- [ ] Task 17.2: Review and optimize compute_type selection logic
- [ ] Task 17.3: Verify cross-platform consistency by comparing outputs
- [ ] Task 17.4: Add unit tests for device resolution, compute_type selection, and MPS fallback
- [ ] Task 18: Verify all 49+ tests pass and Phase 4 is complete

## Git Workflow

After completing Phase 4:
```bash
git add .
git commit -m "Phase 4完了: macOS MPS/CPU対応とクロスプラットフォーム最適化"
git push
```

## Questions or Issues?

If you encounter any issues:
1. Check that venv is activated and HF_TOKEN is set
2. Verify torch and faster-whisper are installed correctly
3. Check MPS availability with `torch.backends.mps.is_available()`
4. Review logs for device fallback warnings
5. Ensure all 49 existing tests still pass before adding new tests
