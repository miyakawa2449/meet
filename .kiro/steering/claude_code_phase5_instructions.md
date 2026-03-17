# Phase 5 Implementation Instructions for Claude Code (Windows CUDA Environment)

## Context

You are implementing Phase 5 (CUDA Environment Verification) of the meeting speaker diarization pipeline. Phase 1-4 are already complete with all 56 tests passing on macOS.

## Current Status

- **Completed**: Phase 1-4 (Basic Pipeline, Markdown, Word-level Alignment, macOS MPS/CPU)
- **Target Environment**: Windows with CUDA
- **macOS Results**: 56/56 tests passing, MPS 12x faster than CPU for diarization
- **Project Structure**: Modular architecture in `src/meeting_pipeline/`

## Phase 5 Tasks Overview

Phase 5 focuses on verifying the pipeline works correctly in Windows CUDA environment and ensuring cross-platform consistency between Windows and macOS.

### Task 19.1: CUDA Environment Pipeline Execution Test

**Objective**: Verify the pipeline runs successfully on Windows with CUDA device.

**Implementation Steps**:

1. **Verify CUDA Availability**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   print(f"CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
   ```

2. **Test CUDA Device**:
   - Run the pipeline with `--device cuda` on the test video
   - Verify output JSON and Markdown are generated correctly
   - Check that CUDA device is properly detected and used
   - Command: `python meeting_pipeline.py --input-file temp/202602017_short_test.mp4 --device cuda --output-dir output/test_cuda`

3. **Test Auto Device Selection**:
   - Run with `--device auto` and verify CUDA is selected (highest priority)
   - Command: `python meeting_pipeline.py --input-file temp/202602017_short_test.mp4 --device auto --output-dir output/test_auto`

4. **Test CPU Device (Fallback Verification)**:
   - Run with `--device cpu` to verify CPU still works on Windows
   - Command: `python meeting_pipeline.py --input-file temp/202602017_short_test.mp4 --device cpu --output-dir output/test_cpu`

**Expected Behavior**:
- CUDA device should be detected and used successfully
- Auto selection should choose CUDA (highest priority)
- compute_type should be "float16" for CUDA
- If float16 fails, should automatically retry with int8
- All outputs should conform to Meeting JSON Schema v1.0

**Files to Review**:
- `src/meeting_pipeline/device.py` - Device resolution logic
- `src/meeting_pipeline/asr.py` - ASR engine with CUDA support and float16 retry logic
- `src/meeting_pipeline/diarization.py` - Diarization engine with CUDA support

### Task 19.2: CUDA Environment Output Consistency Verification

**Objective**: Ensure Windows CUDA outputs are structurally consistent with macOS CPU/MPS outputs.

**Verification Steps**:

1. **Compare JSON Schema Structure**:
   - Run the same input file on Windows CUDA and compare with macOS results
   - Verify all required fields are present
   - Check that field types and formats are identical
   - Confirm `schema_version` is "1.0"

2. **Verify Device Metadata**:
   - Check `pipeline.device.resolved` is "cuda"
   - Verify `pipeline.asr.device` is "cuda"
   - Confirm `pipeline.asr.compute_type` is "float16" (or "int8" if fallback occurred)
   - Verify `pipeline.diarization` records CUDA usage

3. **Markdown Format Validation**:
   - Verify Markdown structure matches macOS output
   - Check timestamp formatting (HH:MM:SS)
   - Confirm speaker grouping is consistent

4. **Content Comparison** (optional):
   - Compare speaker count and segment count
   - Note: Exact text may differ slightly due to model precision differences
   - Focus on structural consistency, not exact content match

**Comparison Checklist**:
```python
# Load both JSON files
with open("output/test_cuda/202602017_short_test_meeting.json") as f:
    cuda_json = json.load(f)

# Verify structure
assert cuda_json["schema_version"] == "1.0"
assert cuda_json["pipeline"]["device"]["resolved"] == "cuda"
assert cuda_json["pipeline"]["asr"]["device"] == "cuda"
assert cuda_json["pipeline"]["asr"]["compute_type"] in ["float16", "int8"]

# Verify all required sections exist
required_sections = ["input", "pipeline", "speakers", "segments", "artifacts", "timing"]
for section in required_sections:
    assert section in cuda_json
```

**Files to Review**:
- Generated JSON files in `output/test_cuda/`, `output/test_auto/`, `output/test_cpu/`
- Compare with macOS results (if available for reference)

### Task 19.3: CUDA Environment Performance Measurement

**Objective**: Measure and record performance metrics for CUDA environment.

**Measurement Steps**:

1. **Run Benchmark Test**:
   ```bash
   python meeting_pipeline.py \
     --input-file temp/202602017_short_test.mp4 \
     --device cuda \
     --output-dir output/bench_cuda \
     --bench-jsonl bench/phase5_cuda.jsonl \
     --run-id phase5_cuda_test \
     --note "Phase 5 CUDA verification"
   ```

2. **Collect Timing Data**:
   - Extract timing information from the generated JSON
   - Record: extract_sec, diarization_sec, asr_load_sec, asr_sec, align_sec, total_sec
   - Compare with macOS CPU/MPS timings (if available)

3. **Expected Performance**:
   - CUDA should be significantly faster than CPU
   - Diarization on CUDA should be comparable to or faster than macOS MPS
   - ASR on CUDA should be much faster than CPU

4. **Document Results**:
   - Create a simple summary of timing comparisons
   - Note any unexpected performance characteristics
   - Record GPU memory usage if possible

**Performance Comparison Table** (example):

| Device | Platform | Diarization | ASR | Total |
|--------|----------|-------------|-----|-------|
| CPU | macOS | 207.9s | ~120s | ~330s |
| MPS | macOS | 17.7s | ~120s (CPU) | ~140s |
| CUDA | Windows | ? | ? | ? |

**Files to Review**:
- `bench/phase5_cuda.jsonl` - Benchmark log
- Generated JSON timing section

## Important Notes for Windows Environment

### Environment Setup

Before running any tests or pipeline commands on Windows:

**PowerShell**:
```powershell
.\venv\Scripts\Activate.ps1
$env:HF_TOKEN = (Get-Content .env | Select-String "HF_TOKEN").ToString().Split("=")[1]
```

**Command Prompt**:
```cmd
venv\Scripts\activate.bat
set HF_TOKEN=<your_token_from_.env>
```

### Path Differences

Windows uses backslashes for paths, but Python handles forward slashes correctly:
- Use: `temp/202602017_short_test.mp4` (works on both platforms)
- Avoid: `temp\202602017_short_test.mp4` (Windows only)

### CUDA-Specific Checks

1. **Verify CUDA Installation**:
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   ```

2. **Check GPU Memory**:
   ```python
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_name(0)}")
       print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
   ```

3. **Monitor Memory Usage**:
   - Watch for out-of-memory errors
   - Verify memory is released after diarization (before ASR)
   - Check `torch.cuda.empty_cache()` is called correctly

### Known CUDA Considerations

1. **float16 Support**:
   - Most modern GPUs support float16
   - If float16 fails, the code automatically retries with int8
   - This is already implemented in `asr.py`

2. **Memory Management**:
   - Diarization model is released before ASR (sequential execution)
   - `torch.cuda.empty_cache()` is called after diarization
   - This prevents GPU memory exhaustion

3. **Compute Type**:
   - CUDA uses float16 by default (optimal for GPU)
   - Automatic fallback to int8 if float16 fails
   - This is already implemented

## Testing

### Run Existing Tests on Windows

```bash
# Activate venv
venv\Scripts\activate.bat

# Run all tests
pytest tests/ -v

# Expected: 56/56 tests pass
```

### Verify Test Compatibility

- All tests should pass on Windows without modification
- Tests use mocking and don't depend on actual CUDA hardware
- If any tests fail, investigate platform-specific issues

## Success Criteria

Phase 5 is complete when:

1. ✅ Pipeline runs successfully on Windows with CUDA device
2. ✅ Auto device selection correctly chooses CUDA on Windows
3. ✅ compute_type is float16 for CUDA (or int8 if fallback)
4. ✅ Output JSON and Markdown are structurally consistent with macOS
5. ✅ All 56 existing tests pass on Windows
6. ✅ Performance metrics are recorded in benchmark JSONL
7. ✅ GPU memory is properly managed (no OOM errors)
8. ✅ Device metadata is correctly recorded in output JSON

## Task Checklist

Use this checklist to track your progress:

- [ ] Task 19.1: Run end-to-end tests on Windows with CUDA and CPU
- [ ] Task 19.2: Verify output consistency between Windows CUDA and macOS
- [ ] Task 19.3: Measure and record CUDA performance metrics
- [ ] Task 20: Verify all 56 tests pass on Windows and Phase 5 is complete

## Expected Outcomes

### Performance Expectations

- **Diarization on CUDA**: Should be significantly faster than CPU, comparable to macOS MPS
- **ASR on CUDA**: Should be much faster than CPU (3-5x speedup expected)
- **Total Pipeline**: Should complete in ~30-60 seconds for 5-minute video (vs ~330s on CPU)

### Output Validation

- JSON schema structure identical to macOS
- Markdown format identical to macOS
- Device metadata correctly shows "cuda"
- Timing measurements are reasonable

## Reporting

After completing Phase 5, report:

1. **CUDA Detection**:
   - CUDA availability status
   - GPU name and memory
   - PyTorch CUDA version

2. **Test Results**:
   - Pipeline execution results (CUDA/Auto/CPU)
   - Output file verification
   - Existing test results (56/56 expected)

3. **Performance Metrics**:
   - Timing comparison table (CUDA vs macOS CPU/MPS)
   - Speedup factors
   - Any performance issues

4. **Consistency Verification**:
   - Schema structure comparison results
   - Any differences found between platforms
   - Metadata accuracy

5. **Issues Encountered** (if any):
   - CUDA-specific errors
   - Memory issues
   - Platform differences

## Git Workflow

After completing Phase 5 on Windows:

```bash
git add .
git commit -m "Phase 5完了: Windows CUDA環境での検証とパフォーマンス測定"
git push
```

## Troubleshooting

### Common CUDA Issues

1. **CUDA Not Available**:
   - Verify NVIDIA GPU is present
   - Check CUDA toolkit is installed
   - Ensure PyTorch CUDA version matches CUDA toolkit

2. **Out of Memory**:
   - Verify diarization model is released before ASR
   - Check `torch.cuda.empty_cache()` is called
   - Try smaller ASR model (tiny or base instead of medium)

3. **float16 Errors**:
   - The code should automatically retry with int8
   - Check logs for retry messages
   - Verify fallback logic is working

### Windows-Specific Issues

1. **Path Issues**:
   - Use forward slashes in Python code
   - Check temp directory creation

2. **ffmpeg**:
   - Verify ffmpeg is in PATH
   - Test: `ffmpeg -version`

3. **HF_TOKEN**:
   - Verify environment variable is set
   - Test: `echo %HF_TOKEN%` (cmd) or `$env:HF_TOKEN` (PowerShell)

## Notes

- This is a verification phase, not a development phase
- No code changes should be needed (unless bugs are found)
- Focus on testing, measurement, and validation
- Document any platform-specific issues discovered
- All tests should pass without modification
