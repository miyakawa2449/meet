# Phase 4 テスト実装依頼

## あなたの役割

あなたは **テスト担当（Codex）** です。Phase 4 の実装が完了したので、Task 17.4 のユニットテストを作成してください。

## 背景

Claude Code が Phase 4（macOS MPS/CPU対応とクロスプラットフォーム最適化）の実装を完了しました。

### 実装完了内容

1. **device.py**: デバイス検出ログの強化
2. **asr.py**: compute_type最適化と耐障害性（CUDA float16失敗時のint8リトライ機能）
3. **diarization.py**: MPSメモリ管理（torch.mps.empty_cache()追加）

### E2E検証結果

- CPU デバイス: 正常動作 ✅
- MPS デバイス: 正常動作（diarization が12倍高速化）✅
- Auto 選択: MPS → CPU フォールバック正常動作 ✅
- 既存49テスト: 全合格 ✅

## 実装タスク: Task 17.4

`tests/test_meeting_pipeline.py` に以下のユニットテストを追加してください。

### 1. compute_type 選択ロジックのテスト

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

### 2. デバイス解決ロジックのテスト（macOS環境）

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

### 3. faster-whisper MPS フォールバックのテスト

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

### 4. クロスプラットフォーム スキーマ一貫性のテスト

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

## 重要な注意事項

1. **既存テストを壊さない**: 全49テストが引き続き合格すること
2. **モックの使用**: 実際のモデルロードを避け、高速にテスト実行
3. **import パス**: `from src.meeting_pipeline.xxx import yyy` の形式を使用
4. **_determine_compute_type**: この関数は private だが、テストのために import 可能

## テスト実行

実装後、以下のコマンドでテストを実行してください:

```bash
# 環境準備
source venv/bin/activate

# 全テスト実行
pytest tests/ -v

# Phase 4 関連テストのみ
pytest tests/ -v -k "compute_type or device_resolution or mps or schema_structure"
```

## 成功基準

- [ ] 新規テスト4件が全て合格
- [ ] 既存テスト49件が引き続き合格
- [ ] 合計53件以上のテストが合格

## 完了後の報告

テスト実装完了後、以下を報告してください:

1. 追加したテスト数
2. 全テストの実行結果（合格数/総数）
3. テスト実行時間
4. 気づいた点や改善提案

よろしくお願いします。
