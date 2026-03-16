# Testing Guide for Codex

## 概要

このドキュメントは、会議録AIパイプライン（meeting_pipeline.py）の検証とテストコード作成を担当するCodexへの指示書です。

Claude Codeが Phase 1（タスク1〜9）とPhase 2（タスク11〜13）の実装を完了しました。あなたの役割は、実装の検証とテストコード作成です。

## プロジェクト情報

- **実装ファイル**: `meeting_pipeline.py`
- **テストファイル作成先**: `tests/test_meeting_pipeline.py`
- **テストフレームワーク**: pytest + Hypothesis（Property-Based Testing）

## 参照ドキュメント

- **要件**: `.kiro/specs/meeting-speaker-diarization-pipeline/requirements.md`
- **設計**: `.kiro/specs/meeting-speaker-diarization-pipeline/design.md`
- **タスク**: `.kiro/specs/meeting-speaker-diarization-pipeline/tasks.md`
- **実装**: `meeting_pipeline.py`

## 実装の検証結果（Claude Codeによる事前チェック）

### ✅ 完了している項目

- タスク1〜9（Phase 1）の全実装
- タスク11〜13（Phase 2）の実装も含まれている
- JSON Schema v1.0準拠
- UNKNOWN話者の保持
- 順次実行（Diarization → モデル解放 → ASR）
- クロスプラットフォーム対応
- エラーハンドリング（exit code 1〜4）

### ⚠️ 確認が必要な項目

以下の項目について、テストコード作成時に確認してください：

1. **asr_load_sec の記録**
   - `timing.asr_load_sec` が正しく設定されているか確認
   - `_run_faster_whisper` と `_run_whisper` 内で計算されているが、Timingオブジェクトに反映されていない可能性

2. **型ヒントの一貫性**
   - 全ての関数に適切な型ヒントが付いているか確認

3. **エッジケースの動作**
   - 話者分離なし（--enable-diarization未指定）の場合の動作
   - 空のASRセグメントの処理
   - 重複ゼロの場合のUNKNOWN割り当て

## あなたのタスク

### Phase 1: ユニットテストとプロパティテストの作成

以下のテストタスクを実装してください（tasks.mdの`*`付きタスク）：

#### 1. コア型定義のテスト（タスク1.1）

**Property 28: JSON Serialization Round-Trip**
- 全てのdataclassがJSON化→パース→復元で同じ構造になることを検証
- Hypothesisを使用してランダムなデータで検証

```python
from hypothesis import given, strategies as st
import json

@given(
    start=st.floats(min_value=0, max_value=3600),
    end=st.floats(min_value=0, max_value=3600),
)
def test_json_serialization_round_trip(start, end):
    """Property 28: JSON Serialization Round-Trip"""
    # Test implementation
    pass
```

#### 2. CLI Parserのテスト（タスク2.3）

- 必須パラメータ欠落時のエラーハンドリング
- 無効なパラメータ値のエラーハンドリング
- デフォルト値の確認

```python
def test_cli_parser_missing_input_file():
    """Test error handling for missing input file"""
    with pytest.raises(SystemExit) as exc_info:
        parse_args([])
    assert exc_info.value.code != 0

def test_cli_parser_invalid_device():
    """Test error handling for invalid device"""
    # Test implementation
    pass
```

#### 3. Device Resolverのテスト（タスク3.3, 3.4）

**Property 4: Device Selection Priority**
- auto指定時にCUDA > MPS > CPUの優先順位が守られることを検証

```python
def test_device_selection_priority():
    """Property 4: Device Selection Priority"""
    # Mock torch availability and test priority
    pass

def test_device_resolver_cuda_unavailable():
    """Test error handling when CUDA requested but unavailable"""
    # Test implementation
    pass
```

#### 4. Audio Extractorのテスト（タスク4.4, 4.5）

**Property 1: Audio Extraction Format Consistency**
- 抽出された音声が16kHz, mono, PCM WAVであることを検証

```python
def test_audio_extraction_format_consistency():
    """Property 1: Audio Extraction Format Consistency"""
    # Create test audio and verify output format
    pass

def test_audio_extractor_unsupported_format():
    """Test error handling for unsupported format"""
    # Test implementation
    pass

def test_audio_extractor_keep_audio_flag():
    """Test keep-audio flag behavior"""
    # Test implementation
    pass
```

#### 5. Diarization Engineのテスト（タスク5.5）

- HF_TOKEN未設定時のエラーハンドリング
- 話者ID割り当てロジック（SPEAKER_00, SPEAKER_01, ...）

```python
def test_diarization_missing_hf_token():
    """Test error handling for missing HF_TOKEN"""
    # Test implementation
    pass

def test_diarization_speaker_id_assignment():
    """Test sequential speaker ID assignment"""
    # Test implementation
    pass
```

#### 6. ASR Engineのテスト（タスク6.4）

- ASRセグメントID生成ロジック（asr_000001, asr_000002, ...）
- パラメータ適用の確認

```python
def test_asr_segment_id_generation():
    """Test sequential ASR segment ID generation"""
    # Test implementation
    pass

def test_asr_parameter_application():
    """Test that ASR parameters are correctly applied"""
    # Test implementation
    pass
```

#### 7. Alignment Moduleのテスト（タスク7.4, 7.5, 7.6）

**Property 11: Alignment Overlap Calculation**
- 重複計算が正しく行われることを検証

**Property 12: UNKNOWN Speaker Assignment**
- 重複ゼロの場合にUNKNOWNが割り当てられることを検証

```python
@given(
    seg_start=st.floats(min_value=0, max_value=100),
    seg_end=st.floats(min_value=0, max_value=100),
    turn_start=st.floats(min_value=0, max_value=100),
    turn_end=st.floats(min_value=0, max_value=100),
)
def test_alignment_overlap_calculation(seg_start, seg_end, turn_start, turn_end):
    """Property 11: Alignment Overlap Calculation"""
    # Test implementation
    pass

def test_alignment_unknown_assignment_zero_overlap():
    """Property 12: UNKNOWN Speaker Assignment"""
    # Test zero overlap case
    pass

def test_alignment_max_overlap_selection():
    """Test that maximum overlap speaker is selected"""
    # Test implementation
    pass
```

#### 8. JSON Generatorのテスト（タスク8.6, 8.7, 8.8）

**Property 14: Speaker Registry Completeness**
- speakers配列に全話者とUNKNOWNが含まれることを検証

**Property 15: Meeting JSON Schema Conformance**
- 生成されたJSONがSchema v1.0に準拠することを検証

```python
def test_speaker_registry_completeness():
    """Property 14: Speaker Registry Completeness"""
    # Test that all speakers including UNKNOWN are in registry
    pass

def test_meeting_json_schema_conformance():
    """Property 15: Meeting JSON Schema Conformance"""
    # Test that generated JSON conforms to schema v1.0
    pass

def test_iso8601_timestamp_format():
    """Test ISO 8601 timestamp format"""
    # Test implementation
    pass

def test_json_filename_generation():
    """Test JSON filename generation logic"""
    # Test implementation
    pass
```

#### 9. メインパイプラインのテスト（タスク9.4）

- エラーハンドリング
- リソース解放タイミング

```python
def test_pipeline_error_handling():
    """Test error handling in pipeline"""
    # Test implementation
    pass

def test_pipeline_resource_cleanup():
    """Test resource cleanup after diarization"""
    # Test implementation
    pass
```

### Phase 2: Markdown生成のテスト

#### 10. Markdown Generatorのテスト（タスク11.5, 11.6, 11.7）

**Property 20: Markdown Speaker Grouping**
- 話者ごとにグループ化されることを検証

**Property 21: Markdown Segment Formatting**
- タイムスタンプとテキストのフォーマットを検証

```python
def test_markdown_speaker_grouping():
    """Property 20: Markdown Speaker Grouping"""
    # Test implementation
    pass

def test_markdown_segment_formatting():
    """Property 21: Markdown Segment Formatting"""
    # Test implementation
    pass

def test_markdown_timestamp_format():
    """Test HH:MM:SS timestamp format with zero-padding"""
    # Test implementation
    pass

def test_markdown_empty_text_skip():
    """Test that empty text segments are skipped"""
    # Test implementation
    pass
```

#### 11. 出力フォーマット制御のテスト（タスク12.2）

```python
def test_output_format_json_only():
    """Test format='json' generates only JSON"""
    # Test implementation
    pass

def test_output_format_md_only():
    """Test format='md' generates only Markdown"""
    # Test implementation
    pass

def test_output_format_both():
    """Test format='both' generates both outputs"""
    # Test implementation
    pass
```

#### 12. Benchmark Loggerのテスト（タスク13.3）

```python
def test_benchmark_run_id_generation():
    """Test automatic run_id generation"""
    # Test implementation
    pass

def test_benchmark_jsonl_append():
    """Test JSONL append functionality"""
    # Test implementation
    pass
```

## テスト実装のガイドライン

### 1. テストファイル構成

```
tests/
├── __init__.py
├── test_meeting_pipeline.py  # メインテストファイル
├── fixtures/
│   ├── sample_audio.wav      # テスト用音声ファイル
│   └── expected_output.json  # 期待される出力例
└── conftest.py               # pytest設定とフィクスチャ
```

### 2. モックの使用

外部依存（pyannote-audio, faster-whisper, ffmpeg）はモック化してください：

```python
import pytest
from unittest.mock import Mock, patch, MagicMock

@pytest.fixture
def mock_diarization_result():
    """Mock diarization result for testing"""
    return DiarizationResult(
        turns=[
            SpeakerTurn(id="turn_001", speaker_id="SPEAKER_00", start=0.0, end=10.0),
            SpeakerTurn(id="turn_002", speaker_id="SPEAKER_01", start=10.0, end=20.0),
        ],
        speakers=["SPEAKER_00", "SPEAKER_01"],
        model="pyannote/speaker-diarization-3.1",
        engine="pyannote-audio",
        hf_token_used=True,
    )
```

### 3. Hypothesis設定

```python
from hypothesis import given, strategies as st, settings

# Configure for CI
settings.register_profile("ci", max_examples=100)
settings.load_profile("ci")
```

### 4. テストタグ

```python
import pytest

@pytest.mark.unit
def test_something():
    pass

@pytest.mark.property
@given(...)
def test_property():
    pass

@pytest.mark.integration
def test_end_to_end():
    pass
```

## 確認が必要な実装の問題

### 問題1: asr_load_sec の記録漏れ

**場所**: `run_pipeline()` 関数内

**問題**: `_run_faster_whisper` と `_run_whisper` 内で `asr_load_sec` を計算しているが、`timing.asr_load_sec` に設定されていない

**確認方法**:
1. `_run_faster_whisper` と `_run_whisper` の戻り値を確認
2. `run_pipeline()` で `timing.asr_load_sec` が設定されているか確認
3. 必要であれば修正を提案

**期待される動作**:
```python
# ASR実行後
timing.asr_load_sec = asr_result.asr_load_sec  # これが設定されているべき
```

### 問題2: 型ヒントの確認

以下の関数の型ヒントを確認してください：
- `_dataclass_to_dict()`
- `_get_audio_duration()`
- `_calculate_overlap()`
- `_format_timestamp()`

## テスト実行コマンド

```bash
# 全テスト実行
pytest tests/

# ユニットテストのみ
pytest tests/ -m unit

# プロパティテストのみ
pytest tests/ -m property

# カバレッジ付き
pytest tests/ --cov=meeting_pipeline --cov-report=html

# 詳細出力
pytest tests/ -v

# 特定のテストのみ
pytest tests/test_meeting_pipeline.py::test_alignment_unknown_assignment_zero_overlap
```

## 成果物

以下のファイルを作成してください：

1. **tests/test_meeting_pipeline.py**
   - 全てのユニットテストとプロパティテスト
   - 適切なモックとフィクスチャ

2. **tests/conftest.py**
   - pytest設定
   - 共通フィクスチャ

3. **tests/fixtures/** (必要に応じて)
   - テスト用のサンプルファイル

4. **テスト実行レポート**
   - 全テストの実行結果
   - カバレッジレポート
   - 発見された問題のリスト

## 実装の問題を発見した場合

問題を発見した場合は、以下の形式でレポートしてください：

```markdown
## 発見された問題

### 問題1: asr_load_sec の記録漏れ

**重要度**: 中
**場所**: `run_pipeline()` 関数、行XXX
**説明**: timing.asr_load_sec が設定されていない
**影響**: Meeting JSONのtiming.asr_load_secが常に0.0になる
**修正案**:
\`\`\`python
# 修正前
asr_result = run_asr(...)

# 修正後
asr_result = run_asr(...)
timing.asr_load_sec = asr_result.asr_load_sec  # 追加
\`\`\`
```

## Checkpoint - Phase 1完了確認

全てのテストを作成・実行した後、以下を確認してください：

- [ ] 全てのユニットテストが作成されている
- [ ] 全てのプロパティテストが作成されている
- [ ] テストカバレッジが80%以上
- [ ] 全テストがパス（または問題が明確に報告されている）
- [ ] 実装の問題が文書化されている
- [ ] モックが適切に使用されている
- [ ] テストコードが読みやすく保守可能

## 質問・不明点がある場合

テスト作成中に不明点があれば、以下を確認してください：

1. design.mdの「Testing Strategy」セクション
2. requirements.mdの該当要件と受入基準
3. meeting_pipeline.pyの実装コード

それでも不明な場合は、ユーザーに質問してください。

---

**テスト開始時の指示例**:

```
Phase 1とPhase 2のテストコードを作成してください。
#TESTING_GUIDE.md の指示に従ってください。
#tasks.md のテストタスク（*付き）を全て実装してください。
#meeting_pipeline.py の実装を検証してください。
```
