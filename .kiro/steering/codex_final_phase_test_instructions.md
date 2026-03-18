# 最終フェーズテスト指示書 for Codex

## Context

Phase 1-5 がすべて完了し、プロジェクト整理も完了しました。最終フェーズでは、包括的なエンドツーエンド統合テストを実施します。

## Current Status

- **完了**: Phase 1-5（基本パイプライン、Markdown生成、単語レベルアライメント、macOS MPS/CPU対応、Windows CUDA検証）
- **完了**: プロジェクト整理（旧スクリプトを legacy/ に移動）
- **テスト**: 56/56 合格
- **環境**: macOS（テスト実施環境）

## 最終フェーズ Test Overview

Task 21.3 では、新パイプラインの包括的なエンドツーエンド統合テストを実施します。

### Task 21.3: エンドツーエンド統合テストの実施

**目的**: 新パイプラインがすべての組み合わせで正常動作することを検証

**テスト対象**:

1. **複数の入力ファイル形式**
   - mp4（動画）
   - wav（音声）
   - m4a（音声）
   - その他対応フォーマット

2. **全出力フォーマット**
   - json（JSON のみ）
   - md（Markdown のみ）
   - both（両方）

3. **全デバイスオプション**
   - auto（自動選択）
   - cuda（CUDA、利用可能な場合）
   - mps（MPS、macOS の場合）
   - cpu（CPU）

4. **話者分離の有無**
   - --enable-diarization あり
   - --enable-diarization なし

5. **アライメント単位**
   - segment（デフォルト）
   - word（単語レベル）

## テスト実装方針

### オプション1: 手動E2Eテスト（推奨）

既存の56テストで十分なカバレッジがあるため、手動で主要な組み合わせをテストする。

**テストケース**:

1. **基本動作**（話者分離なし）:
   ```bash
   python meeting_pipeline.py temp/202602017_short_test.mp4 --device auto
   ```

2. **話者分離（segment-level）**:
   ```bash
   python meeting_pipeline.py temp/202602017_short_test.mp4 \
     --enable-diarization \
     --device auto
   ```

3. **話者分離（word-level）**:
   ```bash
   python meeting_pipeline.py temp/202602017_short_test.mp4 \
     --enable-diarization \
     --align-unit word \
     --device auto
   ```

4. **出力フォーマット（JSON のみ）**:
   ```bash
   python meeting_pipeline.py temp/202602017_short_test.mp4 \
     --enable-diarization \
     --format json
   ```

5. **出力フォーマット（Markdown のみ）**:
   ```bash
   python meeting_pipeline.py temp/202602017_short_test.mp4 \
     --enable-diarization \
     --format md
   ```

6. **デバイス指定（CPU）**:
   ```bash
   python meeting_pipeline.py temp/202602017_short_test.mp4 \
     --device cpu
   ```

7. **デバイス指定（MPS、macOS の場合）**:
   ```bash
   python meeting_pipeline.py temp/202602017_short_test.mp4 \
     --enable-diarization \
     --device mps
   ```

**検証項目**:
- コマンドが正常に実行される
- 出力ファイルが生成される
- エラーや警告が適切に処理される
- 出力内容が期待通り

### オプション2: 自動E2Eテストスクリプト作成

包括的な自動テストスクリプトを作成する（時間がある場合）。

**スクリプト例** (`tests/test_e2e_integration.py`):

```python
import pytest
import subprocess
import json
import os
from pathlib import Path

TEST_VIDEO = "temp/202602017_short_test.mp4"
OUTPUT_DIR = "output/e2e_test"

@pytest.fixture(autouse=True)
def cleanup_output():
    """Clean up output directory before each test."""
    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yield
    # Keep output for inspection

def run_pipeline(args):
    """Run meeting_pipeline.py with given arguments."""
    cmd = ["python", "meeting_pipeline.py", TEST_VIDEO] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600  # 10 minutes timeout
    )
    return result

def test_e2e_basic_no_diarization():
    """Test basic pipeline without diarization."""
    result = run_pipeline([
        "--device", "auto",
        "--output-dir", OUTPUT_DIR,
        "--format", "both"
    ])
    assert result.returncode == 0
    # Verify output files exist
    assert Path(f"{OUTPUT_DIR}/202602017_short_test_meeting.json").exists()
    assert Path(f"{OUTPUT_DIR}/202602017_short_test_transcript.md").exists()

def test_e2e_with_diarization_segment():
    """Test pipeline with diarization (segment-level)."""
    result = run_pipeline([
        "--enable-diarization",
        "--device", "auto",
        "--output-dir", OUTPUT_DIR,
        "--format", "both"
    ])
    assert result.returncode == 0
    # Verify output and check speakers
    json_path = Path(f"{OUTPUT_DIR}/202602017_short_test_meeting.json")
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
        assert len(data["speakers"]) > 0

def test_e2e_with_diarization_word():
    """Test pipeline with diarization (word-level)."""
    result = run_pipeline([
        "--enable-diarization",
        "--align-unit", "word",
        "--device", "auto",
        "--output-dir", OUTPUT_DIR,
        "--format", "both"
    ])
    assert result.returncode == 0
    # Verify align unit is word
    json_path = Path(f"{OUTPUT_DIR}/202602017_short_test_meeting.json")
    with open(json_path) as f:
        data = json.load(f)
        assert data["pipeline"]["align"]["unit"] == "word"

def test_e2e_format_json_only():
    """Test JSON-only output format."""
    result = run_pipeline([
        "--enable-diarization",
        "--format", "json",
        "--output-dir", OUTPUT_DIR
    ])
    assert result.returncode == 0
    assert Path(f"{OUTPUT_DIR}/202602017_short_test_meeting.json").exists()
    assert not Path(f"{OUTPUT_DIR}/202602017_short_test_transcript.md").exists()

def test_e2e_format_md_only():
    """Test Markdown-only output format."""
    result = run_pipeline([
        "--enable-diarization",
        "--format", "md",
        "--output-dir", OUTPUT_DIR
    ])
    assert result.returncode == 0
    assert Path(f"{OUTPUT_DIR}/202602017_short_test_transcript.md").exists()
    # JSON is generated as intermediate but not saved in md-only mode

def test_e2e_device_cpu():
    """Test pipeline with CPU device."""
    result = run_pipeline([
        "--device", "cpu",
        "--output-dir", OUTPUT_DIR
    ])
    assert result.returncode == 0
    json_path = Path(f"{OUTPUT_DIR}/202602017_short_test_meeting.json")
    with open(json_path) as f:
        data = json.load(f)
        assert data["pipeline"]["device"]["resolved"] == "cpu"

@pytest.mark.skipif(
    not __import__("torch").backends.mps.is_available(),
    reason="MPS not available"
)
def test_e2e_device_mps():
    """Test pipeline with MPS device (macOS only)."""
    result = run_pipeline([
        "--enable-diarization",
        "--device", "mps",
        "--output-dir", OUTPUT_DIR
    ])
    assert result.returncode == 0
    json_path = Path(f"{OUTPUT_DIR}/202602017_short_test_meeting.json")
    with open(json_path) as f:
        data = json.load(f)
        assert data["pipeline"]["device"]["resolved"] == "mps"
```

**実行方法**:
```bash
pytest tests/test_e2e_integration.py -v
```

## 推奨アクション

### オプション1: 手動E2Eテスト（推奨）

**理由**:
- 既存56テストで十分なカバレッジ
- 手動テストで主要な組み合わせを確認
- 実装コストが低い

**実施内容**:
- 上記の7つのテストケースを手動で実行
- 結果を記録
- 問題があれば報告

### オプション2: 自動E2Eテストスクリプト作成

**理由**:
- 将来の回帰テストに有用
- 自動化による再現性

**実施内容**:
- `tests/test_e2e_integration.py` を作成
- 上記のテストケースを実装
- pytest で実行

### オプション3: 何もしない

**理由**:
- 既存56テストで十分
- Phase 1-5 で各機能は検証済み

**実施内容**:
- テスト追加なし
- 既存テストの実行のみ

## あなたの判断

以下のいずれかを選択してください:

1. **手動E2Eテスト**（推奨）
   - 7つのテストケースを手動実行
   - 結果を報告

2. **自動E2Eテストスクリプト作成**
   - tests/test_e2e_integration.py を作成
   - 8-10件のテストケースを実装

3. **何もしない**
   - 既存56テストで十分と判断
   - Task 21.3 をスキップ

## Environment Setup

```bash
source venv/bin/activate
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)
```

## Success Criteria

Task 21.3 は以下の条件で完了です:

1. ✅ 主要な組み合わせで動作確認完了
2. ✅ 出力ファイルが正しく生成される
3. ✅ エラーや警告が適切に処理される
4. ✅ 問題が発見された場合は報告

## Reporting

完了後、以下を報告してください:

1. 選択したオプション（1, 2, 3）
2. テスト結果（実施した場合）
3. 発見した問題（あれば）
4. 推奨事項（あれば）

## Notes

- Task 21.3 はオプションです
- 既存56テストで十分なカバレッジがあります
- 時間がなければスキップしても問題ありません

## 詳細手順

詳細な実装方法は `.kiro/steering/codex_final_phase_test_instructions.md` を確認してください。
