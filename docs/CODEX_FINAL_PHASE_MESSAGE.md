# 最終フェーズテスト実装依頼

## あなたの役割

あなたは **テスト担当（Codex）** です。最終フェーズ（Task 21.3）のエンドツーエンド統合テストを実施してください。

## 背景

Phase 1-5 がすべて完了し、プロジェクト整理も完了しました。最終フェーズでは、新パイプラインの包括的な動作確認を行います。

## 現状

- ✅ Phase 1-5 完了
- ✅ プロジェクト整理完了（legacy/, docs/ フォルダ作成）
- ✅ 56/56 テスト合格
- ✅ README.md 更新完了

## 実施タスク: Task 21.3 エンドツーエンド統合テスト

### テスト方針の選択

以下の3つのオプションから選択してください：

#### オプション1: 手動E2Eテスト（推奨）

**理由**:
- 既存56テストで十分なカバレッジ
- 手動テストで主要な組み合わせを確認
- 実装コストが低い

**テストケース**（7つ）:

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

4. **JSON のみ出力**:
   ```bash
   python meeting_pipeline.py temp/202602017_short_test.mp4 \
     --enable-diarization \
     --format json
   ```

5. **Markdown のみ出力**:
   ```bash
   python meeting_pipeline.py temp/202602017_short_test.mp4 \
     --enable-diarization \
     --format md
   ```

6. **CPU デバイス**:
   ```bash
   python meeting_pipeline.py temp/202602017_short_test.mp4 \
     --device cpu
   ```

7. **MPS デバイス**（macOS の場合）:
   ```bash
   python meeting_pipeline.py temp/202602017_short_test.mp4 \
     --enable-diarization \
     --device mps
   ```

**検証項目**:
- ✅ コマンドが正常に実行される（exit code 0）
- ✅ 出力ファイルが生成される
- ✅ JSON が Meeting JSON Schema v1.0 に準拠
- ✅ Markdown が正しくフォーマットされている
- ✅ エラーや警告が適切に処理される

#### オプション2: 自動E2Eテストスクリプト作成

**理由**:
- 将来の回帰テストに有用
- 自動化による再現性

**実施内容**:
- `tests/test_e2e_integration.py` を作成
- 8-10件のテストケースを実装
- pytest で実行

**テンプレート**:
```python
import pytest
import subprocess
import json
from pathlib import Path

TEST_VIDEO = "temp/202602017_short_test.mp4"
OUTPUT_DIR = "output/e2e_test"

def run_pipeline(args):
    """Run meeting_pipeline.py with given arguments."""
    cmd = ["python", "meeting_pipeline.py", TEST_VIDEO] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result

def test_e2e_basic_no_diarization():
    """Test basic pipeline without diarization."""
    result = run_pipeline(["--device", "auto", "--output-dir", OUTPUT_DIR])
    assert result.returncode == 0
    assert Path(f"{OUTPUT_DIR}/202602017_short_test_meeting.json").exists()

# ... 他のテストケース
```

#### オプション3: 何もしない

**理由**:
- 既存56テストで十分
- Phase 1-5 で各機能は検証済み

**実施内容**:
- テスト追加なし
- Task 21.3 をスキップ

## 推奨アクション

**オプション1（手動E2Eテスト）** を推奨します。

**理由**:
1. 既存56テストで機能は十分カバー済み
2. 手動テストで主要な組み合わせを確認すれば十分
3. 実装コストが低く、時間効率が良い
4. Phase 1-5 で各機能は個別に検証済み

## Environment Setup

```bash
source venv/bin/activate
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)
```

## あなたの判断

以下のいずれかを選択して実施してください:

1. **手動E2Eテスト**（推奨）
   - 7つのテストケースを手動実行
   - 結果を記録して報告

2. **自動E2Eテストスクリプト作成**
   - tests/test_e2e_integration.py を作成
   - pytest で実行

3. **何もしない**
   - 既存56テストで十分と判断
   - Task 21.3 をスキップ

## Reporting

完了後、以下を報告してください:

1. **選択したオプション**（1, 2, 3）
2. **テスト結果**（実施した場合）:
   - 実行したテストケース
   - 成功/失敗の結果
   - 出力ファイルの確認結果
3. **発見した問題**（あれば）
4. **推奨事項**（あれば）

## Success Criteria

Task 21.3 は以下の条件で完了です:

1. ✅ 主要な組み合わせで動作確認完了（または判断によりスキップ）
2. ✅ 出力ファイルが正しく生成される（テスト実施の場合）
3. ✅ エラーや警告が適切に処理される（テスト実施の場合）
4. ✅ 結果を報告

## Notes

- Task 21.3 はオプションです
- 既存56テストで十分なカバレッジがあります
- 時間がなければオプション3（何もしない）を選択しても問題ありません
- あなたの判断で最適なオプションを選択してください

## 詳細手順

詳細な実装方法は `.kiro/steering/codex_final_phase_test_instructions.md` を確認してください。
