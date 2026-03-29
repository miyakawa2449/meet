# Phase 6 テスト実装依頼 - Codex へ

## 状況報告

Claude Code が Phase 6（議事録生成機能）の実装を完了し、動作確認も成功しました。

### 実装完了内容

**Task 23: データモデル追加** ✅
- `src/meeting_pipeline/models.py` に以下を追加：
  - `MinutesConfig` - 議事録生成の設定
  - `ActionItem` - アクションアイテム
  - `Decision` - 決定事項
  - `Topic` - トピック
  - `MeetingMinutes` - 議事録全体
  - `PipelineConfig` に `generate_minutes`, `minutes_model`, `minutes_language` フィールド追加
  - `Timing` に `minutes_sec` フィールド追加

**Task 24: 議事録生成コアモジュール** ✅
- `src/meeting_pipeline/minutes.py` を新規作成（5関数実装）：
  - `generate_minutes()` - メイン関数
  - `_load_meeting_json()` - Meeting JSON読み込み・検証
  - `_prepare_prompt()` - プロンプト生成（タイムスタンプ付き文字起こし）
  - `_call_openai_api()` - OpenAI API呼び出し（最大3回リトライ、指数バックオフ）
  - `_parse_api_response()` - レスポンスのパース

**Task 26: 議事録出力生成** ✅
- `src/meeting_pipeline/output.py` に以下を追加：
  - `generate_minutes_markdown()` - Markdown形式の議事録生成
  - `save_minutes_markdown()` - Markdownファイル保存
  - `save_minutes_json()` - JSONファイル保存（検証付き）

**Task 27: CLI統合** ✅
- `src/meeting_pipeline/cli.py` に以下を追加：
  - `--generate-minutes` - 議事録生成を有効化
  - `--minutes-model` - OpenAIモデル選択（デフォルト: gpt-3.5-turbo）
  - `--minutes-language` - 議事録の言語（デフォルト: auto）
  - OPENAI_API_KEY 未設定時の警告ロジック

**Task 28: パイプライン統合** ✅
- `src/meeting_pipeline/pipeline.py` にステージ8（議事録生成）を追加
- 失敗してもパイプライン全体は停止しない設計
- エラーハンドリング完備

**依存関係** ✅
- `requirements.txt` に追加：
  - `openai>=1.0.0`
  - `python-dotenv>=1.0.0`

### 動作確認結果

**テスト実行**: 5分動画（`temp/202602017_short_test.mp4`）で検証

```bash
python meeting_pipeline.py temp/202602017_short_test.mp4 \
  --enable-diarization \
  --generate-minutes \
  --output-dir output/phase6_test_with_minutes
```

**処理時間**:
- 話者分離（MPS）: 16.6秒
- ASR（CPU）: 61.4秒
- 議事録生成（OpenAI API）: 4.6秒
- 総処理時間: 78.7秒

**生成ファイル**:
1. `202602017_short_test_meeting.json` (65KB) - Meeting JSON
2. `202602017_short_test_transcript.md` (5.7KB) - 文字起こしMarkdown
3. `202602017_short_test_minutes.json` (1.3KB) - 議事録JSON ✨
4. `202602017_short_test_minutes.md` (997B) - 議事録Markdown ✨

**議事録の内容**:
- 要約: 会議全体の簡潔な要約（100-300語）✅
- 決定事項: 0件（この会議では明示的な決定なし）
- アクションアイテム: 0件（この会議では明示的なタスクなし）
- トピック: 2件（Google Meetの使用方法、新規顧客獲得とDXの重要性）✅

**検証結果**:
- ✅ OpenAI API統合が正常に動作
- ✅ リトライロジックが実装済み
- ✅ JSON + Markdown形式で出力
- ✅ タイムスタンプ付きトピック分類
- ✅ 参加者情報の抽出
- ✅ エラーハンドリング（議事録失敗でもパイプラインは継続）

### 既存テスト状況

```bash
pytest tests/ -v
```

**結果**: 56/56 テスト合格 ✅

Phase 1-5 のすべてのテストが引き続き合格しています。

---

## あなたのタスク: Phase 6 テスト実装

Phase 6 の実装が完了したので、包括的なユニットテストと統合テストを作成してください。

### テストタスク概要

#### Task 29: 議事録モジュールのユニットテスト

`tests/test_minutes.py` を新規作成し、以下のテストを実装：

1. **Meeting JSON読み込みテスト（3件）**:
   - 正常なJSONファイルの読み込み
   - ファイルが存在しない場合のエラー
   - 無効なJSON形式のエラー

2. **プロンプト準備テスト（3件）**:
   - セグメントからプロンプトへの変換
   - タイムスタンプのフォーマット（HH:MM:SS）
   - 言語指定の処理

3. **OpenAI API呼び出しテスト（5件、モック使用）**:
   - 正常なAPI呼び出し
   - 認証エラー（リトライしない）
   - レート制限エラー（リトライする）
   - タイムアウトエラー（リトライする）
   - 最大リトライ回数超過

4. **APIレスポンスパーステスト（4件）**:
   - 正常なレスポンスのパース
   - 決定事項の抽出
   - アクションアイテムの抽出
   - トピックの抽出

5. **メイン関数テスト（3件）**:
   - エンドツーエンドの議事録生成
   - エラーハンドリング
   - タイミング測定

**期待テスト数**: 18件

#### Task 30: 出力生成のユニットテスト

`tests/test_meeting_pipeline.py` に追加：

1. **Markdown生成テスト（4件）**:
   - 議事録Markdownの構造
   - 要約セクション
   - 決定事項セクション
   - アクションアイテムセクション

2. **JSON保存テスト（2件）**:
   - 議事録JSONの保存
   - スキーマバージョンの検証

3. **Markdown保存テスト（1件）**:
   - 議事録Markdownの保存

**期待テスト数**: 7件

#### Task 31: 統合テスト

`tests/test_meeting_pipeline.py` に追加：

1. **E2Eテスト（3件、モック使用）**:
   - パイプライン全体の実行（議事録生成あり）
   - 議事録生成失敗時のパイプライン継続
   - OPENAI_API_KEY未設定時の警告

2. **CLIテスト（3件）**:
   - `--generate-minutes` オプション
   - `--minutes-model` オプション
   - `--minutes-language` オプション

**期待テスト数**: 6件

### 総テスト数

- 既存テスト: 56件
- 新規テスト: 31件（18 + 7 + 6）
- **合計**: 87件

### 重要なテストポイント

#### OpenAI APIのモック

```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_openai_client(monkeypatch):
    """OpenAI APIクライアントをモックするフィクスチャ"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps({
        "summary": "テスト要約",
        "decisions": [{"text": "決定1", "speaker": "Speaker 1", "timestamp": 10.0}],
        "action_items": [{"task": "タスク1", "assignee": "Speaker 2", "deadline": "2024-03-15", "timestamp": 20.0}],
        "topics": [{"title": "トピック1", "summary": "要約1", "start": 0.0, "end": 30.0}]
    })
    
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    
    with patch("openai.OpenAI", return_value=mock_client):
        yield mock_client
```

#### リトライロジックのテスト

```python
def test_retry_logic(monkeypatch):
    """リトライロジックをテストするためのモック例"""
    call_count = 0
    
    def mock_api_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise openai.RateLimitError("Rate limit exceeded")
        return mock_success_response
    
    monkeypatch.setattr("openai.OpenAI.chat.completions.create", mock_api_call)
```

### テスト実行

```bash
# 環境設定
source venv/bin/activate

# すべてのテストを実行
pytest tests/ -v

# Phase 6のテストのみ実行
pytest tests/test_minutes.py -v
pytest tests/ -v -k "minutes"

# カバレッジ付きで実行
pytest tests/ --cov=src.meeting_pipeline --cov-report=term
```

### 成功基準

1. ✅ `test_minutes.py` が作成されている
2. ✅ 議事録モジュールのユニットテストが18件ある
3. ✅ 出力生成のユニットテストが7件ある
4. ✅ 統合テストが6件ある
5. ✅ すべてのテストが合格する
6. ✅ OpenAI APIがモックされている（実際のAPI呼び出しなし）
7. ✅ `minutes.py` のテストカバレッジ > 80%
8. ✅ 総テスト数: 87件（既存56 + 新規31）

### 注意事項

1. **実際のAPI呼び出しを避ける**: すべてのOpenAI API呼び出しをモックする
2. **テストの独立性**: 各テストは独立して実行可能
3. **エラーケースのカバレッジ**: 正常系だけでなく異常系もテスト
4. **パフォーマンス**: テストは高速に実行される（総実行時間 < 30秒）

### 参考資料

詳細な実装方法は以下を参照してください：
- `docs/CODEX_PHASE6_TEST_MESSAGE.md` - 詳細なテスト実装指示書
- `.kiro/steering/codex_phase6_test_instructions.md` - ステアリングファイル（自動ロード）
- `src/meeting_pipeline/minutes.py` - テスト対象のモジュール
- `src/meeting_pipeline/output.py` - 出力生成関数
- `src/meeting_pipeline/pipeline.py` - パイプライン統合

### 推定時間

- Task 29: 2時間
- Task 30: 1時間
- Task 31: 1時間
- **合計**: 4時間

---

## 開始方法

以下のコマンドでテスト実装を開始してください：

```bash
# 環境設定
source venv/bin/activate

# 新規テストファイル作成
touch tests/test_minutes.py

# テスト実装開始
# Task 29 から順番に実装してください
```

テスト実装が完了したら、以下を報告してください：

1. 追加したテスト数
2. 総テスト数（合格/総数）
3. テスト実行時間
4. カバレッジ（minutes.py）
5. 発見した問題（あれば）

頑張ってください！🚀
