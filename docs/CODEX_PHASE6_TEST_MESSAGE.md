# Phase 6 テスト実装指示書 for Codex

## Context

Claude CodeがPhase 6（議事録生成機能）の実装を完了しました。あなたのタスクは、この機能の包括的なユニットテストと統合テストを作成することです。

## Current Status

- **完了**: Phase 1-5（56/56テスト合格）
- **完了**: Phase 6実装（Claude Codeによる）
- **環境**: macOS
- **新規モジュール**: `src/meeting_pipeline/minutes.py`

## Phase 6 実装概要

Claude Codeが以下を実装しました:

1. **データモデル** (`models.py`):
   - `MinutesConfig`, `ActionItem`, `Decision`, `Topic`, `MeetingMinutes`
   - `PipelineConfig`と`Timing`の拡張

2. **議事録生成モジュール** (`minutes.py`):
   - `generate_minutes()`: メイン関数
   - `_load_meeting_json()`: Meeting JSON読み込み
   - `_prepare_prompt()`: プロンプト準備
   - `_call_openai_api()`: API呼び出し（リトライロジック付き）
   - `_parse_api_response()`: レスポンスパース

3. **出力生成** (`output.py`):
   - `generate_minutes_markdown()`: Markdown生成
   - `save_minutes_markdown()`: Markdown保存
   - `save_minutes_json()`: JSON保存

4. **CLI統合** (`cli.py`):
   - `--generate-minutes`, `--minutes-model`, `--minutes-language`

5. **パイプライン統合** (`pipeline.py`):
   - ステージ8: 議事録生成

## あなたのタスク

### Task 29: 議事録モジュールのユニットテスト

**ファイル**: `tests/test_minutes.py`（新規作成）

**テストケース**:

#### 1. Meeting JSON読み込みテスト

```python
def test_load_meeting_json_success(tmp_path):
    """有効なMeeting JSONを正常に読み込めることをテスト"""
    # Meeting JSONファイルを作成
    # _load_meeting_json()を呼び出し
    # MeetingJSONオブジェクトが返されることを確認

def test_load_meeting_json_file_not_found():
    """存在しないファイルでFileNotFoundErrorが発生することをテスト"""
    # 存在しないパスで_load_meeting_json()を呼び出し
    # FileNotFoundErrorが発生することを確認

def test_load_meeting_json_invalid_format(tmp_path):
    """無効なJSON形式でValueErrorが発生することをテスト"""
    # 無効なJSONファイルを作成
    # _load_meeting_json()を呼び出し
    # ValueErrorが発生することを確認
```

#### 2. プロンプト準備テスト

```python
def test_prepare_prompt_basic():
    """基本的なプロンプト生成をテスト"""
    # モックMeetingJSONを作成
    # _prepare_prompt()を呼び出し
    # プロンプトにシステムメッセージとユーザーメッセージが含まれることを確認

def test_prepare_prompt_includes_speakers():
    """プロンプトに話者情報が含まれることをテスト"""
    # 複数話者のMeetingJSONを作成
    # _prepare_prompt()を呼び出し
    # プロンプトに話者ラベルが含まれることを確認

def test_prepare_prompt_includes_timestamps():
    """プロンプトにタイムスタンプが含まれることをテスト"""
    # タイムスタンプ付きMeetingJSONを作成
    # _prepare_prompt()を呼び出し
    # プロンプトにタイムスタンプが含まれることを確認
```

#### 3. OpenAI API呼び出しテスト（モック使用）

```python
def test_call_openai_api_success(monkeypatch):
    """成功したAPI呼び出しをテスト"""
    # OpenAI APIをモック
    # _call_openai_api()を呼び出し
    # レスポンスが返されることを確認

def test_call_openai_api_retry_on_rate_limit(monkeypatch):
    """レート制限エラーでリトライすることをテスト"""
    # 最初の2回はRateLimitError、3回目は成功するようにモック
    # _call_openai_api()を呼び出し
    # 3回呼び出されることを確認
    # 最終的に成功することを確認

def test_call_openai_api_retry_on_timeout(monkeypatch):
    """タイムアウトエラーでリトライすることをテスト"""
    # 最初の1回はAPITimeoutError、2回目は成功するようにモック
    # _call_openai_api()を呼び出し
    # 2回呼び出されることを確認

def test_call_openai_api_no_retry_on_auth_error(monkeypatch):
    """認証エラーでリトライしないことをテスト"""
    # AuthenticationErrorを発生させるようにモック
    # _call_openai_api()を呼び出し
    # 1回だけ呼び出されることを確認
    # AuthenticationErrorが発生することを確認

def test_call_openai_api_max_retries_exceeded(monkeypatch):
    """最大リトライ回数を超えるとエラーになることをテスト"""
    # 常にRateLimitErrorを発生させるようにモック
    # _call_openai_api()を呼び出し
    # 3回リトライ後にエラーが発生することを確認
```

#### 4. APIレスポンスパーステスト

```python
def test_parse_api_response_valid():
    """有効なAPIレスポンスを正常にパースできることをテスト"""
    # 有効なJSON形式のレスポンスを作成
    # _parse_api_response()を呼び出し
    # MeetingMinutesオブジェクトが返されることを確認
    # すべてのフィールドが正しく設定されていることを確認

def test_parse_api_response_empty_lists():
    """空のリスト（決定、アクション、トピック）を処理できることをテスト"""
    # 空リストを含むレスポンスを作成
    # _parse_api_response()を呼び出し
    # 空リストが正しく処理されることを確認

def test_parse_api_response_missing_fields():
    """必須フィールドが欠落している場合のエラーハンドリングをテスト"""
    # 必須フィールドが欠落したレスポンスを作成
    # _parse_api_response()を呼び出し
    # 適切なエラーが発生することを確認

def test_parse_api_response_invalid_json():
    """無効なJSON形式のレスポンスでエラーが発生することをテスト"""
    # 無効なJSON文字列を作成
    # _parse_api_response()を呼び出し
    # ValueErrorが発生することを確認
```

#### 5. メイン関数テスト

```python
def test_generate_minutes_success(monkeypatch, tmp_path):
    """議事録生成が正常に完了することをテスト"""
    # Meeting JSONファイルを作成
    # OpenAI APIをモック
    # generate_minutes()を呼び出し
    # MeetingMinutesオブジェクトが返されることを確認

def test_generate_minutes_file_not_found(monkeypatch):
    """Meeting JSONが見つからない場合のエラーハンドリングをテスト"""
    # 存在しないパスでgenerate_minutes()を呼び出し
    # FileNotFoundErrorが発生することを確認

def test_generate_minutes_api_error(monkeypatch, tmp_path):
    """API呼び出し失敗時のエラーハンドリングをテスト"""
    # Meeting JSONファイルを作成
    # API呼び出しが失敗するようにモック
    # generate_minutes()を呼び出し
    # 適切なエラーが発生することを確認
```

---

### Task 30: 出力生成のユニットテスト

**ファイル**: `tests/test_meeting_pipeline.py`（既存ファイルに追加）

**テストケース**:

#### 1. Markdown生成テスト

```python
def test_generate_minutes_markdown_structure():
    """Markdown議事録の構造が正しいことをテスト"""
    # MeetingMinutesオブジェクトを作成
    # generate_minutes_markdown()を呼び出し
    # Markdownに必須セクションが含まれることを確認:
    #   - ヘッダー（タイトル、日付、時間、参加者）
    #   - 要約
    #   - 決定事項
    #   - アクションアイテム
    #   - トピック

def test_generate_minutes_markdown_empty_lists():
    """空のリストを含むMarkdown生成をテスト"""
    # 決定、アクション、トピックが空のMeetingMinutesを作成
    # generate_minutes_markdown()を呼び出し
    # エラーなく生成されることを確認

def test_generate_minutes_markdown_timestamp_format():
    """タイムスタンプが正しくフォーマットされることをテスト"""
    # タイムスタンプ付きMeetingMinutesを作成
    # generate_minutes_markdown()を呼び出し
    # タイムスタンプがHH:MM:SS形式であることを確認

def test_generate_minutes_markdown_action_items_table():
    """アクションアイテムがテーブル形式で出力されることをテスト"""
    # アクションアイテムを含むMeetingMinutesを作成
    # generate_minutes_markdown()を呼び出し
    # Markdownテーブル形式が含まれることを確認
```

#### 2. JSON保存テスト

```python
def test_save_minutes_json_valid(tmp_path):
    """議事録JSONが正常に保存されることをテスト"""
    # MeetingMinutesオブジェクトを作成
    # save_minutes_json()を呼び出し
    # ファイルが作成されることを確認
    # ファイル内容が有効なJSONであることを確認

def test_save_minutes_json_round_trip(tmp_path):
    """JSONのラウンドトリップ（保存→読み込み）をテスト"""
    # MeetingMinutesオブジェクトを作成
    # save_minutes_json()で保存
    # ファイルを読み込み
    # 元のデータと一致することを確認
```

#### 3. Markdown保存テスト

```python
def test_save_minutes_markdown(tmp_path):
    """議事録Markdownが正常に保存されることをテスト"""
    # Markdownコンテンツを作成
    # save_minutes_markdown()を呼び出し
    # ファイルが作成されることを確認
    # ファイル内容が正しいことを確認
```

---

### Task 31: 統合テスト

**ファイル**: `tests/test_meeting_pipeline.py`または`tests/test_e2e_minutes.py`

**テストケース**:

#### 1. エンドツーエンドテスト（モック使用）

```python
def test_e2e_with_minutes_generation(monkeypatch, tmp_path):
    """議事録生成付きでパイプライン全体を実行するテスト"""
    # OpenAI APIをモック
    # --generate-minutesフラグ付きでパイプラインを実行
    # すべての出力ファイルが生成されることを確認:
    #   - Meeting JSON
    #   - 文字起こしMarkdown
    #   - 議事録Markdown
    #   - 議事録JSON

def test_e2e_minutes_generation_api_key_missing(monkeypatch, tmp_path):
    """APIキーが設定されていない場合のエラーハンドリングをテスト"""
    # OPENAI_API_KEYを未設定にする
    # --generate-minutesフラグ付きでパイプラインを実行
    # 議事録生成がスキップされることを確認
    # Meeting JSONと文字起こしは正常に保存されることを確認

def test_e2e_minutes_generation_api_failure(monkeypatch, tmp_path):
    """API呼び出し失敗時のエラーハンドリングをテスト"""
    # OpenAI APIが失敗するようにモック
    # --generate-minutesフラグ付きでパイプラインを実行
    # 議事録生成が失敗してもパイプラインは継続することを確認
    # Meeting JSONと文字起こしは正常に保存されることを確認
```

#### 2. CLIテスト

```python
def test_cli_generate_minutes_flag():
    """--generate-minutesフラグが正しくパースされることをテスト"""
    # --generate-minutesフラグ付きでCLIをパース
    # config.generate_minutesがTrueであることを確認

def test_cli_minutes_model_option():
    """--minutes-modelオプションが正しくパースされることをテスト"""
    # --minutes-model gpt-4でCLIをパース
    # config.minutes_modelが"gpt-4"であることを確認

def test_cli_minutes_language_option():
    """--minutes-languageオプションが正しくパースされることをテスト"""
    # --minutes-language enでCLIをパース
    # config.minutes_languageが"en"であることを確認
```

---

## モック戦略

### OpenAI APIのモック

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

### リトライロジックのモック

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
    # テストコード...
```

---

## 環境設定

テスト実行前に環境を設定してください:

```bash
source venv/bin/activate
export OPENAI_API_KEY="test_key_for_mocking"
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)
```

---

## テスト実行

### すべてのテストを実行

```bash
pytest tests/ -v
```

### Phase 6のテストのみ実行

```bash
pytest tests/test_minutes.py -v
pytest tests/ -v -k "minutes"
```

### カバレッジ付きで実行

```bash
pytest tests/ --cov=src.meeting_pipeline --cov-report=term --cov-report=html
```

---

## 成功基準

Phase 6テストは以下の条件で完了です:

1. ✅ `test_minutes.py`が作成されている
2. ✅ 議事録モジュールのユニットテストが15+件ある
3. ✅ 出力生成のユニットテストが5+件ある
4. ✅ 統合テストが5+件ある
5. ✅ すべてのテストが合格する
6. ✅ OpenAI APIがモックされている（実際のAPI呼び出しなし）
7. ✅ エラーシナリオがカバーされている
8. ✅ `minutes.py`のテストカバレッジ > 80%
9. ✅ 総テスト数: 70+件（既存56 + 新規15+）

---

## 期待されるテスト数

| カテゴリ | テスト数 |
|---------|---------|
| Meeting JSON読み込み | 3 |
| プロンプト準備 | 3 |
| OpenAI API呼び出し | 5 |
| APIレスポンスパース | 4 |
| メイン関数 | 3 |
| Markdown生成 | 4 |
| JSON保存 | 2 |
| Markdown保存 | 1 |
| E2Eテスト | 3 |
| CLIテスト | 3 |
| **合計** | **31+** |

既存56テスト + 新規31テスト = **87テスト**

---

## 注意事項

1. **実際のAPI呼び出しを避ける**:
   - すべてのOpenAI API呼び出しをモックする
   - テストでコストが発生しないようにする

2. **テストの独立性**:
   - 各テストは独立して実行可能
   - テスト間で状態を共有しない

3. **エラーケースのカバレッジ**:
   - 正常系だけでなく異常系もテストする
   - エッジケース（空リスト、欠落フィールド）をカバーする

4. **パフォーマンス**:
   - テストは高速に実行される（モック使用）
   - 総実行時間 < 30秒

---

## Git Workflow

テスト実装完了後:

```bash
git add tests/
git commit -m "Phase 6テスト: 議事録生成機能のユニットテストと統合テスト追加

- Task 29: 議事録モジュールのユニットテスト（18テスト）
- Task 30: 出力生成のユニットテスト（7テスト）
- Task 31: 統合テスト（6テスト）
- OpenAI APIモック実装
- 総テスト数: 87テスト（既存56 + 新規31）"
git push
```

---

## 次のステップ

テスト完了後、Task 32（ドキュメント更新）、Task 33（手動テスト）、Task 34（最終チェックポイント）に進みます。
