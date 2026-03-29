# Phase 6 テスト実装指示書 for Codex

## Context

Claude CodeがPhase 6（議事録生成機能）の実装を完了しました。あなたのタスクは、この機能の包括的なユニットテストと統合テストを作成することです。

## Current Status

- **完了**: Phase 1-5（56/56テスト合格）
- **完了**: Phase 6実装（Claude Codeによる）
- **環境**: macOS
- **新規モジュール**: `src/meeting_pipeline/minutes.py`

## テストタスク概要

### Task 29: 議事録モジュールのユニットテスト
- `tests/test_minutes.py`を新規作成
- Meeting JSON読み込みテスト（3件）
- プロンプト準備テスト（3件）
- OpenAI API呼び出しテスト（5件、モック使用）
- APIレスポンスパーステスト（4件）
- メイン関数テスト（3件）

### Task 30: 出力生成のユニットテスト
- `tests/test_meeting_pipeline.py`に追加
- Markdown生成テスト（4件）
- JSON保存テスト（2件）
- Markdown保存テスト（1件）

### Task 31: 統合テスト
- E2Eテスト（3件、モック使用）
- CLIテスト（3件）

## 重要なテストポイント

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

### リトライロジックのテスト

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

## テスト実行

```bash
# すべてのテストを実行
pytest tests/ -v

# Phase 6のテストのみ実行
pytest tests/test_minutes.py -v
pytest tests/ -v -k "minutes"

# カバレッジ付きで実行
pytest tests/ --cov=src.meeting_pipeline --cov-report=term
```

## 成功基準

1. ✅ test_minutes.pyが作成されている
2. ✅ 議事録モジュールのユニットテストが15+件ある
3. ✅ 出力生成のユニットテストが5+件ある
4. ✅ 統合テストが5+件ある
5. ✅ すべてのテストが合格する
6. ✅ OpenAI APIがモックされている（実際のAPI呼び出しなし）
7. ✅ minutes.pyのテストカバレッジ > 80%
8. ✅ 総テスト数: 87+件（既存56 + 新規31）

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

## 注意事項

1. **実際のAPI呼び出しを避ける**: すべてのOpenAI API呼び出しをモックする
2. **テストの独立性**: 各テストは独立して実行可能
3. **エラーケースのカバレッジ**: 正常系だけでなく異常系もテスト
4. **パフォーマンス**: テストは高速に実行される（総実行時間 < 30秒）

## 推定時間

- Task 29: 2時間
- Task 30: 1時間
- Task 31: 1時間
- **合計**: 4時間

詳細は `docs/CODEX_PHASE6_TEST_MESSAGE.md` を参照してください。
