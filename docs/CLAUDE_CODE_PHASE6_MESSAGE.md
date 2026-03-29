# Phase 6 実装指示書 for Claude Code

## Context

Phase 1-5がすべて完了し、プロジェクトは安定稼働しています。Phase 6では、OpenAI APIを使用した議事録生成機能を追加します。

## Current Status

- **完了**: Phase 1-5（基本パイプライン、Markdown生成、単語レベルアライメント、macOS MPS/CPU対応、Windows CUDA検証）
- **テスト**: 56/56 合格
- **環境**: macOS
- **プロジェクト構造**: モジュール化されたアーキテクチャ（`src/meeting_pipeline/`）

## Phase 6 概要

Phase 6では、Meeting JSON（Phase 1-5の出力）を入力として、OpenAI GPTモデルを使用して構造化された議事録を生成します。

### 主要機能

1. **OpenAI API統合**: GPT-4/GPT-3.5-turboを使用
2. **要約生成**: 会議全体の簡潔な要約
3. **決定事項抽出**: 会議中に行われた決定を特定
4. **アクションアイテム抽出**: タスク、担当者、期限を抽出
5. **トピック分類**: 会議を1-10のトピックに分割
6. **出力生成**: Markdown + JSON形式

### アーキテクチャ

```
Meeting JSON (Phase 1-5の出力)
         ↓
   議事録生成器 (minutes.py)
         ↓
    OpenAI API
         ↓
   ┌─────┴─────┐
   ↓           ↓
Minutes.md  Minutes.json
```

## Spec参照

Phase 6の詳細仕様は以下のファイルを参照してください:

- **要件定義**: `.kiro/specs/meeting-minutes-generation/requirements.md`
- **設計書**: `.kiro/specs/meeting-minutes-generation/design.md`
- **タスク一覧**: `.kiro/specs/meeting-minutes-generation/tasks.md`

## 実装タスク

### Task 23: データモデルと設定

**ファイル**: `src/meeting_pipeline/models.py`

**追加するデータクラス**:

```python
@dataclass
class MinutesConfig:
    """議事録生成の設定"""
    enabled: bool
    model: str  # "gpt-4" または "gpt-3.5-turbo"
    language: str  # "auto", "ja", "en", など
    temperature: float = 0.3
    max_tokens: int = 4000

@dataclass
class ActionItem:
    """会議から抽出されたアクションアイテム"""
    task: str
    assignee: str  # 話者ラベルまたは名前
    deadline: Optional[str] = None  # ISO日付または自然言語
    timestamp: float = 0.0  # おおよそのタイムスタンプ（秒）

@dataclass
class Decision:
    """会議中に行われた決定"""
    text: str
    speaker: str  # 話者ラベル
    timestamp: float = 0.0  # おおよそのタイムスタンプ（秒）

@dataclass
class Topic:
    """会議で議論されたトピック"""
    title: str
    summary: str
    start: float = 0.0  # 開始タイムスタンプ（秒）
    end: float = 0.0  # 終了タイムスタンプ（秒）

@dataclass
class MeetingMinutes:
    """構造化された議事録"""
    schema_version: str  # "1.0"
    created_at: str  # ISO 8601タイムスタンプ
    meeting_title: str
    meeting_date: str  # ISO日付
    duration_sec: float
    participants: List[str]  # 話者ラベル
    summary: str  # 会議全体の要約
    decisions: List[Decision]
    action_items: List[ActionItem]
    topics: List[Topic]
    model_info: MinutesConfig
    generation_time_sec: float
```

**既存データクラスの拡張**:

```python
@dataclass
class PipelineConfig:
    # ... 既存フィールド ...
    generate_minutes: bool = False  # 新規
    minutes_model: str = "gpt-3.5-turbo"  # 新規
    minutes_language: str = "auto"  # 新規

@dataclass
class Timing:
    # ... 既存フィールド ...
    minutes_sec: float = 0.0  # 新規
```

---

### Task 24: 議事録生成コアモジュール

**ファイル**: `src/meeting_pipeline/minutes.py`（新規作成）

**実装する関数**:

1. **`generate_minutes(meeting_json_path, config, openai_api_key) -> MeetingMinutes`**
   - メイン関数
   - Meeting JSONを読み込み、OpenAI APIを呼び出し、議事録を生成

2. **`_load_meeting_json(path) -> MeetingJSON`**
   - Meeting JSONファイルを読み込んで検証

3. **`_prepare_prompt(meeting, language) -> str`**
   - OpenAI API用のプロンプトを準備
   - システムプロンプト + ユーザープロンプト（文字起こし）

4. **`_call_openai_api(prompt, config, api_key) -> str`**
   - OpenAI APIを呼び出す
   - リトライロジック: 最大3回、指数バックオフ（1秒、2秒、4秒）
   - エラーハンドリング: レート制限、タイムアウト、認証エラー

5. **`_parse_api_response(response, meeting) -> MeetingMinutes`**
   - APIレスポンス（JSON）をMeetingMinutes構造にパース

**システムプロンプト例**:

```python
SYSTEM_PROMPT = """あなたは専門的な議事録生成者です。会議の文字起こしを分析し、構造化された情報を抽出するのがあなたの仕事です。

出力形式: 以下の構造を持つJSON
{
  "summary": "会議全体の要約（100-300語）",
  "decisions": [{"text": "決定テキスト", "speaker": "Speaker 1", "timestamp": 123.4}],
  "action_items": [{"task": "タスク説明", "assignee": "Speaker 2", "deadline": "2024-03-15", "timestamp": 456.7}],
  "topics": [{"title": "トピック名", "summary": "トピック要約", "start": 0.0, "end": 300.0}]
}

ガイドライン:
- 簡潔かつ正確に
- 明示的に言及された決定とアクションアイテムのみを抽出
- 1-10の明確なトピックを特定
- 文字起こしの話者ラベルを使用
- トレーサビリティのためにタイムスタンプを保持
"""
```

**OpenAI API呼び出し例**:

```python
import openai
from openai import OpenAI

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model=config.model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=config.temperature,
    max_tokens=config.max_tokens,
    response_format={"type": "json_object"}
)

result = response.choices[0].message.content
```

**エラーハンドリング**:
- `FileNotFoundError`: Meeting JSONが見つからない
- `ValueError`: Meeting JSON形式が無効
- `openai.RateLimitError`: レート制限エラー（リトライ）
- `openai.APITimeoutError`: タイムアウト（リトライ）
- `openai.AuthenticationError`: 認証エラー（リトライしない）

---

### Task 25: トークン管理とチャンク化（オプション）

**注意**: ほとんどの会議（2時間未満）はgpt-3.5-turboのトークン制限内に収まるため、このタスクは時間があれば実装してください。優先度は低いです。

**実装する関数**:

1. **`_estimate_tokens(text, language) -> int`**
   - トークン数を推定
   - 英語: 約4文字/トークン
   - 日本語: 約1.5文字/トークン

2. **`_handle_chunking(meeting, max_tokens) -> List[str]`**
   - 文字起こしをチャンクに分割

3. **`_merge_chunked_results(results) -> MeetingMinutes`**
   - チャンク結果をマージ

---

### Task 26: 議事録出力生成

**ファイル**: `src/meeting_pipeline/output.py`

**追加する関数**:

1. **`generate_minutes_markdown(minutes: MeetingMinutes) -> str`**
   - Markdown形式の議事録を生成
   - 構造: ヘッダー、要約、決定事項、アクションアイテム、トピック

2. **`save_minutes_markdown(content: str, output_path: str) -> None`**
   - Markdownをファイルに保存

3. **`save_minutes_json(minutes: MeetingMinutes, output_path: str) -> None`**
   - JSONをファイルに保存（検証付き）

**Markdown形式例**:

```markdown
# 議事録: {title}

**日付**: {date}
**時間**: {duration}秒
**参加者**: {participants}

## 要約

{summary}

## 決定事項

1. [00:05:23] {decision_text} ({speaker}による)
2. ...

## アクションアイテム

| タスク | 担当者 | 期限 | タイムスタンプ |
|--------|--------|------|----------------|
| {task} | {assignee} | {deadline} | 00:10:45 |

## トピック

### {topic_title} [00:00:00 - 00:15:30]

{topic_summary}
```

---

### Task 27: CLI統合

**ファイル**: `src/meeting_pipeline/cli.py`

**追加する引数**:

```python
parser.add_argument(
    "--generate-minutes",
    action="store_true",
    help="OpenAI APIを使用して議事録を生成",
)
parser.add_argument(
    "--minutes-model",
    default="gpt-3.5-turbo",
    choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
    help="議事録生成用のOpenAIモデル（デフォルト: gpt-3.5-turbo）",
)
parser.add_argument(
    "--minutes-language",
    default="auto",
    help="議事録出力の言語（デフォルト: 入力から自動検出）",
)
```

**検証ロジック**:

```python
def validate_config(config: PipelineConfig) -> None:
    # ... 既存の検証 ...
    
    # 議事録生成が有効な場合、APIキーをチェック
    if config.generate_minutes:
        import os
        if not os.getenv("OPENAI_API_KEY"):
            print(
                "警告: OPENAI_API_KEYが設定されていません。議事録生成はスキップされます。",
                file=sys.stderr,
            )
```

---

### Task 28: パイプライン統合

**ファイル**: `src/meeting_pipeline/pipeline.py`

**追加するステージ**:

```python
# --- ステージ8: 議事録生成（オプション） ---
if config.generate_minutes:
    logger.info("ステージ: 議事録生成")
    t0 = time.time()
    
    # OpenAI APIキーを取得
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("環境にOPENAI_API_KEYが見つかりません")
        print("エラー: OPENAI_API_KEYが設定されていません。議事録生成をスキップします。", file=sys.stderr)
    else:
        try:
            # Meeting JSONパス
            meeting_json_path = os.path.join(config.output_dir, f"{basename}_meeting.json")
            
            # 議事録設定
            from .minutes import generate_minutes
            minutes_config = MinutesConfig(
                enabled=True,
                model=config.minutes_model,
                language=config.minutes_language if config.minutes_language != "auto" else config.language,
                temperature=0.3,
                max_tokens=4000,
            )
            
            # 議事録生成
            minutes = generate_minutes(
                meeting_json_path,
                minutes_config,
                openai_api_key,
            )
            
            # 議事録出力を保存
            minutes_md_path = os.path.join(config.output_dir, f"{basename}_minutes.md")
            minutes_json_path = os.path.join(config.output_dir, f"{basename}_minutes.json")
            
            from .output import generate_minutes_markdown, save_minutes_markdown, save_minutes_json
            md_content = generate_minutes_markdown(minutes)
            save_minutes_markdown(md_content, minutes_md_path)
            save_minutes_json(minutes, minutes_json_path)
            
            timing.minutes_sec = round(time.time() - t0, 1)
            logger.info("議事録生成完了: %.1f秒", timing.minutes_sec)
            
        except Exception as e:
            logger.error("議事録生成失敗: %s", e)
            print(f"エラー: 議事録生成失敗: {e}", file=sys.stderr)
            print("Meeting JSONと文字起こしは正常に保存されました。", file=sys.stderr)
            # 終了しない - 議事録生成はオプション
else:
    logger.info("議事録生成無効、スキップ")
```

**重要**: 議事録生成が失敗してもパイプライン全体を失敗させないこと。

---

## 依存関係の追加

**ファイル**: `requirements.txt`

```
openai>=1.0.0
python-dotenv>=1.0.0
```

インストール:
```bash
pip install openai python-dotenv
```

---

## 環境設定

**ファイル**: `.env`

```
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```

---

## テスト

実装完了後、以下のコマンドで動作確認してください:

```bash
# 環境設定
source venv/bin/activate
export OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d= -f2)
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)

# 議事録生成テスト（短い動画）
python meeting_pipeline.py temp/202602017_short_test.mp4 \
  --enable-diarization \
  --generate-minutes \
  --output-dir output/phase6_test

# 出力確認
ls -la output/phase6_test/
cat output/phase6_test/202602017_short_test_minutes.md
```

**期待される出力**:
- `202602017_short_test_meeting.json`（既存）
- `202602017_short_test_transcript.md`（既存）
- `202602017_short_test_minutes.md`（新規）
- `202602017_short_test_minutes.json`（新規）

---

## 実装順序

1. **Task 23**: データモデル（30分）
2. **Task 24**: 議事録生成コア（2時間）
3. **Task 26**: 出力生成（1時間）
4. **Task 27**: CLI統合（30分）
5. **Task 28**: パイプライン統合（1時間）
6. **Task 25**: チャンク化（オプション、1時間）

**推定時間**: 5-6時間（チャンク化なし）、6-7時間（チャンク化あり）

---

## 注意事項

1. **APIキーのセキュリティ**:
   - APIキーをログに出力しない
   - エラーメッセージにAPIキーを含めない

2. **エラーハンドリング**:
   - 議事録生成失敗でパイプラインを停止しない
   - Meeting JSONと文字起こしは常に保存する

3. **パフォーマンス**:
   - gpt-3.5-turboを推奨（高速で安価）
   - temperature=0.3で一貫性を確保

4. **テスト**:
   - 実装中は短い動画でテスト
   - 実際のAPI呼び出しを行うため、コストに注意

---

## Git Workflow

実装完了後:

```bash
git add .
git commit -m "Phase 6実装: 議事録生成機能追加

- Task 23: データモデル追加（MinutesConfig, ActionItem, Decision, Topic, MeetingMinutes）
- Task 24: 議事録生成コアモジュール（minutes.py）
- Task 26: 議事録出力生成（Markdown + JSON）
- Task 27: CLI統合（--generate-minutes, --minutes-model, --minutes-language）
- Task 28: パイプライン統合（ステージ8）
- 依存関係追加: openai, python-dotenv"
git push
```

---

## 成功基準

Phase 6実装は以下の条件で完了です:

1. ✅ すべてのデータモデルが追加されている
2. ✅ `minutes.py`が実装されている
3. ✅ OpenAI API呼び出しが動作する（リトライロジック付き）
4. ✅ Markdown + JSON出力が生成される
5. ✅ CLIオプションが動作する
6. ✅ パイプライン統合が完了している
7. ✅ 手動テストが成功する（短い動画で確認）
8. ✅ エラーハンドリングが適切

---

## 次のステップ

実装完了後、Codexがユニットテストと統合テストを実施します（Task 29-31）。
