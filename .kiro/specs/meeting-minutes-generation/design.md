# 設計書

## 概要

このドキュメントは、Phase 6: 議事録生成機能の設計を説明します。この機能は、OpenAIのGPTモデルを統合して文字起こしデータを処理し、構造化された議事録を生成することで、既存の会議話者分離パイプラインに自動議事録生成機能を追加します。

## アーキテクチャ

### 高レベルアーキテクチャ

```
Meeting JSON (Phase 1-5の出力)
         ↓
   議事録生成器
         ↓
    OpenAI API
    (GPT-4 / GPT-3.5-turbo)
         ↓
   ┌─────┴─────┐
   ↓           ↓
Minutes.md  Minutes.json
```

### モジュール構造

```
src/meeting_pipeline/
├── minutes.py          # 新規: 議事録生成ロジック
├── models.py           # 変更: 議事録関連データモデルを追加
├── output.py           # 変更: 議事録出力関数を追加
├── cli.py              # 変更: --generate-minutesオプションを追加
└── pipeline.py         # 変更: 議事録生成ステージを統合
```

## データモデル

### 新規データモデル (models.py)

```python
@dataclass
class MinutesConfig:
    """議事録生成の設定"""
    enabled: bool
    model: str  # "gpt-4" または "gpt-3.5-turbo"
    language: str  # "auto", "ja", "en", など
    temperature: float  # デフォルト: 0.3
    max_tokens: int  # デフォルト: 4000

@dataclass
class ActionItem:
    """会議から抽出されたアクションアイテム"""
    task: str
    assignee: str  # 話者ラベルまたは名前
    deadline: Optional[str]  # ISO日付または自然言語
    timestamp: float  # おおよそのタイムスタンプ（秒）

@dataclass
class Decision:
    """会議中に行われた決定"""
    text: str
    speaker: str  # 話者ラベル
    timestamp: float  # おおよそのタイムスタンプ（秒）

@dataclass
class Topic:
    """会議で議論されたトピック"""
    title: str
    summary: str
    start: float  # 開始タイムスタンプ（秒）
    end: float  # 終了タイムスタンプ（秒）

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

@dataclass
class PipelineConfig:
    # ... 既存フィールド ...
    generate_minutes: bool = False  # 新規
    minutes_model: str = "gpt-3.5-turbo"  # 新規
    minutes_language: str = "auto"  # 新規
```

### 変更されたデータモデル

```python
@dataclass
class Timing:
    # ... 既存フィールド ...
    minutes_sec: float = 0.0  # 新規: 議事録生成時間
```


## コンポーネント設計

### 1. 議事録生成モジュール (minutes.py)

**目的**: OpenAI APIを使用して議事録を生成するコアロジック。

**主要関数**:

```python
def generate_minutes(
    meeting_json_path: str,
    config: MinutesConfig,
    openai_api_key: str,
) -> MeetingMinutes:
    """
    Meeting JSONから議事録を生成する。
    
    Args:
        meeting_json_path: Meeting JSONファイルへのパス
        config: 議事録生成設定
        openai_api_key: OpenAI APIキー
    
    Returns:
        MeetingMinutesオブジェクト
    
    Raises:
        FileNotFoundError: meeting_json_pathが存在しない場合
        ValueError: Meeting JSONが無効な場合
        OpenAIError: リトライ後にAPI呼び出しが失敗した場合
    """
    pass

def _load_meeting_json(path: str) -> MeetingJSON:
    """Meeting JSONを読み込んで検証する。"""
    pass

def _prepare_prompt(meeting: MeetingJSON, language: str) -> str:
    """
    OpenAI API用のプロンプトを準備する。
    
    含まれる内容:
    - 会議メタデータ（時間、話者）
    - タイムスタンプ付き完全な文字起こし
    - 抽出の指示（要約、決定、アクション、トピック）
    """
    pass

def _call_openai_api(
    prompt: str,
    config: MinutesConfig,
    api_key: str,
) -> str:
    """
    リトライロジック付きでOpenAI APIを呼び出す。
    
    リトライ戦略:
    - 最大3回リトライ
    - 指数バックオフ: 1秒、2秒、4秒
    - リトライ対象: レート制限、タイムアウト、サーバーエラー
    - リトライしない: 認証、無効なリクエスト
    """
    pass

def _parse_api_response(response: str, meeting: MeetingJSON) -> MeetingMinutes:
    """
    OpenAI APIレスポンスをMeetingMinutes構造にパースする。
    
    期待されるレスポンス形式（JSON）:
    {
      "summary": "...",
      "decisions": [{"text": "...", "speaker": "...", "timestamp": 123.4}, ...],
      "action_items": [{"task": "...", "assignee": "...", "deadline": "...", "timestamp": 123.4}, ...],
      "topics": [{"title": "...", "summary": "...", "start": 123.4, "end": 456.7}, ...]
    }
    """
    pass
```

**エラーハンドリング**:
- `FileNotFoundError`: Meeting JSONが見つからない
- `ValueError`: Meeting JSON形式が無効
- `OpenAIError`: APIエラー（レート制限、タイムアウト、サーバーエラー）
- `AuthenticationError`: 無効なAPIキー

**ログ**:
- 議事録生成開始をログ
- API呼び出しの試行とリトライをログ
- チャンク化が適用された場合はログ
- タイミング付きで完了をログ

### 2. 出力モジュール拡張 (output.py)

**新規関数**:

```python
def generate_minutes_markdown(minutes: MeetingMinutes) -> str:
    """
    Markdown形式の議事録を生成する。
    
    構造:
    # 議事録: {title}
    
    **日付**: {date}
    **時間**: {duration}
    **参加者**: {participants}
    
    ## 要約
    {summary}
    
    ## 決定事項
    1. [{timestamp}] {text} ({speaker}による)
    2. ...
    
    ## アクションアイテム
    | タスク | 担当者 | 期限 | タイムスタンプ |
    |--------|--------|------|----------------|
    | ...    | ...    | ...  | ...            |
    
    ## トピック
    ### {topic_title} [{start} - {end}]
    {topic_summary}
    """
    pass

def save_minutes_markdown(content: str, output_path: str) -> None:
    """議事録Markdownをファイルに保存する。"""
    pass

def save_minutes_json(minutes: MeetingMinutes, output_path: str) -> None:
    """検証付きで議事録JSONをファイルに保存する。"""
    pass
```

### 3. CLI拡張 (cli.py)

**新規引数**:

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

**検証**:
- `--generate-minutes`が設定されている場合、環境に`OPENAI_API_KEY`があるか確認
- モデル選択を検証

### 4. パイプライン統合 (pipeline.py)

**新規ステージ**:

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
            # まずMeeting JSONを保存（まだ保存されていない場合）
            meeting_json_path = os.path.join(config.output_dir, f"{basename}_meeting.json")
            
            # 議事録を生成
            minutes_config = MinutesConfig(
                enabled=True,
                model=config.minutes_model,
                language=config.minutes_language if config.minutes_language != "auto" else config.language,
                temperature=0.3,
                max_tokens=4000,
            )
            
            minutes = generate_minutes(
                meeting_json_path,
                minutes_config,
                openai_api_key,
            )
            
            # 議事録出力を保存
            minutes_md_path = os.path.join(config.output_dir, f"{basename}_minutes.md")
            minutes_json_path = os.path.join(config.output_dir, f"{basename}_minutes.json")
            
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

**エラーハンドリング戦略**:
- 議事録生成の失敗はパイプライン全体を失敗させない
- Meeting JSONと文字起こしは常に最初に保存される
- エラーはログに記録されユーザーに報告される
- 議事録生成が失敗してもパイプラインは継続する


## OpenAI API統合

### プロンプト設計

**システムプロンプト**:
```
あなたは専門的な議事録生成者です。会議の文字起こしを分析し、構造化された情報を抽出するのがあなたの仕事です。

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
```

**ユーザープロンプトテンプレート**:
```
会議時間: {duration}秒
参加者: {speaker_list}

文字起こし:
{formatted_transcript}

指定された形式でJSON形式の議事録を生成してください。
```

### API呼び出し設定

```python
{
    "model": config.model,  # "gpt-3.5-turbo" または "gpt-4"
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    "temperature": 0.3,  # 一貫性のための低温度
    "max_tokens": 4000,
    "response_format": {"type": "json_object"}  # JSON出力を強制
}
```

### トークン管理

**推定**:
- 英語: トークンあたり約4文字
- 日本語: トークンあたり約1.5文字
- システムプロンプト: 約200トークン
- レスポンス: 約1000-2000トークン
- 文字起こしに利用可能: max_tokens - system - response ≈ 2000トークン

**チャンク化戦略**（必要な場合）:
1. 文字起こしトークンを推定
2. 2000トークン超の場合、チャンクに分割
3. 各チャンクを個別に処理
4. 結果をマージ

**モデル制限**:
- gpt-3.5-turbo: 16Kコンテキストウィンドウ
- gpt-4: 8Kコンテキストウィンドウ
- gpt-4-turbo: 128Kコンテキストウィンドウ

ほとんどの会議（2時間未満）では、gpt-3.5-turboでチャンク化は不要です。

## ファイル命名規則

| 入力 | Meeting JSON | 文字起こしMD | 議事録MD | 議事録JSON |
|------|--------------|--------------|----------|------------|
| video.mp4 | video_meeting.json | video_transcript.md | video_minutes.md | video_minutes.json |

## テスト戦略

### ユニットテスト (tests/test_minutes.py)

1. **API統合テスト**（モック使用）:
   - 成功したAPI呼び出しをテスト
   - リトライロジックをテスト（レート制限、タイムアウト）
   - 認証エラーをテスト
   - 無効なレスポンス形式をテスト

2. **プロンプト生成テスト**:
   - プロンプトフォーマットをテスト
   - 言語検出をテスト
   - 文字起こし切り詰めをテスト

3. **レスポンスパーステスト**:
   - 有効なJSONレスポンスをテスト
   - 欠落フィールドをテスト
   - 無効なタイムスタンプをテスト
   - 空リストをテスト

4. **チャンク化テスト**:
   - トークン推定をテスト
   - チャンク分割をテスト
   - 結果マージをテスト

5. **出力生成テスト**:
   - Markdownフォーマットをテスト
   - JSONシリアライゼーションをテスト
   - ラウンドトリップ検証をテスト

6. **エラーハンドリングテスト**:
   - APIキー欠落をテスト
   - 無効なMeeting JSONをテスト
   - ファイル未検出をテスト
   - APIエラーをテスト

### 統合テスト

1. **エンドツーエンドテスト**:
   - `--generate-minutes`付きで完全なパイプラインを実行
   - すべての出力が生成されることを検証
   - 議事録コンテンツが妥当であることを検証

2. **CLIテスト**:
   - 引数パースをテスト
   - 検証をテスト（APIキー欠落）
   - エラーメッセージをテスト

## パフォーマンス考慮事項

### 期待されるパフォーマンス

| 会議時間 | 文字起こしサイズ | API呼び出し時間 | 議事録生成総時間 |
|----------|------------------|-----------------|------------------|
| 30分 | 約5Kトークン | 10-20秒 | 15-25秒 |
| 1時間 | 約10Kトークン | 20-30秒 | 25-35秒 |
| 2時間 | 約20Kトークン | 30-40秒（またはチャンク化） | 40-60秒 |

### 最適化戦略

1. **デフォルトでgpt-3.5-turboを使用**（GPT-4より高速で安価）
2. **低温度**（0.3）で一貫性と速度を確保
3. **ストリーミングAPI**（サポートされている場合）で体感レイテンシを削減
4. **並列処理**（複数チャンクの場合） - 将来の拡張
5. **キャッシング**（将来の拡張） - 同一入力のAPIレスポンスをキャッシュ

## セキュリティ考慮事項

1. **APIキー管理**:
   - `.env`ファイルからのみ読み込む
   - APIキーをログや出力に記録しない
   - API呼び出し前にキー形式を検証

2. **入力検証**:
   - Meeting JSONスキーマを検証
   - ファイルパスをサニタイズ
   - 悪用を防ぐために文字起こしサイズを制限

3. **エラーメッセージ**:
   - エラーメッセージにAPIキーを露出しない
   - ユーザー向けエラーに内部パスを露出しない

## 将来の拡張

1. **カスタムプロンプト**: ユーザーがカスタムシステムプロンプトを提供可能に
2. **多言語サポート**: より良い言語検出と翻訳
3. **トピック分類**: より洗練されたトピック検出
4. **話者名マッピング**: 話者ラベルを実名にマッピング
5. **感情分析**: 議論のトーンと感情を検出
6. **会議比較**: 複数の会議の議事録を比較
7. **エクスポート形式**: PDF、DOCX、HTML出力
8. **ストリーミング出力**: 文字起こし中のリアルタイム議事録生成

## 依存関係

### 新規依存関係

```
openai>=1.0.0  # OpenAI Python SDK
python-dotenv>=1.0.0  # .envファイル読み込み用（まだない場合）
```

### 既存依存関係（変更なし）

- torch
- pyannote-audio
- faster-whisper
- openai-whisper（オプション）

## 移行と互換性

### 後方互換性

- すべての既存機能は変更なし
- 議事録生成は`--generate-minutes`フラグでオプトイン
- 既存のAPIや出力に破壊的変更なし

### バージョン互換性

- Meeting JSONスキーマバージョン: 1.0（変更なし）
- 議事録JSONスキーマバージョン: 1.0（新規）
- 両スキーマは独立してバージョン管理される

## ドキュメント更新

### README.md更新

1. 「議事録生成」セクションを追加
2. `--generate-minutes`オプションを文書化
3. 使用例を提供
4. OPENAI_API_KEYセットアップを文書化
5. APIエラーのトラブルシューティングセクションを追加

### 使用例

```bash
# デフォルト設定で議事録を生成
python meeting_pipeline.py video.mp4 \
  --enable-diarization \
  --generate-minutes

# GPT-4で議事録を生成
python meeting_pipeline.py video.mp4 \
  --enable-diarization \
  --generate-minutes \
  --minutes-model gpt-4

# 英語で議事録を生成
python meeting_pipeline.py video.mp4 \
  --enable-diarization \
  --generate-minutes \
  --minutes-language en
```

## 成功基準

Phase 6は以下の条件で完了です:

1. ✅ 議事録生成モジュールが実装されテストされている
2. ✅ OpenAI API統合がリトライロジック付きで動作する
3. ✅ MarkdownとJSON出力が正しく生成される
4. ✅ CLIオプションが追加され検証されている
5. ✅ パイプライン統合がエラーハンドリング付きで完了している
6. ✅ すべてのユニットテストが合格する（目標: 15+新規テスト）
7. ✅ 実際のAPI呼び出しでエンドツーエンドテストが合格する
8. ✅ ドキュメントが更新されている
9. ✅ エラーメッセージが明確で役立つ
10. ✅ パフォーマンスが要件を満たす（1時間の会議で60秒未満）

## タイムライン推定

| タスク | 推定時間 |
|--------|----------|
| データモデル | 30分 |
| 議事録生成コア | 2時間 |
| OpenAI API統合 | 1時間 |
| 出力生成 | 1時間 |
| CLI統合 | 30分 |
| パイプライン統合 | 1時間 |
| ユニットテスト | 2時間 |
| 統合テスト | 1時間 |
| ドキュメント | 1時間 |
| **合計** | **10時間** |

集中して作業すれば、Phase 6は1日で完了できます。
