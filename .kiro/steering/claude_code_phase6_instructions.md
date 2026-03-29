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

## Spec参照

Phase 6の詳細仕様は以下のファイルを参照してください:

- **要件定義**: `.kiro/specs/meeting-minutes-generation/requirements.md`
- **設計書**: `.kiro/specs/meeting-minutes-generation/design.md`
- **タスク一覧**: `.kiro/specs/meeting-minutes-generation/tasks.md`
- **実装指示書**: `docs/CLAUDE_CODE_PHASE6_MESSAGE.md`

## 実装タスク概要

### Task 23: データモデルと設定
- `src/meeting_pipeline/models.py`に新規データクラスを追加
- MinutesConfig, ActionItem, Decision, Topic, MeetingMinutes
- PipelineConfigとTimingを拡張

### Task 24: 議事録生成コアモジュール
- `src/meeting_pipeline/minutes.py`を新規作成
- generate_minutes(), _load_meeting_json(), _prepare_prompt()
- _call_openai_api()（リトライロジック付き）
- _parse_api_response()

### Task 25: トークン管理とチャンク化（オプション）
- 優先度低：時間があれば実装

### Task 26: 議事録出力生成
- `src/meeting_pipeline/output.py`に関数を追加
- generate_minutes_markdown(), save_minutes_markdown()
- save_minutes_json()

### Task 27: CLI統合
- `src/meeting_pipeline/cli.py`に引数を追加
- --generate-minutes, --minutes-model, --minutes-language

### Task 28: パイプライン統合
- `src/meeting_pipeline/pipeline.py`にステージ8を追加
- エラーハンドリング：議事録失敗でパイプラインを停止しない

## 重要な実装ポイント

### OpenAI API呼び出し

```python
from openai import OpenAI

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model=config.model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.3,
    max_tokens=4000,
    response_format={"type": "json_object"}
)
```

### リトライロジック

- 最大3回リトライ
- 指数バックオフ: 1秒、2秒、4秒
- リトライ対象: RateLimitError, APITimeoutError
- リトライしない: AuthenticationError

### エラーハンドリング

- 議事録生成失敗でパイプライン全体を失敗させない
- Meeting JSONと文字起こしは常に最初に保存
- エラーはログに記録しユーザーに報告

## 依存関係

requirements.txtに追加:
```
openai>=1.0.0
python-dotenv>=1.0.0
```

## 環境設定

.envファイル:
```
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```

## テスト

実装完了後、以下で動作確認:

```bash
source venv/bin/activate
export OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d= -f2)
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)

python meeting_pipeline.py temp/202602017_short_test.mp4 \
  --enable-diarization \
  --generate-minutes \
  --output-dir output/phase6_test
```

## 成功基準

1. ✅ すべてのデータモデルが追加されている
2. ✅ minutes.pyが実装されている
3. ✅ OpenAI API呼び出しが動作する
4. ✅ Markdown + JSON出力が生成される
5. ✅ CLIオプションが動作する
6. ✅ パイプライン統合が完了している
7. ✅ 手動テストが成功する

## 推定時間

- Task 23: 30分
- Task 24: 2時間
- Task 26: 1時間
- Task 27: 30分
- Task 28: 1時間
- **合計**: 5-6時間

詳細は `docs/CLAUDE_CODE_PHASE6_MESSAGE.md` を参照してください。
