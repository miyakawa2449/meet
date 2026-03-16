# Implementation Guide for Claude Code

## 概要

このドキュメントは、会議録AIパイプライン（話者分離＋文字起こし）機能の実装を担当するClaude Codeへの指示書です。

## プロジェクト情報

- **機能名**: meeting-speaker-diarization-pipeline
- **目的**: 動画/音声ファイルから話者分離とASRを実行し、話者ラベル付き議事録を生成
- **開発環境**: macOS（MPS/CPU）で開発、Windows（CUDA）でも動作可能な設計
- **新規作成ファイル**: `meeting_pipeline.py`

## 重要な制約

1. **変更禁止**: `bench_transcribe.py` は絶対に変更しない（ASRベンチ専用として保持）
2. **話者数**: 2人固定にしない（1〜N人を動的に検出）
3. **UNKNOWN話者**: 話者割当できない区間を破棄せず、UNKNOWNとして保持
4. **順次実行**: 話者分離とASRを同時実行しない（メモリ効率のため）
5. **クロスプラットフォーム**: Mac/Windows両対応を最初から考慮

## 参照ドキュメント

実装時は以下のドキュメントを参照してください：

- **要件**: `.kiro/specs/meeting-speaker-diarization-pipeline/requirements.md`
  - 16要件、96受入基準
  - 各タスクに紐付く要件番号を確認

- **設計**: `.kiro/specs/meeting-speaker-diarization-pipeline/design.md`
  - システムアーキテクチャ
  - 9コンポーネントの詳細設計
  - データモデル（JSON Schema v1.0）
  - 29個の正確性プロパティ
  - エラーハンドリング戦略

- **タスク**: `.kiro/specs/meeting-speaker-diarization-pipeline/tasks.md`
  - 実装タスクリスト
  - Phase 1（基本パイプライン）とPhase 2（Markdown生成）が必須
  - Phase 3（精度改善）とPhase 4（クロスプラットフォーム最適化）はオプション

- **既存仕様**: `spec/meeting_pipeline_spec_v1.md`
  - 元の詳細仕様書（参考用）

## 実装の進め方

### Phase 1: 基本パイプライン（JSON出力まで）

**目標**: 話者分離→ASR→アライン→JSON出力が正しく動作すること

**実装タスク**:
1. プロジェクト構造とコア型定義の作成（タスク1）
2. CLI Parserの実装（タスク2.1, 2.2）
3. Device Resolverの実装（タスク3.1, 3.2）
4. Audio Extractorの実装（タスク4.1, 4.2, 4.3）
5. Diarization Engineの実装（タスク5.1, 5.2, 5.3, 5.4）
6. ASR Engineの実装（タスク6.1, 6.2, 6.3）
7. Alignment Moduleの実装（タスク7.1, 7.2, 7.3）
8. JSON Generatorの実装（タスク8.1, 8.2, 8.3, 8.4, 8.5）
9. メインパイプライン統合（タスク9.1, 9.2, 9.3）
10. Checkpoint - Phase 1完了確認（タスク10）

**検証**: Codexがテストタスク（1.1, 2.3, 3.3, 3.4, 4.4, 4.5, 5.5, 6.4, 7.4, 7.5, 7.6, 8.6, 8.7, 8.8, 9.4）を実装

### Phase 2: Markdown生成

**目標**: 人間が読める transcript.md を生成できること

**実装タスク**:
11. Markdown Generatorの実装（タスク11.1, 11.2, 11.3, 11.4）
12. 出力フォーマット制御の実装（タスク12.1）
13. Benchmark Loggerの実装（タスク13.1, 13.2）- オプション
14. Checkpoint - Phase 2完了確認（タスク14）

**検証**: Codexがテストタスク（11.5, 11.6, 11.7, 12.2, 13.3）を実装

### Phase 3 & 4: オプション

Phase 3（精度改善）とPhase 4（クロスプラットフォーム最適化）は、Phase 1とPhase 2が完了してから検討します。

## 実装時の注意点

### 1. タスクの進め方

- tasks.mdのタスク番号順に実装
- 各タスクの要件番号（_Requirements: X.X_）を確認
- サブタスクがある場合は、サブタスクから順に実装

### 2. コーディング規約

- Python 3.10+を想定
- dataclassを使用して型定義
- 型ヒントを必ず付ける
- docstringを主要な関数に追加
- エラーメッセージは英語で統一

### 3. エラーハンドリング

design.mdの「Error Handling」セクションを参照：
- 設定エラー: exit code 1
- 環境エラー: exit code 2
- 処理エラー: exit code 3
- データエラー: exit code 4

### 4. リソース管理

```python
# Diarization Engine後のモデル解放
diarization_pipeline = None
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()
```

### 5. JSON Schema v1.0

design.mdの「Data Models」セクションに完全な定義があります。必ず準拠してください。

### 6. デバイス選択ロジック

```python
def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested
```

### 7. max_overlapアルゴリズム

design.mdの「Alignment Module」セクションに詳細な実装例があります。

## 実装開始時のコマンド例

```bash
# 依存パッケージのインストール（必要に応じて）
pip install pyannote-audio faster-whisper torch

# HF_TOKENの設定（話者分離に必要）
export HF_TOKEN=your_token_here

# 実装後のテスト実行例
python meeting_pipeline.py sample.mp4 \
  --device auto \
  --enable-diarization \
  --asr-engine faster-whisper \
  --asr-model medium \
  --language ja \
  --output-dir output \
  --format both
```

## Checkpoint での確認項目

### Phase 1完了時（タスク10）

- [ ] meeting_pipeline.pyが作成されている
- [ ] bench_transcribe.pyが変更されていない
- [ ] サンプル音声でエンドツーエンドテストが成功
- [ ] Meeting JSONが正しく生成される
- [ ] speakers配列にUNKNOWNが含まれる
- [ ] 全要件の受入基準を満たしている

### Phase 2完了時（タスク14）

- [ ] Transcript Markdownが正しく生成される
- [ ] 全出力フォーマット（json, md, both）が動作
- [ ] タイムスタンプがHH:MM:SS形式でゼロパディング
- [ ] Unknown話者ブロックが破綻しない

## 質問・不明点がある場合

実装中に不明点があれば、以下を確認してください：

1. design.mdの該当コンポーネントセクション
2. requirements.mdの該当要件と受入基準
3. spec/meeting_pipeline_spec_v1.mdの詳細説明

それでも不明な場合は、ユーザーに質問してください。

## 最終成果物

Phase 1とPhase 2完了時点で以下が揃います：

- `meeting_pipeline.py` - メインスクリプト
- `output/{basename}_meeting.json` - 統合ログ（JSON）
- `output/{basename}_transcript.md` - 会議ログ（Markdown）
- テストコード（Codexが作成）
- 更新された `README.md` と `requirements.txt`

---

**実装開始時の指示例**:

```
Phase 1の実装を開始してください。
#tasks.md のタスク1から9まで順に実装してください。
#design.md と #requirements.md を参照してください。
bench_transcribe.pyは変更しないでください。
```
