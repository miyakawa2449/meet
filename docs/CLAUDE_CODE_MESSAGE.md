# Phase 4 実装依頼: macOS MPS/CPU対応とクロスプラットフォーム最適化

## あなたの役割

あなたは **実装担当（Claude Code）** です。Phase 4 の実装を行ってください。
テストコードの作成と検証は **Codex** が担当します。

## 背景

meeting speaker diarization pipeline の Phase 1〜3 は完了済みで、全49テストが合格しています。
Phase 4 では macOS 環境での MPS/CPU デバイス対応とクロスプラットフォーム最適化を行います。

**注意**: Phase 1 は Windows CUDA 環境を想定して設計されましたが、実際のテストは macOS CPU 環境で実施されています。CUDA 環境でのテストは未実施ですが、コードはデバイス非依存に設計されているため、Phase 4 で macOS 対応を検証します。将来 CUDA 環境が利用可能になった際に追加検証を行います。

## 実装タスク（`.kiro/specs/meeting-speaker-diarization-pipeline/tasks.md` のタスク17.1〜17.3）

### Task 17.1: macOS MPS/CPU エンドツーエンドテスト

1. MPS デバイスが利用可能か確認:
   ```python
   import torch
   print(f"MPS available: {torch.backends.mps.is_available()}")
   ```

2. 以下のコマンドでテスト実行（MPS が利用可能な場合）:
   ```bash
   python meeting_pipeline.py --input-file temp/202602017_short_test.mp4 --device mps --output-dir output/test_mps
   ```

3. CPU デバイスでテスト実行:
   ```bash
   python meeting_pipeline.py --input-file temp/202602017_short_test.mp4 --device cpu --output-dir output/test_cpu
   ```

4. Auto デバイス選択でテスト実行:
   ```bash
   python meeting_pipeline.py --input-file temp/202602017_short_test.mp4 --device auto --output-dir output/test_auto
   ```

5. 各テストで JSON と Markdown が正しく生成されることを確認

### Task 17.2: compute_type 自動選択ロジックの最適化

1. `src/meeting_pipeline/asr.py` の `_determine_compute_type()` 関数を確認
2. MPS → CPU フォールバックロジックが正しく動作することを確認
3. フォールバック時に明確な警告ログを出力するよう改善
4. 現在の実装:
   - CUDA/MPS: `float16`
   - CPU: `int8`
   - faster-whisper + MPS → CPU へ自動フォールバック

### Task 17.3: クロスプラットフォーム一貫性の検証

1. CPU と MPS（利用可能な場合）で同じ入力ファイルを処理
2. 生成された JSON ファイルの構造を比較:
   - `schema_version` が "1.0" であること
   - 全必須フィールドが存在すること
   - `pipeline.device.resolved` が正しいデバイスを反映していること
3. Markdown フォーマットが一貫していることを確認

## 詳細な実装手順

`.kiro/steering/claude_code_phase4_instructions.md` に詳細な手順、コード例、注意事項が記載されています。必ず確認してください。

## 重要な制約

- **faster-whisper は MPS を直接サポートしていません** → CPU へのフォールバックが必要（既に実装済み）
- pyannote-audio は MPS をサポートしている可能性があります → テストで確認
- テスト用動画: `temp/202602017_short_test.mp4` (5分、324秒)
- 環境変数 `HF_TOKEN` が必要（diarization 用）

## 実行前の準備

```bash
source venv/bin/activate
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)
```

## 成功基準

- [ ] macOS で MPS デバイスが正常動作（利用可能な場合）
- [ ] macOS で CPU デバイスが正常動作
- [ ] デバイス自動選択が正しく機能（CUDA > MPS > CPU）
- [ ] compute_type が各デバイスに最適化されている
- [ ] 出力 JSON/Markdown がデバイス間で構造的に一貫
- [ ] 全既存テスト（49件）が引き続き合格
- [ ] デバイスフォールバック時に適切なログが出力される

## あなたがやること（実装のみ）

1. Task 17.1: エンドツーエンドテストを実行して動作確認
2. Task 17.2: compute_type ロジックの確認と改善（必要に応じて）
3. Task 17.3: クロスプラットフォーム一貫性の検証
4. 既存テストが全て合格することを確認: `pytest tests/ -v`

## あなたがやらないこと（Codex が担当）

- Task 17.4: 新規ユニットテストの作成
- テストコードの実装

## 完了後の報告

実装完了後、以下を報告してください：

1. 各デバイス（MPS/CPU/auto）でのテスト結果
2. 生成された JSON/Markdown の確認結果
3. 既存テスト（49件）の実行結果
4. 改善した点や気づいた点

その後、Codex にテストコード作成を依頼します。

## Git コミット

実装完了後、以下を実行してください：

```bash
git add .
git commit -m "Phase 4実装: macOS MPS/CPU対応とクロスプラットフォーム最適化（テスト前）"
```

よろしくお願いします。
