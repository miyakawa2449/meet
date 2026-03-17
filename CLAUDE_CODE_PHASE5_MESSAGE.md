# Phase 5 実装依頼: Windows CUDA環境での検証

## あなたの役割

あなたは **実装担当（Claude Code）** です。Phase 5 の検証作業を行ってください。
テストコードの作成は **Codex** が担当します（必要に応じて）。

## 背景

meeting speaker diarization pipeline の Phase 1〜4 は macOS 環境で完了済みです（全56テスト合格）。
Phase 5 では Windows CUDA 環境でパイプラインが正常動作することを検証し、パフォーマンスを測定します。

**注意**: Phase 1 は Windows CUDA 環境を想定して設計されましたが、実際の開発とテストは macOS で実施されました。Phase 5 で初めて CUDA 環境での動作を検証します。

## 実装タスク（`.kiro/specs/meeting-speaker-diarization-pipeline/tasks.md` のタスク19.1〜19.3）

### Task 19.1: CUDA環境でのパイプライン実行テスト

**目的**: Windows CUDA 環境でパイプラインが正常動作することを確認

**実施内容**:

1. **CUDA 利用可能性の確認**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA version: {torch.version.cuda}")
   ```

2. **CUDA デバイスでテスト実行**:
   ```bash
   python meeting_pipeline.py --input-file temp/202602017_short_test.mp4 --device cuda --output-dir output/test_cuda
   ```

3. **Auto デバイス選択でテスト実行**:
   ```bash
   python meeting_pipeline.py --input-file temp/202602017_short_test.mp4 --device auto --output-dir output/test_auto
   ```
   - CUDA が最優先で選択されることを確認

4. **CPU デバイスでテスト実行**（フォールバック確認）:
   ```bash
   python meeting_pipeline.py --input-file temp/202602017_short_test.mp4 --device cpu --output-dir output/test_cpu
   ```

**期待される動作**:
- CUDA デバイスが検出され、正常に使用される
- compute_type は float16（失敗時は int8 に自動リトライ）
- JSON と Markdown が正しく生成される
- メモリ管理が正常（OOM エラーなし）

### Task 19.2: CUDA環境での出力一貫性検証

**目的**: Windows CUDA の出力が macOS と構造的に一貫していることを確認

**検証内容**:

1. **JSON スキーマ構造の確認**:
   - 生成された JSON ファイルを開く
   - 全必須フィールドが存在することを確認
   - `schema_version` が "1.0" であることを確認

2. **デバイスメタデータの確認**:
   ```python
   import json
   with open("output/test_cuda/202602017_short_test_meeting.json") as f:
       cuda_json = json.load(f)
   
   print(f"Device resolved: {cuda_json['pipeline']['device']['resolved']}")
   print(f"ASR device: {cuda_json['pipeline']['asr']['device']}")
   print(f"ASR compute_type: {cuda_json['pipeline']['asr']['compute_type']}")
   ```
   - `pipeline.device.resolved` が "cuda"
   - `pipeline.asr.device` が "cuda"
   - `pipeline.asr.compute_type` が "float16" または "int8"

3. **Markdown フォーマットの確認**:
   - タイムスタンプが HH:MM:SS 形式
   - 話者グループ化が正しい
   - macOS 出力と同じ構造

### Task 19.3: CUDA環境でのパフォーマンス測定

**目的**: CUDA 環境のパフォーマンスを測定し、記録する

**測定手順**:

1. **ベンチマーク実行**:
   ```bash
   python meeting_pipeline.py --input-file temp/202602017_short_test.mp4 --device cuda --output-dir output/bench_cuda --bench-jsonl bench/phase5_cuda.jsonl --run-id phase5_cuda_test --note "Phase 5 CUDA verification"
   ```

2. **タイミングデータの収集**:
   - 生成された JSON から timing セクションを抽出
   - 各ステージの処理時間を記録
   - macOS CPU/MPS と比較

3. **期待されるパフォーマンス**:
   - Diarization: CUDA は CPU より大幅に高速、MPS と同等以上
   - ASR: CUDA は CPU より 3-5倍高速
   - Total: 5分動画を 30-60秒で処理（CPU は ~330秒）

4. **結果の記録**:
   - パフォーマンス比較表を作成
   - GPU メモリ使用量を記録（可能であれば）
   - 気づいた点をメモ

**参考: macOS での結果**:

| Device | Platform | Diarization | ASR | Total |
|--------|----------|-------------|-----|-------|
| CPU | macOS | 207.9s | ~120s | ~330s |
| MPS | macOS | 17.7s | ~120s (CPU fallback) | ~140s |

## 詳細な実装手順

`.kiro/steering/claude_code_phase5_instructions.md` に詳細な手順、トラブルシューティング、注意事項が記載されています。必ず確認してください。

## 重要な制約（Windows 環境）

### 環境準備

**PowerShell**:
```powershell
.\venv\Scripts\Activate.ps1
$env:HF_TOKEN = (Get-Content .env | Select-String "HF_TOKEN").ToString().Split("=")[1]
```

**Command Prompt**:
```cmd
venv\Scripts\activate.bat
set HF_TOKEN=<your_token_from_.env>
```

### パス指定

- Python コードでは forward slash を使用: `temp/202602017_short_test.mp4`
- Windows でも正しく動作します

### テスト実行

```bash
# 全テスト実行
pytest tests/ -v

# 期待結果: 56/56 tests pass
```

## 成功基準

- [ ] Windows で CUDA デバイスが正常動作
- [ ] Auto デバイス選択で CUDA が選択される
- [ ] compute_type が float16（または int8 フォールバック）
- [ ] 出力 JSON/Markdown が macOS と構造的に一貫
- [ ] 全56テストが Windows でも合格
- [ ] パフォーマンスメトリクスが記録される
- [ ] GPU メモリが適切に管理される（OOM なし）

## あなたがやること（検証作業）

1. Task 19.1: CUDA/Auto/CPU デバイスでパイプライン実行
2. Task 19.2: 出力の一貫性検証（JSON/Markdown）
3. Task 19.3: パフォーマンス測定とベンチマーク記録
4. 既存テストが全て合格することを確認: `pytest tests/ -v`

## あなたがやらないこと

- 新規コード実装（検証フェーズのため）
- テストコードの作成（Codex が担当、必要に応じて）

## 完了後の報告

実装完了後、以下を報告してください：

1. **CUDA 検出結果**:
   - CUDA 利用可能性
   - GPU 名とメモリ
   - PyTorch CUDA バージョン

2. **テスト結果**:
   - CUDA/Auto/CPU での実行結果
   - 出力ファイルの検証結果
   - 既存テスト結果（56/56 期待）

3. **パフォーマンスメトリクス**:
   - タイミング比較表（CUDA vs macOS）
   - 高速化率
   - パフォーマンス上の問題

4. **一貫性検証**:
   - スキーマ構造の比較結果
   - プラットフォーム間の差異
   - メタデータの正確性

5. **問題点**（あれば）:
   - CUDA 固有のエラー
   - メモリ問題
   - プラットフォーム差異

## Git コミット

Phase 5 完了後、以下を実行してください：

```bash
git add .
git commit -m "Phase 5完了: Windows CUDA環境での検証とパフォーマンス測定"
git push
```

## トラブルシューティング

### CUDA が利用不可の場合

1. NVIDIA GPU が搭載されているか確認
2. CUDA toolkit がインストールされているか確認
3. PyTorch の CUDA バージョンが CUDA toolkit と一致しているか確認

### Out of Memory エラー

1. Diarization モデルが ASR 前に解放されているか確認
2. `torch.cuda.empty_cache()` が呼ばれているか確認
3. より小さい ASR モデル（tiny または base）を試す

### float16 エラー

- 自動的に int8 でリトライされるはず
- ログで retry メッセージを確認
- フォールバックロジックが動作しているか確認

よろしくお願いします。
