# Meeting Speaker Diarization Pipeline

会議音声・動画ファイルから話者分離とASRを実行し、話者ラベル付き議事録を生成するパイプラインです。

## 概要

このパイプラインは以下の機能を提供します：

- **音声抽出**: 動画ファイルから音声を抽出（ffmpeg）
- **話者分離**: pyannote-audio を使用した話者識別（オプション）
- **音声認識**: faster-whisper または OpenAI whisper による文字起こし
- **アライメント**: ASR結果と話者情報の統合（segment-level / word-level）
- **出力生成**: JSON形式の構造化データと Markdown 形式の議事録

## アーキテクチャ

```
src/meeting_pipeline/
├── models.py          # データクラス定義
├── cli.py             # CLI Parser & Validation
├── device.py          # Device Resolver (CUDA/MPS/CPU)
├── audio.py           # Audio Extractor (ffmpeg)
├── diarization.py     # Diarization Engine (pyannote-audio)
├── asr.py             # ASR Engine (faster-whisper/whisper)
├── alignment.py       # Alignment Module (segment/word-level)
├── output.py          # JSON & Markdown Generator
└── pipeline.py        # Main Pipeline Orchestration
```

## インストール

### 必須要件

- Python 3.8+
- ffmpeg
- PyTorch

### 依存パッケージ

```bash
pip install -r requirements.txt
```

### Hugging Face Token（話者分離を使用する場合）

```bash
export HF_TOKEN=your_token_here
```

Token取得: https://huggingface.co/settings/tokens

## 使用方法

### 基本的な使い方

```bash
# ASRのみ（話者分離なし）
python meeting_pipeline.py input.mp4

# 話者分離を有効化
python meeting_pipeline.py input.mp4 --enable-diarization

# デバイスとモデルを指定
python meeting_pipeline.py input.mp4 \
  --enable-diarization \
  --device cuda \
  --asr-model medium
```

### 主要オプション

```
必須:
  input_file              入力ファイル (mp4, avi, mkv, wav, m4a, mp3, flac)

デバイス:
  --device {auto,cuda,mps,cpu}
                          計算デバイス (default: auto)
                          auto: CUDA > MPS > CPU の優先順位で自動選択

話者分離:
  --enable-diarization    話者分離を有効化
  --diar-model MODEL      話者分離モデル (default: pyannote/speaker-diarization-3.1)

ASR:
  --asr-engine {faster-whisper,whisper}
                          ASRエンジン (default: faster-whisper)
  --asr-model {tiny,base,small,medium,large}
                          ASRモデルサイズ (default: medium)
  --language LANG         言語コード (default: ja)
  --beam-size N           ビームサイズ (default: 1)
  --best-of N             Best-of (default: 1)
  --vad-filter            VADフィルタを有効化

アライメント:
  --align-unit {segment,word}
                          アライメント単位 (default: segment)
                          word: 単語レベルの精密なアライメント

出力:
  --output-dir DIR        出力ディレクトリ (default: output)
  --format {json,md,both} 出力フォーマット (default: both)
  --temp-dir DIR          一時ディレクトリ (default: temp)
  --keep-audio            抽出した音声ファイルを保持

ベンチマーク:
  --bench-jsonl FILE      ベンチマークログファイル
  --run-id ID             実行ID
  --note NOTE             メモ
```

### 使用例

```bash
# 1. 基本的な文字起こし（話者分離なし）
python meeting_pipeline.py meeting.mp4

# 2. 話者分離付き文字起こし
python meeting_pipeline.py meeting.mp4 --enable-diarization

# 3. 単語レベルアライメント（高精度）
python meeting_pipeline.py meeting.mp4 \
  --enable-diarization \
  --align-unit word

# 4. GPU使用、大きいモデル
python meeting_pipeline.py meeting.mp4 \
  --enable-diarization \
  --device cuda \
  --asr-model large

# 5. JSON出力のみ
python meeting_pipeline.py meeting.mp4 \
  --enable-diarization \
  --format json

# 6. ベンチマークログ付き
python meeting_pipeline.py meeting.mp4 \
  --enable-diarization \
  --bench-jsonl bench/results.jsonl \
  --run-id exp001 \
  --note "baseline test"
```

## 出力フォーマット

### JSON出力 (`{basename}_meeting.json`)

Meeting JSON Schema v1.0 に準拠した構造化データ：

```json
{
  "schema_version": "1.0",
  "created_at": "2026-03-16T16:19:56+09:00",
  "title": "",
  "input": {
    "path": "input.mp4",
    "audio": {...},
    "duration_sec": 324.8
  },
  "pipeline": {
    "device": {...},
    "diarization": {...},
    "asr": {...},
    "align": {...}
  },
  "speakers": [
    {"id": "SPEAKER_00", "label": "Speaker 1"},
    {"id": "SPEAKER_01", "label": "Speaker 2"}
  ],
  "segments": [
    {
      "id": "seg_000001",
      "start": 0.0,
      "end": 2.84,
      "speaker_id": "SPEAKER_00",
      "speaker_label": "Speaker 1",
      "text": "こんにちは",
      "confidence": null,
      "source": {...}
    }
  ],
  "artifacts": {...},
  "timing": {...},
  "notes": ""
}
```

### Markdown出力 (`{basename}_transcript.md`)

話者ごとにグループ化された読みやすい議事録：

```markdown
# 会議ログ（話者付き）

## Transcript

### Speaker 1
- [00:00:00 - 00:00:02] こんにちは
- [00:00:04 - 00:00:06] よろしくお願いします

### Speaker 2
- [00:00:09 - 00:00:10] はい、お願いします
```

## 開発フェーズ

### Phase 1: 基本パイプライン ✅ 完了
- CLI Parser
- Device Resolver
- Audio Extractor
- Diarization Engine
- ASR Engine
- Alignment Module (segment-level)
- JSON Generator

### Phase 2: Markdown生成 ✅ 完了
- Markdown Generator
- 出力フォーマット制御
- Benchmark Logger

### Phase 3: 精度改善 ✅ 完了
- 単語レベルアライメント（word-level alignment）
- 単語タイムスタンプの活用
- 連続単語の統合

### Phase 4: クロスプラットフォーム最適化 ✅ 完了
- macOS MPS 対応の検証
- CPU 最適化
- クロスプラットフォーム一貫性の確認

### Phase 5: Windows CUDA 環境検証 ✅ 完了
- Windows WSL2 + NVIDIA RTX 5070 (CUDA) での動作検証
- CUDA / auto デバイスモードでの正常動作確認
- パフォーマンス測定（5分動画: CUDA 24.5秒 vs macOS CPU 330秒、13.5倍高速）
- 全56テスト合格

## テスト

```bash
# 全テスト実行
pytest tests/ -v

# 特定のテストのみ
pytest tests/ -k "alignment" -v

# カバレッジ付き
pytest tests/ --cov=src/meeting_pipeline --cov-report=html
```

現在のテスト状況: **56 passed** (1.53s)

## パフォーマンス

5分の動画（324秒）での実行時間比較（medium モデル、diarization有効）：

| Device | Platform | Diarization | ASR | Total |
|--------|----------|-------------|-----|-------|
| CPU | macOS | 207.9s | ~120s | ~330s |
| MPS | macOS | 17.7s | ~120s (CPU fallback) | ~140s |
| CUDA | WSL2/RTX 5070 | 11.4s | 10.8s | 24.5s |

## トラブルシューティング

### HF_TOKEN エラー
```
Error: HF_TOKEN environment variable is required for diarization
```
→ Hugging Face Token を環境変数に設定してください

### CUDA/MPS 利用不可エラー
```
Error: CUDA device requested but CUDA is not available
```
→ `--device auto` または `--device cpu` を使用してください

### ffmpeg not found
```
Error: ffmpeg is required but not found in PATH
```
→ ffmpeg をインストールしてください: https://ffmpeg.org/download.html

### faster-whisper MPS 警告
```
WARNING: faster-whisper does not support MPS, using CPU instead
```
→ これは正常な動作です。faster-whisper は自動的に CPU にフォールバックします

## 既知の制限事項

- 本環境の CTranslate2 4.7.1 カスタムビルドは CUDA 専用であり、CPU SGEMM バックエンドを含みません。そのため、この環境では CPU での ASR 実行はサポートされません。
- Phase 5 では CUDA および auto デバイスモードでの正常動作を確認済みです。

## ライセンス

このプロジェクトは以下のオープンソースライブラリを使用しています：

- [pyannote-audio](https://github.com/pyannote/pyannote-audio) - MIT License
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - MIT License
- [OpenAI Whisper](https://github.com/openai/whisper) - MIT License

## 貢献

バグ報告や機能リクエストは Issue でお願いします。
