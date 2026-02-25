# 会議動画文字起こし・議事録作成ツール

MP4動画から自動で文字起こしと議事録を作成するツールです。
M4 Pro（MPS）とRTX（CUDA）の両方に対応しています。

## 機能

1. MP4動画から音声を抽出
2. Whisperで文字起こし（OpenAI Whisper / faster-whisper）
3. LLMで要約して議事録を作成

※ Whisperは稀に存在しない文言を生成することがあるので、議事録用途は「最終チェック必須」

## 📘 Articles

- 🧠 Design & Philosophy (Miyakawa Codes)
  https://miyakawa.codes/blog/local-ai-meeting-minutes-10-minutes

- ⚙️ Technical Guide (Qiita)
  https://qiita.com/miyakawa2449@github/items/be7a1e5c2a16ac934f13


## どのスクリプトを使うか

- `transcribe.py`  
  Apple SiliconでPyTorchを試したバージョン。Macで一番高いパフォーマンスを出します。MPSに対応しており、M4 Pro 48GBで90分の動画を10分台で議事録までまとめます。
- `transcribe_fw.py`  
  Apple Siliconで fast whisper を試した実験版。MPS未対応のためCPUで実行。
- `transcribe_cuda.py`  
  RTXなどNVIDIA GPUで **OpenAI Whisper (PyTorch)** をCUDA利用する場合。
- `transcribe_fw_cuda.py`  
  RTXなどNVIDIA GPUで **faster-whisper (CTranslate2 CUDA)** を利用する場合。

## bench_transcribe.py（統合ベンチ）

`bench_transcribe.py` は、Whisper / faster-whisper を1本で切り替えて計測するベンチ用スクリプトです。

- `--engine whisper | faster-whisper`
- `--device auto | cpu | mps | cuda`
- 計測項目: `extract / load / asr / summary / total`
- ログ: `--bench-jsonl`（JSONL追記）, `--bench-md`（Markdown表追記）

### 基本コマンド

```bash
python3 bench_transcribe.py meeting.mp4 \
  --engine faster-whisper \
  --device auto \
  --no-summary \
  --bench-jsonl bench/bench.jsonl \
  --bench-md bench/bench.md
```

### 実行例（環境別）

#### 1) Mac + Whisper（MPS）

```bash
python3 bench_transcribe.py meeting.mp4 \
  --engine whisper \
  --device mps \
  --beam-size 1 \
  --best-of 1 \
  --no-summary \
  --bench-jsonl bench/bench.jsonl \
  --bench-md bench/bench.md
```

#### 2) Mac + faster-whisper（CPU / auto）

```bash
python3 bench_transcribe.py meeting.mp4 \
  --engine faster-whisper \
  --device auto \
  --beam-size 1 \
  --best-of 1 \
  --no-summary \
  --bench-jsonl bench/bench.jsonl \
  --bench-md bench/bench.md
```

#### 3) WSL2 (Windows上のUbuntu) + Whisper（CUDA）

```bash
python3 bench_transcribe.py meeting.mp4 \
  --engine whisper \
  --device cuda \
  --beam-size 1 \
  --best-of 1 \
  --no-summary \
  --bench-jsonl bench/bench.jsonl \
  --bench-md bench/bench.md
```

#### 4) WSL2 (Windows上のUbuntu) + faster-whisper（CUDA）

```bash
python3 bench_transcribe.py meeting.mp4 \
  --engine faster-whisper \
  --device cuda \
  --beam-size 1 \
  --best-of 1 \
  --no-summary \
  --bench-jsonl bench/bench.jsonl \
  --bench-md bench/bench.md
```

## セットアップ

### 簡単セットアップ（推奨）

```bash
# 実行権限を付与
chmod +x setup.sh

# セットアップスクリプトを実行
./setup.sh
```

これで仮想環境の作成、パッケージのインストール、設定ファイルの準備が完了します。

### 手動セットアップ

```bash
# 1. 仮想環境を作成
python3 -m venv whisper

# 2. 仮想環境を有効化
source whisper/bin/activate

# 3. ffmpegのインストール
brew install ffmpeg

# 4. Pythonパッケージのインストール
pip install --upgrade pip
pip install -r requirements.txt

# 5. 環境変数の設定
cp .env.example .env
# .envファイルを編集してOpenAI API Keyを設定
```

## 使い方

### 仮想環境の有効化（毎回必要）

```bash
source whisper/bin/activate
```

### 基本的な使い方（mediumモデル推奨）

```bash
python transcribe.py meeting.mp4
```

### 終了時

```bash
deactivate
```

### 最高精度で処理

```bash
python transcribe.py meeting.mp4 --model large
```

### 高速処理（精度は少し落ちる）

```bash
python transcribe.py meeting.mp4 --model small
```

### 要約をスキップ（文字起こしのみ）

```bash
python transcribe.py meeting.mp4 --no-summary
```

## 参考パフォーマンス（90分動画）

- faster-whisper CUDA: 2分50秒
- OpenAI Whisper CUDA: 6分30秒前後
- MPS（M4 Pro）: 10分40秒前後

## 出力ファイル

- `output/[ファイル名]_transcript.txt` - 文字起こし結果
- `output/[ファイル名]_minutes.txt` - 議事録

## トラブルシューティング

### MPSバックエンドが使えない場合

最新のPyTorchをインストール：

```bash
pip install --upgrade torch
```

### メモリ不足エラー

M4 Pro 48GBでは通常発生しませんが、他のアプリを閉じてください。

## パフォーマンスTips

1. **バックグラウンドアプリを閉じる**: より多くのメモリをWhisperに割り当て
2. **電源接続**: バッテリー駆動時より高速
3. **初回実行**: モデルのダウンロードに時間がかかります（2回目以降は高速）
