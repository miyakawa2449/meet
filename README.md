# 会議動画文字起こし・議事録作成ツール（M4 Pro最適化版）

MP4動画から自動で文字起こしと議事録を作成するツールです。
MacBook Pro M4 Pro 48GB向けに最適化されています。

## 機能

1. MP4動画から音声を抽出
2. OpenAI Whisperで文字起こし（ローカル実行、Apple Silicon最適化）
3. LLMで要約して議事録を作成

## M4 Pro 48GBでの性能

あなたのMacBook Pro M4 Pro 48GBは以下の点で優れています：

- **メモリ**: 48GBあれば`large`モデルも余裕で動作
- **Neural Engine**: M4 Proのニューラルエンジンで高速処理
- **推奨モデル**: `medium`または`large`（高精度＋高速）
- **処理時間目安**: 90分の動画を10-20分程度で処理可能（mediumモデル）

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
python3 -m venv venv

# 2. 仮想環境を有効化
source venv/bin/activate

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
source venv/bin/activate
```

### 基本的な使い方（mediumモデル推奨）

```bash
python transcribe.py meeting.mp4
```

### 終了時

```bash
deactivate
```

### 最高精度で処理（M4 Pro 48GBなら快適）

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

## M4 Pro 48GBでの処理時間目安（90分動画）

| モデル | メモリ使用量 | 処理時間 | 精度 | 推奨度 |
|--------|-------------|---------|------|--------|
| small  | ~2GB        | 5-8分   | 良   | ○     |
| medium | ~5GB        | 10-20分 | 優   | ★★★  |
| large  | ~10GB       | 20-40分 | 最高 | ★★   |

**推奨**: `medium`モデルが精度と速度のバランスが最適です。

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
