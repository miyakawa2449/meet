# Meeting Speaker Diarization Pipeline

会議音声・動画ファイルから**話者分離**と**文字起こし**を実行し、話者ラベル付き議事録を生成するパイプラインです。

## 🎯 プロジェクト概要

このプロジェクトは「Mac で Whisper を動かして議事録を作る」ところから始まり、現在は**話者分離機能を備えた本格的なパイプライン**に進化しました。

### 🆕 新パイプライン（推奨）

**`meeting_pipeline.py`** - 話者分離対応の最新版

- ✅ **話者分離**: pyannote-audio による自動話者識別
- ✅ **議事録生成**: OpenAI API による要約・決定事項・アクションアイテム抽出
- ✅ **クロスプラットフォーム**: Mac（MPS/CPU）と Windows（CUDA）対応
- ✅ **高精度**: 単語レベルアライメントによる精密な話者割り当て
- ✅ **構造化出力**: JSON + Markdown 形式
- ✅ **本番環境**: Windows + CUDA RTX 5070 で最適化済み

**詳細ドキュメント**: [src/README.md](src/README.md)

### 📚 旧スクリプト（Legacy）

シンプルな文字起こしのみが必要な場合や、過去の記事からアクセスした場合は [legacy/](legacy/) フォルダを参照してください。

---

## 🚀 クイックスタート

### インストール

```bash
# 1. リポジトリをクローン
git clone https://github.com/miyakawa2449/meet.git
cd meet

# 2. セットアップスクリプトを実行
chmod +x setup.sh
./setup.sh

# 3. 環境変数を設定
cp .env.example .env
# .env ファイルを編集して HF_TOKEN を設定
```

### 基本的な使い方

```bash
# 仮想環境を有効化
source venv/bin/activate

# 話者分離付き文字起こし
python meeting_pipeline.py input.mp4 --enable-diarization

# デバイスを指定（auto, cuda, mps, cpu）
python meeting_pipeline.py input.mp4 --enable-diarization --device auto
```

---

## 📊 パフォーマンス

5分の動画（324秒）での処理時間比較（medium モデル、diarization 有効）：

| Device | Platform | Diarization | ASR | Total |
|--------|----------|-------------|-----|-------|
| CPU | macOS | 207.9s | ~120s | ~330s |
| MPS | macOS | 17.7s | ~120s (CPU fallback) | ~140s |
| **CUDA** | **WSL2/RTX 5070** | **11.4s** | **10.8s** | **24.5s** ⭐ |

**CUDA 環境で 13.5倍高速化！**

---

## 🎨 主な機能

### 1. 話者分離（Diarization）

pyannote-audio を使用して、誰がいつ話したかを自動識別します。

```bash
python meeting_pipeline.py meeting.mp4 --enable-diarization
```

### 2. 単語レベルアライメント

単語単位で話者を割り当て、高精度な議事録を生成します。

```bash
python meeting_pipeline.py meeting.mp4 \
  --enable-diarization \
  --align-unit word
```

### 3. 構造化出力

**JSON 出力** (`{basename}_meeting.json`):
- Meeting JSON Schema v1.0 準拠
- 完全な構造化データ
- メタデータとタイミング情報

**Markdown 出力** (`{basename}_transcript.md`):
- 話者ごとにグループ化
- タイムスタンプ付き
- 読みやすい議事録形式

### 4. クロスプラットフォーム対応

- **macOS**: MPS（Metal）または CPU
- **Windows/Linux**: CUDA または CPU
- **自動選択**: `--device auto` で最適なデバイスを自動選択

---

## 📖 ドキュメント

- **新パイプライン詳細**: [src/README.md](src/README.md)
- **旧スクリプト**: [legacy/README.md](legacy/README.md)
- **Phase 4 レポート**: [reports/phase4_report.md](reports/phase4_report.md)
- **Phase 5 レポート**: [reports/phase5_report.md](reports/phase5_report.md)

---

## 🔧 開発情報

### プロジェクト構造

```
.
├── meeting_pipeline.py          # メインエントリーポイント（新パイプライン）
├── src/meeting_pipeline/        # モジュール群
│   ├── models.py               # データクラス定義
│   ├── cli.py                  # CLI Parser
│   ├── device.py               # Device Resolver
│   ├── audio.py                # Audio Extractor
│   ├── diarization.py          # Diarization Engine
│   ├── asr.py                  # ASR Engine
│   ├── alignment.py            # Alignment Module
│   ├── output.py               # JSON & Markdown Generator
│   └── pipeline.py             # Main Pipeline Orchestration
├── tests/                       # テストスイート（56 テスト）
├── legacy/                      # 旧スクリプト
└── reports/                     # 開発レポート
```

### テスト

```bash
# 全テスト実行
pytest tests/ -v

# カバレッジ付き
pytest tests/ --cov=src/meeting_pipeline --cov-report=html
```

**現在のテスト状況**: 87/87 合格 ✅

### 開発フェーズ

- ✅ Phase 1: 基本パイプライン
- ✅ Phase 2: Markdown 生成
- ✅ Phase 3: 単語レベルアライメント
- ✅ Phase 4: macOS MPS/CPU 対応
- ✅ Phase 5: Windows CUDA 環境検証
- ✅ Phase 6: 議事録生成機能（OpenAI API統合）

---

## 📝 関連記事

- 🧠 **Design & Philosophy** (Miyakawa Codes)  
  https://miyakawa.codes/blog/local-ai-meeting-minutes-10-minutes

- ⚙️ **Technical Guide** (Qiita)  
  https://qiita.com/miyakawa2449@github/items/be7a1e5c2a16ac934f13

---

## 🤝 コントリビューション

バグ報告や機能リクエストは Issue でお願いします。

---

## 📄 ライセンス

このプロジェクトは以下のオープンソースライブラリを使用しています：

- [pyannote-audio](https://github.com/pyannote/pyannote-audio) - MIT License
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - MIT License
- [OpenAI Whisper](https://github.com/openai/whisper) - MIT License

---

## 🎯 推奨環境

### 本番環境（最高性能）
- **OS**: Windows 11 + WSL2
- **GPU**: NVIDIA RTX 5070（または同等）
- **VRAM**: 12GB 以上
- **処理時間**: 5分動画を約25秒で処理

### 開発環境
- **OS**: macOS（Apple Silicon）
- **デバイス**: MPS（Metal Performance Shaders）
- **処理時間**: 5分動画を約140秒で処理

### フォールバック
- **デバイス**: CPU
- **処理時間**: 5分動画を約330秒で処理
- すべての環境で動作保証

---

**新しいプロジェクトでは `meeting_pipeline.py` を使用してください！**

詳細は [src/README.md](src/README.md) を参照してください。
