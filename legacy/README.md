# Legacy Scripts（旧バージョン）

⚠️ **注意**: これらは旧バージョンのスクリプトです。新しいプロジェクトでは **ルートの `meeting_pipeline.py`** を使用してください。

## 📌 このフォルダについて

このフォルダには、プロジェクトの初期バージョンで作成されたスクリプトが含まれています。これらは以下の理由で保持されています：

- Qiita や技術ブログからの外部リンク対応
- 過去のバージョンの参照用
- シンプルな文字起こしのみが必要な場合の選択肢

## 🆕 新パイプラインへの移行

**話者分離機能**が必要な場合は、新しいパイプラインを使用してください：

```bash
# ルートディレクトリで実行
python meeting_pipeline.py input.mp4 --enable-diarization
```

詳細は [src/README.md](../src/README.md) を参照してください。

---

## 📚 旧スクリプト一覧

### transcribe.py
- **用途**: Mac（Apple Silicon）で OpenAI Whisper を使用
- **デバイス**: MPS（Metal Performance Shaders）
- **特徴**: M4 Pro などで高速動作
- **外部リンク**: Qiita/ブログ記事からリンクあり

### transcribe_fw.py
- **用途**: Mac で faster-whisper を使用（実験版）
- **デバイス**: CPU（MPS 非対応）
- **特徴**: faster-whisper の動作確認用

### transcribe_cuda.py
- **用途**: NVIDIA GPU で OpenAI Whisper を使用
- **デバイス**: CUDA
- **特徴**: RTX シリーズなどで高速動作

### transcribe_fw_cuda.py
- **用途**: NVIDIA GPU で faster-whisper を使用
- **デバイス**: CUDA
- **特徴**: CTranslate2 CUDA バックエンド使用

### bench_transcribe.py
- **用途**: ベンチマーク測定用統合スクリプト
- **特徴**: Whisper / faster-whisper を切り替えて計測
- **オプション**: `--engine`, `--device`, `--bench-jsonl`, `--bench-md`

### config.py
- **用途**: 設定ファイル
- **特徴**: 旧スクリプトで使用される共通設定

### transcribe_解説.md
- **用途**: 旧スクリプトの解説ドキュメント
- **言語**: 日本語

---

## 🔗 外部リンクからのアクセス

Qiita や技術ブログからこのファイルにアクセスした場合：

### 旧 URL → 新 URL マッピング

| 旧 URL | 新 URL |
|--------|--------|
| `transcribe.py` | `legacy/transcribe.py` |
| `transcribe_fw.py` | `legacy/transcribe_fw.py` |
| `transcribe_cuda.py` | `legacy/transcribe_cuda.py` |
| `transcribe_fw_cuda.py` | `legacy/transcribe_fw_cuda.py` |
| `bench_transcribe.py` | `legacy/bench_transcribe.py` |
| `transcribe_解説.md` | `legacy/transcribe_解説.md` |

---

## 🚀 新パイプラインの利点

新しい `meeting_pipeline.py` は以下の機能を提供します：

- ✅ **話者分離**: pyannote-audio による自動話者識別
- ✅ **クロスプラットフォーム**: Mac（MPS/CPU）と Windows（CUDA）対応
- ✅ **モジュール化**: 保守性の高いアーキテクチャ
- ✅ **単語レベルアライメント**: 高精度な話者割り当て
- ✅ **構造化出力**: JSON + Markdown 形式
- ✅ **包括的テスト**: 56 テストによる品質保証

詳細は [src/README.md](../src/README.md) を参照してください。

---

## 📖 関連記事

- 🧠 Design & Philosophy (Miyakawa Codes)  
  https://miyakawa.codes/blog/local-ai-meeting-minutes-10-minutes

- ⚙️ Technical Guide (Qiita)  
  https://qiita.com/miyakawa2449@github/items/be7a1e5c2a16ac934f13

---

## ⚠️ サポート状況

これらの旧スクリプトは**メンテナンスされていません**。新機能の追加やバグ修正は新パイプラインで行われます。

新しいプロジェクトでは、必ず **`meeting_pipeline.py`** を使用してください。
