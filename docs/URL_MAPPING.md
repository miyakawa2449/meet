# URL マッピング表（外部リンク更新用）

## 概要

プロジェクト整理により、旧スクリプトを `legacy/` フォルダに移動しました。
Qiita や技術ブログからの外部リンクを更新する際は、以下のマッピング表を参照してください。

## 📋 ファイル移動マッピング

### Python スクリプト

| 旧 URL | 新 URL | 説明 |
|--------|--------|------|
| `transcribe.py` | `legacy/transcribe.py` | Mac MPS版（OpenAI Whisper） |
| `transcribe_fw.py` | `legacy/transcribe_fw.py` | Mac CPU版（faster-whisper） |
| `transcribe_cuda.py` | `legacy/transcribe_cuda.py` | CUDA版（OpenAI Whisper） |
| `transcribe_fw_cuda.py` | `legacy/transcribe_fw_cuda.py` | CUDA版（faster-whisper） |
| `bench_transcribe.py` | `legacy/bench_transcribe.py` | ベンチマークスクリプト |
| `config.py` | `legacy/config.py` | 設定ファイル |

### ドキュメント

| 旧 URL | 新 URL | 説明 |
|--------|--------|------|
| `transcribe_解説.md` | `legacy/transcribe_解説.md` | 旧スクリプト解説 |

### プロジェクト管理ファイル

| 旧 URL | 新 URL | 説明 |
|--------|--------|------|
| `CLAUDE_CODE_MESSAGE.md` | `docs/CLAUDE_CODE_MESSAGE.md` | Phase 4 指示 |
| `CLAUDE_CODE_PHASE5_MESSAGE.md` | `docs/CLAUDE_CODE_PHASE5_MESSAGE.md` | Phase 5 指示 |
| `CODEX_PHASE4_TEST_MESSAGE.md` | `docs/CODEX_PHASE4_TEST_MESSAGE.md` | Phase 4 テスト指示 |
| `CODEX_PHASE5_TEST_MESSAGE.md` | `docs/CODEX_PHASE5_TEST_MESSAGE.md` | Phase 5 テスト指示 |
| `SESSION_HANDOFF.md` | `docs/SESSION_HANDOFF.md` | セッション引き継ぎ |

## 🔗 GitHub URL 形式

### Raw ファイルへの直接リンク

**旧 URL 形式**:
```
https://raw.githubusercontent.com/miyakawa2449/meet/main/transcribe.py
```

**新 URL 形式**:
```
https://raw.githubusercontent.com/miyakawa2449/meet/main/legacy/transcribe.py
```

### ブラウザ表示用リンク

**旧 URL 形式**:
```
https://github.com/miyakawa2449/meet/blob/main/transcribe.py
```

**新 URL 形式**:
```
https://github.com/miyakawa2449/meet/blob/main/legacy/transcribe.py
```

## 📝 更新が必要な記事

以下の記事で URL 更新が必要な可能性があります：

### Qiita
- https://qiita.com/miyakawa2449@github/items/be7a1e5c2a16ac934f13

### Miyakawa Codes
- https://miyakawa.codes/blog/local-ai-meeting-minutes-10-minutes

### その他
- （該当する記事があれば追記）

## 🎯 推奨対応

### オプション1: 新パイプラインへの誘導（推奨）

記事に以下の注意書きを追加：

```markdown
⚠️ **更新情報**: このプロジェクトは進化し、話者分離機能を備えた新パイプラインが利用可能です。

- **新パイプライン**: [meeting_pipeline.py](https://github.com/miyakawa2449/meet)
- **旧スクリプト**: [legacy/](https://github.com/miyakawa2449/meet/tree/main/legacy)

新しいプロジェクトでは新パイプラインの使用を推奨します。
```

### オプション2: URL のみ更新

記事内のリンクを `legacy/` パスに更新：

```markdown
- 変更前: `transcribe.py`
- 変更後: `legacy/transcribe.py`
```

## 📌 Git 履歴

ファイル移動は `git mv` コマンドで実行されているため、Git 履歴は保持されています。

```bash
# 履歴確認
git log --follow legacy/transcribe.py
```

## 🔄 移動日時

- **実行日**: 2026年3月18日
- **コミット**: "プロジェクト整理: 旧スクリプトを legacy/ に、管理ファイルを docs/ に移動"

## ✅ チェックリスト

外部リンク更新時のチェックリスト：

- [ ] Qiita 記事の URL 更新
- [ ] Miyakawa Codes ブログの URL 更新
- [ ] その他の技術ブログの URL 更新
- [ ] SNS 投稿の URL 確認
- [ ] README.md の関連記事リンク確認

---

**注意**: 新しいプロジェクトでは `meeting_pipeline.py`（新パイプライン）の使用を推奨してください。
