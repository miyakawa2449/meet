# 最終フェーズ実装指示書 for Claude Code

## Context

Phase 1-5 がすべて完了し、プロジェクト整理も完了しました。最終フェーズでは、ドキュメントの最終確認と requirements.txt の更新を行います。

## Current Status

- **完了**: Phase 1-5（基本パイプライン、Markdown生成、単語レベルアライメント、macOS MPS/CPU対応、Windows CUDA検証）
- **完了**: プロジェクト整理（旧スクリプトを legacy/ に移動）
- **環境**: macOS
- **テスト**: 56/56 合格
- **プロジェクト構造**: モジュール化されたアーキテクチャ

## 最終フェーズ Tasks Overview

最終フェーズでは、プロジェクトの完成に向けた最終確認とドキュメント整備を行います。

### Task 21.1: README.md の最終確認

**目的**: ルートとsrc/のREADME.mdが最新の状態であることを確認

**確認項目**:

1. **ルート README.md**:
   - ✅ 新パイプラインへの誘導が明確
   - ✅ legacy/ フォルダへの参照
   - ✅ クイックスタート手順
   - ✅ パフォーマンス比較表
   - ✅ 関連記事リンク

2. **src/README.md**:
   - ✅ インストール手順（依存パッケージ、HF_TOKEN設定）
   - ✅ 使用例（基本的なコマンド例）
   - ✅ パラメータ一覧
   - ✅ トラブルシューティング
   - ✅ Phase 1-5 完了マーク
   - ✅ パフォーマンス比較表
   - ✅ 既知の制限事項

**実施内容**:
- 両方の README.md を読んで内容を確認
- 不足している情報や古い情報があれば更新
- 特に問題がなければ「確認完了」と報告

**期待される結果**:
- README.md が最新の状態
- 新規ユーザーがすぐに使い始められる内容
- Phase 1-5 の成果が適切に記載されている

### Task 21.2: requirements.txt の検証と更新

**目的**: すべての依存パッケージが正確に記載されていることを確認

**確認項目**:

1. **必須パッケージ**:
   - torch（PyTorch）
   - pyannote-audio（話者分離）
   - faster-whisper（ASR）
   - openai-whisper（ASR、オプション）
   - その他の依存関係

2. **バージョン指定**:
   - 重要なパッケージはバージョンを固定すべきか検討
   - 互換性の問題がないか確認

**実施手順**:

1. 現在の requirements.txt を確認:
   ```bash
   cat requirements.txt
   ```

2. 実際にインストールされているパッケージを確認:
   ```bash
   pip list --format=freeze > /tmp/current_packages.txt
   ```

3. 新パイプラインで使用しているパッケージを確認:
   ```bash
   # src/meeting_pipeline/ 内の import 文を確認
   grep -r "^import\|^from" src/meeting_pipeline/*.py | sort -u
   ```

4. 不足しているパッケージがあれば追加

5. 不要なパッケージがあれば削除（慎重に）

**期待される結果**:
- requirements.txt が最新の状態
- 新規インストール時に必要なパッケージがすべて含まれる
- バージョン互換性の問題がない

**注意事項**:
- 旧スクリプト（legacy/）用のパッケージは含めない
- 新パイプライン（meeting_pipeline.py）で使用するパッケージのみ

### Task 21.3: 最終動作確認（オプション）

**目的**: 新パイプラインが正常に動作することを最終確認

**確認項目**:

1. **基本動作**:
   ```bash
   python meeting_pipeline.py temp/202602017_short_test.mp4 --device auto
   ```

2. **話者分離**:
   ```bash
   python meeting_pipeline.py temp/202602017_short_test.mp4 --enable-diarization --device auto
   ```

3. **出力フォーマット**:
   ```bash
   # JSON のみ
   python meeting_pipeline.py temp/202602017_short_test.mp4 --format json
   
   # Markdown のみ
   python meeting_pipeline.py temp/202602017_short_test.mp4 --format md
   ```

**期待される結果**:
- すべてのコマンドが正常に実行される
- 出力ファイルが正しく生成される
- エラーや警告がない（または適切に処理される）

**注意事項**:
- これは簡単な動作確認です
- 詳細なテストは Codex が Task 21.3 で実施します

## Important Notes

### Environment Setup

```bash
source venv/bin/activate
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)
```

### Git Workflow

Task 21.1-21.2 完了後:
```bash
git add .
git commit -m "最終フェーズ: ドキュメント最終確認と requirements.txt 更新"
git push
```

## Success Criteria

Task 21 は以下の条件で完了です:

1. ✅ README.md（ルートと src/）が最新の状態
2. ✅ requirements.txt が正確で最新
3. ✅ 新パイプラインが正常に動作（簡単な確認）
4. ✅ ドキュメントに不足や古い情報がない

## Task Checklist

- [ ] Task 21.1: README.md の最終確認（ルートと src/）
- [ ] Task 21.2: requirements.txt の検証と更新
- [ ] Task 21.3: 新パイプラインの最終動作確認（オプション）

## Reporting

完了後、以下を報告してください:

1. README.md の確認結果（更新の有無）
2. requirements.txt の確認結果（更新の有無）
3. 動作確認の結果（実施した場合）
4. 発見した問題（あれば）

## Notes

- Task 21 は主に確認作業です
- 大きな変更は不要の想定
- 問題が見つかった場合のみ修正
- Codex が Task 21.3 で詳細なE2Eテストを実施します
