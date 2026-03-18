# 最終フェーズ実装依頼

## あなたの役割

あなたは **実装担当（Claude Code）** です。最終フェーズ（Task 21）のドキュメント確認と requirements.txt 更新を実施してください。

## 背景

Phase 1-5 がすべて完了し、プロジェクト整理も完了しました。最終フェーズでは、プロジェクトの完成に向けた最終確認を行います。

## 実施タスク

### Task 21.1: README.md の最終確認

以下の2つの README.md を確認してください：

1. **ルート README.md**
2. **src/README.md**

**確認ポイント**:
- 内容が最新か
- 不足している情報はないか
- 古い情報や誤った情報はないか
- 新規ユーザーがすぐに使い始められる内容か

問題があれば修正、なければ「確認完了」と報告してください。

### Task 21.2: requirements.txt の検証と更新

現在の `requirements.txt` を確認し、新パイプライン（meeting_pipeline.py）で使用するパッケージがすべて含まれているか検証してください。

**確認手順**:
```bash
# 1. 現在の requirements.txt を確認
cat requirements.txt

# 2. 新パイプラインで使用しているパッケージを確認
grep -r "^import\|^from" src/meeting_pipeline/*.py | sort -u

# 3. 不足があれば追加
```

**注意**:
- 旧スクリプト（legacy/）用のパッケージは含めない
- 新パイプラインで使用するパッケージのみ

### Task 21.3: 最終動作確認（オプション）

時間があれば、簡単な動作確認を実施してください：

```bash
# 基本動作
python meeting_pipeline.py temp/202602017_short_test.mp4 --device auto

# 話者分離
python meeting_pipeline.py temp/202602017_short_test.mp4 --enable-diarization --device auto
```

## 完了後

以下を報告してください：

1. README.md の確認結果（更新の有無）
2. requirements.txt の確認結果（更新の有無）
3. 動作確認の結果（実施した場合）
4. 発見した問題（あれば）

変更があれば git commit & push してください。

## 詳細手順

詳細な実装方法は `.kiro/steering/claude_code_final_phase_instructions.md` を確認してください。
