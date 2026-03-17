会議録AIパイプライン（話者分離＋文字起こし）仕様書 v1.0

（Claude Code 実装用 / 段階的開発プロセス付き）

作成日: 2026-03-04
対象: Windows（WSL2 Ubuntu）+ NVIDIA GPU（CUDA）を第一ターゲットに、後から macOS（MPS/CPU）へ拡張する。
前提: 既存の bench_transcribe.py は変更しない（ASRベンチ専用として保持）。

⸻

0. 背景と狙い（いきさつ）
	•	会議録画から議事録を作る目的で Whisper / faster-whisper を検証し、環境（MPS/CUDA）や推論エンジン（PyTorch/CTranslate2）によって性能・安定性が大きく変わることを確認した。
	•	速度比較のフェーズを経て、次の段階として「誰が話したか」を付けた議事録（=話者分離）へ進む。
	•	話者分離は難易度が高く、実装を急ぐと失敗しやすい。そのため、先に**データ設計（JSON）と出力（Markdown）**を固定し、段階的に品質を上げる。
	•	bench用スクリプトは資産として残し、新機能は新しいスクリプトで実装する。

⸻

1. 目標（非機能要件含む）

1.1 目標
	1.	動画/音声ファイルから 話者分離（Diarization） を行う
	2.	同じ音声から ASR（音声認識） を行う
	3.	話者区間とASR区間を突合し、話者ラベル付きテキストを生成する
	4.	出力として以下を生成する
	•	meeting.json（正：再処理可能な統合ログ）
	•	transcript.md（人間が読む会議ログ）

1.2 非機能（重要）
	•	再現性：利用したモデル名、デバイス、主要パラメータ、処理時間をJSONに残す
	•	汎用性：話者数「2人固定」にしない（1〜N人を許容）
	•	堅牢性：話者割当できない区間は UNKNOWN として落とさず残す
	•	拡張性：将来GUI化しやすいCLI/JSON構造にする
	•	段階的開発：Phaseごとに、ユーザーが「確認」「合否判定」できる成果物を作る

⸻

2. スコープ

Phase 1（Windows先行・v1完成）
	•	Windows（WSL2 Ubuntu）+ CUDA 前提で、話者分離＋ASR＋アライン＋JSON出力を完成させる
	•	まずは アライン単位＝ASRセグメント で実装する
	•	要約（LLM）はオプション（後回し可）

Phase 2（Markdown生成の確定・運用可能化）
	•	JSON → Markdown 変換を実装し、人間が読める会議ログを生成
	•	UNKNOWN を含めても破綻しない表現にする

Phase 3（精度・品質向上）
	•	単語/句単位アライン（WhisperX系の考え方）へ拡張
	•	オーバーラップ発話（同時発話）や区間の細分化を改善

Phase 4（macOS対応）
	•	--device auto|cuda|mps|cpu を活用して macOS（MPS/CPU）へ拡張
	•	安定優先（MPSが不安定ならCPUフォールバック）

⸻

3. 新規作成ファイル
	•	新スクリプト（例）：meeting_pipeline.py
	•	bench_transcribe.py をベースに思想と構成を継承するが、ファイルは別にする
	•	目的はベンチではなく「会議ログ生成（話者付き）」のパイプライン

注: 名前は meeting_pipeline.py を推奨。リポジトリ方針に合わせて変更可。

⸻

4. CLI仕様（v1）

実行形式:

python meeting_pipeline.py INPUT_FILE [options]

4.1 入力
	•	INPUT_FILE（必須）
	•	mp4 / wav / m4a / mp3 / flac など
	•	動画の場合は ffmpeg で音声抽出（16kHz, mono, pcm_s16le）

4.2 デバイス
	•	--device auto|cuda|mps|cpu（default: auto）
	•	auto: CUDA→MPS→CPU の優先順
	•	v1の実装ターゲットは Windows+CUDA だが、IFは先に用意する

4.3 話者分離
	•	--enable-diarization（指定時のみ実行。default: OFFでもOK / v1ではON推奨）
	•	--diar-model MODEL（defaultはコード内で固定可）
	•	HFトークンは環境変数 HF_TOKEN から取得（CLIで受けない）
	•	無い場合は明確なエラーで停止

4.4 ASR
	•	--asr-engine faster-whisper|whisper（default: faster-whisper）
	•	v1は faster-whisper 実装を最優先
	•	whisper は分岐だけ用意（未実装でも可）
	•	--asr-model tiny|base|small|medium|large（default: medium）
	•	--language ja（default: ja）
	•	--beam-size N（default: 1）
	•	--best-of N（default: 1）
	•	--vad-filter（default: False）

4.5 出力
	•	--output-dir DIR（default: output）
	•	--temp-dir DIR（default: temp）
	•	--keep-audio（default: False）
	•	--format json|md|both（default: both）
	•	Phase1では json を必須成果物
	•	Phase2で md を必須成果物

4.6 ベンチログ（任意）
	•	--bench-jsonl PATH（指定時のみ追記）
	•	--run-id ID（指定がなければ日時で自動採番）
	•	--note TEXT（任意メモ）

⸻

5. 処理フロー（v1）
	1.	音声抽出（必要な場合のみ）
	2.	話者分離（pyannote-audio）
	3.	ASR（faster-whisper）
	4.	アライン（max_overlap, unit=asr_segment）
	5.	統合ログ出力（Meeting JSON Schema v1）
	6.	（Phase2）Markdown生成
	7.	（オプション）要約

重要:
	•	話者分離とASRは同時並行ではなくシーケンシャルで実行する
	•	diarizationを終えたら、必要ならモデル参照を解放してからASRへ進む

⸻

6. Meeting JSON Schema v1（最終確定）

6.1 概要
	•	1ファイル統合
	•	話者は id（機械用）＋ label（人間用）を併記
	•	UNKNOWN を speakers に含める

6.2 トップレベル

{
  "schema_version": "1.0",
  "created_at": "YYYY-MM-DDTHH:MM:SS±TZ",
  "title": "",
  "input": {},
  "pipeline": {},
  "speakers": [],
  "segments": [],
  "artifacts": {},
  "timing": {},
  "notes": ""
}

6.3 input

"input": {
  "path": "meeting.mp4",
  "audio": {
    "path": "temp/meeting.wav",
    "sample_rate": 16000,
    "channels": 1
  },
  "duration_sec": 5432.1
}

6.4 pipeline

"pipeline": {
  "device": {
    "requested": "auto",
    "resolved": "cuda"
  },
  "diarization": {
    "enabled": true,
    "engine": "pyannote-audio",
    "model": "pyannote/speaker-diarization",
    "hf_token_used": true
  },
  "asr": {
    "engine": "faster-whisper",
    "model": "medium",
    "device": "cuda",
    "compute_type": "float16",
    "language": "ja",
    "beam_size": 1,
    "best_of": 1,
    "vad_filter": false
  },
  "align": {
    "method": "max_overlap",
    "unit": "asr_segment"
  }
}

6.5 speakers（UNKNOWN含む）

"speakers": [
  { "id": "SPEAKER_00", "label": "Speaker 1" },
  { "id": "SPEAKER_01", "label": "Speaker 2" },
  { "id": "UNKNOWN", "label": "Unknown" }
]

6.6 segments（主データ）

"segments": [
  {
    "id": "seg_000001",
    "start": 12.34,
    "end": 18.90,
    "speaker_id": "SPEAKER_00",
    "speaker_label": "Speaker 1",
    "text": "今日はこの件から始めます。",
    "confidence": null,
    "source": {
      "asr_segment_id": "asr_000104",
      "diarization_turn_id": "turn_000087",
      "overlap_sec": 6.12
    }
  }
]

6.7 artifacts（生データ）

"artifacts": {
  "diarization_turns": [
    { "id": "turn_000087", "speaker_id": "SPEAKER_00", "start": 11.80, "end": 19.10 }
  ],
  "asr_segments": [
    { "id": "asr_000104", "start": 12.34, "end": 18.90, "text": "今日はこの件から始めます。" }
  ]
}

6.8 timing

"timing": {
  "extract_sec": 2.9,
  "diarization_sec": 210.3,
  "asr_load_sec": 1.8,
  "asr_sec": 127.1,
  "align_sec": 0.2,
  "summary_sec": 0.0,
  "total_sec": 360.0
}

6.9 v1アライン方式（max_overlap）
	•	ASRセグメント S と話者区間 T_i の重なり秒 overlap(S,T_i) を計算
	•	最大重なりの話者を割り当てる
	•	最大重なりが 0 の場合は speaker_id="UNKNOWN" を割り当てる

⸻

7. Markdownフォーマット v1（確定）

7.1 目的
	•	話者分離が不完全でも成立（Unknown許容）
	•	人間が読む会議ログ
	•	LLM要約の入力に使いやすい

7.2 出力例

# 会議ログ（話者付き）

## Transcript

### Speaker 1
- [00:12:34 - 00:12:39] 今日はこの件から始めます。
- [00:12:40 - 00:12:50] まず前提を整理します。

### Speaker 2
- [00:12:51 - 00:13:05] はい、資料を確認しました。

### Unknown
- [00:13:06 - 00:13:12] （音声が重なって聞き取りづらい）

7.3 変換ルール（JSON→MD）
	•	meeting.json の segments[] のみを使用
	•	start 昇順で処理
	•	speaker_label が変わったら ### {speaker_label} 見出しを出す
	•	各行は - [START - END] TEXT
	•	時刻は hh:mm:ss（ゼロ埋め）
	•	text が空/空白ならその行はスキップ

⸻

8. Phaseごとの成果物・確認手順（ユーザー検証が必須）

Phase 1: Windowsで話者分離＋JSON出力（最優先）

目的: 話者分離→ASR→アライン→JSONが正しく出ること。

成果物
	•	output/..._meeting.json
	•	可能なら --bench-jsonl への記録

ユーザー検証項目
	1.	HF_TOKEN 未設定時に、明確なエラーで停止するか
	2.	speakers に UNKNOWN が含まれるか
	3.	segments が start/end/speaker_id/speaker_label/text を持つか
	4.	話者割当できない区間が UNKNOWN で残るか
	5.	pipeline と timing が最低限埋まるか

実行例

python meeting_pipeline.py meeting.mp4 \
  --device cuda \
  --enable-diarization \
  --asr-engine faster-whisper \
  --asr-model medium \
  --language ja \
  --beam-size 1 --best-of 1 \
  --output-dir output --temp-dir temp \
  --format json \
  --bench-jsonl bench/meeting_pipeline.jsonl


⸻

Phase 2: JSON→Markdown生成（運用の入口）

目的: 人間が読める transcript.md を生成できること。

成果物
	•	output/..._transcript.md

ユーザー検証項目
	1.	見出し ### Speaker N が適切に切り替わるか
	2.	時刻が hh:mm:ss でゼロ埋めされているか
	3.	Unknown ブロックが破綻しないか
	4.	長時間会議でも読みやすいか

⸻

Phase 3: 精度改善（単語/句単位アライン）

目的: 話者割り当て精度を上げる。

成果物例
	•	artifacts.asr_words の追加
	•	align.method の拡張（例: word_level）

ユーザー検証項目
	1.	連続発話の誤割当が減ったか
	2.	同時発話・短い相槌などの誤りが許容範囲か
	3.	v1（segment）との比較でメリットが見えるか

⸻

Phase 4: macOS対応

目的: 同一CLIで macOS（MPS/CPU）でも動く。

成果物
	•	--device auto|mps|cpu による実行確認
	•	不安定時のフォールバック（mps→cpu）

ユーザー検証項目
	1.	macOSで --device auto が期待通りに解決されるか
	2.	MPSが不安定な場合にCPUへ切替できるか
	3.	出力JSON/MDがWindowsと同じ仕様で生成されるか

⸻

9. Claude Code への重要な作業指針（失敗防止）
	•	実装を急がず、Phase1では「JSONの正しさ」を最優先にする
	•	まずは「動く」より「壊れない」を優先（Unknownで落とさない）
	•	2人固定にしない（SPEAKER_00... を出現分だけ列挙）
	•	diarizationとASRを同時に回さない（VRAMを圧迫しない）
	•	未実装項目がある場合は、理由と次アクションを明記し、ユーザーに確認を求める

⸻

10. 参考（既存資産）
	•	既存ベンチ（変更禁止）：bench_transcribe.py
	•	目標：本仕様の meeting_pipeline.py を新規で作成し、段階的に完成させる

⸻

付録A: ユーザーが最初に用意するもの（Phase1前）
	•	Windows（WSL2 Ubuntu）+ CUDAが使える環境
	•	HF_TOKEN（pyannoteモデルDL用）
	•	Python環境（venvなど）
	•	依存関係（例）
	•	ffmpeg
	•	pyannote-audio（+ torch）
	•	faster-whisper
	•	ctranslate2

（依存の固定や最小セットはPhase1実装後に確定する）

⸻