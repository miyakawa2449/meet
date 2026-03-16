# Implementation Plan: Meeting Speaker Diarization Pipeline

## Overview

このタスクリストは、会議音声・動画ファイルから話者分離とASRを実行し、話者ラベル付き議事録を生成するパイプラインの実装計画です。段階的な開発アプローチを採用し、Phase 1（基本パイプライン）とPhase 2（Markdown生成）を必須タスク、Phase 3（精度改善）とPhase 4（クロスプラットフォーム最適化）をオプションタスクとして定義します。

## 役割分担

- **Claude Code (Kiro)**: 実装担当
  - コード作成、リファクタリング、バグ修正
  - タスクリストに従った段階的実装
  
- **Codex**: 検証担当
  - テストコード作成（ユニットテスト、プロパティテスト）
  - 実装の検証と品質チェック
  - 受入基準の確認

## 実装の進め方

1. Claude Code が実装タスク（例: タスク1, 2.1, 2.2）を完了
2. Codex がテストタスク（例: タスク2.3）を実装して検証
3. Checkpoint で両者が確認
4. 次のフェーズへ進む

## Tasks

### Phase 1: 基本パイプライン（JSON出力まで）

- [x] 1. プロジェクト構造とコア型定義の作成 **[Claude Code]**
  - meeting_pipeline.pyファイルを作成
  - データクラス定義（AudioInfo, DeviceInfo, SpeakerTurn, ASRSegment, AlignedSegment等）をPythonで実装
  - PipelineConfig型を定義し、全てのCLIパラメータを含める
  - _Requirements: 1.5, 2.4, 3.5, 4.6, 5.6, 12.1, 12.2, 12.3_

- [x]* 1.1 コア型定義のプロパティテストを作成 **[Codex]**
  - **Property 28: JSON Serialization Round-Trip**
  - **Validates: Requirements 16.3**

- [x] 2. CLI Parserの実装 **[Claude Code]**
  - [x] 2.1 argparseを使用してコマンドライン引数を解析
    - 全てのパラメータ（input-file, device, enable-diarization, asr-engine, asr-model, language, beam-size, best-of, vad-filter, output-dir, temp-dir, keep-audio, format, bench-jsonl, run-id, note）を定義
    - デフォルト値を設定（device=auto, asr-engine=faster-whisper, asr-model=medium, language=ja, format=both等）
    - _Requirements: 1.1, 1.2, 2.1, 3.1, 4.1, 4.4, 4.5, 9.4_
  
  - [x] 2.2 入力検証ロジックを実装
    - 入力ファイルの存在確認
    - 出力ディレクトリの作成可能性確認
    - パラメータ値の妥当性検証
    - _Requirements: 11.1, 11.2_

- [x]* 2.3 CLI Parserのユニットテストを作成 **[Codex]**
  - 必須パラメータ欠落時のエラーハンドリングをテスト
  - 無効なパラメータ値のエラーハンドリングをテスト
  - _Requirements: 11.1_

- [x] 3. Device Resolverの実装 **[Claude Code]**
  - [x] 3.1 デバイス検出ロジックを実装
    - CUDA利用可能性チェック（torch.cuda.is_available()）
    - MPS利用可能性チェック（torch.backends.mps.is_available()）
    - auto指定時の優先順位（CUDA > MPS > CPU）
    - _Requirements: 2.1, 2.2, 2.3, 14.1, 14.2, 14.3_
  
  - [x] 3.2 DeviceInfo生成とロギング
    - requested/resolved deviceを記録
    - 選択されたデバイスをログ出力
    - _Requirements: 2.4_

- [x]* 3.3 Device Resolverのプロパティテストを作成 **[Codex]**
  - **Property 4: Device Selection Priority**
  - **Validates: Requirements 2.1**

- [x]* 3.4 Device Resolverのユニットテストを作成 **[Codex]**
  - CUDA/MPS利用不可時のエラーハンドリングをテスト
  - _Requirements: 2.2, 2.3_

- [x] 4. Audio Extractorの実装 **[Claude Code]**
  - [x] 4.1 ffmpegを使用した音声抽出機能を実装
    - subprocess経由でffmpegコマンドを実行
    - 16kHz mono PCM WAV形式に変換
    - 一時ファイル管理（temp_dir使用）
    - _Requirements: 1.1, 1.2_
  
  - [x] 4.2 AudioInfo生成とメタデータ記録
    - 抽出された音声ファイルのパス、サンプルレート、チャンネル数、再生時間を記録
    - keep-audioフラグに応じた一時ファイル保持/削除
    - _Requirements: 1.4, 1.5_
  
  - [x] 4.3 エラーハンドリングを実装
    - ffmpeg利用不可時のエラーメッセージ
    - 非対応フォーマット時のエラーメッセージ
    - _Requirements: 1.3, 11.3_

- [x]* 4.4 Audio Extractorのプロパティテストを作成 **[Codex]**
  - **Property 1: Audio Extraction Format Consistency**
  - **Validates: Requirements 1.1, 1.2**

- [x]* 4.5 Audio Extractorのユニットテストを作成 **[Codex]**
  - 非対応フォーマットのエラーハンドリングをテスト
  - keep-audioフラグの動作をテスト
  - _Requirements: 1.3, 1.4_

- [x] 5. Diarization Engineの実装 **[Claude Code]**
  - [x] 5.1 pyannote-audioを使用した話者分離機能を実装
    - HF_TOKEN環境変数の確認
    - pyannote.audio.Pipelineのロードと実行
    - 話者数を動的に検出（2人固定にしない）
    - _Requirements: 3.1, 3.2, 3.4_
  
  - [x] 5.2 SpeakerTurn生成とID割り当て
    - 出現順にSPEAKER_00, SPEAKER_01, ...を割り当て
    - turn_000001, turn_000002, ...の連番IDを生成
    - start/end時刻を記録
    - _Requirements: 3.3, 3.4_
  
  - [x] 5.3 DiarizationResult生成とメタデータ記録
    - モデル名、エンジン名、HF_Token使用状況を記録
    - 検出された全話者のリストを生成
    - _Requirements: 3.5_
  
  - [x] 5.4 リソース解放ロジックを実装
    - モデル参照をNoneに設定
    - gc.collect()とtorch.cuda.empty_cache()を呼び出し
    - _Requirements: 3.6, 13.1, 13.2_

- [x]* 5.5 Diarization Engineのユニットテストを作成 **[Codex]**
  - HF_TOKEN未設定時のエラーハンドリングをテスト
  - 話者ID割り当てロジックをテスト
  - _Requirements: 3.2, 3.4_

- [x] 6. ASR Engineの実装 **[Claude Code]**
  - [x] 6.1 faster-whisper実装を作成
    - WhisperModelのロードと設定（model_size, device, compute_type）
    - transcribeメソッドの実行（language, beam_size, best_of, vad_filter）
    - ASRSegment生成（asr_000001, asr_000002, ...の連番ID）
    - _Requirements: 4.1, 4.3, 4.4, 4.5, 4.7_
  
  - [x] 6.2 whisper実装の分岐を用意（Phase 1では基本構造のみ）
    - asr_engine='whisper'時の分岐処理を追加
    - whisper.load_modelとtranscribeの呼び出し構造を定義
    - _Requirements: 4.2_
  
  - [x] 6.3 ASRResult生成とメタデータ記録
    - モデル名、エンジン名、デバイス、compute_type、言語、beam_size、best_of、vad_filterを記録
    - _Requirements: 4.6_

- [x]* 6.4 ASR Engineのユニットテストを作成 **[Codex]**
  - ASRセグメントID生成ロジックをテスト
  - パラメータ適用を検証
  - _Requirements: 4.7, 4.4, 4.5_

- [x] 7. Alignment Moduleの実装 **[Claude Code]**
  - [x] 7.1 max_overlapアルゴリズムを実装
    - 各ASRセグメントと全SpeakerTurnの時間的重複を計算
    - 最大重複を持つSpeakerTurnのspeaker_idを割り当て
    - 重複がゼロの場合はspeaker_id="UNKNOWN"を割り当て
    - _Requirements: 5.1, 5.2, 5.3, 15.1_
  
  - [x] 7.2 AlignedSegment生成
    - seg_000001, seg_000002, ...の連番IDを生成
    - speaker_id, speaker_label, text, start, endを設定
    - SegmentSource（asr_segment_id, diarization_turn_id, overlap_sec）を記録
    - _Requirements: 5.4, 5.5_
  
  - [x] 7.3 UNKNOWN話者の保持を確認
    - 全てのASRセグメントがAlignedSegmentに変換されることを確認
    - UNKNOWNセグメントが破棄されないことを確認
    - _Requirements: 15.2, 15.4_

- [x]* 7.4 Alignment Moduleのプロパティテストを作成 **[Codex]**
  - **Property 11: Alignment Overlap Calculation**
  - **Validates: Requirements 5.1, 5.2**

- [x]* 7.5 Alignment Moduleのプロパティテストを作成 **[Codex]**
  - **Property 12: UNKNOWN Speaker Assignment**
  - **Validates: Requirements 5.3, 15.1, 15.2, 15.3, 15.4**

- [x]* 7.6 Alignment Moduleのユニットテストを作成 **[Codex]**
  - ゼロ重複時のUNKNOWN割り当てをテスト
  - 複数話者との重複時の最大重複選択をテスト
  - _Requirements: 5.3, 15.1_

- [x] 8. JSON Generatorの実装 **[Claude Code]**
  - [x] 8.1 Meeting JSON Schema v1.0構造を実装
    - schema_version, created_at, title, input, pipeline, speakers, segments, artifacts, timing, notesの全セクションを定義
    - ISO 8601形式のタイムスタンプ生成
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.6, 7.8, 12.5_
  
  - [x] 8.2 speakers配列の生成
    - 全識別済み話者を含める
    - UNKNOWN話者エントリ（id: "UNKNOWN", label: "Unknown"）を追加
    - speaker_labelを"Speaker 1", "Speaker 2", ...形式で生成
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [x] 8.3 segments配列の生成
    - AlignedSegmentを時刻順にソート
    - 全フィールド（id, start, end, speaker_id, speaker_label, text, confidence, source）を含める
    - _Requirements: 7.5_
  
  - [x] 8.4 artifacts、timing、その他メタデータの記録
    - diarization_turns配列（生のSpeakerTurn）
    - asr_segments配列（生のASRSegment）
    - 各ステージのタイミング（extract_sec, diarization_sec, asr_load_sec, asr_sec, align_sec, total_sec）
    - _Requirements: 7.6, 7.7, 12.4_
  
  - [x] 8.5 JSON検証とファイル保存
    - json.dumpsとjson.loadsでラウンドトリップ検証
    - {input_basename}_meeting.json形式でファイル保存
    - UTF-8エンコーディング、ensure_ascii=False、indent=2
    - _Requirements: 7.9, 16.1, 16.2, 16.3, 16.4_

- [x]* 8.6 JSON Generatorのプロパティテストを作成 **[Codex]**
  - **Property 14: Speaker Registry Completeness**
  - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 7.4**

- [x]* 8.7 JSON Generatorのプロパティテストを作成 **[Codex]**
  - **Property 15: Meeting JSON Schema Conformance**
  - **Validates: Requirements 7.1, 7.2, 7.3, 7.6, 16.1**

- [x]* 8.8 JSON Generatorのユニットテストを作成 **[Codex]**
  - ISO 8601タイムスタンプ形式をテスト
  - ファイル名生成ロジックをテスト
  - JSON検証エラーハンドリングをテスト
  - _Requirements: 7.8, 7.9, 16.2_

- [x] 9. メインパイプライン統合 **[Claude Code]**
  - [x] 9.1 パイプライン実行フローを実装
    - CLI Parser → Device Resolver → Audio Extractor → Diarization Engine → ASR Engine → Alignment Module → JSON Generator の順次実行
    - 各ステージの実行時間を計測
    - _Requirements: 13.1, 13.3_
  
  - [x] 9.2 エラーハンドリングとロギングを実装
    - 各ステージでのtry-exceptブロック
    - 失敗ステージの明示とスタックトレース出力
    - 中間出力の保存
    - _Requirements: 11.4, 11.5_
  
  - [x] 9.3 リソース管理を実装
    - Diarization Engine後のモデル解放
    - 一時ファイルのクリーンアップ（keep-audioフラグに応じて）
    - _Requirements: 3.6, 13.2_

- [x]* 9.4 メインパイプラインのユニットテストを作成 **[Codex]**
  - エラーハンドリングをテスト
  - リソース解放タイミングをテスト
  - _Requirements: 11.5, 13.2_

- [x] 10. Checkpoint - Phase 1完了確認 **[Claude Code + Codex + User]**
  - Phase 1の全実装タスクが完了していることを確認
  - サンプル音声ファイルでエンドツーエンドテストを実行
  - Meeting JSONが正しく生成されることを確認
  - 全テストがパスすることを確認
  - ユーザーに質問や問題がないか確認

### Phase 2: Markdown生成

- [x] 11. Markdown Generatorの実装 **[Claude Code]**
  - [x] 11.1 Meeting JSONからMarkdown生成ロジックを実装
    - segments配列を時刻順に処理（既にソート済みの想定）
    - speaker_label変更時に"### {speaker_label}"見出しを挿入
    - 各セグメントを"- [HH:MM:SS - HH:MM:SS] {text}"形式で出力
    - _Requirements: 8.1, 8.2, 8.3_
  
  - [x] 11.2 タイムスタンプフォーマット処理を実装
    - 秒数をHH:MM:SS形式に変換
    - ゼロパディングを適用
    - _Requirements: 8.4, 8.6_
  
  - [x] 11.3 空テキストのスキップロジックを実装
    - textが空または空白のみのセグメントをスキップ
    - _Requirements: 8.5_
  
  - [x] 11.4 Markdownファイル保存
    - {input_basename}_transcript.md形式でファイル保存
    - UTF-8エンコーディング
    - _Requirements: 8.7_

- [x]* 11.5 Markdown Generatorのプロパティテストを作成 **[Codex]**
  - **Property 20: Markdown Speaker Grouping**
  - **Validates: Requirements 8.2, 8.3**

- [x]* 11.6 Markdown Generatorのプロパティテストを作成 **[Codex]**
  - **Property 21: Markdown Segment Formatting**
  - **Validates: Requirements 8.4, 8.5, 8.6**

- [x]* 11.7 Markdown Generatorのユニットテストを作成 **[Codex]**
  - タイムスタンプフォーマットをテスト
  - 空テキストスキップをテスト
  - ファイル名生成をテスト
  - _Requirements: 8.4, 8.5, 8.6, 8.7_

- [x] 12. 出力フォーマット制御の実装 **[Claude Code]**
  - [x] 12.1 --formatフラグに応じた出力制御
    - format='json': Meeting JSONのみ生成
    - format='md': Transcript Markdownのみ生成（Meeting JSONを中間生成してから変換）
    - format='both': 両方生成
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x]* 12.2 出力フォーマット制御のユニットテストを作成 **[Codex]**
  - 各formatオプションの動作をテスト
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 13. Benchmark Loggerの実装（オプション機能） **[Claude Code]**
  - [x] 13.1 BenchmarkRecord生成ロジックを実装
    - run_id, timestamp, input_file, device, models, timing, noteを含める
    - run_id未指定時はタイムスタンプから生成
    - _Requirements: 10.1, 10.2, 10.3, 10.4_
  
  - [x] 13.2 JSONL追記機能を実装
    - 指定されたJSONLファイルに1行追記
    - ファイルが存在しない場合は新規作成
    - _Requirements: 10.1_

- [x]* 13.3 Benchmark Loggerのユニットテストを作成 **[Codex]**
  - run_id自動生成をテスト
  - JSONL追記をテスト
  - _Requirements: 10.3_

- [x] 14. Checkpoint - Phase 2完了確認 **[Claude Code + Codex + User]**
  - Phase 2の全実装タスクが完了していることを確認
  - サンプル音声ファイルでMarkdown生成をテスト
  - 全出力フォーマット（json, md, both）をテスト
  - 全テストがパスすることを確認
  - ユーザーに質問や問題がないか確認

### Phase 3: 精度改善（オプション）

- [ ]* 15. 単語/句単位アライン機能の実装
  - [ ]* 15.1 faster-whisperのword_timestampsオプションを有効化
    - transcribe時にword_timestamps=Trueを指定
    - 単語レベルのタイムスタンプを取得
    - _Requirements: 該当なし（Phase 3拡張機能）_
  
  - [ ]* 15.2 単語レベルアライン処理を実装
    - 各単語とSpeakerTurnの重複を計算
    - 単語単位でspeaker_idを割り当て
    - 同一話者の連続単語を1つのAlignedSegmentに統合
    - _Requirements: 該当なし（Phase 3拡張機能）_
  
  - [ ]* 15.3 align.unitメタデータを更新
    - unit='word'または'phrase'を記録
    - _Requirements: 該当なし（Phase 3拡張機能）_

- [ ]* 15.4 単語レベルアラインのユニットテストを作成
  - 単語タイムスタンプ取得をテスト
  - 単語レベル重複計算をテスト
  - 連続単語統合ロジックをテスト

- [ ]* 16. Checkpoint - Phase 3完了確認（オプション）
  - Phase 3の全実装タスクが完了していることを確認
  - 単語レベルアラインの精度向上を検証
  - 全テストがパスすることを確認

### Phase 4: クロスプラットフォーム最適化（オプション）

- [ ]* 17. macOS MPS/CPU対応の検証と最適化
  - [ ]* 17.1 macOS環境でのエンドツーエンドテストを実施
    - MPS deviceでの実行を検証
    - CPU deviceでの実行を検証
    - _Requirements: 14.2, 14.3_
  
  - [ ]* 17.2 compute_type自動選択ロジックを最適化
    - MPS使用時のcompute_type設定を調整
    - CPU使用時のint8設定を確認
    - _Requirements: 該当なし（Phase 4最適化）_
  
  - [ ]* 17.3 クロスプラットフォーム一貫性を検証
    - 同一入力ファイルでWindows/macOS出力を比較
    - スキーマ構造の一貫性を確認
    - Markdownフォーマットの一貫性を確認
    - _Requirements: 14.4, 14.5_

- [ ]* 17.4 クロスプラットフォーム検証のユニットテストを作成
  - 各プラットフォームでのデバイス選択をテスト
  - compute_type選択ロジックをテスト

- [ ]* 18. Checkpoint - Phase 4完了確認（オプション）
  - Phase 4の全実装タスクが完了していることを確認
  - Windows/macOS両環境での動作を確認
  - 全テストがパスすることを確認

### 最終統合

- [ ] 19. 最終統合とドキュメント整備 **[Claude Code]**
  - [ ] 19.1 README.mdを更新
    - インストール手順（依存パッケージ、HF_TOKEN設定）
    - 使用例（基本的なコマンド例）
    - パラメータ一覧
    - トラブルシューティング
  
  - [ ] 19.2 requirements.txtを更新
    - 全依存パッケージとバージョンを記載
    - pyannote-audio, faster-whisper, torch等
  
  - [ ] 19.3 エンドツーエンド統合テストを実施 **[Codex]**
    - 複数の入力ファイル形式（mp4, wav, m4a）でテスト
    - 全出力フォーマット（json, md, both）でテスト
    - 全デバイスオプション（auto, cuda, mps, cpu）でテスト

- [ ] 20. 最終Checkpoint **[User]**
  - 全必須タスク（Phase 1, Phase 2）が完了していることを確認
  - bench_transcribe.pyが変更されていないことを確認
  - meeting_pipeline.pyが正しく動作することを確認
  - 全テストがパスすることを確認
  - ユーザーに最終確認と質問がないか確認

## Notes

- タスクに`*`が付いているものはオプションタスクで、スキップ可能です
- Phase 1とPhase 2は必須、Phase 3とPhase 4はオプションです
- 各タスクは要件番号を参照しており、トレーサビリティを確保しています
- Checkpointタスクで段階的な検証を行い、問題を早期発見します
- プロパティテストは設計ドキュメントのプロパティ番号を明示しています
