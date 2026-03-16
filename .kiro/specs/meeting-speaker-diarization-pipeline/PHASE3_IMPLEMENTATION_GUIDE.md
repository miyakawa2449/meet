# Phase 3 実装指示書: 単語/句単位アライン機能

## 概要

Phase 1とPhase 2が完了し、基本的な話者分離パイプラインが動作しています。Phase 3では、**単語/句単位のアライン機能**を実装して、話者割り当ての精度を向上させます。

現在の実装では、ASRセグメント全体（通常数秒〜十数秒）と話者ターンの重複を計算していますが、Phase 3では単語レベルのタイムスタンプを活用して、より細かい粒度でアライメントを行います。

## 実装担当

- **Claude Code (Kiro)**: 実装担当
- **Codex**: テスト・検証担当

## 対象タスク

- **タスク15**: 単語/句単位アライン機能の実装
  - タスク15.1: faster-whisperのword_timestampsオプション有効化
  - タスク15.2: 単語レベルアライン処理の実装
  - タスク15.3: align.unitメタデータの更新

## 参照ドキュメント

- `.kiro/specs/meeting-speaker-diarization-pipeline/SESSION_SUMMARY.md` - プロジェクト全体の状況
- `.kiro/specs/meeting-speaker-diarization-pipeline/tasks.md` - タスクリスト
- `.kiro/specs/meeting-speaker-diarization-pipeline/design.md` - 設計詳細
- `meeting_pipeline.py` - 現在の実装

## 現在の実装状況

### ✅ 完了している機能


- ASRセグメント単位のアライメント（max_overlapアルゴリズム）
- UNKNOWN話者の保持
- JSON/Markdown出力
- 全38テストがパス（カバレッジ90%）

### 🎯 Phase 3で実装する機能

faster-whisperの`word_timestamps`機能を使用して、単語レベルのタイムスタンプを取得し、より細かい粒度でアライメントを実行します。

**メリット**:
- 話者が頻繁に切り替わる会議でも正確な話者割り当て
- 長いASRセグメント内で話者が変わった場合も対応可能
- UNKNOWN割り当ての削減

## タスク15.1: faster-whisperのword_timestampsオプション有効化

### 実装箇所

`meeting_pipeline.py` の `_run_faster_whisper()` 関数

### 現在のコード

```python
def _run_faster_whisper(
    audio_path: str,
    device: str,
    compute_type: str,
    config: PipelineConfig,
) -> ASRResult:
    # ... モデルロード処理 ...
    
    segments_iter, info = model.transcribe(
        audio_path,
        language=config.language,
        beam_size=config.beam_size,
        best_of=config.best_of,
        vad_filter=config.vad_filter,
    )
```

### 実装内容

1. **transcribeメソッドに`word_timestamps=True`を追加**


```python
segments_iter, info = model.transcribe(
    audio_path,
    language=config.language,
    beam_size=config.beam_size,
    best_of=config.best_of,
    vad_filter=config.vad_filter,
    word_timestamps=True,  # 追加
)
```

2. **単語タイムスタンプの取得と保存**

faster-whisperのセグメントオブジェクトには`.words`属性があり、各単語のタイムスタンプが含まれます。

```python
# 各セグメントから単語情報を取得
for seg in segments_iter:
    words = []
    if hasattr(seg, 'words') and seg.words:
        for word in seg.words:
            words.append({
                'word': word.word,
                'start': word.start,
                'end': word.end,
                'probability': word.probability
            })
```

3. **ASRSegmentデータクラスの拡張**

単語情報を保存するため、`ASRSegment`に`words`フィールドを追加します。

```python
@dataclass
class ASRSegment:
    id: str
    start: float
    end: float
    text: str
    words: Optional[List[Dict[str, Any]]] = None  # 追加
```

**注意**: この変更は後方互換性があります。既存のコードは`words=None`として動作します。

### 検証ポイント

- `word_timestamps=True`が正しく設定されている
- 単語情報が`ASRSegment.words`に保存されている
- 単語情報がない場合（whisperエンジン使用時など）も正常動作する



## タスク15.2: 単語レベルアライン処理の実装

### 実装箇所

`meeting_pipeline.py` に新しい関数 `align_segments_word_level()` を追加

### アルゴリズム概要

1. 各ASRセグメント内の単語を順次処理
2. 各単語と全SpeakerTurnの重複を計算
3. 最大重複を持つSpeakerTurnのspeaker_idを割り当て
4. 同一話者の連続単語を1つのAlignedSegmentに統合
5. 重複ゼロの単語はUNKNOWNに割り当て

### 実装コード

```python
def align_segments_word_level(
    asr_segments: List[ASRSegment],
    speaker_turns: List[SpeakerTurn],
) -> List[AlignedSegment]:
    """
    Align ASR segments with speaker turns at word level.
    
    Args:
        asr_segments: List of ASR segments with word timestamps
        speaker_turns: List of speaker turns from diarization
    
    Returns:
        List of aligned segments with speaker labels
    """
    aligned_segments = []
    seg_counter = 1
    
    for asr_seg in asr_segments:
        # 単語情報がない場合は従来のセグメント単位アライメントにフォールバック
        if not asr_seg.words:
            # 既存のmax_overlapロジックを使用
            aligned = _align_single_segment(asr_seg, speaker_turns, seg_counter)
            aligned_segments.append(aligned)
            seg_counter += 1
            continue
        
        # 単語レベルアライメント
        word_alignments = []
        for word_info in asr_seg.words:
            word_start = word_info['start']
            word_end = word_info['end']
            word_text = word_info['word']
            
            # 各単語と全話者ターンの重複を計算
            max_overlap = 0.0
            best_turn = None
            
            for turn in speaker_turns:
                overlap = _calculate_overlap(
                    word_start, word_end,
                    turn.start, turn.end
                )
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_turn = turn
            
            # 話者割り当て
            if max_overlap > 0 and best_turn:
                speaker_id = best_turn.speaker_id
                turn_id = best_turn.id
            else:
                speaker_id = "UNKNOWN"
                turn_id = None
            
            word_alignments.append({
                'word': word_text,
                'start': word_start,
                'end': word_end,
                'speaker_id': speaker_id,
                'turn_id': turn_id,
                'overlap': max_overlap
            })
        
        # 同一話者の連続単語を統合
        merged_segments = _merge_consecutive_words(word_alignments, asr_seg.id)
        
        for merged in merged_segments:
            aligned_segments.append(AlignedSegment(
                id=f"seg_{seg_counter:06d}",
                start=merged['start'],
                end=merged['end'],
                speaker_id=merged['speaker_id'],
                speaker_label=_get_speaker_label(merged['speaker_id']),
                text=merged['text'],
                confidence=None,
                source=SegmentSource(
                    asr_segment_id=asr_seg.id,
                    diarization_turn_id=merged['turn_id'],
                    overlap_sec=merged['overlap']
                )
            ))
            seg_counter += 1
    
    return aligned_segments
```


### ヘルパー関数の実装

#### 1. 単語統合関数

```python
def _merge_consecutive_words(
    word_alignments: List[Dict],
    asr_segment_id: str
) -> List[Dict]:
    """
    Merge consecutive words with the same speaker into segments.
    
    Args:
        word_alignments: List of word-level alignments
        asr_segment_id: Source ASR segment ID
    
    Returns:
        List of merged segments
    """
    if not word_alignments:
        return []
    
    merged = []
    current_segment = {
        'speaker_id': word_alignments[0]['speaker_id'],
        'turn_id': word_alignments[0]['turn_id'],
        'start': word_alignments[0]['start'],
        'end': word_alignments[0]['end'],
        'words': [word_alignments[0]['word']],
        'overlap': word_alignments[0]['overlap']
    }
    
    for word_align in word_alignments[1:]:
        # 同じ話者なら統合
        if word_align['speaker_id'] == current_segment['speaker_id']:
            current_segment['end'] = word_align['end']
            current_segment['words'].append(word_align['word'])
            current_segment['overlap'] += word_align['overlap']
        else:
            # 話者が変わったら現在のセグメントを保存
            current_segment['text'] = ''.join(current_segment['words']).strip()
            merged.append(current_segment)
            
            # 新しいセグメント開始
            current_segment = {
                'speaker_id': word_align['speaker_id'],
                'turn_id': word_align['turn_id'],
                'start': word_align['start'],
                'end': word_align['end'],
                'words': [word_align['word']],
                'overlap': word_align['overlap']
            }
    
    # 最後のセグメントを追加
    current_segment['text'] = ''.join(current_segment['words']).strip()
    merged.append(current_segment)
    
    return merged
```


#### 2. 話者ラベル取得関数

```python
def _get_speaker_label(speaker_id: str) -> str:
    """
    Convert speaker_id to human-readable label.
    
    Args:
        speaker_id: Speaker ID (SPEAKER_00, SPEAKER_01, ..., or UNKNOWN)
    
    Returns:
        Human-readable label (Speaker 1, Speaker 2, ..., or Unknown)
    """
    if speaker_id == "UNKNOWN":
        return "Unknown"
    
    # SPEAKER_00 -> Speaker 1, SPEAKER_01 -> Speaker 2, ...
    if speaker_id.startswith("SPEAKER_"):
        try:
            num = int(speaker_id.split("_")[1])
            return f"Speaker {num + 1}"
        except (IndexError, ValueError):
            return speaker_id
    
    return speaker_id
```

#### 3. 単一セグメントアライメント関数（フォールバック用）

```python
def _align_single_segment(
    asr_seg: ASRSegment,
    speaker_turns: List[SpeakerTurn],
    seg_counter: int
) -> AlignedSegment:
    """
    Align a single ASR segment using segment-level overlap (fallback).
    
    Args:
        asr_seg: ASR segment to align
        speaker_turns: List of speaker turns
        seg_counter: Current segment counter
    
    Returns:
        Aligned segment
    """
    max_overlap = 0.0
    best_turn = None
    
    for turn in speaker_turns:
        overlap = _calculate_overlap(
            asr_seg.start, asr_seg.end,
            turn.start, turn.end
        )
        if overlap > max_overlap:
            max_overlap = overlap
            best_turn = turn
    
    if max_overlap > 0 and best_turn:
        speaker_id = best_turn.speaker_id
        turn_id = best_turn.id
    else:
        speaker_id = "UNKNOWN"
        turn_id = None
    
    return AlignedSegment(
        id=f"seg_{seg_counter:06d}",
        start=asr_seg.start,
        end=asr_seg.end,
        speaker_id=speaker_id,
        speaker_label=_get_speaker_label(speaker_id),
        text=asr_seg.text,
        confidence=None,
        source=SegmentSource(
            asr_segment_id=asr_seg.id,
            diarization_turn_id=turn_id,
            overlap_sec=max_overlap
        )
    )
```



### 検証ポイント

- `word_timestamps=True`が設定されている
- 単語情報が取得できている（`seg.words`が存在する）
- `ASRSegment.words`に単語情報が保存されている
- 単語情報がない場合も正常動作する（後方互換性）

## タスク15.3: align.unitメタデータの更新

### 実装箇所

`meeting_pipeline.py` の `run_pipeline()` 関数と `AlignConfig` データクラス

### 現在のコード

```python
@dataclass
class AlignConfig:
    method: str
    unit: str
```

### 実装内容

1. **CLIパラメータの追加**

`parse_args()` 関数に `--align-unit` パラメータを追加します。

```python
parser.add_argument(
    "--align-unit",
    type=str,
    choices=["segment", "word"],
    default="segment",
    help="Alignment unit: segment (default) or word"
)
```

2. **PipelineConfigへの追加**

```python
@dataclass
class PipelineConfig:
    # ... 既存のフィールド ...
    align_unit: str = "segment"  # 追加
```

3. **run_pipeline()での分岐処理**

```python
def run_pipeline(config: PipelineConfig) -> None:
    # ... 既存の処理 ...
    
    # Alignment
    logger.info("Starting alignment...")
    align_start = time.time()
    
    if config.align_unit == "word":
        aligned_segments = align_segments_word_level(
            asr_result.segments,
            diarization_result.turns
        )
    else:
        aligned_segments = align_segments(
            asr_result.segments,
            diarization_result.turns,
            method="max_overlap"
        )
    
    align_sec = time.time() - align_start
    logger.info(f"Alignment completed in {align_sec:.2f}s")
```


4. **AlignConfigへの反映**

```python
# PipelineInfo生成時
align_config = AlignConfig(
    method="max_overlap",
    unit=config.align_unit  # "segment" または "word"
)
```

### 検証ポイント

- `--align-unit` パラメータが正しく解析される
- `unit="word"` の場合、`align_segments_word_level()` が呼ばれる
- `unit="segment"` の場合、既存の`align_segments()` が呼ばれる（デフォルト動作）
- Meeting JSONの`pipeline.align.unit`に正しい値が記録される

## 実装の注意事項

### 1. 後方互換性の維持

- デフォルトは`--align-unit segment`（既存の動作）
- 単語情報がない場合は自動的にセグメント単位にフォールバック
- 既存のテストは全てパスする必要がある

### 2. エラーハンドリング

```python
# 単語情報がない場合の警告
if config.align_unit == "word" and not any(seg.words for seg in asr_result.segments):
    logger.warning(
        "Word-level alignment requested but no word timestamps available. "
        "Falling back to segment-level alignment."
    )
```

### 3. パフォーマンス考慮

- 単語レベルアライメントは計算量が増加する（単語数 × 話者ターン数）
- 大規模な会議（1時間以上）では処理時間が増加する可能性
- 必要に応じて最適化（例: 時間範囲でフィルタリング）

### 4. メモリ効率

- 単語情報は`ASRSegment.words`に保存されるため、メモリ使用量が増加
- artifactsセクションに単語情報を含めるかは検討の余地あり（現時点では含めない）



## 実装の順序

Phase 3の実装は以下の順序で進めてください：

### ステップ1: データクラスの拡張

1. `ASRSegment`に`words`フィールドを追加
2. `PipelineConfig`に`align_unit`フィールドを追加

### ステップ2: faster-whisperの単語タイムスタンプ取得

1. `_run_faster_whisper()`で`word_timestamps=True`を設定
2. 単語情報を`ASRSegment.words`に保存

### ステップ3: 単語レベルアライメント関数の実装

1. `align_segments_word_level()`関数を実装
2. `_merge_consecutive_words()`ヘルパー関数を実装
3. `_get_speaker_label()`ヘルパー関数を実装
4. `_align_single_segment()`ヘルパー関数を実装（フォールバック用）

### ステップ4: CLIパラメータとパイプライン統合

1. `parse_args()`に`--align-unit`パラメータを追加
2. `run_pipeline()`で`align_unit`に応じて分岐処理
3. `AlignConfig`に`unit`を反映

### ステップ5: 動作確認

1. `--align-unit segment`で既存の動作を確認（全テストパス）
2. `--align-unit word`で単語レベルアライメントを確認

## テスト戦略（Codex担当）

### タスク15.4: 単語レベルアラインのユニットテスト

Codexは以下のテストを作成してください：

#### 1. 単語タイムスタンプ取得のテスト

```python
def test_word_timestamps_extraction():
    """Test that word timestamps are extracted from faster-whisper"""
    # Mock faster-whisper with word timestamps
    # Verify ASRSegment.words contains word info
    pass
```

#### 2. 単語レベル重複計算のテスト

```python
def test_word_level_overlap_calculation():
    """Test overlap calculation at word level"""
    # Create word alignments with known overlaps
    # Verify correct speaker assignment
    pass
```


#### 3. 連続単語統合のテスト

```python
def test_consecutive_word_merging():
    """Test merging of consecutive words with same speaker"""
    word_alignments = [
        {'word': 'こんにちは', 'start': 0.0, 'end': 1.0, 'speaker_id': 'SPEAKER_00', 'turn_id': 'turn_001', 'overlap': 1.0},
        {'word': 'です', 'start': 1.0, 'end': 1.5, 'speaker_id': 'SPEAKER_00', 'turn_id': 'turn_001', 'overlap': 0.5},
        {'word': 'ありがとう', 'start': 2.0, 'end': 3.0, 'speaker_id': 'SPEAKER_01', 'turn_id': 'turn_002', 'overlap': 1.0},
    ]
    
    merged = _merge_consecutive_words(word_alignments, "asr_000001")
    
    assert len(merged) == 2
    assert merged[0]['speaker_id'] == 'SPEAKER_00'
    assert merged[0]['text'] == 'こんにちはです'
    assert merged[1]['speaker_id'] == 'SPEAKER_01'
    assert merged[1]['text'] == 'ありがとう'
```

#### 4. フォールバック動作のテスト

```python
def test_word_level_fallback_to_segment():
    """Test fallback to segment-level when word timestamps unavailable"""
    # Create ASR segment without words
    asr_seg = ASRSegment(id="asr_000001", start=0.0, end=5.0, text="test", words=None)
    
    # Should use segment-level alignment
    aligned = align_segments_word_level([asr_seg], speaker_turns)
    
    assert len(aligned) == 1
    # Verify it used segment-level logic
```

#### 5. エンドツーエンドテスト

```python
def test_word_level_alignment_end_to_end():
    """Test complete pipeline with word-level alignment"""
    config = PipelineConfig(
        input_file="tests/fixtures/sample.wav",
        enable_diarization=True,
        align_unit="word",  # 単語レベル指定
        format="both",
        output_dir="tests/output"
    )
    
    run_pipeline(config)
    
    # Verify output
    with open("tests/output/sample_meeting.json") as f:
        meeting = json.load(f)
    
    assert meeting['pipeline']['align']['unit'] == 'word'
    # Verify segments are properly aligned
```



## 実装チェックリスト

Claude Codeは以下の項目を確認しながら実装してください：

### データモデル
- [ ] `ASRSegment`に`words: Optional[List[Dict[str, Any]]]`フィールドを追加
- [ ] `PipelineConfig`に`align_unit: str`フィールドを追加

### faster-whisper統合
- [ ] `_run_faster_whisper()`で`word_timestamps=True`を設定
- [ ] 単語情報を`ASRSegment.words`に保存
- [ ] 単語情報がない場合も正常動作（`words=None`）

### アライメント関数
- [ ] `align_segments_word_level()`関数を実装
- [ ] `_merge_consecutive_words()`ヘルパー関数を実装
- [ ] `_get_speaker_label()`ヘルパー関数を実装
- [ ] `_align_single_segment()`ヘルパー関数を実装

### CLI統合
- [ ] `parse_args()`に`--align-unit`パラメータを追加
- [ ] `run_pipeline()`で`align_unit`に応じて分岐
- [ ] `AlignConfig`に`unit`を設定

### エラーハンドリング
- [ ] 単語情報がない場合の警告ログ
- [ ] フォールバック動作の実装

### 後方互換性
- [ ] デフォルトは`--align-unit segment`
- [ ] 既存のテストが全てパス
- [ ] 既存の`align_segments()`関数は変更しない（新関数を追加）

## 実行例

### セグメント単位アライメント（既存の動作）

```bash
python meeting_pipeline.py 202602017_mtg.mp4 \
  --enable-diarization \
  --device auto \
  --asr-engine faster-whisper \
  --asr-model medium \
  --language ja \
  --align-unit segment \
  --format both
```

### 単語単位アライメント（Phase 3の新機能）

```bash
python meeting_pipeline.py 202602017_mtg.mp4 \
  --enable-diarization \
  --device auto \
  --asr-engine faster-whisper \
  --asr-model medium \
  --language ja \
  --align-unit word \
  --format both
```



## 期待される出力の変化

### Meeting JSON

```json
{
  "pipeline": {
    "align": {
      "method": "max_overlap",
      "unit": "word"  // "segment" から "word" に変更
    }
  },
  "segments": [
    {
      "id": "seg_000001",
      "start": 12.34,
      "end": 14.56,
      "speaker_id": "SPEAKER_00",
      "speaker_label": "Speaker 1",
      "text": "こんにちは",
      "source": {
        "asr_segment_id": "asr_000001",
        "diarization_turn_id": "turn_000001",
        "overlap_sec": 2.22
      }
    },
    {
      "id": "seg_000002",
      "start": 14.56,
      "end": 16.78,
      "speaker_id": "SPEAKER_01",
      "speaker_label": "Speaker 2",
      "text": "ありがとう",
      "source": {
        "asr_segment_id": "asr_000001",  // 同じASRセグメントから分割
        "diarization_turn_id": "turn_000002",
        "overlap_sec": 2.22
      }
    }
  ]
}
```

**変化点**:
- 1つのASRセグメントが複数のAlignedSegmentに分割される可能性
- より細かい粒度で話者が切り替わる
- `pipeline.align.unit`が"word"になる

### Transcript Markdown

```markdown
### Speaker 1
- [00:12:34 - 00:14:56] こんにちは

### Speaker 2
- [00:14:56 - 00:16:78] ありがとう
```

**変化点**:
- 話者切り替わりがより頻繁になる
- 短いセグメントが増える



## トラブルシューティング

### 問題1: 単語タイムスタンプが取得できない

**症状**: `seg.words`が空またはNone

**原因**:
- faster-whisperのバージョンが古い
- `word_timestamps=True`が設定されていない

**解決方法**:
```bash
pip install --upgrade faster-whisper
```

### 問題2: メモリ不足エラー

**症状**: 単語レベルアライメント中にOOMエラー

**原因**: 単語数が多すぎる（長時間の会議）

**解決方法**:
- `--align-unit segment`にフォールバック
- または、時間範囲でフィルタリングする最適化を追加

### 問題3: 処理時間が長すぎる

**症状**: 単語レベルアライメントに時間がかかる

**原因**: 単語数 × 話者ターン数の計算量

**解決方法**:
- 時間範囲でフィルタリング（単語の前後5秒以内の話者ターンのみ計算）
- または、`--align-unit segment`を使用

## 完了基準

Phase 3の実装が完了したと判断する基準：

### 実装完了
- [ ] 全てのコード変更が完了
- [ ] `--align-unit word`で実行できる
- [ ] `--align-unit segment`で既存の動作を維持
- [ ] Meeting JSONの`pipeline.align.unit`が正しく記録される

### テスト完了（Codex担当）
- [ ] 単語タイムスタンプ取得のテストがパス
- [ ] 単語レベル重複計算のテストがパス
- [ ] 連続単語統合のテストがパス
- [ ] フォールバック動作のテストがパス
- [ ] エンドツーエンドテストがパス
- [ ] 既存の全38テストが引き続きパス

### 動作確認
- [ ] 実際の音声ファイルで`--align-unit word`を実行
- [ ] 出力JSONとMarkdownを目視確認
- [ ] 話者切り替わりが適切に検出されている



## 実装時の重要な制約

### 絶対に守るべきこと

1. **bench_transcribe.pyは変更しない**
2. **既存のテストは全てパスする**（後方互換性）
3. **デフォルトは`--align-unit segment`**（既存の動作）
4. **UNKNOWN話者を破棄しない**
5. **単語情報がない場合はセグメント単位にフォールバック**

### 推奨事項

1. **段階的な実装**: データクラス → faster-whisper → アライメント → CLI統合
2. **こまめなテスト**: 各ステップで動作確認
3. **ログ出力**: 単語レベルアライメントの動作をログで確認できるようにする
4. **エラーハンドリング**: 単語情報がない場合の警告を出力

## 参考情報

### faster-whisperのword_timestamps

faster-whisperのドキュメント: https://github.com/SYSTRAN/faster-whisper

```python
# word_timestamps=Trueを指定すると、各セグメントに.words属性が追加される
segments, info = model.transcribe(audio, word_timestamps=True)

for segment in segments:
    for word in segment.words:
        print(f"{word.start:.2f}s - {word.end:.2f}s: {word.word}")
```

### 単語情報の構造

```python
# word.word: str - 単語テキスト
# word.start: float - 開始時刻（秒）
# word.end: float - 終了時刻（秒）
# word.probability: float - 信頼度（0.0〜1.0）
```

## 実装開始時の指示例

```
Phase 3の実装を開始してください。

#PHASE3_IMPLEMENTATION_GUIDE.md の指示に従ってください。
#tasks.md のタスク15を実装してください。
#meeting_pipeline.py を編集してください。

実装の順序:
1. データクラスの拡張（ASRSegment、PipelineConfig）
2. faster-whisperの単語タイムスタンプ取得
3. 単語レベルアライメント関数の実装
4. CLIパラメータとパイプライン統合

後方互換性を維持し、既存のテストが全てパスすることを確認してください。
```



## コード変更の詳細マップ

### 変更ファイル: meeting_pipeline.py

#### 1. データクラスの変更（行46付近）

```python
# 変更前
@dataclass
class ASRSegment:
    id: str
    start: float
    end: float
    text: str

# 変更後
@dataclass
class ASRSegment:
    id: str
    start: float
    end: float
    text: str
    words: Optional[List[Dict[str, Any]]] = None  # 追加
```

#### 2. PipelineConfigの変更（行196付近）

```python
# 変更前
@dataclass
class PipelineConfig:
    # ... 既存のフィールド ...
    note: Optional[str] = None

# 変更後
@dataclass
class PipelineConfig:
    # ... 既存のフィールド ...
    note: Optional[str] = None
    align_unit: str = "segment"  # 追加
```

#### 3. parse_args()の変更（行221付近）

```python
# 追加するパラメータ
parser.add_argument(
    "--align-unit",
    type=str,
    choices=["segment", "word"],
    default="segment",
    help="Alignment unit: segment (default) or word-level"
)

# return文の変更
return PipelineConfig(
    # ... 既存のフィールド ...
    align_unit=args.align_unit,  # 追加
)
```

#### 4. _run_faster_whisper()の変更（行657付近）

```python
# transcribe呼び出しの変更
segments_iter, info = model.transcribe(
    audio_path,
    language=config.language,
    beam_size=config.beam_size,
    best_of=config.best_of,
    vad_filter=config.vad_filter,
    word_timestamps=True,  # 追加
)

# セグメント処理の変更
for seg in segments_iter:
    words = None
    if hasattr(seg, 'words') and seg.words:
        words = [
            {
                'word': w.word,
                'start': w.start,
                'end': w.end,
                'probability': w.probability
            }
            for w in seg.words
        ]
    
    asr_segments.append(
        ASRSegment(
            id=f"asr_{seg_counter:06d}",
            start=seg.start,
            end=seg.end,
            text=seg.text,
            words=words,  # 追加
        )
    )
```



#### 5. 新関数の追加（行792付近、align_segments()の後）

```python
def align_segments_word_level(
    asr_segments: List[ASRSegment],
    speaker_turns: List[SpeakerTurn],
) -> List[AlignedSegment]:
    """実装内容は上記参照"""
    pass

def _merge_consecutive_words(
    word_alignments: List[Dict],
    asr_segment_id: str
) -> List[Dict]:
    """実装内容は上記参照"""
    pass

def _get_speaker_label(speaker_id: str) -> str:
    """実装内容は上記参照"""
    pass

def _align_single_segment(
    asr_seg: ASRSegment,
    speaker_turns: List[SpeakerTurn],
    seg_counter: int
) -> AlignedSegment:
    """実装内容は上記参照"""
    pass
```

#### 6. run_pipeline()の変更（行1021付近）

```python
# Alignment処理の変更
logger.info("Starting alignment...")
align_start = time.time()

if config.align_unit == "word":
    # 単語情報の確認
    has_words = any(seg.words for seg in asr_result.segments)
    if not has_words:
        logger.warning(
            "Word-level alignment requested but no word timestamps available. "
            "Falling back to segment-level alignment."
        )
        aligned_segments = align_segments(
            asr_result.segments,
            diarization_result.turns,
            method="max_overlap"
        )
    else:
        aligned_segments = align_segments_word_level(
            asr_result.segments,
            diarization_result.turns
        )
else:
    aligned_segments = align_segments(
        asr_result.segments,
        diarization_result.turns,
        method="max_overlap"
    )

align_sec = time.time() - align_start
```



#### 7. AlignConfig生成の変更（行1100付近）

```python
# 変更前
align_config = AlignConfig(
    method="max_overlap",
    unit="asr_segment"
)

# 変更後
align_config = AlignConfig(
    method="max_overlap",
    unit=config.align_unit  # "segment" または "word"
)
```

## 実装後の動作確認

### 1. 既存機能の確認

```bash
# 既存のテストが全てパス
pytest tests/ -v

# セグメント単位アライメントで実行
python meeting_pipeline.py 202602017_mtg.mp4 \
  --enable-diarization \
  --align-unit segment \
  --format both
```

### 2. 新機能の確認

```bash
# 単語単位アライメントで実行
python meeting_pipeline.py 202602017_mtg.mp4 \
  --enable-diarization \
  --align-unit word \
  --format both

# 出力確認
cat output/202602017_mtg_meeting.json | jq '.pipeline.align.unit'
# 期待値: "word"

cat output/202602017_mtg_transcript.md | head -20
# 話者切り替わりが細かくなっていることを確認
```

### 3. パフォーマンス確認

```bash
# ベンチマークログで処理時間を確認
python meeting_pipeline.py 202602017_mtg.mp4 \
  --enable-diarization \
  --align-unit word \
  --bench-jsonl bench/phase3.jsonl \
  --run-id phase3-word-test

cat bench/phase3.jsonl | jq '.timing.align_sec'
# セグメント単位と比較して処理時間の増加を確認
```



## Codexへの引き継ぎ

Claude Codeが実装を完了したら、Codexに以下を依頼してください：

```
Phase 3の実装が完了しました。

#PHASE3_IMPLEMENTATION_GUIDE.md のテスト戦略セクションを参照してください。
#meeting_pipeline.py の変更内容を確認してください。

以下のテストを作成してください：
- タスク15.4: 単語レベルアラインのユニットテスト
  - 単語タイムスタンプ取得のテスト
  - 単語レベル重複計算のテスト
  - 連続単語統合のテスト
  - フォールバック動作のテスト
  - エンドツーエンドテスト

既存の38テストが全てパスすることも確認してください。
```

## まとめ

Phase 3では、faster-whisperの単語タイムスタンプ機能を活用して、より細かい粒度での話者割り当てを実現します。

**主な変更点**:
- `ASRSegment`に`words`フィールド追加
- `--align-unit`パラメータ追加
- `align_segments_word_level()`関数追加
- 後方互換性を維持（デフォルトは既存の動作）

**期待される効果**:
- 話者切り替わりの検出精度向上
- UNKNOWN割り当ての削減
- より読みやすい議事録

実装完了後、実際の音声ファイルでテストして、精度向上を確認してください。

---

**作成日**: 2026-03-16  
**対象フェーズ**: Phase 3（オプション）  
**前提条件**: Phase 1とPhase 2が完了していること

