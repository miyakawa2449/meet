# Phase 3 Testing Guide for Codex

## 概要

Claude CodeがPhase 3（単語/句単位アライン機能）の実装を完了しました。あなたの役割は、新機能の検証とテストコード作成です。

## 実装完了内容

### ✅ Claude Codeが実装した機能

1. **データクラスの拡張**
   - `ASRSegment`に`words: Optional[List[Dict[str, Any]]]`フィールド追加
   - `PipelineConfig`に`align_unit: str = "segment"`フィールド追加

2. **faster-whisper統合**
   - `_run_faster_whisper()`で`word_timestamps=True`を設定
   - 単語情報を`ASRSegment.words`に保存

3. **単語レベルアライメント関数**
   - `align_segments_word_level()` - メイン関数
   - `_merge_consecutive_words()` - 連続単語統合
   - `_get_speaker_label()` - 話者ラベル変換
   - `_align_single_segment()` - フォールバック用

4. **CLI統合**
   - `--align-unit` パラメータ追加（choices: segment, word）
   - `run_pipeline()`で分岐処理
   - 単語情報がない場合の警告とフォールバック

5. **後方互換性**
   - デフォルトは`--align-unit segment`
   - 既存の38テストが全てパス

## あなたのタスク

### タスク15.4: 単語レベルアラインのユニットテスト作成

以下のテストを`tests/test_meeting_pipeline.py`に追加してください。



### テスト1: 単語タイムスタンプ取得のテスト

```python
def test_word_timestamps_extraction():
    """
    Test that word timestamps are extracted from faster-whisper.
    Validates: Phase 3, Task 15.1
    """
    # モックのfaster-whisperセグメントを作成
    # seg.wordsに単語情報が含まれることを確認
    # ASRSegment.wordsに正しく保存されることを確認
    pass
```

**検証ポイント**:
- `word_timestamps=True`が設定されている
- 単語情報（word, start, end, probability）が取得できる
- `ASRSegment.words`に保存される

### テスト2: 単語レベル重複計算のテスト

```python
def test_word_level_overlap_calculation():
    """
    Test overlap calculation at word level.
    Validates: Phase 3, Task 15.2
    """
    # 単語と話者ターンの重複を計算
    # 最大重複を持つ話者が選択されることを確認
    
    # テストケース例:
    # 単語: 0.0-1.0秒
    # SPEAKER_00: 0.0-0.5秒（重複0.5秒）
    # SPEAKER_01: 0.5-1.5秒（重複0.5秒）
    # → 同じ重複の場合、最初に見つかった話者を選択
    pass
```

**検証ポイント**:
- 単語と各話者ターンの重複が正しく計算される
- 最大重複を持つ話者が選択される
- 重複ゼロの場合はUNKNOWNが割り当てられる

### テスト3: 連続単語統合のテスト

```python
def test_consecutive_word_merging():
    """
    Test merging of consecutive words with same speaker.
    Validates: Phase 3, Task 15.2
    """
    word_alignments = [
        {'word': 'こんにちは', 'start': 0.0, 'end': 1.0, 'speaker_id': 'SPEAKER_00', 'turn_id': 'turn_001', 'overlap': 1.0},
        {'word': 'です', 'start': 1.0, 'end': 1.5, 'speaker_id': 'SPEAKER_00', 'turn_id': 'turn_001', 'overlap': 0.5},
        {'word': 'ありがとう', 'start': 2.0, 'end': 3.0, 'speaker_id': 'SPEAKER_01', 'turn_id': 'turn_002', 'overlap': 1.0},
    ]
    
    merged = _merge_consecutive_words(word_alignments, "asr_000001")
    
    # 期待値: 2つのセグメント
    assert len(merged) == 2
    assert merged[0]['speaker_id'] == 'SPEAKER_00'
    assert merged[0]['text'] == 'こんにちはです'
    assert merged[0]['start'] == 0.0
    assert merged[0]['end'] == 1.5
    assert merged[1]['speaker_id'] == 'SPEAKER_01'
    assert merged[1]['text'] == 'ありがとう'
```

**検証ポイント**:
- 同一話者の連続単語が1つのセグメントに統合される
- 話者が変わるとセグメントが分割される
- テキストが正しく結合される（空白処理含む）
- start/endが正しく設定される



### テスト4: フォールバック動作のテスト

```python
def test_word_level_fallback_to_segment():
    """
    Test fallback to segment-level when word timestamps unavailable.
    Validates: Phase 3, backward compatibility
    """
    # 単語情報がないASRセグメントを作成
    asr_seg = ASRSegment(
        id="asr_000001",
        start=0.0,
        end=5.0,
        text="test text",
        words=None  # 単語情報なし
    )
    
    speaker_turns = [
        SpeakerTurn(id="turn_001", speaker_id="SPEAKER_00", start=0.0, end=10.0)
    ]
    
    # 単語レベルアライメントを呼び出し
    aligned = align_segments_word_level([asr_seg], speaker_turns)
    
    # セグメント単位のアライメントにフォールバックすることを確認
    assert len(aligned) == 1
    assert aligned[0].speaker_id == 'SPEAKER_00'
    assert aligned[0].text == 'test text'
    assert aligned[0].start == 0.0
    assert aligned[0].end == 5.0
```

**検証ポイント**:
- `words=None`の場合、セグメント単位アライメントにフォールバック
- エラーが発生しない
- 既存の動作と同じ結果が得られる

### テスト5: 話者ラベル変換のテスト

```python
def test_get_speaker_label():
    """
    Test speaker ID to label conversion.
    Validates: Phase 3, speaker label generation
    """
    assert _get_speaker_label("SPEAKER_00") == "Speaker 1"
    assert _get_speaker_label("SPEAKER_01") == "Speaker 2"
    assert _get_speaker_label("SPEAKER_09") == "Speaker 10"
    assert _get_speaker_label("UNKNOWN") == "Unknown"
```

**検証ポイント**:
- SPEAKER_00 → Speaker 1（0ベースから1ベースへ変換）
- UNKNOWN → Unknown
- 不正な形式でもエラーにならない



### テスト6: 単語レベルアライメントのエンドツーエンドテスト

```python
@pytest.mark.integration
def test_word_level_alignment_end_to_end(tmp_path, mock_audio_file):
    """
    Test complete pipeline with word-level alignment.
    Validates: Phase 3, end-to-end integration
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # モックを使用してパイプライン実行
    with patch('meeting_pipeline.run_diarization') as mock_diar, \
         patch('meeting_pipeline.run_asr') as mock_asr:
        
        # モックの設定
        mock_diar.return_value = DiarizationResult(
            turns=[
                SpeakerTurn(id="turn_001", speaker_id="SPEAKER_00", start=0.0, end=2.0),
                SpeakerTurn(id="turn_002", speaker_id="SPEAKER_01", start=2.0, end=4.0),
            ],
            speakers=["SPEAKER_00", "SPEAKER_01"],
            model="pyannote/speaker-diarization-3.1",
            engine="pyannote-audio",
            hf_token_used=True
        )
        
        mock_asr.return_value = ASRResult(
            segments=[
                ASRSegment(
                    id="asr_000001",
                    start=0.0,
                    end=4.0,
                    text="こんにちはありがとう",
                    words=[
                        {'word': 'こんにちは', 'start': 0.5, 'end': 1.5, 'probability': 0.9},
                        {'word': 'ありがとう', 'start': 2.5, 'end': 3.5, 'probability': 0.9},
                    ]
                )
            ],
            model="medium",
            engine="faster-whisper",
            device="cpu",
            compute_type="int8",
            language="ja",
            beam_size=1,
            best_of=1,
            vad_filter=False,
            asr_load_sec=1.0
        )
        
        config = PipelineConfig(
            input_file=str(mock_audio_file),
            device="cpu",
            enable_diarization=True,
            align_unit="word",  # 単語レベル指定
            format="json",
            output_dir=str(output_dir)
        )
        
        run_pipeline(config)
    
    # 出力確認
    json_path = output_dir / f"{mock_audio_file.stem}_meeting.json"
    assert json_path.exists()
    
    with open(json_path) as f:
        meeting = json.load(f)
    
    # 検証
    assert meeting['pipeline']['align']['unit'] == 'word'
    assert len(meeting['segments']) == 2  # 1つのASRセグメントが2つに分割
    assert meeting['segments'][0]['speaker_id'] == 'SPEAKER_00'
    assert meeting['segments'][1]['speaker_id'] == 'SPEAKER_01'
```

**検証ポイント**:
- `--align-unit word`で実行できる
- 1つのASRセグメントが話者ごとに分割される
- Meeting JSONの`pipeline.align.unit`が"word"になる
- 各セグメントに正しい話者が割り当てられる



### テスト7: UNKNOWN話者の単語レベル割り当てテスト

```python
def test_word_level_unknown_assignment():
    """
    Test UNKNOWN assignment at word level when no overlap.
    Validates: Phase 3, UNKNOWN preservation
    """
    asr_seg = ASRSegment(
        id="asr_000001",
        start=10.0,
        end=12.0,
        text="テストテキスト",
        words=[
            {'word': 'テスト', 'start': 10.5, 'end': 11.0, 'probability': 0.9},
            {'word': 'テキスト', 'start': 11.0, 'end': 11.5, 'probability': 0.9},
        ]
    )
    
    # 重複しない話者ターン
    speaker_turns = [
        SpeakerTurn(id="turn_001", speaker_id="SPEAKER_00", start=0.0, end=5.0),
        SpeakerTurn(id="turn_002", speaker_id="SPEAKER_01", start=15.0, end=20.0),
    ]
    
    aligned = align_segments_word_level([asr_seg], speaker_turns)
    
    # 全ての単語がUNKNOWNに割り当てられる
    assert len(aligned) == 1
    assert aligned[0].speaker_id == 'UNKNOWN'
    assert aligned[0].speaker_label == 'Unknown'
    assert aligned[0].text == 'テストテキスト'
    assert aligned[0].source.diarization_turn_id is None
    assert aligned[0].source.overlap_sec == 0.0
```

**検証ポイント**:
- 重複ゼロの単語はUNKNOWNに割り当てられる
- UNKNOWN話者が破棄されない
- 複数の単語が統合される

### テスト8: 話者切り替わりの検出テスト

```python
def test_word_level_speaker_transition():
    """
    Test detection of speaker transitions within ASR segment.
    Validates: Phase 3, speaker transition accuracy
    """
    asr_seg = ASRSegment(
        id="asr_000001",
        start=0.0,
        end=6.0,
        text="こんにちはありがとう",
        words=[
            {'word': 'こんにちは', 'start': 0.5, 'end': 1.5, 'probability': 0.9},
            {'word': 'ありがとう', 'start': 4.5, 'end': 5.5, 'probability': 0.9},
        ]
    )
    
    speaker_turns = [
        SpeakerTurn(id="turn_001", speaker_id="SPEAKER_00", start=0.0, end=2.0),
        SpeakerTurn(id="turn_002", speaker_id="SPEAKER_01", start=4.0, end=6.0),
    ]
    
    aligned = align_segments_word_level([asr_seg], speaker_turns)
    
    # 1つのASRセグメントが2つのAlignedSegmentに分割される
    assert len(aligned) == 2
    assert aligned[0].speaker_id == 'SPEAKER_00'
    assert aligned[0].text == 'こんにちは'
    assert aligned[1].speaker_id == 'SPEAKER_01'
    assert aligned[1].text == 'ありがとう'
```

**検証ポイント**:
- 1つのASRセグメント内で話者が切り替わる場合を検出
- 適切に複数のAlignedSegmentに分割される
- 各セグメントに正しい話者が割り当てられる



### テスト9: CLI パラメータのテスト

```python
def test_cli_align_unit_parameter():
    """
    Test --align-unit CLI parameter parsing.
    Validates: Phase 3, CLI integration
    """
    # segment指定
    args = parse_args(['input.mp4', '--align-unit', 'segment'])
    assert args.align_unit == 'segment'
    
    # word指定
    args = parse_args(['input.mp4', '--align-unit', 'word'])
    assert args.align_unit == 'word'
    
    # デフォルト
    args = parse_args(['input.mp4'])
    assert args.align_unit == 'segment'
    
    # 無効な値
    with pytest.raises(SystemExit):
        parse_args(['input.mp4', '--align-unit', 'invalid'])
```

**検証ポイント**:
- `--align-unit`パラメータが正しく解析される
- デフォルトは"segment"
- 無効な値はエラーになる

### テスト10: AlignConfig メタデータのテスト

```python
def test_align_config_unit_metadata():
    """
    Test that align.unit is correctly recorded in Meeting JSON.
    Validates: Phase 3, Task 15.3
    """
    # モックを使用してパイプライン実行
    # Meeting JSONのpipeline.align.unitを確認
    
    # segment指定の場合
    # assert meeting['pipeline']['align']['unit'] == 'segment'
    
    # word指定の場合
    # assert meeting['pipeline']['align']['unit'] == 'word'
    pass
```

**検証ポイント**:
- `pipeline.align.unit`に正しい値が記録される
- JSON Schemaに準拠している

### テスト11: 後方互換性のテスト

```python
def test_phase3_backward_compatibility():
    """
    Test that existing tests still pass with Phase 3 changes.
    Validates: Phase 3, backward compatibility
    """
    # 既存の38テストが全てパスすることを確認
    # このテストは実際には pytest の実行結果で確認
    pass
```

**検証方法**:
```bash
pytest tests/ -v
# 全38テスト + Phase 3の新規テストがパスすることを確認
```



## Property-Based Testing（オプション）

余裕があれば、以下のプロパティテストも追加してください：

### Property 30: Word-Level Alignment Completeness

```python
from hypothesis import given, strategies as st

@given(
    num_words=st.integers(min_value=1, max_value=20),
    num_turns=st.integers(min_value=1, max_value=5)
)
def test_property_word_level_alignment_completeness(num_words, num_turns):
    """
    Property 30: For any ASR segment with word timestamps,
    all words should be assigned to some speaker (including UNKNOWN).
    """
    # ランダムな単語と話者ターンを生成
    # 全ての単語がAlignedSegmentに含まれることを確認
    pass
```

### Property 31: Word-Level Segment Continuity

```python
@given(
    num_words=st.integers(min_value=2, max_value=20)
)
def test_property_word_level_segment_continuity(num_words):
    """
    Property 31: For any sequence of words with same speaker,
    they should be merged into a single AlignedSegment.
    """
    # 同一話者の連続単語が統合されることを確認
    pass
```

## テスト実行

### 新規テストのみ実行

```bash
# Phase 3のテストのみ
pytest tests/test_meeting_pipeline.py::test_word_timestamps_extraction -v
pytest tests/test_meeting_pipeline.py::test_word_level_overlap_calculation -v
pytest tests/test_meeting_pipeline.py::test_consecutive_word_merging -v
pytest tests/test_meeting_pipeline.py::test_word_level_fallback_to_segment -v
pytest tests/test_meeting_pipeline.py::test_get_speaker_label -v
pytest tests/test_meeting_pipeline.py::test_word_level_alignment_end_to_end -v
pytest tests/test_meeting_pipeline.py::test_word_level_unknown_assignment -v
pytest tests/test_meeting_pipeline.py::test_word_level_speaker_transition -v
pytest tests/test_meeting_pipeline.py::test_cli_align_unit_parameter -v
pytest tests/test_meeting_pipeline.py::test_align_config_unit_metadata -v
```

### 全テスト実行

```bash
# 既存38テスト + Phase 3の新規テスト
pytest tests/ -v

# カバレッジ付き
pytest tests/ --cov=meeting_pipeline --cov-report=term --cov-report=html
```

### 期待される結果

- 既存38テスト: 全てパス
- Phase 3新規テスト: 10テスト程度、全てパス
- カバレッジ: 90%以上を維持



## 実装の検証ポイント

### 1. データクラスの確認

```python
# ASRSegment に words フィールドがあるか
from meeting_pipeline import ASRSegment
import inspect

sig = inspect.signature(ASRSegment)
assert 'words' in sig.parameters

# PipelineConfig に align_unit フィールドがあるか
from meeting_pipeline import PipelineConfig
sig = inspect.signature(PipelineConfig)
assert 'align_unit' in sig.parameters
```

### 2. 関数の存在確認

```python
# 新規関数が実装されているか
from meeting_pipeline import (
    align_segments_word_level,
    _merge_consecutive_words,
    _get_speaker_label,
    _align_single_segment
)

# 関数が呼び出し可能か
assert callable(align_segments_word_level)
assert callable(_merge_consecutive_words)
assert callable(_get_speaker_label)
assert callable(_align_single_segment)
```

### 3. 既存機能の動作確認

```bash
# セグメント単位アライメント（既存の動作）
python meeting_pipeline.py tests/fixtures/sample.wav \
  --enable-diarization \
  --align-unit segment \
  --format json \
  --output-dir tests/output

# 出力確認
cat tests/output/sample_meeting.json | jq '.pipeline.align.unit'
# 期待値: "segment"
```

## 発見された問題の報告

テスト作成中に問題を発見した場合は、以下の形式でレポートしてください：

```markdown
## 発見された問題

### 問題1: [問題のタイトル]

**重要度**: 高/中/低
**場所**: `meeting_pipeline.py` の関数名、行番号
**説明**: 問題の詳細
**影響**: どのような影響があるか
**再現手順**:
1. ステップ1
2. ステップ2

**修正案**:
\`\`\`python
# 修正コード
\`\`\`
```



## 成果物

以下を提出してください：

### 1. テストコード

`tests/test_meeting_pipeline.py`に以下を追加：
- テスト1〜11（上記参照）
- 必要に応じてフィクスチャやヘルパー関数

### 2. テスト実行レポート

```bash
# 実行コマンド
pytest tests/ -v --cov=meeting_pipeline --cov-report=term

# 出力を保存
pytest tests/ -v > phase3_test_results.txt
```

### 3. カバレッジレポート

```bash
pytest tests/ --cov=meeting_pipeline --cov-report=html
# htmlcov/index.html を確認
```

### 4. 問題レポート（あれば）

発見された問題のリストと修正案

## Checkpoint - Phase 3完了確認

全てのテストを作成・実行した後、以下を確認してください：

- [ ] Phase 3の新規テスト（10テスト程度）が作成されている
- [ ] 既存の38テストが全てパス
- [ ] 新規テストが全てパス
- [ ] カバレッジが90%以上を維持
- [ ] 実装の問題が文書化されている（あれば）
- [ ] `--align-unit word`で実行できることを確認
- [ ] `--align-unit segment`で既存の動作を維持

## 次のステップ

テスト完了後、以下のいずれかを選択：

### オプション1: 実際の音声ファイルでテスト

```bash
python meeting_pipeline.py 202602017_mtg.mp4 \
  --enable-diarization \
  --align-unit word \
  --format both

# 出力を比較
diff output/202602017_mtg_transcript.md output/202602017_mtg_transcript_segment.md
```

### オプション2: Phase 4（クロスプラットフォーム最適化）へ進む

### オプション3: ドキュメント整備（タスク19）

---

**作成日**: 2026-03-16  
**対象**: Codex（テスト・検証担当）  
**前提条件**: Claude CodeがPhase 3実装を完了、全38テストがパス

