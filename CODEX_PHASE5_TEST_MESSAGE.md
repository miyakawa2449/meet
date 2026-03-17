# Phase 5 テスト実装依頼

## あなたの役割

あなたは **テスト担当（Codex）** です。Phase 5 の検証が完了したので、必要に応じてテストを追加してください。

## 背景

Claude Code が Phase 5（Windows CUDA環境での検証）を完了しました。

### 検証完了内容

#### 1. CUDA 検出結果

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5070 |
| VRAM | 12.82 GB |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| CTranslate2 | 4.7.1 (カスタムビルド) |

#### 2. テスト結果

| デバイス | 結果 | 備考 |
|----------|------|------|
| CUDA | ✅ 成功 | float16, 81セグメント, 2話者 |
| Auto | ✅ 成功 | CUDAが正しく選択された |
| CPU | ⚠️ Diarization成功, ASR失敗 | CTranslate2カスタムビルドにCPU SGEMMバックエンドなし |
| ユニットテスト | ✅ 56/56 合格 | |

#### 3. パフォーマンス比較

| Device | Platform | Diarization | ASR | Total |
|--------|----------|-------------|-----|-------|
| CPU | macOS | 207.9s | ~120s | ~330s |
| MPS | macOS | 17.7s | ~120s (CPU fallback) | ~140s |
| CUDA | WSL2/RTX 5070 | 11.4s | 10.8s | 24.5s |

**高速化率（vs macOS CPU）**:
- Diarization: 18.2x 高速
- ASR: 11.1x 高速
- Total: 13.5x 高速

#### 4. 発見した問題と修正

- **CPU int8 フォールバック不足**: `asr.py` に CPU int8 → float32 フォールバックを追加
  - ただし、カスタムビルドでは float32 でも SGEMM 未対応で失敗
- **CPU ASR 不可**: CTranslate2 カスタムビルドが CUDA 専用のため、CPU での ASR は動作しません

## 実装タスク: 追加テストの検討

### 現状分析

1. **既存テスト**: 56/56 合格（Windows でも動作確認済み）
2. **CUDA 動作**: 正常（float16 で動作）
3. **問題**: Windows 環境の CTranslate2 カスタムビルドが CPU 非対応

### 推奨テスト追加

#### オプション1: CPU フォールバックテストの追加（推奨しない）

理由: Windows 環境の CTranslate2 が CPU 非対応のため、実環境でテストできない

#### オプション2: CUDA 固有のテストを追加（推奨）

Windows CUDA 環境で実際に動作することを確認するテストを追加:

```python
def test_cuda_device_detection_windows(monkeypatch):
    """Test CUDA device is correctly detected on Windows."""
    import torch
    from src.meeting_pipeline.device import resolve_device
    
    # Mock torch to simulate Windows with CUDA
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    
    device_info = resolve_device("auto")
    assert device_info.resolved == "cuda"

def test_cuda_float16_compute_type():
    """Test CUDA uses float16 compute type by default."""
    from src.meeting_pipeline.asr import _determine_compute_type
    
    compute_type = _determine_compute_type("cuda")
    assert compute_type == "float16"
```

#### オプション3: 何もしない（推奨）

理由:
- 既存56テストが Windows でも合格している
- CUDA 動作は E2E テストで確認済み
- Phase 4 で追加したテストが CUDA もカバーしている
- 新規テスト追加の必要性が低い

## 推奨アクション

**オプション3（何もしない）** を推奨します。

理由:
1. 既存テストが Windows CUDA 環境で全て合格
2. E2E テストで CUDA の実動作を確認済み
3. Phase 4 のテストが CUDA ロジックをカバー済み
4. CPU 非対応は環境固有の問題（コード問題ではない）

## ドキュメント更新の提案

代わりに、以下のドキュメント更新を推奨します:

### 1. README.md に Windows 環境の注意事項を追加

```markdown
## Known Issues

### Windows Environment with Custom CTranslate2 Build

If you are using a custom CUDA-only build of CTranslate2 (faster-whisper dependency):
- CPU device may not work for ASR
- Use `--device cuda` or `--device auto` (which will select CUDA)
- Diarization works on CPU, but ASR requires CUDA

This is an environment-specific limitation, not a code issue.
```

### 2. tasks.md に Phase 5 の結果を記録

Task 19.1-19.3 を完了済みにマークし、結果を記載。

## あなたの判断

以下のいずれかを選択してください:

1. **何もしない**（推奨）
   - 既存テストで十分
   - ドキュメント更新のみ

2. **CUDA 固有テストを追加**
   - Windows CUDA 環境での動作を明示的にテスト
   - 2-3件の簡単なテスト追加

3. **CPU フォールバックテストを追加**
   - 将来の標準ビルド対応のため
   - ただし現環境では実行できない

どの方針で進めますか？
