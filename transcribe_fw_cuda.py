#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from typing import Optional
from faster_whisper import WhisperModel
import ffmpeg
from openai import OpenAI
import config
import time


def extract_audio(video_path, audio_path):
    """MP4から音声を抽出"""
    print(f"音声を抽出中: {video_path}")
    try:
        (
            ffmpeg.input(video_path)
            .output(audio_path, acodec="pcm_s16le", ac=1, ar="16k")
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"音声抽出完了: {audio_path}")
    except ffmpeg.Error as e:
        print(f"エラー: 音声抽出に失敗しました")
        raise


def transcribe_audio(audio_path, model_name="medium"):
    """faster-whisperで文字起こし（CUDA/フォールバック対応版）"""
    print(f"faster-whisper モデルをロード中: {model_name}")

    def run_transcribe(model: WhisperModel, log_suffix: Optional[str] = None):
        print(f"文字起こし中...{log_suffix or ''}")
        segments, _ = model.transcribe(
            audio_path,
            language="ja",
            beam_size=1,
            best_of=1,
            vad_filter=False,
            temperature=0.0
        )
        result_text_parts = [seg.text for seg in segments]
        return {"text": "".join(result_text_parts).strip()}

    # CUDA利用可否を判定（NVIDIAデバイスが無い環境での即失敗を回避）
    cuda_available = False
    try:
        import ctranslate2
        cuda_available = ctranslate2.get_cuda_device_count() > 0
    except Exception as e:
        print(f"⚠️ CUDA デバイス判定に失敗したため、CPU を使用します: {e}")

    if cuda_available:
        # GPUでは float16 を優先し、失敗時に int8_float16 にフォールバック
        for compute_type in ("float16", "int8_float16"):
            try:
                model = WhisperModel(
                    model_name,
                    device="cuda",
                    compute_type=compute_type
                )
                print(f"最終設定 - デバイス: cuda, 計算タイプ: {compute_type}")
                return run_transcribe(model)
            except Exception as e:
                print(
                    f"⚠️ CUDA 実行に失敗しました (compute_type={compute_type})。"
                    f" 次の設定を試します: {e}"
                )
    else:
        print("⚠️ CUDA デバイスが検出されなかったため、CPU にフォールバックします。")

    # CPUにフォールバック
    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=8,
        num_workers=2
    )
    print("最終設定 - デバイス: cpu, 計算タイプ: int8")
    return run_transcribe(model, " (CPU)")


def summarize_with_llm(transcript_text):
    """LLMで要約して議事録を作成"""
    if not config.OPENAI_API_KEY:
        print("警告: OPENAI_API_KEYが設定されていません。要約をスキップします。")
        return None

    print("LLMで議事録を作成中...")
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    prompt = f"""以下は会議の文字起こしです。この内容から議事録を作成してください。

要件：
- 会議の主要なトピックを箇条書きで整理
- 決定事項を明確に記載
- アクションアイテムがあれば抽出
- 重要な議論のポイントをまとめる

文字起こし：
{transcript_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは議事録作成の専門家です。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


def main():
    # 実行時間計測開始
    start_time = time.time()
    
    parser = argparse.ArgumentParser(
        description="MP4動画から議事録を作成（faster-whisper版 / CUDA最適化）"
    )
    parser.add_argument("video_file", help="入力するMP4ファイル")
    parser.add_argument(
        "--model",
        default=config.WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisperモデル (デフォルト: medium)",
    )
    parser.add_argument(
        "--no-summary", action="store_true", help="LLMによる要約をスキップ"
    )

    args = parser.parse_args()

    # ディレクトリ作成
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.TEMP_DIR, exist_ok=True)

    video_path = Path(args.video_file)
    if not video_path.exists():
        print(f"エラー: ファイルが見つかりません: {video_path}")
        sys.exit(1)

    base_name = video_path.stem
    audio_path = Path(config.TEMP_DIR) / f"{base_name}.wav"
    transcript_path = Path(config.OUTPUT_DIR) / f"{base_name}_transcript.txt"
    minutes_path = Path(config.OUTPUT_DIR) / f"{base_name}_minutes.txt"

    try:
        # ステップ1: 音声抽出
        extract_audio(str(video_path), str(audio_path))

        # ステップ2: 文字起こし
        result = transcribe_audio(str(audio_path), args.model)

        # 文字起こし結果を保存
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"\n文字起こし完了: {transcript_path}")

        # ステップ3: 要約
        if not args.no_summary:
            minutes = summarize_with_llm(result["text"])
            if minutes:
                with open(minutes_path, "w", encoding="utf-8") as f:
                    f.write(minutes)
                print(f"議事録作成完了: {minutes_path}")
    finally:
        # 失敗時も一時ファイルを削除
        if audio_path.exists():
            audio_path.unlink()
            print(f"\n一時ファイルを削除: {audio_path}")

    # 実行時間計測終了
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 時間を見やすく表示
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    
    print("\n" + "="*50)
    print(f"処理完了！ 総実行時間: {minutes}分 {seconds:.2f}秒")
    print("="*50)


if __name__ == "__main__":
    main()
