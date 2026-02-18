#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
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
    """faster-whisperで文字起こし（CPU最適化・速度検証版）"""
    print(f"faster-whisper モデルをロード中: {model_name}")
    
    # ★検証はCPU固定（mps分岐を外す）
    device = "cpu"
    compute_type = "int8"
    
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=8,   # 8 / 10 / 12 など試す
        num_workers=2    # 0 / 1 / 2 / 4 など試す
    )
    
    print(f"最終設定 - デバイス: {device}, 計算タイプ: {compute_type}")
    print("文字起こし中...")
    
    segments, info = model.transcribe(
        audio_path,
        language="ja",
        beam_size=1,        # ★最重要：まず1で測る
        best_of=1,
        vad_filter=False,   # ★まずOFFで測る（ONも後で比較）
        temperature=0.0
    )
    
    # ★速度検証時はprintしない（ログI/Oを減らす）
    result_text_parts = [seg.text for seg in segments]
    
    return {"text": "".join(result_text_parts).strip()}


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
        description="MP4動画から議事録を作成（faster-whisper版 / M4 Pro最適化）"
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

    # 一時ファイル削除
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
