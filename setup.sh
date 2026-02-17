#!/bin/bash

echo "=== 会議動画文字起こしツール セットアップ ==="
echo ""

# Python3がインストールされているか確認
if ! command -v python3 &> /dev/null; then
    echo "エラー: Python3がインストールされていません"
    exit 1
fi

# ffmpegがインストールされているか確認
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpegがインストールされていません。インストールしますか？ (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        brew install ffmpeg
    else
        echo "ffmpegは必須です。後でインストールしてください: brew install ffmpeg"
        exit 1
    fi
fi

# 仮想環境を作成
echo "仮想環境を作成中..."
python3 -m venv venv

# 仮想環境を有効化
echo "仮想環境を有効化中..."
source venv/bin/activate

# パッケージをインストール
echo "必要なパッケージをインストール中..."
pip install --upgrade pip
pip install -r requirements.txt

# .envファイルを作成
if [ ! -f .env ]; then
    echo ".envファイルを作成中..."
    cp .env.example .env
    echo ""
    echo "⚠️  .envファイルを編集してOpenAI API Keyを設定してください"
fi

echo ""
echo "=== セットアップ完了！ ==="
echo ""
echo "次回以降の使い方："
echo "1. 仮想環境を有効化: source venv/bin/activate"
echo "2. スクリプト実行: python transcribe.py your_video.mp4"
echo "3. 終了時: deactivate"
echo ""
