import os
from dotenv import load_dotenv

load_dotenv()

# Whisper設定
# M4 Pro 48GBなら "large" モデルも快適に動作します
WHISPER_MODEL = "medium"  # tiny, base, small, medium, large から選択

# OpenAI API設定（要約用）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# 出力設定
OUTPUT_DIR = "output"
TEMP_DIR = "temp"
