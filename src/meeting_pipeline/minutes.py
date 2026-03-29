"""Meeting minutes generation using OpenAI API."""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from .models import (
    ActionItem,
    Decision,
    MeetingJSON,
    MeetingMinutes,
    MinutesConfig,
    Topic,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """あなたは専門的な議事録生成者です。会議の文字起こしを分析し、構造化された情報を抽出するのがあなたの仕事です。

出力形式: 以下の構造を持つJSON
{
  "summary": "会議全体の要約（100-300語）",
  "decisions": [{"text": "決定テキスト", "speaker": "Speaker 1", "timestamp": 123.4}],
  "action_items": [{"task": "タスク説明", "assignee": "Speaker 2", "deadline": "2024-03-15", "timestamp": 456.7}],
  "topics": [{"title": "トピック名", "summary": "トピック要約", "start": 0.0, "end": 300.0}]
}

ガイドライン:
- 簡潔かつ正確に
- 明示的に言及された決定とアクションアイテムのみを抽出
- 1-10の明確なトピックを特定
- 文字起こしの話者ラベルを使用
- トレーサビリティのためにタイムスタンプを保持
"""


def _load_meeting_json(path: str) -> MeetingJSON:
    """Meeting JSONファイルを読み込んで検証する。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Meeting JSONが見つかりません: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Meeting JSON形式が無効です: {e}")

    # 必須フィールドの検証
    required_fields = ["schema_version", "title", "segments", "speakers"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Meeting JSONに必須フィールドがありません: {field}")

    # MeetingJSONオブジェクトに変換せず、dictのまま返す簡易ラッパー
    # 実際の利用ではdictとして扱う
    return data  # type: ignore[return-value]


def _prepare_prompt(meeting: Dict[str, Any], language: str) -> str:
    """OpenAI API用のユーザープロンプトを準備する。"""
    lines = []
    for seg in meeting.get("segments", []):
        start = seg.get("start", 0.0)
        speaker = seg.get("speaker_label", "Unknown")
        text = seg.get("text", "").strip()
        if text:
            minutes, secs = divmod(int(start), 60)
            hours, minutes = divmod(minutes, 60)
            ts = f"{hours:02d}:{minutes:02d}:{secs:02d}"
            lines.append(f"[{ts}] {speaker}: {text}")

    transcript = "\n".join(lines)

    if language and language != "auto":
        lang_instruction = f"\n\n出力言語: {language}"
    else:
        lang_instruction = ""

    return f"以下の会議の文字起こしを分析し、議事録をJSON形式で生成してください。{lang_instruction}\n\n{transcript}"


def _call_openai_api(prompt: str, config: MinutesConfig, api_key: str) -> str:
    """OpenAI APIを呼び出す（リトライロジック付き）。"""
    try:
        from openai import APITimeoutError, AuthenticationError, OpenAI, RateLimitError
    except ImportError as e:
        raise ImportError(f"openaiパッケージがインストールされていません: {e}")

    client = OpenAI(api_key=api_key)

    max_retries = 3
    backoff_seconds = [1, 2, 4]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        except AuthenticationError as e:
            raise AuthenticationError(f"OpenAI認証エラー: {e}") from e

        except (RateLimitError, APITimeoutError) as e:
            if attempt < max_retries - 1:
                wait = backoff_seconds[attempt]
                logger.warning(
                    "OpenAI APIエラー（試行 %d/%d）: %s。%d秒後にリトライ...",
                    attempt + 1,
                    max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                raise


def _parse_api_response(
    response: str, meeting: Dict[str, Any], config: MinutesConfig, generation_time_sec: float
) -> MeetingMinutes:
    """APIレスポンス（JSON）をMeetingMinutes構造にパースする。"""
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"APIレスポンスのJSONパースに失敗しました: {e}")

    decisions = [
        Decision(
            text=d.get("text", ""),
            speaker=d.get("speaker", "Unknown"),
            timestamp=float(d.get("timestamp", 0.0)),
        )
        for d in data.get("decisions", [])
    ]

    action_items = [
        ActionItem(
            task=a.get("task", ""),
            assignee=a.get("assignee", "Unknown"),
            deadline=a.get("deadline"),
            timestamp=float(a.get("timestamp", 0.0)),
        )
        for a in data.get("action_items", [])
    ]

    topics = [
        Topic(
            title=t.get("title", ""),
            summary=t.get("summary", ""),
            start=float(t.get("start", 0.0)),
            end=float(t.get("end", 0.0)),
        )
        for t in data.get("topics", [])
    ]

    participants: List[str] = [s.get("label", s.get("id", "")) for s in meeting.get("speakers", [])]
    duration_sec: float = meeting.get("input", {}).get("duration_sec", 0.0)

    created_at = datetime.now(timezone.utc).astimezone().isoformat()
    meeting_date = created_at[:10]

    return MeetingMinutes(
        schema_version="1.0",
        created_at=created_at,
        meeting_title=meeting.get("title", ""),
        meeting_date=meeting_date,
        duration_sec=duration_sec,
        participants=participants,
        summary=data.get("summary", ""),
        decisions=decisions,
        action_items=action_items,
        topics=topics,
        model_info=config,
        generation_time_sec=generation_time_sec,
    )


def generate_minutes(
    meeting_json_path: str,
    config: MinutesConfig,
    openai_api_key: str,
) -> MeetingMinutes:
    """Meeting JSONを読み込み、OpenAI APIを呼び出して議事録を生成する。"""
    t0 = time.time()

    meeting = _load_meeting_json(meeting_json_path)
    prompt = _prepare_prompt(meeting, config.language)

    logger.info("OpenAI API呼び出し中（モデル: %s）...", config.model)
    response = _call_openai_api(prompt, config, openai_api_key)

    generation_time_sec = round(time.time() - t0, 1)
    minutes = _parse_api_response(response, meeting, config, generation_time_sec)

    logger.info(
        "議事録生成完了: 決定事項=%d件、アクションアイテム=%d件、トピック=%d件",
        len(minutes.decisions),
        len(minutes.action_items),
        len(minutes.topics),
    )
    return minutes
