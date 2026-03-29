from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.meeting_pipeline.minutes import (
    _call_openai_api,
    _load_meeting_json,
    _parse_api_response,
    _prepare_prompt,
    generate_minutes,
)
from src.meeting_pipeline.models import MinutesConfig


@pytest.fixture
def sample_minutes_config() -> MinutesConfig:
    return MinutesConfig(
        enabled=True,
        model="gpt-4",
        language="ja",
        temperature=0.3,
        max_tokens=4000,
    )


@pytest.fixture
def sample_meeting_dict() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "title": "定例会議",
        "input": {"duration_sec": 120.0},
        "segments": [
            {"start": 0.0, "speaker_label": "Speaker 1", "text": "開始します"},
            {"start": 65.0, "speaker_label": "Speaker 2", "text": "次の議題です"},
            {"start": 3661.0, "speaker_label": "Speaker 1", "text": "長時間会議の例です"},
        ],
        "speakers": [
            {"id": "SPEAKER_00", "label": "Speaker 1"},
            {"id": "SPEAKER_01", "label": "Speaker 2"},
        ],
    }


@pytest.fixture
def sample_api_response() -> str:
    return json.dumps(
        {
            "summary": "テスト要約",
            "decisions": [{"text": "決定1", "speaker": "Speaker 1", "timestamp": 10.0}],
            "action_items": [
                {
                    "task": "タスク1",
                    "assignee": "Speaker 2",
                    "deadline": "2026-04-01",
                    "timestamp": 20.0,
                }
            ],
            "topics": [
                {"title": "トピック1", "summary": "要約1", "start": 0.0, "end": 30.0}
            ],
        },
        ensure_ascii=False,
    )


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    create_callable,
) -> ModuleType:
    module = ModuleType("openai")

    class AuthenticationError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class OpenAI:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=create_callable),
            )

    module.AuthenticationError = AuthenticationError
    module.RateLimitError = RateLimitError
    module.APITimeoutError = APITimeoutError
    module.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return module


def test_load_meeting_json_success(tmp_path: Path, sample_meeting_dict: dict[str, object]) -> None:
    path = tmp_path / "meeting.json"
    path.write_text(json.dumps(sample_meeting_dict, ensure_ascii=False), encoding="utf-8")

    loaded = _load_meeting_json(str(path))

    assert loaded["title"] == "定例会議"
    assert len(loaded["segments"]) == 3


def test_load_meeting_json_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="Meeting JSONが見つかりません"):
        _load_meeting_json("/tmp/not-found-meeting.json")


def test_load_meeting_json_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="Meeting JSON形式が無効です"):
        _load_meeting_json(str(path))


def test_prepare_prompt_converts_segments(sample_meeting_dict: dict[str, object]) -> None:
    prompt = _prepare_prompt(sample_meeting_dict, "ja")

    assert "Speaker 1: 開始します" in prompt
    assert "Speaker 2: 次の議題です" in prompt


def test_prepare_prompt_formats_timestamp_hhmmss(sample_meeting_dict: dict[str, object]) -> None:
    prompt = _prepare_prompt(sample_meeting_dict, "ja")

    assert "[00:00:00] Speaker 1: 開始します" in prompt
    assert "[00:01:05] Speaker 2: 次の議題です" in prompt
    assert "[01:01:01] Speaker 1: 長時間会議の例です" in prompt


def test_prepare_prompt_includes_language_instruction(sample_meeting_dict: dict[str, object]) -> None:
    prompt = _prepare_prompt(sample_meeting_dict, "en")
    auto_prompt = _prepare_prompt(sample_meeting_dict, "auto")

    assert "出力言語: en" in prompt
    assert "出力言語:" not in auto_prompt


def test_call_openai_api_success(
    monkeypatch: pytest.MonkeyPatch,
    sample_minutes_config: MinutesConfig,
    sample_api_response: str,
) -> None:
    captured: dict[str, object] = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=sample_api_response))]
        )

    _install_fake_openai(monkeypatch, create)

    response = _call_openai_api("prompt body", sample_minutes_config, "sk-test")

    assert response == sample_api_response
    assert captured["model"] == "gpt-4"
    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 4000
    assert captured["response_format"] == {"type": "json_object"}


def test_call_openai_api_authentication_error_no_retry(
    monkeypatch: pytest.MonkeyPatch,
    sample_minutes_config: MinutesConfig,
) -> None:
    calls = {"n": 0}

    def create(**kwargs):
        calls["n"] += 1
        raise openai_mod.AuthenticationError("bad key")

    openai_mod = _install_fake_openai(monkeypatch, create)
    sleep_calls: list[int] = []
    monkeypatch.setattr("src.meeting_pipeline.minutes.time.sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(openai_mod.AuthenticationError, match="OpenAI認証エラー"):
        _call_openai_api("prompt", sample_minutes_config, "sk-test")

    assert calls["n"] == 1
    assert sleep_calls == []


def test_call_openai_api_rate_limit_retries(
    monkeypatch: pytest.MonkeyPatch,
    sample_minutes_config: MinutesConfig,
    sample_api_response: str,
) -> None:
    calls = {"n": 0}
    sleep_calls: list[int] = []

    def create(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise openai_mod.RateLimitError("slow down")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=sample_api_response))]
        )

    openai_mod = _install_fake_openai(monkeypatch, create)
    monkeypatch.setattr("src.meeting_pipeline.minutes.time.sleep", lambda seconds: sleep_calls.append(seconds))

    response = _call_openai_api("prompt", sample_minutes_config, "sk-test")

    assert response == sample_api_response
    assert calls["n"] == 3
    assert sleep_calls == [1, 2]


def test_call_openai_api_timeout_retries(
    monkeypatch: pytest.MonkeyPatch,
    sample_minutes_config: MinutesConfig,
    sample_api_response: str,
) -> None:
    calls = {"n": 0}
    sleep_calls: list[int] = []

    def create(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise openai_mod.APITimeoutError("timeout")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=sample_api_response))]
        )

    openai_mod = _install_fake_openai(monkeypatch, create)
    monkeypatch.setattr("src.meeting_pipeline.minutes.time.sleep", lambda seconds: sleep_calls.append(seconds))

    response = _call_openai_api("prompt", sample_minutes_config, "sk-test")

    assert response == sample_api_response
    assert calls["n"] == 2
    assert sleep_calls == [1]


def test_call_openai_api_max_retries_exceeded(
    monkeypatch: pytest.MonkeyPatch,
    sample_minutes_config: MinutesConfig,
) -> None:
    calls = {"n": 0}
    sleep_calls: list[int] = []

    def create(**kwargs):
        calls["n"] += 1
        raise openai_mod.RateLimitError("still limited")

    openai_mod = _install_fake_openai(monkeypatch, create)
    monkeypatch.setattr("src.meeting_pipeline.minutes.time.sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(openai_mod.RateLimitError):
        _call_openai_api("prompt", sample_minutes_config, "sk-test")

    assert calls["n"] == 3
    assert sleep_calls == [1, 2]


def test_parse_api_response_success(
    monkeypatch: pytest.MonkeyPatch,
    sample_meeting_dict: dict[str, object],
    sample_minutes_config: MinutesConfig,
    sample_api_response: str,
) -> None:
    fixed_now = SimpleNamespace(
        astimezone=lambda: SimpleNamespace(isoformat=lambda: "2026-03-29T12:00:00+09:00")
    )
    monkeypatch.setattr("src.meeting_pipeline.minutes.datetime", SimpleNamespace(now=lambda tz: fixed_now))

    minutes = _parse_api_response(sample_api_response, sample_meeting_dict, sample_minutes_config, 1.7)

    assert minutes.summary == "テスト要約"
    assert minutes.meeting_title == "定例会議"
    assert minutes.meeting_date == "2026-03-29"
    assert minutes.duration_sec == 120.0
    assert minutes.participants == ["Speaker 1", "Speaker 2"]
    assert minutes.generation_time_sec == 1.7


def test_parse_api_response_extracts_decisions(
    sample_meeting_dict: dict[str, object],
    sample_minutes_config: MinutesConfig,
    sample_api_response: str,
) -> None:
    minutes = _parse_api_response(sample_api_response, sample_meeting_dict, sample_minutes_config, 0.5)

    assert len(minutes.decisions) == 1
    assert minutes.decisions[0].text == "決定1"
    assert minutes.decisions[0].speaker == "Speaker 1"
    assert minutes.decisions[0].timestamp == 10.0


def test_parse_api_response_extracts_action_items(
    sample_meeting_dict: dict[str, object],
    sample_minutes_config: MinutesConfig,
    sample_api_response: str,
) -> None:
    minutes = _parse_api_response(sample_api_response, sample_meeting_dict, sample_minutes_config, 0.5)

    assert len(minutes.action_items) == 1
    assert minutes.action_items[0].task == "タスク1"
    assert minutes.action_items[0].assignee == "Speaker 2"
    assert minutes.action_items[0].deadline == "2026-04-01"


def test_parse_api_response_extracts_topics(
    sample_meeting_dict: dict[str, object],
    sample_minutes_config: MinutesConfig,
    sample_api_response: str,
) -> None:
    minutes = _parse_api_response(sample_api_response, sample_meeting_dict, sample_minutes_config, 0.5)

    assert len(minutes.topics) == 1
    assert minutes.topics[0].title == "トピック1"
    assert minutes.topics[0].summary == "要約1"
    assert minutes.topics[0].start == 0.0
    assert minutes.topics[0].end == 30.0


def test_generate_minutes_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_meeting_dict: dict[str, object],
    sample_minutes_config: MinutesConfig,
    sample_api_response: str,
) -> None:
    path = tmp_path / "meeting.json"
    path.write_text(json.dumps(sample_meeting_dict, ensure_ascii=False), encoding="utf-8")
    times = iter([100.0, 101.64])
    monkeypatch.setattr("src.meeting_pipeline.minutes.time.time", lambda: next(times))
    monkeypatch.setattr("src.meeting_pipeline.minutes._call_openai_api", lambda prompt, config, api_key: sample_api_response)

    minutes = generate_minutes(str(path), sample_minutes_config, "sk-test")

    assert minutes.summary == "テスト要約"
    assert minutes.generation_time_sec == 1.6
    assert minutes.model_info.model == "gpt-4"


def test_generate_minutes_error_handling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_meeting_dict: dict[str, object],
    sample_minutes_config: MinutesConfig,
) -> None:
    path = tmp_path / "meeting.json"
    path.write_text(json.dumps(sample_meeting_dict, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(
        "src.meeting_pipeline.minutes._call_openai_api",
        lambda prompt, config, api_key: (_ for _ in ()).throw(RuntimeError("api failed")),
    )

    with pytest.raises(RuntimeError, match="api failed"):
        generate_minutes(str(path), sample_minutes_config, "sk-test")


def test_generate_minutes_measures_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_meeting_dict: dict[str, object],
    sample_minutes_config: MinutesConfig,
) -> None:
    path = tmp_path / "meeting.json"
    path.write_text(json.dumps(sample_meeting_dict, ensure_ascii=False), encoding="utf-8")
    times = iter([10.0, 12.26])
    captured: dict[str, float] = {}

    monkeypatch.setattr("src.meeting_pipeline.minutes.time.time", lambda: next(times))
    monkeypatch.setattr("src.meeting_pipeline.minutes._call_openai_api", lambda prompt, config, api_key: "{}")

    def fake_parse(response: str, meeting: dict[str, object], config: MinutesConfig, generation_time_sec: float):
        captured["generation_time_sec"] = generation_time_sec
        return SimpleNamespace(decisions=[], action_items=[], topics=[])

    monkeypatch.setattr("src.meeting_pipeline.minutes._parse_api_response", fake_parse)

    result = generate_minutes(str(path), sample_minutes_config, "sk-test")

    assert captured["generation_time_sec"] == 2.3
    assert result.topics == []
