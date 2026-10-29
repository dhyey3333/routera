"""llm_client.py and summary.py: provider chain, request shapes, validation, and LLM safety.

No network: the SDK clients are replaced by fakes that record what would be sent.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest

from agents.sentinel import llm_client
from agents.sentinel.llm_client import (
    NEWS_SCHEMA,
    NEWS_SYSTEM,
    ChatLLM,
    LLMReplyError,
    LLMUnavailable,
    build_llm_client,
    parse_json_object,
)
from agents.sentinel.models import Disruption
from agents.sentinel.normalise import Gazetteer
from agents.sentinel.replay import (
    SCENARIO_NAMES,
    compare_with_expected,
    load_expected,
    load_scenario,
    make_rig,
    replay,
    rig_for,
)
from agents.sentinel.settings import SentinelConfig
from agents.sentinel.summary import Summariser, summary_facts, template_summary
from agents.sentinel.tests.builders import T0, news, port
from agents.sentinel.tests.conftest import ClassifyFn

CLOSURE_JSON = '{"type": "PORT_CLOSURE", "location": "Jebel Ali", "confidence": 0.8}'


def ns(**kwargs: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(**kwargs)


def fixed(reply: str) -> llm_client.Complete:
    return lambda system, user, schema, max_tokens: reply


# --- choosing a provider --------------------------------------------------------------------------


def test_llm_is_off_by_default() -> None:
    selection = build_llm_client(environ={})
    assert (selection.client, selection.label) == (None, "off")


def test_unknown_mode_is_an_error() -> None:
    with pytest.raises(ValueError, match="SENTINEL_LLM"):
        build_llm_client(environ={"SENTINEL_LLM": "gemini"})


def test_chain_skips_unavailable_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(model: str, timeout: float, env: Any) -> llm_client.Complete:
        raise LLMUnavailable("no credentials")

    monkeypatch.setitem(llm_client.BACKENDS, "genai_hub", unavailable)
    monkeypatch.setitem(llm_client.BACKENDS, "anthropic", lambda m, t, e: fixed("{}"))
    selection = build_llm_client(environ={"SENTINEL_LLM": "genai_hub"})
    assert selection.label == "anthropic:claude-opus-5-5"
    assert selection.notes == ["genai_hub skipped: no credentials"]

    custom = build_llm_client(
        environ={"SENTINEL_LLM": "anthropic", "SENTINEL_ANTHROPIC_MODEL": "claude-haiku-5-5"}
    )
    assert custom.label == "anthropic:claude-haiku-5-5"


def test_chain_ends_with_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    selection = build_llm_client("openai", environ={})
    assert selection.client is None
    assert selection.label == "off (no LLM provider available)"
    assert selection.notes[0].startswith("openai skipped:")


def test_genai_hub_needs_aicore_credentials(tmp_path: Path) -> None:
    with pytest.raises(LLMUnavailable, match="AICORE"):
        llm_client.genai_hub_backend("gpt-4o-mini", 5, {"AICORE_HOME": str(tmp_path)})


# --- request shapes -------------------------------------------------------------------------------


class FakeAnthropic:
    instances: ClassVar[list[FakeAnthropic]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init = kwargs
        self.calls: list[dict[str, Any]] = []
        self.reply = ns(
            stop_reason="end_turn",
            content=[ns(type="thinking", thinking=""), ns(type="text", text=CLOSURE_JSON)],
        )
        self.beta = ns(messages=ns(create=self._create))
        FakeAnthropic.instances.append(self)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.reply


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> type[FakeAnthropic]:
    anthropic = pytest.importorskip("anthropic")

    FakeAnthropic.instances = []
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    return FakeAnthropic


def test_anthropic_request_shape(fake_anthropic: type[FakeAnthropic]) -> None:
    llm = ChatLLM(
        "anthropic", "claude-opus-5-5", llm_client.anthropic_backend("claude-opus-5-5", 8, {}), 8
    )
    assert llm.classify_news("Jebel Ali port closed", "") == json.loads(CLOSURE_JSON)
    client = fake_anthropic.instances[0]
    assert client.init == {"timeout": 8, "max_retries": 0}
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == NEWS_SYSTEM
    assert call["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": NEWS_SCHEMA},
    }
    assert (call["betas"], call["fallbacks"]) == (["server-side-fallback-2026-07-01"], "default")
    assert "temperature" not in call and "thinking" not in call

    llm.summarise({"type": "PORT_CLOSURE"})
    assert client.calls[1]["output_config"] == {"effort": "low"}  # free text, no schema


def test_anthropic_refusal_and_truncation_are_errors(fake_anthropic: type[FakeAnthropic]) -> None:
    llm = ChatLLM(
        "anthropic", "claude-opus-5-5", llm_client.anthropic_backend("claude-opus-5-5", 8, {}), 8
    )
    for stop in ("refusal", "max_tokens"):
        fake_anthropic.instances[0].reply = ns(stop_reason=stop, content=[])
        with pytest.raises(LLMReplyError, match=stop):
            llm.classify_news("h", "b")


def test_refusal_fallbacks_only_on_models_that_accept_them(
    fake_anthropic: type[FakeAnthropic],
) -> None:
    llm_client.anthropic_backend("claude-haiku-5-5", 8, {})("s", "u", None, 10)
    call = fake_anthropic.instances[0].calls[0]
    assert "fallbacks" not in call and "betas" not in call


def test_openai_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    openai = pytest.importorskip("openai")

    calls: list[dict[str, Any]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs == {"timeout": 8, "max_retries": 0}
            self.chat = ns(completions=ns(create=self._create))

        def _create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return ns(choices=[ns(finish_reason="stop", message=ns(content=CLOSURE_JSON))])

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    complete = llm_client.openai_backend("gpt-4o-mini", 8, {})
    assert parse_json_object(complete(NEWS_SYSTEM, "u", NEWS_SCHEMA, 100))["type"] == "PORT_CLOSURE"
    call = calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["max_completion_tokens"] == 100
    assert call["response_format"]["json_schema"] == {
        "name": "news_classification",
        "strict": True,
        "schema": NEWS_SCHEMA,
    }
    assert call["messages"][0] == {"role": "system", "content": NEWS_SYSTEM}


def test_genai_hub_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return ns(choices=[ns(finish_reason="stop", message=ns(content=CLOSURE_JSON))])

    proxy = types.ModuleType("gen_ai_hub.proxy.native.openai")
    proxy.chat = ns(completions=ns(create=create))  # type: ignore[attr-defined]
    for name in ("gen_ai_hub", "gen_ai_hub.proxy", "gen_ai_hub.proxy.native"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy.native.openai", proxy)
    env = {"AICORE_CLIENT_ID": "id", "AICORE_CLIENT_SECRET": "secret"}
    complete = llm_client.genai_hub_backend("gpt-4o-mini", 8, env)
    complete(NEWS_SYSTEM, "u", NEWS_SCHEMA, 100)
    assert calls[0]["model_name"] == "gpt-4o-mini"  # the proxy's keyword, not `model`
    assert calls[0]["response_format"] == {"type": "json_object"}


# --- ChatLLM --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        CLOSURE_JSON,
        f"```json\n{CLOSURE_JSON}\n```",
        f"Here you go: {CLOSURE_JSON}",
    ],
)
def test_json_is_found_in_the_reply(reply: str) -> None:
    assert ChatLLM("p", "m", fixed(reply), 1).classify_news("h", "b")["type"] == "PORT_CLOSURE"


@pytest.mark.parametrize("reply", ["no json here", "[1, 2]", "{not json"])
def test_unparseable_reply_raises(reply: str) -> None:
    with pytest.raises(ValueError):
        ChatLLM("p", "m", fixed(reply), 1).classify_news("h", "b")


def test_slow_provider_times_out() -> None:
    def slow(system: str, user: str, schema: Any, max_tokens: int) -> str:
        time.sleep(1)
        return CLOSURE_JSON

    started = time.perf_counter()
    with pytest.raises(TimeoutError, match=r"0\.05 s"):
        ChatLLM("p", "m", slow, 0.05).classify_news("h", "b")
    assert time.perf_counter() - started < 0.5


def test_empty_reply_is_an_error() -> None:
    with pytest.raises(LLMReplyError, match="empty"):
        ChatLLM("p", "m", fixed("  "), 1).summarise({})


def test_classifier_uses_the_llm_and_falls_back_on_timeout(
    config: SentinelConfig, gazetteer: Gazetteer, classify: ClassifyFn
) -> None:
    good = ChatLLM("p", "m", fixed(CLOSURE_JSON), 1)
    signal = classify("news", news("Something at the port"), llm=good)
    assert (signal.classified_by, signal.location) == ("llm", "AEJEA")

    def slow(system: str, user: str, schema: Any, max_tokens: int) -> str:
        time.sleep(1)
        return CLOSURE_JSON

    signal = classify("news", news("Jebel Ali port closed"), llm=ChatLLM("p", "m", slow, 0.05))
    assert signal.classified_by == "keywords"
    assert signal.reason.startswith("LLM classification rejected (TimeoutError")


# --- summaries ------------------------------------------------------------------------------------


@pytest.fixture
def closure() -> Disruption:
    rig = make_rig(None, T0)
    rig.sentinel.ingest(
        {
            "event_id": "S-1",
            "source": "port_authority",
            "received_at": T0.isoformat(),
            "payload": port("AEJEA", "closed"),
        }
    )
    return rig.sentinel.tracker.get("DSR-0001").disruption


class ScriptedSummaries:
    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.summaries = 0

    def classify_news(self, headline: str, body: str) -> dict[str, Any]:
        raise TimeoutError("not used")

    def summarise(self, disruption: dict[str, Any]) -> str:
        self.summaries += 1
        reply = self.replies.pop(0) if self.replies else "Jebel Ali is closed."
        if isinstance(reply, BaseException):
            raise reply
        return reply


GOOD = (
    "Jebel Ali (AEJEA) is closed indefinitely: 12 grape containers face a ~14-day delay, "
    "severity 5/5."
)


def test_template_without_llm(closure: Disruption) -> None:
    result = Summariser(None).summarise(closure)
    assert (result.text, result.by) == (template_summary(closure), "template")


def test_valid_llm_summary_is_used(closure: Disruption) -> None:
    result = Summariser(ScriptedSummaries(GOOD)).summarise(closure)
    assert (result.text, result.by) == (GOOD, "llm")


def test_llm_sees_only_the_fact_subset(closure: Disruption) -> None:
    facts = summary_facts(closure)
    assert facts["affected_containers"] == 12
    assert "rationale" not in facts and "sources" not in facts


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("word " * 30, id="too-many-words"),
        pytest.param("Jebel Ali closed.\nSecond line.", id="two-lines"),
        pytest.param("Jebel Ali closed; 99 containers stuck for 14 days.", id="invented-number"),
        pytest.param("", id="empty"),
        pytest.param(42, id="not-text"),
        pytest.param(TimeoutError("slow"), id="timeout"),
        pytest.param(RuntimeError("HTTP 500"), id="server-error"),
    ],
)
def test_bad_llm_summary_falls_back_to_template(closure: Disruption, reply: Any) -> None:
    result = Summariser(ScriptedSummaries(reply)).summarise(closure)
    assert (result.text, result.by) == (template_summary(closure), "template")
    assert result.note.startswith("LLM summary rejected")


def test_sentinel_publishes_llm_summaries_but_never_for_candidates() -> None:
    llm = ScriptedSummaries(GOOD)
    rig = make_rig(None, T0, llm=llm)
    feed = {
        "source": "news",
        "received_at": T0.isoformat(),
        "payload": news("Jebel Ali port closed"),
    }
    rig.sentinel.ingest({"event_id": "N-1", **feed})
    assert llm.summaries == 0  # an unconfirmed candidate costs no LLM call
    rig.sentinel.ingest(
        {
            "event_id": "P-1",
            "source": "port_authority",
            "received_at": T0.isoformat(),
            "payload": port("AEJEA", "closed"),
        }
    )
    rig.sentinel.ingest(
        {
            "event_id": "N-2",
            **feed,
            "payload": news("Jebel Ali port closed", outlet="Other Wire (SIM)"),
        }
    )
    assert llm.summaries == 1  # cached: the second news report changed nothing material
    [(_topic, message)] = rig.bus.messages
    assert message["disruption"]["summary"] == GOOD
    assert rig.sentinel.describe("DSR-0001")["summary_by"] == "llm"
    assert rig.sentinel.active()[0].summary == GOOD


class BrokenLLM:
    """Every call fails: classification times out, summaries come back malformed."""

    def classify_news(self, headline: str, body: str) -> dict[str, Any]:
        raise TimeoutError("LLM call timed out")

    def summarise(self, disruption: dict[str, Any]) -> str:
        return "Not one line\nand mentions 999 containers"


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_a_broken_llm_changes_nothing(name: str) -> None:
    scenario = load_scenario(name)
    rig = rig_for(scenario, llm=BrokenLLM())
    result = replay(scenario, rig)
    assert compare_with_expected(result, load_expected(name)) == []
    for _topic, message in rig.bus.messages:
        disruption = Disruption.model_validate(message["disruption"])
        assert disruption.summary == template_summary(disruption)
