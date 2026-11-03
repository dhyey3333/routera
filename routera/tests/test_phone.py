"""Phone approvals: the bot talks to a stand-in for Telegram, in Telegram's documented formats."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest

from routera.app import RouteraService
from routera.phone import TelegramBot, TelegramError

MINE, STRANGER = 111, 999


class FakeTelegram:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.next_id = 500

    def __call__(self, method: str, payload: dict[str, Any]) -> Any:
        self.calls.append((method, payload))
        if method == "sendMessage":
            self.next_id += 1
            return {"message_id": self.next_id}
        return True

    def sent(self, method: str = "sendMessage") -> list[dict[str, Any]]:
        return [p for m, p in self.calls if m == method]

    def alerts(self) -> list[str]:
        return [p["text"] for p in self.sent("answerCallbackQuery") if p.get("show_alert")]


@pytest.fixture
def rig() -> Iterator[tuple[RouteraService, TelegramBot, FakeTelegram]]:
    service = RouteraService(feeds=[])
    service.start_replay("gulf_closure", 0)
    service.wait_for_replay(30)
    telegram = FakeTelegram()
    bot = TelegramBot(service, "SECRET-TOKEN", chat_id=MINE, call=telegram)
    service.bot = bot
    yield service, bot, telegram
    service.stop_replay()


def button_plan(telegram: FakeTelegram) -> tuple[str, str]:
    """The first plan sent to the phone: its Approve button's data, and its plan id."""
    message = next(p for p in telegram.sent() if "reply_markup" in p)
    data = message["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    return data, data.split("|")[2]


def tap(data: str, who: int = MINE, chat: int | None = None) -> dict[str, Any]:
    return {"callback_query": {"id": "q", "data": data, "from": {"id": who},
                               "message": {"chat": {"id": chat or who}}}}


def say(text: str, who: int = MINE, kind: str = "private") -> dict[str, Any]:
    return {"message": {"text": text, "chat": {"id": who, "type": kind, "first_name": "Asha", "last_name": "K."}}}


def test_a_new_plan_buzzes_the_phone_once_and_one_tap_approves_it_on_the_record(rig: Any) -> None:
    service, bot, telegram = rig
    bot.sync()
    bot.sync()
    to_decide = [p for p in service.orchestrator.plans.values() if p["needs_approval"]]
    assert len([p for p in telegram.sent() if "reply_markup" in p]) == len(to_decide)  # no repeats
    data, plan_id = button_plan(telegram)
    assert "Saves" in telegram.sent()[0]["text"] and "sandbox" in telegram.sent()[0]["text"]
    bot.handle_update(tap(data))
    plan = service.orchestrator.plans[plan_id]
    assert plan["status"] == "approved" and plan["approved_by"] == "Phone approver via Telegram"
    assert [b for b in service.orchestrator.bookings if b["plan_id"] == plan_id]
    entries = [e for e in service.chain.entries if e["action"] == "plan.approved"]
    assert entries[-1]["actor"] == "Phone approver via Telegram" and service.chain.verify()["ok"]
    assert "Approved by" in telegram.sent("editMessageText")[-1]["text"]
    assert "reply_markup" not in telegram.sent("editMessageText")[-1]  # no buttons left to tap


def test_a_stranger_cannot_decide_and_a_group_chat_cannot_pair(rig: Any) -> None:
    service, bot, telegram = rig
    bot.sync()
    data, plan_id = button_plan(telegram)
    bot.handle_update(tap(data, who=STRANGER))
    bot.handle_update(tap(data, who=STRANGER, chat=MINE))  # someone else tapping in my chat
    assert service.orchestrator.plans[plan_id]["status"] == "pending"
    assert len(telegram.alerts()) == 2
    bot.chat_id = None
    bot.handle_update(say(f"/pair {bot.code}", who=STRANGER, kind="group"))
    assert bot.chat_id is None


def test_pairing_needs_the_code_and_wrong_guesses_change_it(rig: Any) -> None:
    service, _, telegram = rig
    bot = TelegramBot(service, "SECRET-TOKEN", call=telegram)
    bot.sync()
    assert not telegram.sent() and bot.status()["state"] == "pairing"  # nothing leaves before pairing
    first = bot.code
    for _ in range(5):
        bot.handle_update(say("/pair 000000"))
    assert bot.chat_id is None and bot.code != first
    bot.handle_update(say(f"/pair {first}"))
    assert bot.chat_id is None  # the old code is dead
    bot.handle_update(say(f"/pair {bot.code}"))
    assert bot.chat_id == MINE and bot.status() == {"state": "on", "bot": "", "code": None, "error": "",
                                                    "approver": "Asha K. via Telegram"}  # named as Telegram knows them
    assert any("reply_markup" in p for p in telegram.sent())  # waiting plans arrive straight away


def test_a_tap_only_approves_the_version_it_was_shown(rig: Any) -> None:
    service, bot, telegram = rig
    bot.sync()
    data, plan_id = button_plan(telegram)
    plan = service.orchestrator.plans[plan_id]
    plan["version"] += 1  # re-planned while the phone was in a pocket
    bot.handle_update(tap(data))
    assert plan["status"] == "pending" and "changed" in telegram.alerts()[-1]
    edited = telegram.sent("editMessageText")[-1]  # the phone now shows the new plan, with new buttons
    fresh = edited["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    assert fresh != data
    bot.handle_update(tap(fresh))
    assert plan["status"] == "approved"


def test_a_tap_from_an_earlier_run_does_nothing(rig: Any) -> None:
    service, bot, telegram = rig
    bot.sync()
    data, plan_id = button_plan(telegram)
    service.start_replay("gulf_closure", 0)  # plan numbers start again in the new run
    service.wait_for_replay(30)
    bot.sync()
    bot.handle_update(tap(data))
    assert "earlier run" in telegram.alerts()[-1]
    assert service.orchestrator.plans[plan_id]["status"] == "pending"


def test_approving_on_the_screen_first_clears_the_phone_and_blocks_a_second_approval(rig: Any) -> None:
    service, bot, telegram = rig
    bot.sync()
    data, plan_id = button_plan(telegram)
    service.run(service.orchestrator.approve, plan_id, "Asha K.")
    bot.sync()
    assert "Approved by Asha K." in telegram.sent("editMessageText")[-1]["text"]
    bot.handle_update(tap(data))
    assert "Already approved" in telegram.alerts()[-1]
    assert [e["actor"] for e in service.chain.entries if e["action"] == "plan.approved"] == ["Asha K."]


def test_reject_and_why_work_from_the_phone(rig: Any) -> None:
    service, bot, telegram = rig
    bot.sync()
    data, plan_id = button_plan(telegram)
    bot.handle_update(tap("w" + data[1:]))
    assert any("💬" in p["text"] for p in telegram.sent())
    assert service.orchestrator.plans[plan_id]["status"] == "pending"
    bot.handle_update(tap("r" + data[1:]))
    plan = service.orchestrator.plans[plan_id]
    assert plan["status"] == "rejected" and plan["rejected_by"] == "Phone approver via Telegram"
    assert not [b for b in service.orchestrator.bookings if b["plan_id"] == plan_id]


def test_an_escalation_is_sent_to_the_phone(rig: Any) -> None:
    service, bot, telegram = rig
    bot.sync()
    _, plan_id = button_plan(telegram)
    bot.sent[plan_id]["level"] = 0
    plan = service.orchestrator.plans[plan_id]
    plan["created_at"] = service.rig.clock.now().isoformat()
    plan["escalation"] = None
    bot.sync()
    assert not any("escalated" in p["text"] for p in telegram.sent())
    service.rig.clock.set(service.rig.clock.now() + timedelta(hours=3))
    bot.sync()
    assert any("escalated to Export manager" in p["text"] for p in telegram.sent())
    assert plan["status"] == "pending"


def test_the_token_never_shows_and_no_token_means_no_bot(rig: Any) -> None:
    service, _, _ = rig
    assert TelegramBot.from_env(service, env={}) is None
    made = TelegramBot.from_env(service, env={"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "42"})
    assert made is not None and made.chat_id == 42

    def down(method: str, payload: dict[str, Any]) -> Any:
        raise TelegramError("failed https://api.telegram.org/botSECRET-TOKEN/getMe")

    bot = TelegramBot(service, "SECRET-TOKEN", call=down)
    bot._try("getMe", {})
    assert "SECRET-TOKEN" not in bot.error and "SECRET-TOKEN" not in str(bot.status())


def test_the_snapshot_reports_the_phone(rig: Any) -> None:
    service, _, _ = rig
    assert service.snapshot()["phone"]["state"] == "on"
    service.bot = None
    assert service.snapshot()["phone"] == {"state": "off"}
