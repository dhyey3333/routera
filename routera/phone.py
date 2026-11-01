"""Real phone approvals through a Telegram bot: a plan buzzes the phone, one tap approves it.

The bot only makes outbound HTTPS calls (long polling), so the phone never has to reach the app and SAP BAS
needs no public address. The token stays in the TELEGRAM_BOT_TOKEN environment variable.

The phone is a second Approve button on the same plan, with these safeguards:
- Only the paired private chat counts. Pairing needs a one-time code shown in the control room, or a
  chat id in TELEGRAM_CHAT_ID. Everyone else is refused, and five wrong codes change the code.
- A tap approves only the plan version it was shown. If the plan has changed, the phone shows the new one.
- A tap from an earlier run does nothing, even though plan numbers repeat between runs.
- Compliance blocks stay blocked and route changes stay on the big screen: the phone takes the plan as it is.
Every decision goes through Orchestrator.approve or reject, so it lands on the audit chain under its name.
"""

from __future__ import annotations

import html
import json
import logging
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from routera.planner import fmt_inr

log = logging.getLogger("routera.phone")
API = "https://api.telegram.org/bot"
LONG_POLL_S = 25
SYNC_EVERY_S = 15.0
MAX_WRONG_CODES = 5
REJECT_REASON = "rejected on the phone"
Call = Callable[[str, dict[str, Any]], Any]


class TelegramError(Exception):
    pass


class TelegramBot:
    def __init__(self, service: Any, token: str, chat_id: str | int | None = None,
                 approver: str = "Sunil P. via Telegram", call: Call | None = None) -> None:
        self.service, self.token, self.approver = service, token, approver
        self.call = call or self._http_call
        self.chat_id = int(chat_id) if chat_id else None
        self.code = self._new_code()
        self.username = ""
        self.error = ""
        self.offset = 0
        self.wrong_codes = 0
        self.run_token = ""
        self.rig: Any = None
        self.sent: dict[str, dict[str, Any]] = {}  # plan id -> message id, version, status, escalation level
        self._kick = threading.Event()
        self._sync_lock = threading.RLock()
        self._announced: dict[str, Any] = {}

    @classmethod
    def from_env(cls, service: Any, env: dict[str, str] | None = None) -> TelegramBot | None:
        source = os.environ if env is None else env
        token = source.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            return None
        return cls(service, token, source.get("TELEGRAM_CHAT_ID", "").strip() or None,
                   source.get("TELEGRAM_APPROVER", "").strip() or "Sunil P. via Telegram")

    @staticmethod
    def _new_code() -> str:
        return str(secrets.randbelow(900000) + 100000)

    # --- talking to Telegram ------------------------------------------------------------------------

    def _http_call(self, method: str, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(f"{API}{self.token}/{method}", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=LONG_POLL_S + 10) as response:
                return json.load(response)["result"]
        except urllib.error.HTTPError as exc:
            detail = json.loads(exc.read() or b"{}").get("description", f"HTTP {exc.code}")
            raise TelegramError(detail) from None
        except OSError as exc:
            raise TelegramError(self._clean(exc)) from None

    def _clean(self, exc: Exception) -> str:
        return str(exc).replace(self.token, "***")[:200]

    def _try(self, method: str, payload: dict[str, Any]) -> Any:
        try:
            return self.call(method, payload)
        except TelegramError as exc:
            if "message is not modified" not in str(exc):
                self.error = self._clean(exc)
                log.warning("telegram %s failed: %s", method, self.error)
            return None

    def _say(self, chat_id: int, text: str) -> Any:
        return self._try("sendMessage", {"chat_id": chat_id, "text": text, "parse_mode": "HTML"})

    # --- what the control room shows ----------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        if self.chat_id is not None:
            state = "on"
        elif self.error and not self.username:
            state = "error"
        else:
            state = "pairing"
        return {"state": state, "bot": self.username, "code": self.code if state == "pairing" else None,
                "error": self.error if state == "error" else ""}

    def _announce(self) -> None:
        now = self.status()
        if now != self._announced:
            self._announced = now
            self.service._broadcast("phone", now)

    # --- plans to the phone ---------------------------------------------------------------------------

    def kick(self) -> None:
        self._kick.set()

    def _buttons(self, plan: dict[str, Any]) -> dict[str, Any]:
        tag = f"{self.run_token}|{plan['plan_id']}|{plan['version']}"
        return {"inline_keyboard": [[{"text": "✅ Approve", "callback_data": f"a|{tag}"},
                                     {"text": "❌ Reject", "callback_data": f"r|{tag}"}],
                                    [{"text": "💬 Why?", "callback_data": f"w|{tag}"}]]}

    @staticmethod
    def _plan_text(plan: dict[str, Any]) -> str:
        lines = [f"⚠️ <b>{html.escape(plan['title'])}</b>", html.escape(plan["headline"])]
        wait = plan.get("escalation")
        if wait:
            lines.append(f"⏳ Waiting on {html.escape(wait['role'])}. Silence is never consent.")
        lines += [f"• {html.escape(g)}" for g in plan.get("guardrails", [])[:3]]
        lines.append(f"<i>{html.escape(plan['plan_id'])} v{plan['version']} · simulated fleet and bookings</i>")
        return "\n".join(lines)[:3900]

    @staticmethod
    def _final_text(plan: dict[str, Any]) -> str:
        title = f"<b>{html.escape(plan['title'])}</b>"
        if plan["status"] == "approved":
            moved = len(plan.get("packs") or {})
            return (f"✅ {title}\nApproved by {html.escape(str(plan.get('approved_by')))}. {moved} moves booked in "
                    f"SAP TM (simulated). Saves {fmt_inr(plan['saved_inr'])}. Logged on the audit chain.")
        if plan["status"] == "rejected":
            return (f"❌ {title}\nRejected by {html.escape(str(plan.get('rejected_by')))}. "
                    "The containers stay on their bookings. Logged on the audit chain.")
        return f"ℹ️ {title}\nClosed: {html.escape(str(plan.get('closed_reason') or 'no longer needed'))}."

    def sync(self) -> None:
        """Bring the phone up to date: new plans, changed plans, escalations and finished plans."""
        with self._sync_lock:
            self._announce()
            with self.service._lock:
                rig = self.service.rig
                self.service.orchestrator.snapshot()  # brings escalations up to date
                plans = [dict(p) for p in self.service.orchestrator.plans.values()]
            if rig is not self.rig:  # a new run: plan numbers start again, so forget the old messages
                self.rig, self.run_token, self.sent = rig, secrets.token_hex(3), {}
            if self.chat_id is None:
                return
            for plan in plans:
                self._sync_plan(plan)

    def _sync_plan(self, plan: dict[str, Any]) -> None:
        pid, seen = plan["plan_id"], self.sent.get(plan["plan_id"])
        level = (plan.get("escalation") or {}).get("level", 0)
        if plan["status"] == "pending":
            if not plan.get("needs_approval"):
                return
            if seen is None:
                sent = self._try("sendMessage", {"chat_id": self.chat_id, "text": self._plan_text(plan),
                                                 "parse_mode": "HTML", "reply_markup": self._buttons(plan)})
                if sent:
                    self.sent[pid] = {"message_id": sent["message_id"], "version": plan["version"], "level": level,
                                      "status": "pending"}
                return
            if seen["version"] != plan["version"]:  # re-planned: the buttons must approve what's on the screen
                self._try("editMessageText", {"chat_id": self.chat_id, "message_id": seen["message_id"],
                                              "text": self._plan_text(plan), "parse_mode": "HTML",
                                              "reply_markup": self._buttons(plan)})
                seen["version"] = plan["version"]
            if level > seen["level"]:
                seen["level"] = level
                role = html.escape(plan["escalation"]["role"])
                self._say(self.chat_id, f"⏫ <b>{html.escape(pid)}</b> escalated to {role}. "
                                        "Nothing moves until someone approves.")
        elif seen is not None and seen["status"] == "pending":
            seen["status"] = plan["status"]
            self._try("editMessageText", {"chat_id": self.chat_id, "message_id": seen["message_id"],
                                          "text": self._final_text(plan), "parse_mode": "HTML"})

    # --- the phone to ROUTERA ---------------------------------------------------------------------------

    def handle_update(self, update: dict[str, Any]) -> None:
        if "callback_query" in update:
            self._on_tap(update["callback_query"])
        elif update.get("message", {}).get("text"):
            self._on_message(update["message"])

    def _on_message(self, message: dict[str, Any]) -> None:
        chat, text = message["chat"], message["text"].strip()
        word, _, rest = text.partition(" ")
        word = word.split("@")[0].lower()
        if self.chat_id is None:
            if word == "/pair":
                self._pair(chat, rest.strip())
            else:
                self._say(chat["id"], "Send <b>/pair</b> and the code shown in the ROUTERA control room.")
            return
        if chat["id"] != self.chat_id:
            self._say(chat["id"], "This bot is paired with someone else's phone.")
            return
        if word == "/status":
            kpis = self.service.run(lambda: self.service.orchestrator.snapshot()["kpis"])
            self._say(chat["id"], f"Watching <b>{kpis['containers']}</b> containers. <b>{kpis['pending_plans']}</b> "
                                  f"plan(s) waiting, {fmt_inr(kpis['protected_inr'])} protected so far.")
        else:
            self._say(chat["id"], "You're connected. New plans arrive here with Approve and Reject buttons. "
                                  "Send /status for a summary.")

    def _pair(self, chat: dict[str, Any], code: str) -> None:
        if chat.get("type") != "private":
            self._say(chat["id"], "Pair from a private chat, so only you can approve.")
        elif secrets.compare_digest(code, self.code):
            self.chat_id, self.wrong_codes = chat["id"], 0
            self._say(chat["id"], "✅ <b>Connected.</b> New ROUTERA plans will arrive here. "
                                  "Only this chat can approve.")
            self.sync()
        else:
            self.wrong_codes += 1
            if self.wrong_codes >= MAX_WRONG_CODES:
                self.code, self.wrong_codes = self._new_code(), 0
                self._announce()
            self._say(chat["id"], "That code isn't right. Check the one shown in the control room.")

    def _answer(self, query: dict[str, Any], text: str, alert: bool = False) -> None:
        self._try("answerCallbackQuery", {"callback_query_id": query["id"], "text": text, "show_alert": alert})

    def _on_tap(self, query: dict[str, Any]) -> None:
        chat = query.get("message", {}).get("chat", {}).get("id")
        if self.chat_id is None or chat != self.chat_id or query.get("from", {}).get("id") != self.chat_id:
            self._answer(query, "You're not allowed to decide this plan.", alert=True)
            return
        try:
            action, run, plan_id, version = query["data"].split("|")
        except ValueError:
            self._answer(query, "I didn't understand that button.", alert=True)
            return
        if run != self.run_token:
            self._answer(query, "That plan is from an earlier run. Nothing changed.", alert=True)
            return
        try:
            plan = self.service.run(lambda: dict(self.service.orchestrator.plans[plan_id]))
        except KeyError:
            self._answer(query, "That plan no longer exists.", alert=True)
            return
        if action == "w":
            self._why(query, plan)
        elif plan["status"] != "pending":
            self._answer(query, f"Already {plan['status']}.", alert=True)
        elif str(plan["version"]) != version:
            self._answer(query, "The plan just changed, so nothing was approved. Check the new one.", alert=True)
        elif action == "a":
            self._decide(query, plan_id, "Approved", lambda o: o.approve(plan_id, self.approver))
        elif action == "r":
            self._decide(query, plan_id, "Rejected", lambda o: o.reject(plan_id, self.approver, REJECT_REASON))
        self.sync()

    def _why(self, query: dict[str, Any], plan: dict[str, Any]) -> None:
        rows = sorted(plan["rows"], key=lambda r: -r["value_saved_inr"])
        self._answer(query, "Here's why")
        if rows:
            answer = self.service.run(self.service.orchestrator.why, plan["plan_id"], rows[0]["container_id"])
            self._say(query["message"]["chat"]["id"], f"💬 {html.escape(answer['text'])[:3800]}")

    def _decide(self, query: dict[str, Any], plan_id: str, done: str, act: Callable[[Any], Any]) -> None:
        try:
            self.service.run(lambda: act(self.service.orchestrator))
        except (PermissionError, ValueError, KeyError) as exc:
            self._answer(query, f"Not done: {exc}"[:190], alert=True)
            return
        self._answer(query, f"{done} {plan_id}")

    # --- the two background loops -----------------------------------------------------------------------

    def start(self) -> None:
        threading.Thread(target=self._poll_loop, name="telegram-poll", daemon=True).start()
        threading.Thread(target=self._sync_loop, name="telegram-sync", daemon=True).start()

    def _poll_loop(self) -> None:
        while True:
            try:
                if not self.username:
                    self.username = self.call("getMe", {})["username"]
                    self.error = ""
                    self._announce()
                updates = self.call("getUpdates", {"offset": self.offset, "timeout": LONG_POLL_S,
                                                   "allowed_updates": ["message", "callback_query"]})
                for update in updates:
                    self.offset = update["update_id"] + 1
                    try:
                        self.handle_update(update)
                    except Exception:  # one bad message must not stop the bot
                        log.exception("telegram update failed")
            except TelegramError as exc:
                self.error = self._clean(exc)
                self._announce()
                time.sleep(5)

    def _sync_loop(self) -> None:
        while True:
            self._kick.wait(SYNC_EVERY_S)
            self._kick.clear()
            try:
                self.sync()
            except Exception:  # the phone must never take the control room down
                log.exception("telegram sync failed")
