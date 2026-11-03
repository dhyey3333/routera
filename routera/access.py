"""Who's on duty: a name, a role and the team's shared access code.

Not accounts. The team shares one code (ROUTERA_ACCESS_CODE), the server checks it, and the name a person
signs in with goes on the audit log next to every decision they make. With no code set the control room
stays open, as before: the sign-in then only records who is at the desk.

The session is a cookie signed with HMAC-SHA256, so nobody can edit the name in it. The signing secret is
ROUTERA_SESSION_SECRET, or a fresh random one per run (everyone signs in again after a restart, which also
wipes the in-memory plans, so nothing is lost).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from typing import Any

COOKIE = "routera_session"
ROLES = {"ops": "Operations lead", "export": "Export manager", "fpo": "FPO director"}
SESSION_S = 12 * 3600  # one shift
TRIES, TRIES_WINDOW_S = 5, 300.0  # wrong codes allowed per client in five minutes
NAME_MAX = 40


class Access:
    def __init__(self, code: str | None = None, secret: str | bytes | None = None, *,
                 clock: Any = time.time) -> None:
        self.code = (code or "").strip()
        key = secret or secrets.token_bytes(32)
        self.key = key.encode() if isinstance(key, str) else key
        self.clock = clock
        self.misses: dict[str, list[float]] = {}

    @property
    def required(self) -> bool:
        return bool(self.code)

    # --- the cookie -------------------------------------------------------------------------------
    def _sign(self, body: bytes) -> str:
        return hmac.new(self.key, body, hashlib.sha256).hexdigest()

    def issue(self, name: str, role: str) -> str:
        body = json.dumps({"name": name, "role": role, "at": int(self.clock())}, separators=(",", ":")).encode()
        return f"{base64.urlsafe_b64encode(body).decode().rstrip('=')}.{self._sign(body)}"

    def read(self, cookie: str | None) -> dict[str, str] | None:
        if not cookie or "." not in cookie:
            return None
        packed, sig = cookie.rsplit(".", 1)
        try:
            body = base64.urlsafe_b64decode(packed + "=" * (-len(packed) % 4))
        except ValueError:
            return None
        if not hmac.compare_digest(sig, self._sign(body)):
            return None
        try:
            data = json.loads(body)
        except ValueError:
            return None
        if data.get("role") not in ROLES or not data.get("name") or self.clock() - data.get("at", 0) > SESSION_S:
            return None
        return {"name": data["name"], "role": data["role"], "role_label": ROLES[data["role"]]}

    # --- signing in -------------------------------------------------------------------------------
    @staticmethod
    def clean_name(raw: Any) -> str:
        name = re.sub(r"\s+", " ", str(raw or "")).strip()
        name = "".join(ch for ch in name if ch.isprintable())
        return name[:NAME_MAX].strip()

    def code_ok(self, given: Any) -> bool:
        return hmac.compare_digest(str(given or "").strip().encode(), self.code.encode())

    def blocked(self, client: str) -> bool:
        now = self.clock()
        recent = [t for t in self.misses.get(client, []) if now - t < TRIES_WINDOW_S]
        self.misses[client] = recent
        return len(recent) >= TRIES

    def missed(self, client: str) -> None:
        self.misses.setdefault(client, []).append(self.clock())

    @staticmethod
    def person(session: dict[str, str]) -> str:
        """How a signed-in person appears on the audit log."""
        return f"{session['name']} ({session['role_label']})"
