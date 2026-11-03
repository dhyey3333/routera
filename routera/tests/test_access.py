"""Signing in: a name and a role always, and the team access code when the server sets one."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from routera.access import COOKIE, SESSION_S, TRIES, Access
from routera.app import RouteraService, create_routera_app

CODE = "grapes-at-dawn"


@pytest.fixture
def open_room() -> Iterator[TestClient]:
    service = RouteraService()
    with TestClient(create_routera_app(service)) as c:
        c.service = service  # type: ignore[attr-defined]
        yield c


@pytest.fixture
def locked_room() -> Iterator[TestClient]:
    with TestClient(create_routera_app(RouteraService(), Access(CODE))) as c:
        yield c


def test_the_signed_in_name_and_role_approve_whatever_the_request_says(open_room: TestClient) -> None:
    c = open_room
    assert '<form id="signin"' in c.get("/").text
    assert c.get("/room", follow_redirects=False).headers["location"] == "/"
    assert c.get("/api/session").json() == {"signed_in": False, "code_required": False,
                                            "roles": {"ops": "Operations lead", "export": "Export manager",
                                                      "fpo": "FPO director"}}
    assert c.post("/api/session", json={"name": "   ", "role": "ops"}).status_code == 422
    assert c.post("/api/session", json={"name": "Asha K.", "role": "boss"}).status_code == 422
    me = c.post("/api/session", json={"name": "  Asha \n  K. ", "role": "export"}).json()
    assert me == {"signed_in": True, "name": "Asha K.", "role": "export", "role_label": "Export manager"}
    assert "Control room" in c.get("/room").text

    c.service.start_replay("gulf_closure", 0)  # type: ignore[attr-defined]
    c.service.wait_for_replay(30)  # type: ignore[attr-defined]
    plan = next(p for p in c.get("/api/state").json()["routera"]["plans"] if p["status"] == "pending")
    assert c.post(f"/api/plans/{plan['plan_id']}/approve", json={"approver": "Somebody Else"}).status_code == 200
    entry = next(e for e in c.get("/api/audit").json()["entries"] if e["action"] == "plan.approved")
    assert entry["actor"] == "Asha K. (Export manager)" and "Somebody" not in entry["rationale"]

    assert c.delete("/api/session").json() == {"signed_in": False}
    assert c.get("/room", follow_redirects=False).status_code == 303


def test_with_a_code_only_the_sign_in_page_is_reachable_until_the_code_is_right(locked_room: TestClient) -> None:
    c, page = locked_room, {"accept": "text/html"}
    assert c.get("/").status_code == 200 and c.get("/api/session").json()["code_required"] is True
    assert c.get("/api/state").status_code == 401
    assert c.post("/sentinel/replay/gulf_closure", params={"speed": 0}).status_code == 401
    for path in ("/room", "/sentinel", "/docs"):
        assert c.get(path, headers=page, follow_redirects=False).headers["location"] == "/"

    wrong = c.post("/api/session", json={"name": "Asha K.", "role": "ops", "code": "grapes"})
    assert wrong.status_code == 401 and "access code" in wrong.json()["detail"]
    assert c.post("/api/session", json={"name": "Asha K.", "role": "ops", "code": f" {CODE} "}).status_code == 200
    assert c.get("/api/state").status_code == 200 and c.get("/room").status_code == 200
    # already signed in: a new role or name needs no code again
    assert c.post("/api/session", json={"name": "Asha K.", "role": "fpo"}).json()["role_label"] == "FPO director"


def test_wrong_codes_are_throttled_even_when_the_next_one_is_right(locked_room: TestClient) -> None:
    for _ in range(TRIES):
        assert locked_room.post("/api/session", json={"name": "X", "role": "ops", "code": "nope"}).status_code == 401
    assert locked_room.post("/api/session", json={"name": "X", "role": "ops", "code": CODE}).status_code == 429


def test_the_session_cannot_be_edited_reused_elsewhere_or_outlive_a_shift() -> None:
    now = [1_000_000.0]
    access = Access(CODE, "secret", clock=lambda: now[0])
    cookie = access.issue("Asha K.", "ops")
    assert access.read(cookie) == {"name": "Asha K.", "role": "ops", "role_label": "Operations lead"}
    body, sig = cookie.rsplit(".", 1)
    assert access.read(f"{body[:-2]}xx.{sig}") is None
    assert Access(CODE, "another secret", clock=lambda: now[0]).read(cookie) is None
    assert access.read("garbage") is None and access.read(None) is None
    now[0] += SESSION_S + 1
    assert access.read(cookie) is None
    assert COOKIE == "routera_session"
