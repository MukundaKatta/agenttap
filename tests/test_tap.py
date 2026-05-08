import json
import httpx
import pytest

from agenttap import Tap, Redactor, diff


# ---- httpx MockTransport gives us a deterministic backend without network ----

def echo_handler(request: httpx.Request) -> httpx.Response:
    body = request.content.decode("utf-8") if request.content else ""
    try:
        parsed = json.loads(body) if body else {}
    except json.JSONDecodeError:
        parsed = {"raw": body}
    return httpx.Response(
        status_code=200,
        json={"echoed": parsed, "method": request.method, "path": request.url.path},
    )


def make_client(tap: Tap) -> httpx.Client:
    # Compose: tap layered on top of MockTransport
    parent = httpx.MockTransport(echo_handler)
    return httpx.Client(transport=tap.transport(parent=parent))


def test_records_request_and_response():
    t = Tap()
    with make_client(t) as client:
        r = client.post(
            "https://api.example.com/v1/messages",
            json={"model": "claude", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-ant-actualsecret123456789", "x-api-key": "leaky"},
        )
    assert r.status_code == 200
    assert t.last is not None
    assert t.last.method == "POST"
    assert t.last.url == "https://api.example.com/v1/messages"
    assert t.last.response_status == 200


def test_redacts_sensitive_headers():
    # httpx lowercases header names internally.
    t = Tap()
    with make_client(t) as client:
        client.post(
            "https://api.example.com/x",
            json={},
            headers={"Authorization": "Bearer sk-ant-secret9876543210xyz", "x-api-key": "abc123"},
        )
    h = t.last.request_headers
    assert h["authorization"] == "***REDACTED***"
    assert h["x-api-key"] == "***REDACTED***"


def test_redacts_api_key_patterns_in_body():
    t = Tap()
    with make_client(t) as client:
        client.post(
            "https://api.example.com/x",
            json={"system": "Use this key sk-ant-thiseekrit1234567890 internally"},
        )
    sys_text = t.last.request_body["system"]
    assert "sk-ant" not in sys_text
    assert "***REDACTED***" in sys_text


def test_redactor_none_disables_scrubbing():
    t = Tap(redactor=Redactor.none())
    with make_client(t) as client:
        client.post(
            "https://api.example.com/x",
            json={"system": "key sk-ant-thiseekrit1234567890"},
            headers={"Authorization": "Bearer plain-token"},
        )
    assert "sk-ant" in t.last.request_body["system"]
    assert t.last.request_headers["authorization"] == "Bearer plain-token"


def test_request_body_decoded_as_dict():
    t = Tap()
    with make_client(t) as client:
        client.post("https://api.example.com/x", json={"foo": [1, 2, 3]})
    assert t.last.request_body == {"foo": [1, 2, 3]}


def test_diff_shows_changed_field():
    t = Tap()
    with make_client(t) as client:
        client.post("https://api.example.com/x", json={"system": "A", "user": "hi"})
        client.post("https://api.example.com/x", json={"system": "B", "user": "hi"})
    d = diff(t.all[0], t.all[1])
    assert '-  "system": "A"' in d
    assert '+  "system": "B"' in d


def test_history_trim():
    t = Tap(history_size=3)
    with make_client(t) as client:
        for i in range(5):
            client.post("https://api.example.com/x", json={"i": i})
    assert len(t.all) == 3
    assert t.all[-1].request_body == {"i": 4}
    # Earliest two were dropped: remaining are i=2, i=3, i=4
    assert t.all[0].request_body == {"i": 2}


def test_session_context_appends_to_parent():
    t = Tap()
    with make_client(t) as client:
        client.post("https://api.example.com/x", json={"a": 1})
    with t.session() as sub:
        with httpx.Client(
            transport=sub.transport(parent=httpx.MockTransport(echo_handler))
        ) as sub_client:
            sub_client.post("https://api.example.com/y", json={"b": 2})
        assert len(sub.all) == 1
    assert len(t.all) == 2


def test_reset_clears():
    t = Tap()
    with make_client(t) as client:
        client.post("https://api.example.com/x", json={})
    assert len(t.all) == 1
    t.reset()
    assert t.all == []


@pytest.mark.asyncio
async def test_async_transport():
    async def handler(request):
        return httpx.Response(200, json={"ok": True})

    t = Tap()
    parent = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=t.async_transport(parent=parent)) as client:
        r = await client.post("https://api.example.com/x", json={"k": "v"})
    assert r.status_code == 200
    assert t.last.request_body == {"k": "v"}
