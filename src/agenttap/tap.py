"""Wire-level introspection of LLM SDK HTTP calls.

`agenttap` plugs into the SDK's underlying httpx client at the transport
layer. The Anthropic and OpenAI Python SDKs both accept a custom
`http_client=httpx.Client(...)` argument; pass one constructed with
`Tap().transport()` and every request/response pair is captured, with
sensitive headers redacted by default.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Pattern

import httpx


# Header names whose values get redacted by default.
DEFAULT_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "x-api-key",
        "api-key",
        "x-amz-security-token",
        "x-google-api-key",
        "openai-organization",
        "anthropic-api-key",
        "cookie",
        "set-cookie",
    }
)

# Regex patterns matched against any string field to redact known key shapes.
DEFAULT_VALUE_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),  # OpenAI / Anthropic
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),  # Google API key
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack token
)


@dataclass(frozen=True)
class TappedCall:
    """One captured request/response pair, redacted."""

    timestamp: float
    method: str
    url: str
    request_headers: dict[str, str]
    request_body: Any  # dict if JSON-decodable, else str
    response_status: int
    response_headers: dict[str, str]
    response_body: Any
    elapsed_ms: float

    def pretty_request(self) -> str:
        body = (
            json.dumps(self.request_body, indent=2, sort_keys=True)
            if not isinstance(self.request_body, str)
            else self.request_body
        )
        return f"{self.method} {self.url}\n\n{body}"


@dataclass
class Redactor:
    sensitive_headers: frozenset[str] = DEFAULT_SENSITIVE_HEADERS
    value_patterns: tuple[Pattern[str], ...] = DEFAULT_VALUE_PATTERNS
    placeholder: str = "***REDACTED***"

    @classmethod
    def default(cls) -> Redactor:
        return cls()

    @classmethod
    def strict(cls) -> Redactor:
        # Future: more aggressive scrubbing; for now alias of default.
        return cls()

    @classmethod
    def none(cls) -> Redactor:
        return cls(sensitive_headers=frozenset(), value_patterns=())

    def headers(self, headers: dict[str, str]) -> dict[str, str]:
        out = {}
        for k, v in headers.items():
            if k.lower() in self.sensitive_headers:
                out[k] = self.placeholder
            else:
                out[k] = self._scrub_value(v)
        return out

    def _scrub_value(self, s: str) -> str:
        if not isinstance(s, str):
            return s
        for pat in self.value_patterns:
            s = pat.sub(self.placeholder, s)
        return s

    def body(self, body: Any) -> Any:
        if isinstance(body, dict):
            return {k: self.body(v) for k, v in body.items()}
        if isinstance(body, list):
            return [self.body(v) for v in body]
        if isinstance(body, str):
            return self._scrub_value(body)
        return body


class Tap:
    """Records LLM SDK HTTP traffic with redaction.

    Usage with Anthropic:
        import anthropic
        t = Tap()
        client = anthropic.Anthropic(http_client=httpx.Client(transport=t.transport()))
        client.messages.create(...)
        print(t.last)
        print(t.last.pretty_request())

    Usage with OpenAI:
        import openai
        t = Tap()
        client = openai.OpenAI(http_client=httpx.Client(transport=t.transport()))
    """

    def __init__(self, redactor: Optional[Redactor] = None, history_size: int = 1_000):
        self.redactor = redactor or Redactor.default()
        self._calls: list[TappedCall] = []
        self._lock = threading.Lock()
        self._history_size = history_size

    def transport(self, parent: Optional[httpx.HTTPTransport] = None) -> _TappedTransport:
        return _TappedTransport(self, parent or httpx.HTTPTransport())

    def async_transport(
        self, parent: Optional[httpx.AsyncHTTPTransport] = None
    ) -> _TappedAsyncTransport:
        return _TappedAsyncTransport(self, parent or httpx.AsyncHTTPTransport())

    @contextlib.contextmanager
    def session(self) -> Iterator["Tap"]:
        """Context manager that returns a fresh sub-tap; useful for scoping."""
        sub = Tap(redactor=self.redactor)
        yield sub
        with self._lock:
            self._calls.extend(sub._calls)
            self._trim()

    @property
    def last(self) -> Optional[TappedCall]:
        with self._lock:
            return self._calls[-1] if self._calls else None

    @property
    def all(self) -> list[TappedCall]:
        with self._lock:
            return list(self._calls)

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()

    def _record(self, call: TappedCall) -> None:
        with self._lock:
            self._calls.append(call)
            self._trim()

    def _trim(self) -> None:
        excess = len(self._calls) - self._history_size
        if excess > 0:
            del self._calls[:excess]


class _TappedTransport(httpx.BaseTransport):
    def __init__(self, tap: Tap, parent: httpx.HTTPTransport):
        self._tap = tap
        self._parent = parent

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        t0 = time.perf_counter()
        response = self._parent.handle_request(request)
        # Read into memory so we can record without consuming the user's stream.
        response.read()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._tap._record(_capture(self._tap.redactor, request, response, elapsed_ms))
        return response

    def close(self) -> None:
        self._parent.close()


class _TappedAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, tap: Tap, parent: httpx.AsyncHTTPTransport):
        self._tap = tap
        self._parent = parent

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        t0 = time.perf_counter()
        response = await self._parent.handle_async_request(request)
        await response.aread()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._tap._record(_capture(self._tap.redactor, request, response, elapsed_ms))
        return response

    async def aclose(self) -> None:
        await self._parent.aclose()


def _capture(
    redactor: Redactor, request: httpx.Request, response: httpx.Response, elapsed_ms: float
) -> TappedCall:
    req_body = _decode_body(request.content) if request.content else None
    resp_body = _decode_body(response.content) if response.content else None
    return TappedCall(
        timestamp=time.time(),
        method=request.method,
        url=str(request.url),
        request_headers=redactor.headers(dict(request.headers)),
        request_body=redactor.body(req_body),
        response_status=response.status_code,
        response_headers=redactor.headers(dict(response.headers)),
        response_body=redactor.body(resp_body),
        elapsed_ms=elapsed_ms,
    )


def _decode_body(content: bytes) -> Any:
    if not content:
        return None
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        return f"<{len(content)} bytes binary>"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def diff(a: TappedCall, b: TappedCall) -> str:
    """Unified diff of two captured request bodies. Useful for 'why did this work
    yesterday and not today.'"""

    def pretty(c: TappedCall) -> str:
        return json.dumps(c.request_body, indent=2, sort_keys=True, default=str)

    lines = difflib.unified_diff(
        pretty(a).splitlines(keepends=False),
        pretty(b).splitlines(keepends=False),
        fromfile="a",
        tofile="b",
        lineterm="",
    )
    return "\n".join(lines)
