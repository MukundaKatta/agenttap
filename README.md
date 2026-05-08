# agenttap

Wire-level prompt introspection for LLM SDK calls. See exactly what was sent. Credentials redacted by default. Works with any httpx-based SDK (Anthropic, OpenAI, etc).

```bash
pip install agenttap
```

## Why

Five years into the SDK era, "what was actually sent to the model?" remains a hard question. SDK debug logging is verbose, leaks API keys, and reformats payloads. Callbacks scatter across vendor-specific abstractions. `agenttap` taps the wire at the httpx transport layer so you see the exact request body the provider received — with `Authorization`, `x-api-key`, and known key patterns scrubbed automatically.

## Quick start

**Anthropic**

```python
import httpx
import anthropic
from agenttap import Tap

tap = Tap()
client = anthropic.Anthropic(http_client=httpx.Client(transport=tap.transport()))

client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    messages=[{"role": "user", "content": "Hello"}],
)

print(tap.last.url)                  # https://api.anthropic.com/v1/messages
print(tap.last.pretty_request())     # exact JSON sent
print(tap.last.response_status)      # 200
```

**OpenAI**

```python
import httpx
import openai
from agenttap import Tap

tap = Tap()
client = openai.OpenAI(http_client=httpx.Client(transport=tap.transport()))

client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "Hi"}]
)

print(tap.last.request_body)
```

**Diff two requests**

```python
from agenttap import diff

# After two calls
print(diff(tap.all[0], tap.all[1]))
# - "system": "v1: be helpful"
# + "system": "v2: be concise"
```

## What's redacted by default

- Headers: `Authorization`, `x-api-key`, `api-key`, `cookie`, `anthropic-api-key`, `openai-organization`, `x-amz-security-token`, `x-google-api-key`
- Body string values matching: OpenAI/Anthropic `sk-…`, AWS `AKIA…`, Google `AIza…`, Slack `xox[baprs]-…`

Override with a custom `Redactor`:

```python
from agenttap import Tap, Redactor

tap = Tap(redactor=Redactor.none())   # no scrubbing
tap = Tap(redactor=Redactor(placeholder="<hidden>"))
```

## API

```python
Tap(redactor=None, history_size=1000)

tap.transport(parent=None)         # for httpx.Client(transport=...)
tap.async_transport(parent=None)   # for httpx.AsyncClient(transport=...)

tap.last                           # most recent TappedCall
tap.all                            # list[TappedCall]
tap.reset()
with tap.session() as sub: ...     # scoped sub-tap, results merged on exit

TappedCall.method, .url
TappedCall.request_headers, .request_body
TappedCall.response_status, .response_headers, .response_body
TappedCall.elapsed_ms
TappedCall.pretty_request()

diff(a, b)                         # unified diff of request bodies
```

## What it doesn't do

- Not a proxy. Not a server. No UI. No persistence (write `tap.all` to JSON yourself).
- Not full observability — for traces, ship the recorded calls into Phoenix/Langfuse/OTel.
- Doesn't normalize across providers. The whole point is to show what each provider actually received.

## License

MIT
