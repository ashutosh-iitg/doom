# doom

A panel of judges. Starting point: a constitutional classifier in the sense
of [Cunningham et al., *Constitutional Classifiers++*](2601.04603v1.pdf) —
specifically the paper's **exchange classifier** (§3): a judge that
evaluates a full conversational exchange, in context, against a fixed set of
rules, rather than examining the request and the response separately (which
is what lets a request fragmented across turns, or an answer obfuscated in
metaphor, slip past a classifier that only ever sees one side at a time).

```python
from doom import Exchange, ExchangeClassifier
from doom.constitutions.cbrn_example import CBRN_EXAMPLE

exchange = Exchange.of(
    user="...",
    assistant="...",
)

verdict = ExchangeClassifier().judge(exchange, CBRN_EXAMPLE)
print(verdict.flagged, verdict.reasoning, verdict.rule_ids)
```

`ExchangeClassifier` calls out through
[`emissary`](../emissary) — any provider it supports works here, including a
locally-hosted [vLLM](https://github.com/vllm-project/vllm) server:

```bash
export DOOM_JUDGE_PROVIDER="vllm:my-local-model"
export VLLM_BASE_URL="http://localhost:8000/v1"
```

`constitutions/cbrn_example.py` is an **illustrative example only** — five
rules modeled on the paper's CBRN framing (including its reconstruction- and
obfuscation-attack analysis from §2), not a production policy.

## Where this goes next

`Panel` (`src/doom/panel.py`) holds a sequence of judges today and flags if
any one does — deliberately thin. The paper's later sections are the
roadmap for what grows in front of `Panel.evaluate`, without changing
`Verdict` or `ExchangeClassifier`:

- §4 — a cheap first-stage judge screens every exchange; only flagged ones
  escalate to an expensive second-stage judge.
- §5 — linear probes reading a *local* model's activations mid-generation
  (this is the other reason `emissary`'s vLLM support matters: probes need
  access to an open-weight model's internals, which an API-only provider
  can't give you).
- §6 — a weighted ensemble of probe and classifier scores.

## Development

```bash
uv sync
uv run pytest          # mocked — no network, no API key needed
uv run ruff check .
```

An end-to-end run against a real provider needs a credential, e.g.
`ANTHROPIC_API_KEY` for the default `anthropic` provider:

```bash
export ANTHROPIC_API_KEY=...
uv run python -c "
from doom import ExchangeClassifier
from doom.constitutions.cbrn_example import CBRN_EXAMPLE
from tests.fixtures.exchanges import BENIGN_EXCHANGE, FLAGGED_EXCHANGE

judge = ExchangeClassifier()
print(judge.judge(BENIGN_EXCHANGE, CBRN_EXAMPLE))
print(judge.judge(FLAGGED_EXCHANGE, CBRN_EXAMPLE))
"
```
