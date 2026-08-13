# doom

A panel of judges. Starting point: a constitutional classifier in the sense
of [Cunningham et al., *Constitutional Classifiers++*](https://arxiv.org/abs/2601.04603) —
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

## The two-stage cascade (paper §4)

Running the expensive judge on every exchange is what §4 exists to avoid. A
cheap `Screen` scores all traffic and only suspicious exchanges reach the
judge:

```python
from doom import Cascade, ExchangeClassifier, Screen
from doom.constitutions.cbrn_example import CBRN_EXAMPLE
import emissary

cascade = Cascade(
    screen=Screen(emissary.parse_spec("vllm:my-local-model")),  # one token
    judge=ExchangeClassifier(),                                  # full verdict
    constitution=CBRN_EXAMPLE,
    threshold=0.15,
)

out = cascade.evaluate(exchange)
out.flagged      # the decision
out.score        # what the screen thought (kept on both routes)
out.escalated    # whether the judge ever saw it
```

The screen generates **one token** and scores it from the model's own
logprobs — not from asking the model to rate its own confidence, which isn't
calibrated. That means the screen needs a logprob-capable provider: a local
vLLM model or OpenAI. **The Anthropic API exposes no logprobs**, so a Claude
model can't be the screen; `emissary.call_choice` refuses it rather than
degrading silently.

Because a flagged exchange is *escalated, not refused*, the screen is
deliberately biased toward flagging — a false positive costs one extra judge
call and nothing else. `threshold` is the compute/robustness knob: lower it
and more traffic escalates.

> The default `threshold` of 0.15 is a placeholder. A real value comes from
> measuring escalation and missed-violation rates on labelled exchanges.

## Where this goes next

- **§5** — linear probes reading a local model's activations mid-generation.
  A probe exposes the same `exchange → score` interface `Screen` does, so it
  drops into stage 1 without touching `Cascade`. (An off-the-shelf safety
  classifier served via vLLM is the cheaper step on the way there.)
- **§6** — a weighted ensemble of probe and classifier scores, which is why
  `CascadeVerdict` keeps the screen's score even after escalating.

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

> **Status:** nothing here has run against a live model yet. The tests are
> fully mocked, and `Screen`'s logprob scoring is verified only against
> hand-built response payloads.

## Reference

This project implements ideas from:

> Hoagy Cunningham, Jerry Wei, Zihan Wang, Andrew Persic, Alwin Peng, Jordan
> Abderrachid, et al. **Constitutional Classifiers++: Efficient
> Production-Grade Defenses against Universal Jailbreaks.** arXiv:2601.04603
> [cs.CR], January 2026. <https://arxiv.org/abs/2601.04603>

```bibtex
@article{cunningham2026constitutional,
  title   = {Constitutional Classifiers++: Efficient Production-Grade
             Defenses against Universal Jailbreaks},
  author  = {Cunningham, Hoagy and Wei, Jerry and Wang, Zihan and
             Persic, Andrew and Peng, Alwin and Abderrachid, Jordan and
             Perez, Ethan and Sharma, Mrinank},
  journal = {arXiv preprint arXiv:2601.04603},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.04603}
}
```

The paper is the authors' work and is not distributed with this repository —
fetch it from arXiv. This repository's own code is MIT licensed (see
[LICENSE](LICENSE)); the `constitutions/cbrn_example.py` ruleset is an
illustrative example written for testing, not a production safety policy.

## License

[MIT](LICENSE) © ashutosh-iitg
