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

## §4 calibration

`DEFAULT_THRESHOLD` is a placeholder. `doom.calibration.sweep_thresholds`
measures, rather than guesses, a real value — given a `Screen`, a
`Constitution`, and labelled exchanges (`doom.LabelledExchange`, loadable from
JSONL via `doom.dataset.load_jsonl`), it reports escalation rate and
missed-violation rate at each candidate threshold:

```python
from doom.calibration import sweep_thresholds
from doom.dataset import load_jsonl

labelled = load_jsonl("labelled_exchanges.jsonl")
for stats in sweep_thresholds(screen, CBRN_EXAMPLE, labelled, thresholds=[0.05, 0.15, 0.3, 0.5]):
    print(stats.threshold, stats.escalation_rate, stats.missed_violation_rate)
```

This has only ever run against small synthetic fixtures in its own tests —
picking a real threshold needs a real labelled dataset, which doesn't exist
yet.

## §5 linear probe (PyTorch)

A separate, optional `probe` dependency group (`uv sync --group probe` —
torch, transformers, safetensors, pyarrow; not installed by default, and not
imported by `doom/__init__.py`). `doom.probe.Backbone` runs one frozen
forward pass over a local HF `transformers` model (`DOOM_PROBE_BACKBONE`) and
pools the last-token hidden state; `doom.probe.ProbeHead` is a small trained
PyTorch head (linear, or a shallow MLP via `hidden_dims`); `doom.probe.
LinearProbe` combines them into the same `exchange, constitution -> score in
[0, 1]` shape `Screen` has, so it's a drop-in `Cascade` screen on its own:

```python
from doom import Cascade, ExchangeClassifier
from doom.probe import Backbone, LinearProbe, ProbeHead
from doom.probe.data import extract_and_cache, load_cached
from doom.constitutions.cbrn_example import CBRN_EXAMPLE

backbone = Backbone()  # reads DOOM_PROBE_BACKBONE
extract_and_cache(labelled_exchanges, backbone, CBRN_EXAMPLE, "activations/")
# uv run python -m doom.probe.train --cls-dir activations/ --checkpoint-dir ckpts/ --run-name v0

head = ProbeHead.load("ckpts/v0/step_002000", emb_dim=...)
probe = LinearProbe(backbone, head)
cascade = Cascade(screen=probe, judge=ExchangeClassifier(), constitution=CBRN_EXAMPLE)
```

Never run against a real backbone or real labelled data — training/checkpoint
logic is verified with a real, tiny, synthetic, CPU-only training run in
`tests/probe/test_train.py`; `Backbone` is exercised only on its pooling
logic. The paper's sliding-window logit smoothing and softmax-weighted loss
aren't implemented here — this is a plain linear (or shallow-MLP) probe.

## §6 weighted ensemble

`doom.ensemble.EnsembleScreen` blends the §5 probe with `Screen` — not with
`ExchangeClassifier`, which is prompted/tool-forced and has no calibrated
probability to blend — `z = alpha * z_probe + (1 - alpha) * z_screen`. It
satisfies the same screen interface, so it drops into `Cascade` unmodified:

```python
from doom.ensemble import EnsembleScreen

ensemble = EnsembleScreen(screen=screen, probe=probe, alpha=0.5)
cascade = Cascade(screen=ensemble, judge=ExchangeClassifier(), constitution=CBRN_EXAMPLE)
```

`alpha` is unset/untuned — that needs both a working probe and labelled data
in a real, non-synthetic form.

## Reasoning Jury (`doom.jury`)

A second, sibling judge family — not an extension of §3–§6. It implements
Wang, Singh, Gao, Matsoukas, Liu, Namazifar (Amazon AGI), *Reasoning Jury:
Multi-Model Consensus for Evaluating Reasoning Traces* (arXiv:2608.12585),
answering a different question than the judges above: not "does this
exchange violate a policy," but "where exactly, and how badly, does this
chain-of-thought reasoning trace go wrong." No new dependency — everything
routes through `emissary.call_tool`, same as `ExchangeClassifier` — but it's
not re-exported from `doom/__init__.py`; import it explicitly from
`doom.jury`.

A `ReasoningTrace` is a problem, a raw trace (auto-segmented into `[STEP-x]`
units), and a final solution. A panel of `Juror`s independently lists
trace-grounded defects (Phase 1); one of two interchangeable Phase 2
policies then reconciles them into a `JuryVerdict`:

```python
from doom.jury import Consolidator, Deliberator, Juror, Moderator, ReasoningJury, ReasoningTrace

trace = ReasoningTrace.of(problem="...", raw_trace="...", final_solution="...")
jurors = (Juror(juror_id="jury-0"), Juror(juror_id="jury-1"), Juror(juror_id="jury-2"))

# Cheap mode: one extra call reconciles the panel's findings.
jury = ReasoningJury(jurors=jurors, phase2=Consolidator())
verdict = jury.evaluate(trace)

# Thorough mode: a content-blind moderator runs a multi-turn debate among jurors.
jury = ReasoningJury(jurors=jurors, phase2=Deliberator(jurors=jurors, moderator=Moderator()))
verdict = jury.evaluate(trace)

for defect in verdict.negatives:
    print(defect.impact, defect.statement_refs, defect.what_went_wrong)
```

The moderator in `Deliberator` never sees the problem/trace/solution — only
the deliberation transcript — so its control over turn-taking can't be
biased by opinions about the content. Fault tolerance is real: a juror
failing mid-deliberation is skipped and dropped after repeated failures, a
moderator failure falls back to deterministic round-robin, and
`ReasoningJury` proceeds on a Phase 1 juror failure as long as a configured
minimum still succeeds — every dropped juror shows up in
`verdict.failed_jurors`, never silently.

Never run against a real juror, moderator, or deliberation — code-complete
and tested with mocked `emissary.call_tool` calls and stub jurors/moderators
only, same standard as everything else in this repo.

## Development

```bash
uv sync
uv run pytest          # mocked — no network, no API key needed
uv run ruff check .

uv sync --group probe   # optional: torch/transformers/pyarrow/safetensors,
                         # needed for tests/probe/ (§5) — skipped cleanly without it
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
> hand-built response payloads. §4's calibration sweep, §5's probe, and §6's
> ensemble are code-complete and tested against synthetic/mocked data (§5's
> training loop is verified with a real tiny CPU training run), but none of
> them has run against a real labelled dataset or a real backbone.
> `doom.jury` (Reasoning Jury) is likewise code-complete and tested only
> with mocked calls and stub jurors/moderators — no real juror, moderator,
> or deliberation has ever run.

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
