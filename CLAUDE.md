# doom: a panel of judges

---

# Rules — read these before every task and apply them throughout

These rules apply to every task in this project unless explicitly overridden.
Bias: caution over speed on non-trivial work. Use judgment on trivial tasks.

**These rules are shared verbatim with `stria` and `emissary`.** They are copied
rather than referenced because each repo must stand alone when cloned. If you
change a rule, change it in all three — a rule that disagrees across repos is
worse than no rule.

## Rule 1 — Think Before Coding
State assumptions explicitly. If uncertain, ask rather than guess.
Present multiple interpretations when ambiguity exists.
Push back when a simpler approach exists.
Stop when confused. Name what's unclear.

## Rule 2 — Simplicity First
Minimum code that solves the problem. Nothing speculative.
No features beyond what was asked. No abstractions for single-use code.
Test: would a senior engineer say this is overcomplicated? If yes, simplify.

## Rule 3 — Surgical Changes
Touch only what you must. Clean up only your own mess.
Don't "improve" adjacent code, comments, or formatting.
Don't refactor what isn't broken. Match existing style.

## Rule 4 — Goal-Driven Execution
Define success criteria. Loop until verified.
Don't follow steps. Define success and iterate.
Strong success criteria let you loop independently.

## Rule 5 — Use the model only for judgment calls
Use me for: classification, drafting, summarization, extraction.
Do NOT use me for: routing, retries, deterministic transforms.
If code can answer, code answers.

## Rule 6 — Token budgets are not advisory
Per-task: 4,000 tokens. Per-session: 30,000 tokens.
These bind routine work. A long-running or genuinely high-complexity task —
multi-repo changes, extractions, anything needing sustained context — may exceed
them; say so when you pass, and why the task warranted it.
Never silently overrun, and never let a budget truncate work into something
half-done. If you are approaching budget on *routine* work, summarize and start
fresh — that is the case the budget exists for.

## Rule 7 — Surface conflicts, don't average them
If two patterns contradict, pick one (more recent / more tested).
Explain why. Flag the other for cleanup.
Don't blend conflicting patterns.

## Rule 8 — Read before you write
Before adding code, read exports, immediate callers, shared utilities.
"Looks orthogonal" is dangerous. If unsure why code is structured a way, ask.

## Rule 9 — Tests verify intent, not just behavior
Tests must encode WHY behavior matters, not just WHAT it does.
A test that can't fail when business logic changes is wrong.

## Rule 10 — Checkpoint after every significant step
Summarize what was done, what's verified, what's left.
Don't continue from a state you can't describe back.
If you lose track, stop and restate.

## Rule 11 — Match the codebase's conventions, even if you disagree
Conformance > taste inside the codebase.
If you genuinely think a convention is harmful, surface it. Don't fork silently.

## Rule 12 — Fail loud
"Completed" is wrong if anything was skipped silently.
"Tests pass" is wrong if any were skipped.
Default to surfacing uncertainty, not hiding it.

## Rule 13 — Read the architecture and summarise it whenever you lose sight of the big picture
Before diving into any isolated part, re-read the architecture section and state
in one sentence how the piece you are about to touch fits into the overall data
flow. If you cannot articulate that connection, stop and resolve it before
writing code. Use that connection to validate that local decisions (naming,
granularity, output shape) are consistent with the system's end-to-end contract,
not just locally convenient.

## Rule 14 — Every architecture decision must be defensible
Apply KISS, DRY, YAGNI, and SOLID throughout — not as a checklist but as a
standard of craft. Every non-trivial design choice (a new abstraction, a pattern,
a module boundary, a data model decision) must have a clear rationale grounded in
the system's actual constraints and the architecture it lives inside. "It's a
common pattern" is not a rationale. "It solves X because Y, and the alternative
has cost Z" is. If a decision cannot withstand a pointed question from a senior
engineer, it should not be made. Surface the trade-off, name the alternative you
rejected, and state why this system favors the chosen approach. Uninformed or
arbitrary decisions — even small ones — are not acceptable.

## Rule 15 — Good code is self-explanatory; bad code needs verbose comments
If a comment is explaining *what* the code does, the code is wrong — fix the
naming, the structure, or the decomposition instead. Reaching for a comment to
make a block legible is a signal, not a solution.

Comments are for the things the code genuinely cannot say:
- **Non-obvious catches** — why this order, why this guard, what breaks without it.
- **Assumptions made** — what we are relying on that the reader cannot see from here.
- **Docstrings on function, module, and API definitions** — contract, not narration.

Not required for simple functions. A docstring restating an obvious signature is
noise and should be deleted.

---

# Project context

**doom** is a panel of judges for evaluating model behaviour. It starts from
Cunningham et al., *Constitutional Classifiers++* (arXiv:2601.04603,
<https://arxiv.org/abs/2601.04603>) — **read the paper before changing the
judging architecture**; the section numbers below refer to it. The PDF is not
tracked in this repo; fetch it from arXiv.

## Where the project actually is

Built and tested (mocked): the paper's **§3 exchange classifier**, the **§4
two-stage cascade** (routing logic and a calibration harness), **§5's linear
probe** (PyTorch, frozen backbone + trained head), and **§6's weighted
ensemble**. `ExchangeClassifier.judge` sends one full exchange plus a
constitution to a judge model and gets back a structured `Verdict`. `Screen`
scores an exchange from one token's logprobs; `Cascade` routes on that score.
`Panel` holds a sequence of judges and flags if any one does.

**§4/§5/§6 are code-complete but unverified against anything real** — see the
per-section notes in the Roadmap below. None of them has run against a real
labelled dataset, a real backbone, or a real threshold sweep; every test is
mocked or uses hand-built synthetic data, same as §3.

Not built: the fine-tuning behind the paper's classifiers. Do not describe
this repo as implementing Constitutional Classifiers — the judges here are
*prompted*, not fine-tuned, which is the single biggest gap against the
paper's numbers. That gap is unaffected by §4/§5/§6 being code-complete.

**A second judge family, `doom.jury`, is also built and tested (mocked).**
It implements Wang, Singh, Gao, Matsoukas, Liu, Namazifar (Amazon AGI),
*Reasoning Jury: Multi-Model Consensus for Evaluating Reasoning Traces*
(arXiv:2608.12585) — a genuinely different question from §3–§6 ("where does
this reasoning trace go wrong", not "does this exchange violate a policy"),
over a different input/output shape, so it lives as its own module tree
sharing no types with `constitution.py`/`verdict.py`/`panel.py`/`cascade.py`.
Like §4/§5/§6, it is code-complete and mock-tested only — no real juror, no
real moderator, no real deliberation has ever run. See the "Decisions"
section below for why it's structured this way, and "The shape" for its
file layout.

**Nothing in this repo has ever run against a live model.** There is no vLLM
server and no funded API key, so `Screen.score` in particular is verified only
against hand-built logprob payloads. Do not claim the cascade works end to end
until someone runs it.

## The shape

```
exchange.py       Exchange, Turn — the conversation under evaluation
constitution.py   Constitution, Rule — the ruleset judged against
constitutions/    cbrn_example.py — ILLUSTRATIVE ONLY, not policy
verdict.py        Verdict — flagged, reasoning, rule_ids
judge.py          ExchangeClassifier — stage 2: one prompted judge, tool-forced
screen.py         Screen — stage 1: one token, scored from logprobs
cascade.py        Cascade — threshold routing; CascadeVerdict records the route
panel.py          Panel — sequence of judges; the seam for §6's ensemble
dataset.py        LabelledExchange — the one labelled-exchange shape §4 and §5 both read
calibration.py    sweep_thresholds — measures Cascade.threshold instead of guessing it
ensemble.py       EnsembleScreen — §6: alpha * probe.score + (1 - alpha) * screen.score
probe/            §5's linear probe — PyTorch, opt-in (see "Dependencies" below):
                    backbone.py  frozen HF causal LM, one forward pass, hidden states out
                    head.py      ProbeHead — the trained half; safetensors + MANIFEST.json
                    data.py      activation cache over parquet (extract_and_cache/load_cached)
                    train.py     CLI training loop (warmup+cosine LR, non-finite hard-stop)
                    probe.py     LinearProbe — combines backbone+head into Screen's own shape
jury/             Reasoning Jury — a second, sibling judge family (reasoning-trace
                  quality, not policy compliance); no new dependency, not re-exported
                  from doom/__init__.py:
                    trace.py         segment_trace + ReasoningTrace (problem/steps/solution)
                    defect.py        Defect — one shape for Phase 1 and Phase 2 findings
                    verdict.py       JuryVerdict — the unifying Phase 2 output
                    phase1.py        Juror — independent judgement + deliberation turns
                    consolidation.py Consolidator — Phase 2's single-call consensus mode
                    deliberation.py  Moderator + Deliberator — the multi-turn debate mode
                    jury.py          ReasoningJury — Phase 1 fan-out, then delegates to
                                     whichever Phase 2 policy (Consolidator/Deliberator)
                                     it's given
```

Two data flows, by cost:

- **Screen** (every exchange): `Exchange` + `Constitution` →
  `emissary.call_choice` → one generated token, scored from its logprobs → a
  float in [0, 1].
- **Judge** (escalated only): `Exchange` + `Constitution` →
  `emissary.call_tool` → `Verdict` with reasoning and rule ids.

`Cascade` runs the first and, above `threshold`, the second. Nothing here
holds state between calls.

## Decisions — do not re-open without new information

**Why the exchange classifier first (§3, not §2).** The paper's §2 shows
input-only and output-only classifiers falling to two attack classes:
reconstruction (a request fragmented across benign-looking turns) and output
obfuscation (a harmful answer disguised in metaphor or code words). Both are
invisible to a classifier that sees one side at a time and visible to one that
reads the exchange whole. Starting anywhere else would mean building the
known-broken thing first.

**Prompted, not fine-tuned.** The paper's classifiers are fine-tuned; this one
is prompted, because there is no training pipeline or labelled data here yet.
That is a real capability gap, not a design preference — it should be named as
such and not papered over when comparing results to the paper's numbers.

**`Verdict` is uniform across judge kinds.** A prompted judge, a fine-tuned
judge, and (per §5) a linear probe reading activations all produce the same
shape, so `Panel` never learns which kind it holds. This is the one abstraction
in the repo that earns its keep: it is what makes §4's cascade and §6's ensemble
additive rather than a rewrite.

**`Panel` is deliberately thin — flag if any judge flags.** It is not a cascade
and does not pretend to be. §6 (weighted probe + classifier logits) belongs in a
policy *in front of* `Panel.evaluate`, leaving `Verdict` and
`ExchangeClassifier` untouched. Adding weighting to `Panel` itself would couple
the two. `Cascade` is exactly that shape and is what §4 turned out to need.

**The screen scores from logprobs, never from self-reported confidence.** The
paper's cascade thresholds a classifier's logits; the faithful analogue for a
prompted judge is to generate one constrained token and read the probability
the model actually assigned it. Asking a model "how confident are you, 0-1?"
returns a number that is not calibrated, and thresholding it only looks like
measurement. This is the reason `Screen` cannot use the Anthropic API at all:
**it exposes no logprobs** — no parameter, no derivation — so the screen needs
a locally served open-weight model (`vllm:<model>`) or OpenAI.
`emissary.call_choice` refuses an Anthropic spec explicitly rather than
silently degrading.

**Escalation is not refusal, and that asymmetry is load-bearing.** A flagged
exchange costs one extra judge call; nothing is blocked on the screen's word.
That is what lets the screening prompt be deliberately recall-biased ("when
unsure, FLAG") without the false-positive cost a refusing classifier would
face — and in turn what lets stage 1 be small and cheap. `CascadeVerdict`
records `escalated` alongside the verdict because "the judge cleared it" and
"the screen never sent it" are both `flagged=False` and must stay
distinguishable when auditing flag rates.

**`DEFAULT_THRESHOLD = 0.15` is a guess, not a calibration.** A real value
comes from measuring escalation rate and missed-violation rate against
labelled exchanges. Treat the constant as a placeholder that happens to lean
the right way (low, per §4's "flag the vast majority of known-bad examples").

**The constitution is cache-marked; the exchange is not.** The constitution is
identical across every exchange a judge evaluates, the exchange is unique per
call. Same reasoning as stria's per-provision extraction calls.

**`cbrn_example.py` is an example, and must stay labelled as one.** Five rules
modelled on the paper's CBRN framing so the classifier is runnable end to end.
It is not a production policy and must never be presented as one. Rules name
what is disallowed and why — never operational detail.

**§5's activation access lives entirely inside `doom`, not `emissary`.**
`emissary` has exactly two wire adapters (`anthropic_wire.py`,
`openai_wire.py`), both HTTP calls to a remote or local API — `vllm` there
means vLLM's OpenAI-compatible HTTP server, never the vLLM Python API or a
direct `transformers` model load. Reading hidden states needs an in-process
forward pass over local model weights: a different transport, not another
wire format. Bolting that onto `emissary` would force every consumer
(including `stria`) to carry a torch/transformers stack they don't need, so
`probe/backbone.py` loads a local HF model directly and is doom's only place
that imports torch — kept out of `doom/__init__.py`'s eager exports (and
`ensemble.py` is typed structurally against `calibration.ScreenLike` instead
of importing `LinearProbe` by name), so `import doom` never requires torch.

**§6 combines `Screen` + the probe, not the judge + the probe.** The paper's
`z = α·z_probe + (1−α)·z_classifier` needs two continuous scores.
`ExchangeClassifier` is prompted and tool-forced — it returns a boolean
`Verdict`, and the same objection `screen.py` already makes against
self-reported confidence applies to giving the judge a confidence field for
this purpose. `Screen` already has a real logprob-derived score, so that's
the classifier side of the blend in this codebase.

**Reasoning Jury (`doom.jury`) is a separate module tree, not an extension
of `Verdict`/`Panel`/`Cascade`.** It answers a different question ("where
does this reasoning trace go wrong", not "does this exchange violate a
policy") over a different input (`problem`/segmented trace/`final_solution`,
not `Exchange`+`Constitution`) and a structurally richer output (a *list* of
step-grounded, severity-tagged, evidence-backed defects, not one boolean
`Verdict`). Forcing it into the existing types would overload the one
abstraction this file already calls out as load-bearing specifically because
it's uniform and simple.

**One `Defect` type serves both Phase 1 and Phase 2** — not a parallel
raw/consensus type hierarchy. `vote_count`/`voters` default to "not
applicable" (`0`/`()`) until a Phase 2 policy has actually run, the same
"uniform shape, meaningful defaults" move `Verdict` itself makes.

**`Consolidator` and `Deliberator` are duck-typed as one `Phase2Policy`
shape** (`run(trace, phase1) -> JuryVerdict`) — the same structural-typing
move already used for `Screen`/`LinearProbe`/`EnsembleScreen` via
`calibration.ScreenLike`. A policy is swapped by passing a different object,
never by branching on a mode flag.

**Deliberation's juror-contribution call is a single-field tool call, not a
new emissary capability.** `emissary` exposes no free-text completion call
(only `call_choice` for one token and `call_tool` for tool-forced structured
output) — the same reason §5's activation access stays inside `doom`, not
`emissary`, applies here too: adapting to what emissary already offers beats
growing a shared dependency for one call site.

**Fault tolerance (paper Appendix F) degrades without hiding it.** A juror
failing during deliberation has its turn skipped and is dropped after
repeated consecutive failures; a moderator failure falls back to
deterministic round-robin; `ReasoningJury` proceeds on a Phase 1 juror
failure as long as a configured minimum still succeeds. None of this is
silent — every dropped juror ends up in `JuryVerdict.failed_jurors`.
Deliberately **not** built: exponential backoff on API throttling (a
transport concern `emissary` doesn't implement either), a wall-clock ceiling
on the Phase 1 fan-out, and the optional grounding-checker/evidence-arbiter
mechanisms the paper mentions only in passing as opt-in extensions beyond
its core pipeline.

## Roadmap — the paper's later sections, in order

- **§4 two-stage cascade — built** (`cascade.py`, `screen.py`), unverified
  against a live model. `calibration.py::sweep_thresholds` now measures
  escalation rate and missed-violation rate across candidate thresholds given
  a `Screen` and labelled exchanges — but it has only ever been run against
  the small synthetic fixtures in its own tests. `DEFAULT_THRESHOLD` is still
  the placeholder constant; setting a real value needs a real labelled
  dataset run through the sweep, which has not happened.
- **§5 linear probe — built** (`probe/`), in PyTorch, per project decision
  (see "Decisions" above: activation access lives in `doom`, not `emissary`,
  because emissary's wire adapters are HTTP-only and cannot expose hidden
  states). `Backbone` runs one frozen forward pass over a local HF
  `transformers` causal LM (`DOOM_PROBE_BACKBONE` env var) and pools the
  last-token hidden state; `ProbeHead` is a linear-or-shallow-MLP trained on
  cached activations (`train.py`, safetensors + MANIFEST.json checkpoints);
  `LinearProbe` exposes the same `exchange, constitution → score in [0,1]`
  shape `Screen` has, so it drops into `Cascade` with no change to `Cascade`
  itself. **Never run against a real backbone or real labelled data** — the
  training loop is verified with a real (but tiny, synthetic, CPU) training
  run in `tests/probe/test_train.py`; `Backbone` itself is exercised only on
  its pooling logic, never a real model download. The paper's sliding-window
  logit smoothing and softmax-weighted loss are not implemented — this is a
  plain linear probe, not the paper's full §5 recipe. The suggested
  off-the-shelf-classifier-over-vLLM intermediate step doesn't need new code:
  point `Screen` at a different `vllm:<model>` with an adapted prompt.
- **§6 weighted ensemble — built** (`ensemble.py::EnsembleScreen`):
  `z = alpha * z_probe + (1 - alpha) * z_screen`. Combines `Screen` and the
  §5 probe, not the probe and `ExchangeClassifier` — the judge is prompted
  and tool-forced, with no calibrated probability to blend (self-reported
  confidence is exactly the kind of number `screen.py`'s own docstring
  already rejects as uncalibrated). `EnsembleScreen` satisfies the same
  screen interface, so it too drops into `Cascade` unmodified. `alpha` is
  unset/untuned — that needs both a working probe and labelled data, neither
  of which exists yet in a real form.

## Testing

`uv run pytest` — mocked, no network, no key needed. Everything in `tests/`
except `tests/probe/` runs with only the `dev` dependency group. A live
end-to-end run needs a funded provider credential; the snippet is in
`README.md`. **As of the last session that run has never been executed** —
Kimi was the only key available and its account was out of balance. Do not
claim end-to-end verification until someone actually runs it.

`tests/probe/` (§5) needs the optional `probe` dependency group — `uv sync
--group probe` (pulls in torch/transformers/pyarrow/safetensors). Without it,
those tests skip cleanly via `pytest.importorskip` rather than failing; `uv
run pytest` stays green either way. Even with the group installed, this only
verifies the training/checkpoint/data-cache *logic* on synthetic data and a
hand-built pooling test — no test in this repo has ever loaded a real
backbone model.

`tests/jury/` needs no optional group — `doom.jury` has no new dependency.
Every test mocks `emissary.call_tool` (same pattern as `test_judge.py`) or
uses stub jurors/moderators (same pattern as `test_cascade.py`'s
`_StubScreen`); the deliberation turn loop, termination, and every
fault-tolerance path are exercised this way, with no real model call
anywhere.

## Dependencies

`emissary` (editable path dep, `../emissary`) is the only way this repo talks to
a model for the §3/§4 judges. Set `DOOM_JUDGE_PROVIDER` to pick one —
including `vllm:<model>` with `VLLM_BASE_URL` for a local server.

§5's probe (`doom.probe`) is a separate, optional `probe` dependency group
(torch, transformers, safetensors, pyarrow) — not a core dependency, and not
imported by `doom/__init__.py`. `Backbone`'s model comes from
`DOOM_PROBE_BACKBONE`, matching the `DOOM_JUDGE_PROVIDER`/
`DOOM_SCREEN_PROVIDER` convention. It does not go through `emissary` at all
(see the §5 activation-access decision above).

`doom.jury` needs no new dependency — every call routes through
`emissary.call_tool`, same as `judge.py`. `Juror`'s provider comes from
`DOOM_JURY_JUROR_PROVIDER`, `Consolidator`/`Moderator`'s from
`DOOM_JURY_MODERATOR_PROVIDER`, both defaulting to `anthropic` like the
other provider env vars in this project.
