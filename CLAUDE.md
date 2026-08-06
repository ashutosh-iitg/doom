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
Cunningham et al., *Constitutional Classifiers++* (`2601.04603v1.pdf`, in this
repo) — **read the paper before changing the judging architecture**; the section
numbers below refer to it.

## Where the project actually is

Built and tested: the paper's **§3 exchange classifier**, at the prompting level.
`ExchangeClassifier.judge` sends one full exchange plus a constitution to a judge
model and gets back a structured `Verdict`. `Panel` holds a sequence of judges
and flags if any one does.

Not built: everything else in the paper. Do not describe this repo as
implementing Constitutional Classifiers — it implements the simplest of the four
systems the paper compares, without the fine-tuning, cascade, or probes that
make the production version what it is.

## The shape

```
exchange.py       Exchange, Turn — the conversation under evaluation
constitution.py   Constitution, Rule — the ruleset judged against
constitutions/    cbrn_example.py — ILLUSTRATIVE ONLY, not policy
verdict.py        Verdict — flagged, reasoning, rule_ids
judge.py          ExchangeClassifier — one prompted judge, via emissary
panel.py          Panel — sequence of judges; the seam for what comes next
```

Data flow: `Exchange` + `Constitution` → `ExchangeClassifier.judge` → one
tool-forced `emissary.call_tool` → `Verdict`. `Panel` fans that over several
judges. Nothing here holds state between calls.

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
and does not pretend to be. §4 (cheap judge screens, expensive judge escalates)
and §6 (weighted probe + classifier logits) belong in a routing policy *in front
of* `Panel.evaluate`, leaving `Verdict` and `ExchangeClassifier` untouched.
Adding weighting or routing to `Panel` itself would couple the two.

**The constitution is cache-marked; the exchange is not.** The constitution is
identical across every exchange a judge evaluates, the exchange is unique per
call. Same reasoning as stria's per-provision extraction calls.

**`cbrn_example.py` is an example, and must stay labelled as one.** Five rules
modelled on the paper's CBRN framing so the classifier is runnable end to end.
It is not a production policy and must never be presented as one. Rules name
what is disallowed and why — never operational detail.

## Roadmap — the paper's later sections, in order

- **§4** two-stage cascade: a cheap first-stage judge screens all traffic; only
  flagged exchanges reach an expensive second stage. Escalating rather than
  refusing is what lets stage one tolerate a high false-positive rate.
- **§5** linear probes over a local open-weight model's activations, with
  sliding-window logit smoothing and softmax-weighted loss. This needs model
  internals, which is why `emissary`'s vLLM support matters — an API-only
  provider cannot give you activations.
- **§6** weighted ensemble of probe and classifier scores.

## Testing

`uv run pytest` — mocked, no network, no key needed. A live end-to-end run needs
a funded provider credential; the snippet is in `README.md`. **As of the last
session that run has never been executed** — Kimi was the only key available and
its account was out of balance. Do not claim end-to-end verification until
someone actually runs it.

## Dependencies

`emissary` (editable path dep, `../emissary`) is the only way this repo talks to
a model. Set `DOOM_JUDGE_PROVIDER` to pick one — including `vllm:<model>` with
`VLLM_BASE_URL` for a local server.
