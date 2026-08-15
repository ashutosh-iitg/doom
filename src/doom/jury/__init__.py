"""Reasoning Jury: a second, sibling judge family in doom, evaluating
*reasoning quality* (where a chain-of-thought goes wrong) rather than
*policy compliance* (the §3/§4 judges' job). See CLAUDE.md's jury decision
for why this is a separate module tree sharing no types with
`constitution.py`/`verdict.py`/`panel.py`/`cascade.py`.

Not re-exported from `doom/__init__.py` — a large enough new surface that it
stays a deliberate `from doom.jury import ...` import, the same
namespace-hygiene call already made for `doom.probe`/`doom.ensemble`.
"""

from .consolidation import Consolidator
from .defect import Defect, Impact
from .deliberation import Deliberator, Moderator
from .jury import ReasoningJury
from .phase1 import Juror
from .trace import ReasoningTrace, segment_trace
from .verdict import JuryVerdict

__all__ = [
    "Consolidator",
    "Defect",
    "Deliberator",
    "Impact",
    "Juror",
    "JuryVerdict",
    "Moderator",
    "ReasoningJury",
    "ReasoningTrace",
    "segment_trace",
]
