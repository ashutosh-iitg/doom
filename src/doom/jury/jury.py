"""`ReasoningJury`: orchestrates Phase 1 fan-out (with a minimum-juror
threshold) and delegates consensus to whichever Phase 2 policy it's given —
`Consolidator` or `Deliberator`, both implementing the same
`run(trace, phase1) -> JuryVerdict` shape. Swapping the policy is passing a
different object, not branching on a mode flag — the same structural-typing
move `Screen`/`LinearProbe`/`EnsembleScreen` already share via
`calibration.ScreenLike`.
"""

import logging
from dataclasses import dataclass, replace
from typing import Protocol

from .defect import Defect
from .phase1 import Juror
from .trace import ReasoningTrace
from .verdict import JuryVerdict

log = logging.getLogger(__name__)

Phase1Result = tuple[tuple[str, tuple[Defect, ...]], ...]


class Phase2Policy(Protocol):
    def run(self, trace: ReasoningTrace, phase1: Phase1Result) -> JuryVerdict: ...


@dataclass(frozen=True)
class ReasoningJury:
    jurors: tuple[Juror, ...]
    phase2: Phase2Policy
    min_jurors: int = 1

    def evaluate(self, trace: ReasoningTrace) -> JuryVerdict:
        phase1, failed = self._run_phase1(trace)
        verdict = self.phase2.run(trace, phase1)
        if failed:
            verdict = replace(verdict, failed_jurors=verdict.failed_jurors + failed)
        return verdict

    def _run_phase1(self, trace: ReasoningTrace) -> tuple[Phase1Result, tuple[str, ...]]:
        results: list[tuple[str, tuple[Defect, ...]]] = []
        failed: list[str] = []
        for juror in self.jurors:
            try:
                results.append((juror.juror_id, juror.judge(trace)))
            # Any juror failure degrades Phase 1 to its remaining successes (Appendix F).
            except Exception as exc:  # noqa: BLE001
                log.warning(f"juror {juror.juror_id} failed Phase 1: {exc}")
                failed.append(juror.juror_id)
        if len(results) < self.min_jurors:
            raise RuntimeError(
                f"only {len(results)}/{len(self.jurors)} jurors succeeded in Phase 1, "
                f"minimum is {self.min_jurors} — failed: {failed}"
            )
        return tuple(results), tuple(failed)
