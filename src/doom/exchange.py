from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Turn:
    role: Role
    content: str


@dataclass(frozen=True)
class Exchange:
    """A full conversational exchange, evaluated together.

    The paper's exchange classifiers (§3) read outputs in the context of
    their inputs specifically because a request fragmented across turns, or
    an answer obfuscated in metaphor or coded substitution, only reads as
    harmful next to the turns around it — not in isolation.
    """

    turns: tuple[Turn, ...]

    @classmethod
    def of(cls, *, user: str, assistant: str, system: str | None = None) -> "Exchange":
        """The common case: one user turn, one assistant turn, an optional
        system prompt."""
        turns: list[Turn] = []
        if system:
            turns.append(Turn("system", system))
        turns.append(Turn("user", user))
        turns.append(Turn("assistant", assistant))
        return cls(tuple(turns))

    def render(self) -> str:
        return "\n\n".join(f"[{turn.role}]\n{turn.content}" for turn in self.turns)
