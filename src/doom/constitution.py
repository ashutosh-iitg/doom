from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Constitution:
    """A named, ordered set of rules a judge evaluates an exchange against."""

    name: str
    rules: tuple[Rule, ...]

    def render(self) -> str:
        return "\n\n".join(f"{rule.id}. {rule.title}\n{rule.text}" for rule in self.rules)
