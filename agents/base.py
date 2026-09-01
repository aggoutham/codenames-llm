from __future__ import annotations

from abc import ABC, abstractmethod

from engine.board import Team
from engine.game import Game


class SpymasterAgent(ABC):
    team: Team

    @abstractmethod
    def get_clue(self, game: Game) -> tuple[str, int]:
        """Return (word, count) for the current board."""


class OperativeAgent(ABC):
    team: Team

    @abstractmethod
    def get_guess(self, game: Game) -> str | None:
        """Return a word to guess, or None to pass/stop guessing."""


class HumanSpymaster(SpymasterAgent):
    """Marker class: the UI collects the clue directly and calls game.give_clue().

    get_clue() is never actually invoked for a human seat - the Streamlit UI
    checks isinstance(agent, HumanSpymaster) and renders input widgets instead
    of calling the agent.
    """

    def __init__(self, team: Team):
        self.team = team

    def get_clue(self, game: Game) -> tuple[str, int]:
        raise NotImplementedError("human seats are driven by the UI, not called directly")


class HumanOperative(OperativeAgent):
    def __init__(self, team: Team):
        self.team = team

    def get_guess(self, game: Game) -> str | None:
        raise NotImplementedError("human seats are driven by the UI, not called directly")
