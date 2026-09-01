from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from engine.board import Board, CardColor, Team, generate_board


class Phase(str, Enum):
    CLUE = "CLUE"           # waiting on the current team's spymaster
    GUESS = "GUESS"         # waiting on the current team's operative
    GAME_OVER = "GAME_OVER"


@dataclass
class Clue:
    word: str
    count: int
    guesses_left: int


@dataclass
class LogEntry:
    team: Team
    role: str  # "spymaster" | "operative"
    text: str


def _team_color(team: Team) -> CardColor:
    return CardColor.RED if team == Team.RED else CardColor.BLUE


def _other_team(team: Team) -> Team:
    return Team.BLUE if team == Team.RED else Team.RED


class InvalidMove(Exception):
    pass


@dataclass
class Game:
    board: Board
    current_team: Team
    phase: Phase = Phase.CLUE
    clue: Clue | None = None
    winner: Team | None = None
    log: list[LogEntry] = field(default_factory=list)

    @classmethod
    def new(cls, word_pool: list[str], starting_team: Team = Team.RED) -> "Game":
        return cls(board=generate_board(word_pool, starting_team), current_team=starting_team)

    def give_clue(self, team: Team, word: str, count: int) -> None:
        if self.phase != Phase.CLUE:
            raise InvalidMove(f"not waiting on a clue (phase={self.phase})")
        if team != self.current_team:
            raise InvalidMove(f"it's {self.current_team}'s turn, not {team}'s")
        word = word.strip().upper()
        if " " in word or not word.isalpha():
            raise InvalidMove("clue must be a single alphabetic word")
        if word in self.board.words():
            raise InvalidMove("clue word cannot appear on the board")
        if count < 0:
            raise InvalidMove("count must be >= 0")

        self.clue = Clue(word=word, count=count, guesses_left=count + 1 if count > 0 else 1)
        self.phase = Phase.GUESS
        self.log.append(LogEntry(team, "spymaster", f"{word} {count}"))

    def guess(self, team: Team, word: str) -> CardColor:
        """Reveal a card. Returns the revealed color. Raises InvalidMove on bad input."""
        if self.phase != Phase.GUESS:
            raise InvalidMove(f"not waiting on a guess (phase={self.phase})")
        if team != self.current_team:
            raise InvalidMove(f"it's {self.current_team}'s turn, not {team}'s")
        card = self.board.card_for(word)
        if card is None:
            raise InvalidMove(f"{word!r} is not on the board")
        if card.revealed:
            raise InvalidMove(f"{word!r} is already revealed")

        card.revealed = True
        self.log.append(LogEntry(team, "operative", f"guessed {card.word} -> {card.color.value}"))
        self._check_win()
        if self.winner:
            self.phase = Phase.GAME_OVER
            return card.color

        my_color = _team_color(team)
        if card.color == CardColor.ASSASSIN:
            self.winner = _other_team(team)
            self.phase = Phase.GAME_OVER
        elif card.color == my_color:
            self.clue.guesses_left -= 1
            if self.clue.guesses_left <= 0:
                self._end_turn()
        else:
            # wrong team's card or neutral: turn ends immediately
            self._end_turn()

        return card.color

    def pass_turn(self, team: Team) -> None:
        """Operative voluntarily stops guessing."""
        if self.phase != Phase.GUESS:
            raise InvalidMove(f"not waiting on a guess (phase={self.phase})")
        if team != self.current_team:
            raise InvalidMove(f"it's {self.current_team}'s turn, not {team}'s")
        self.log.append(LogEntry(team, "operative", "passed"))
        self._end_turn()

    def _end_turn(self) -> None:
        self.current_team = _other_team(self.current_team)
        self.clue = None
        self.phase = Phase.CLUE

    def _check_win(self) -> None:
        if self.board.remaining_for(CardColor.RED) == 0:
            self.winner = Team.RED
        elif self.board.remaining_for(CardColor.BLUE) == 0:
            self.winner = Team.BLUE
