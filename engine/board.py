from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


class Team(str, Enum):
    RED = "RED"
    BLUE = "BLUE"


class CardColor(str, Enum):
    RED = "RED"
    BLUE = "BLUE"
    NEUTRAL = "NEUTRAL"
    ASSASSIN = "ASSASSIN"


@dataclass
class Card:
    word: str
    color: CardColor
    revealed: bool = False


@dataclass
class Board:
    cards: list[Card] = field(default_factory=list)

    def words(self) -> list[str]:
        return [c.word for c in self.cards]

    def unrevealed_words(self) -> list[str]:
        return [c.word for c in self.cards if not c.revealed]

    def card_for(self, word: str) -> Card | None:
        word = word.strip().upper()
        for c in self.cards:
            if c.word == word:
                return c
        return None

    def remaining_for(self, color: CardColor) -> int:
        return sum(1 for c in self.cards if c.color == color and not c.revealed)


def generate_board(word_pool: list[str], starting_team: Team, size: int = 25) -> Board:
    """9 cards for the starting team, 8 for the other, 7 neutral, 1 assassin (classic ratio)."""
    if size != 25:
        raise ValueError("only the classic 5x5 (25 card) layout is supported")

    words = random.sample(word_pool, size)

    starting_color = CardColor.RED if starting_team == Team.RED else CardColor.BLUE
    other_color = CardColor.BLUE if starting_team == Team.RED else CardColor.RED

    colors = (
        [starting_color] * 9
        + [other_color] * 8
        + [CardColor.NEUTRAL] * 7
        + [CardColor.ASSASSIN] * 1
    )
    random.shuffle(colors)

    cards = [Card(word=w, color=c) for w, c in zip(words, colors)]
    return Board(cards=cards)
