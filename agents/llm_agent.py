from __future__ import annotations

import time

from engine.board import Team
from engine.game import Game
from llm.client import LLMClient

from .base import OperativeAgent, SpymasterAgent
from .prompts import (
    operative_system,
    operative_user_turn,
    spymaster_system,
    spymaster_user_turn,
)


class LLMSpymaster(SpymasterAgent):
    def __init__(self, team: Team, client: LLMClient, max_attempts: int = 3):
        self.team = team
        self.client = client
        self.max_attempts = max_attempts
        # The system prompt is said ONCE, here. Everything after this is turns
        # appended to this same list - this list IS the model's memory.
        self.messages: list[dict] = [{"role": "system", "content": spymaster_system(team)}]
        self.last_reasoning: str | None = None
        self.last_duration_sec: float = 0.0
        self.last_attempts: int = 0
        self.last_usage: dict = {}

    def get_clue(self, game: Game) -> tuple[str, int]:
        start = time.perf_counter()
        self.messages.append({"role": "user", "content": spymaster_user_turn(game, self.team)})
        board_words = set(game.board.words())

        tag = f"{self.team.value}-spymaster"
        for attempt in range(self.max_attempts):
            data, raw, usage = self.client.chat_json(self.messages, tag=tag)
            self.messages.append({"role": "assistant", "content": raw})
            self.last_usage = usage

            word = str(data.get("clue", "")).strip().upper()
            count = int(data.get("count", 0))

            if not word.isalpha():
                self.messages.append(
                    {"role": "user", "content": f"'{word}' is not a single alphabetic word. Try again."}
                )
                continue
            if any(word in w or w in word for w in board_words):
                self.messages.append(
                    {"role": "user", "content": f"'{word}' overlaps a board word. Choose a different word. Try again."}
                )
                continue

            self.last_reasoning = data.get("reasoning")
            self.last_attempts = attempt + 1
            self.last_duration_sec = time.perf_counter() - start
            return word, count

        self.last_reasoning = "(fell back to PASS after repeated invalid clues)"
        self.last_attempts = self.max_attempts
        self.last_duration_sec = time.perf_counter() - start
        return "PASS", 0


class LLMOperative(OperativeAgent):
    def __init__(self, team: Team, client: LLMClient, max_attempts: int = 3):
        self.team = team
        self.client = client
        self.max_attempts = max_attempts
        self.messages: list[dict] = [{"role": "system", "content": operative_system(team)}]
        self._guessed_this_turn: list[str] = []
        self.last_reasoning: str | None = None
        self.last_duration_sec: float = 0.0
        self.last_attempts: int = 0
        self.last_usage: dict = {}

    def get_guess(self, game: Game) -> str | None:
        if game.clue is None:
            return None
        if game.log and game.log[-1].role == "spymaster":
            self._guessed_this_turn = []  # new turn started

        start = time.perf_counter()
        self.messages.append(
            {"role": "user", "content": operative_user_turn(game, self._guessed_this_turn)}
        )
        unrevealed = set(game.board.unrevealed_words())
        tag = f"{self.team.value}-operative"

        for attempt in range(self.max_attempts):
            data, raw, usage = self.client.chat_json(self.messages, tag=tag)
            self.messages.append({"role": "assistant", "content": raw})
            self.last_usage = usage

            self.last_reasoning = data.get("reasoning")
            guess = data.get("guess")
            if guess is None:
                self.last_attempts = attempt + 1
                self.last_duration_sec = time.perf_counter() - start
                return None
            guess = str(guess).strip().upper()
            if guess in unrevealed:
                self._guessed_this_turn.append(guess)
                self.last_attempts = attempt + 1
                self.last_duration_sec = time.perf_counter() - start
                return guess

            self.messages.append(
                {"role": "user", "content": f"'{guess}' is not an unrevealed board word. Try again or use null."}
            )

        self.last_attempts = self.max_attempts
        self.last_duration_sec = time.perf_counter() - start
        return None
