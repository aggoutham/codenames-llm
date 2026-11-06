from __future__ import annotations

import time

from engine.board import Team
from engine.game import Game
from llm.client import LLMClient

from .base import OperativeAgent, SpymasterAgent
from .prompts import (
    operative_system,
    operative_user_turn,
    spymaster_fallback_user_turn,
    spymaster_system,
    spymaster_user_turn,
)

# Absolute last resort if BOTH LLM tiers fail - guaranteed-legal, verified against the
# word pool (words.py) to contain none of these. A spymaster must never skip a turn;
# this is what makes that guarantee unconditional rather than "usually."
_DETERMINISTIC_FALLBACK_WORDS = ["THING", "ITEM", "OBJECT", "STUFF", "IDEA", "CONCEPT"]


class LLMSpymaster(SpymasterAgent):
    def __init__(self, team: Team, client: LLMClient, max_attempts: int = 3):
        self.team = team
        self.client = client
        self.max_attempts = max_attempts
        # Said once, reused every call. NOT a growing conversation - each call below builds
        # a fresh [system, user] pair from the CURRENT game state + a compact history log
        # (agents/prompts.py:format_history), instead of resending an ever-larger transcript
        # full of now-stale board snapshots. Within-call retries are still a short back-and-
        # forth (the model needs to see its own rejected attempt to correct it), they just
        # don't persist into the next turn.
        self.system_message = {"role": "system", "content": spymaster_system(team)}
        self.last_reasoning: str | None = None
        self.last_duration_sec: float = 0.0
        self.last_attempts: int = 0
        self.last_usage: dict = {}
        self.last_tier: str = ""  # "holistic" | "safe_single" | "deterministic"

    def get_clue(self, game: Game) -> tuple[str, int]:
        start = time.perf_counter()
        board_words = set(game.board.words())
        tag = f"{self.team.value}-spymaster"

        holistic = self._try_holistic(game, board_words, tag)
        if holistic is not None:
            word, count, reasoning, attempts, tier = holistic
        else:
            safe = self._try_safe_single_card(game, board_words, tag)
            if safe is not None:
                word, count, reasoning, safe_attempts, tier = safe
                attempts = self.max_attempts + safe_attempts  # + the failed holistic tries
            else:
                word, count, reasoning, tier = self._deterministic_fallback(board_words)
                attempts = self.max_attempts + 2  # both LLM tiers fully exhausted

        self.last_reasoning = reasoning
        self.last_tier = tier
        self.last_attempts = attempts
        self.last_duration_sec = time.perf_counter() - start
        return word, count

    def _try_holistic(self, game: Game, board_words: set[str], tag: str):
        """Tier 1 (cheap, usual case): one holistic ask across the whole board,
        may relate to multiple words. Up to self.max_attempts tries."""
        messages = [self.system_message, {"role": "user", "content": spymaster_user_turn(game, self.team)}]

        for attempt in range(self.max_attempts):
            data, raw, usage = self.client.chat_json(messages, tag=tag)
            messages.append({"role": "assistant", "content": raw})
            self.last_usage = usage

            word = str(data.get("clue", "")).strip().upper()
            count = int(data.get("count", 0) or 0)

            if not word.isalpha():
                messages.append(
                    {"role": "user", "content": f"'{word}' is not a single alphabetic word. Try again."}
                )
                continue
            if any(word in w or w in word for w in board_words):
                messages.append(
                    {"role": "user", "content": f"'{word}' overlaps a board word. Choose a different word. Try again."}
                )
                continue

            return word, count, data.get("reasoning"), attempt + 1, "holistic"
        return None

    def _try_safe_single_card(self, game: Game, board_words: set[str], tag: str):
        """Tier 2 (fallback): narrow the ask to ONE easy, safe word instead of re-asking
        the same hard multi-word search. Small retry budget - this task is much easier."""
        messages = [self.system_message, {"role": "user", "content": spymaster_fallback_user_turn(game, self.team)}]
        fallback_attempts = 2

        for attempt in range(fallback_attempts):
            data, raw, usage = self.client.chat_json(messages, tag=f"{tag}-fallback")
            messages.append({"role": "assistant", "content": raw})
            self.last_usage = usage

            word = str(data.get("clue", "")).strip().upper()

            if not word.isalpha():
                messages.append(
                    {"role": "user", "content": f"'{word}' is not a single alphabetic word. One safe word, try again."}
                )
                continue
            if any(word in w or w in word for w in board_words):
                messages.append(
                    {"role": "user", "content": f"'{word}' overlaps a board word. Pick a different safe word."}
                )
                continue

            return word, 1, data.get("reasoning"), attempt + 1, "safe_single"
        return None

    def _deterministic_fallback(self, board_words: set[str]) -> tuple[str, int, str, str]:
        """Tier 3: no LLM call at all. Guarantees a legal clue exists no matter what."""
        for word in _DETERMINISTIC_FALLBACK_WORDS:
            if not any(word in w or w in word for w in board_words):
                return word, 1, "(deterministic fallback - both LLM tiers failed to produce a valid clue)", "deterministic"
        # Should be unreachable given words.py doesn't contain any of the fallback words,
        # but never return something that could violate give_clue()'s own rules.
        return "MISCELLANEOUS", 1, "(last-resort fallback)", "deterministic"


class LLMOperative(OperativeAgent):
    def __init__(self, team: Team, client: LLMClient, max_attempts: int = 3):
        self.team = team
        self.client = client
        self.max_attempts = max_attempts
        self.system_message = {"role": "system", "content": operative_system(team)}
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
        messages = [
            self.system_message,
            {"role": "user", "content": operative_user_turn(game, self._guessed_this_turn)},
        ]
        unrevealed = set(game.board.unrevealed_words())
        tag = f"{self.team.value}-operative"

        for attempt in range(self.max_attempts):
            data, raw, usage = self.client.chat_json(messages, tag=tag)
            messages.append({"role": "assistant", "content": raw})
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

            messages.append(
                {"role": "user", "content": f"'{guess}' is not an unrevealed board word. Try again or use null."}
            )

        self.last_attempts = self.max_attempts
        self.last_duration_sec = time.perf_counter() - start
        return None
