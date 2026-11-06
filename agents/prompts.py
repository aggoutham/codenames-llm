import json

from engine.board import CardColor, Team
from engine.game import Game

SPYMASTER_SYSTEM = """You are the spymaster in a game of Codenames, playing for team {team}.
You can see the secret color of every word on the board. Your job is to give one-word
clues that help your operative guess your team's remaining words without touching the
opposing team's words, the neutral words, or - fatally - the assassin word.

Priority order, most important first:
1. NEVER give a clue that could plausibly lead your operative to guess the assassin word.
   This is an instant loss - it is far worse than an inefficient or overly cautious clue.
2. Avoid leading them to the opponent's words or neutral words.
3. Only after 1 and 2 are satisfied, prefer clues that relate to more of your own words.

Rules for a valid clue:
- Exactly one English word, not a phrase, not hyphenated.
- The clue word must NOT appear on the board, and must not be a board word with a
  prefix/suffix added (no substrings either direction).
- `count` is how many of your team's words this clue relates to (0 means "one word,
  unlimited related guesses").

Each turn you'll be given the game history so far and the current board state. The
current state only ever lists words still in play - once a word is revealed it's gone
for good and can never be guessed again, so don't spend a clue on it.

Always reply with ONLY a JSON object: {{"clue": "WORD", "count": N, "reasoning": "..."}}
"""

OPERATIVE_SYSTEM = """You are the operative for team {team} in a game of Codenames.
You do NOT know the secret colors. Your spymaster just gave you a clue and a count.
Guess board words you think relate to the clue, one at a time. Stop guessing (guess=null)
once you run out of good matches, even if you have guesses remaining - a wrong guess can
end your turn or reveal the opponent's word or the assassin.

Each turn you'll be given the game history so far and the current board state. Only
words still listed as unrevealed are valid guesses.

Always reply with ONLY a JSON object: {{"guess": "WORD_ON_BOARD" or null, "reasoning": "..."}}
"""


def spymaster_system(team: Team) -> str:
    return SPYMASTER_SYSTEM.format(team=team.value)


def operative_system(team: Team) -> str:
    return OPERATIVE_SYSTEM.format(team=team.value)


def format_history(game: Game) -> list[dict]:
    """Compact step-by-step log instead of a growing raw transcript - this is what gives
    the model continuity across turns, without re-sending old board-state dumps that can
    contradict the current (authoritative) state. Excludes the UI-only "-reasoning" log
    rows that ui/app.py appends for display purposes."""
    return [
        {"step": i, "team": entry.team.value, "role": entry.role, "action": entry.text}
        for i, entry in enumerate(
            (e for e in game.log if e.role in ("spymaster", "operative")), start=1
        )
    ]


def spymaster_user_turn(game: Game, team: Team) -> str:
    """Built fresh every call - NOT appended to a growing conversation. History gives
    continuity; the state below is always the authoritative, current truth."""
    my_color = CardColor.RED if team == Team.RED else CardColor.BLUE
    opp_color = CardColor.BLUE if team == Team.RED else CardColor.RED

    remaining_mine = [c.word for c in game.board.cards if c.color == my_color and not c.revealed]
    remaining_opp = [c.word for c in game.board.cards if c.color == opp_color and not c.revealed]
    remaining_neutral = [c.word for c in game.board.cards if c.color == CardColor.NEUTRAL and not c.revealed]
    assassin_word = next(c.word for c in game.board.cards if c.color == CardColor.ASSASSIN and not c.revealed)

    history = format_history(game)

    return (
        f"Game history so far, in order (JSON):\n{json.dumps(history)}\n\n"
        "Current state - ONLY unrevealed words are listed below; anything already "
        "revealed is gone for good, ignore it entirely:\n"
        f"Your team's remaining words: {remaining_mine}\n"
        f"Opponent's remaining words: {remaining_opp}\n"
        f"Neutral remaining words: {remaining_neutral}\n"
        f"Assassin word (instant loss if guessed): {assassin_word}\n\n"
        "Give your next clue."
    )


def spymaster_fallback_user_turn(game: Game, team: Team) -> str:
    """Used after the holistic multi-word search has repeatedly failed. Narrows the ask
    to a single easy word instead of asking the model to solve the same hard search again."""
    my_color = CardColor.RED if team == Team.RED else CardColor.BLUE
    opp_color = CardColor.BLUE if team == Team.RED else CardColor.RED

    remaining_mine = [c.word for c in game.board.cards if c.color == my_color and not c.revealed]
    remaining_opp = [c.word for c in game.board.cards if c.color == opp_color and not c.revealed]
    remaining_neutral = [c.word for c in game.board.cards if c.color == CardColor.NEUTRAL and not c.revealed]
    assassin_word = next(c.word for c in game.board.cards if c.color == CardColor.ASSASSIN and not c.revealed)

    return (
        "You were unable to find a safe multi-word clue. Stop trying to be clever - pick "
        "just ONE word from your team's remaining words below (whichever is easiest to "
        "clue for), and give the most direct, obvious single-word synonym or association "
        "for it. count must be exactly 1.\n\n"
        f"Your team's remaining words (pick ONE to target): {remaining_mine}\n"
        "Words your clue must NOT point to:\n"
        f"  - Opponent's words: {remaining_opp}\n"
        f"  - Neutral words: {remaining_neutral}\n"
        f"  - THE ASSASSIN WORD (instant loss - avoid this above all else): {assassin_word}\n\n"
        "Give your clue now."
    )


def operative_user_turn(game: Game, guessed_this_turn: list[str]) -> str:
    unrevealed = game.board.unrevealed_words()
    history = format_history(game)

    return (
        f"Game history so far, in order (JSON):\n{json.dumps(history)}\n\n"
        f"Current clue: {game.clue.word} {game.clue.count}\n"
        f"Guesses remaining this turn: {game.clue.guesses_left}\n"
        f"Already guessed this turn: {guessed_this_turn}\n"
        f"Unrevealed words still on the board (only these are valid guesses): {unrevealed}\n\n"
        "Give your next guess, or null to stop."
    )
