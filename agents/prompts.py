from engine.board import CardColor, Team
from engine.game import Game

SPYMASTER_SYSTEM = """You are the spymaster in a game of Codenames, playing for team {team}.
You can see the secret color of every word on the board. Your job is to give one-word
clues that help your operative guess your team's remaining words without touching the
opposing team's words, the neutral words, or - fatally - the assassin word.

Rules for a valid clue:
- Exactly one English word, not a phrase, not hyphenated.
- The clue word must NOT appear on the board, and must not be a board word with a
  prefix/suffix added (no substrings either direction).
- `count` is how many of your team's words this clue relates to (0 means "one word,
  unlimited related guesses").

Always reply with ONLY a JSON object: {{"clue": "WORD", "count": N, "reasoning": "..."}}
"""

OPERATIVE_SYSTEM = """You are the operative for team {team} in a game of Codenames.
You do NOT know the secret colors. Your spymaster just gave you a clue and a count.
Guess board words you think relate to the clue, one at a time. Stop guessing (guess=null)
once you run out of good matches, even if you have guesses remaining - a wrong guess can
end your turn or reveal the opponent's word or the assassin.

Always reply with ONLY a JSON object: {{"guess": "WORD_ON_BOARD" or null, "reasoning": "..."}}
"""


def spymaster_system(team: Team) -> str:
    return SPYMASTER_SYSTEM.format(team=team.value)


def operative_system(team: Team) -> str:
    return OPERATIVE_SYSTEM.format(team=team.value)


def spymaster_user_turn(game: Game, team: Team) -> str:
    """Built fresh every turn and appended to the agent's growing conversation.
    Only the delta the model needs to act *now* - past turns are already in history."""
    my_color = CardColor.RED if team == Team.RED else CardColor.BLUE
    opp_color = CardColor.BLUE if team == Team.RED else CardColor.RED

    lines = []
    for c in game.board.cards:
        status = "REVEALED" if c.revealed else "hidden"
        lines.append(f"- {c.word}: {c.color.value} ({status})")

    remaining_mine = [c.word for c in game.board.cards if c.color == my_color and not c.revealed]
    remaining_opp = [c.word for c in game.board.cards if c.color == opp_color and not c.revealed]

    return (
        f"Current board state (word: true color):\n" + "\n".join(lines) +
        f"\n\nYour team's remaining words: {remaining_mine}"
        f"\nOpponent's remaining words: {remaining_opp}"
        "\n\nGive your next clue."
    )


def operative_user_turn(game: Game, guessed_this_turn: list[str]) -> str:
    unrevealed = game.board.unrevealed_words()
    return (
        f"Clue: {game.clue.word} {game.clue.count}\n"
        f"Guesses remaining this turn: {game.clue.guesses_left}\n"
        f"Already guessed this turn: {guessed_this_turn}\n"
        f"Unrevealed words on the board: {unrevealed}\n\n"
        "Give your next guess, or null to stop."
    )
