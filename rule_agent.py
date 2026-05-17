"""Rule-based Guandan agent for v6 (level/wildcard rules)."""
from __future__ import annotations
import numpy as np

from env import legal_moves, PASS, BOMB, NUM_TYPES


def _team(idx: int) -> int:
    return idx % 2


def _min_opp_size(hand_sizes, my_idx) -> int:
    return min(int(hand_sizes[(my_idx + 1) % 4]), int(hand_sizes[(my_idx + 3) % 4]))


def rule_choose(hand: np.ndarray,
                wildcards: int,
                level_rank: int,
                last_play,
                last_player,
                hand_sizes: np.ndarray,
                my_idx: int):
    """Return a legal Move for player `my_idx`."""
    legal = legal_moves(hand, int(wildcards), int(level_rank), last_play)
    if not legal:
        return None

    free = (last_play is None) or (last_play[0] == PASS)

    if not free and last_player is not None and _team(last_player) == _team(my_idx) and last_player != my_idx:
        for m in legal:
            if m[0] == PASS:
                return m

    min_opp = _min_opp_size(hand_sizes, my_idx)
    threat = min_opp <= 5

    non_bomb = [m for m in legal if m[0] != BOMB and m[0] != PASS]
    bombs = [m for m in legal if m[0] == BOMB]
    pass_move = next((m for m in legal if m[0] == PASS), None)

    # Among same-effect moves, prefer the one that consumes the fewest wildcards.
    def _wild_cost(m):
        return m[4]

    if free:
        candidates = non_bomb if non_bomb else bombs
        candidates.sort(key=lambda m: (m[0], m[1], _wild_cost(m)))
        return candidates[0]

    if pass_move is not None and not threat:
        return pass_move

    if non_bomb:
        non_bomb.sort(key=lambda m: (m[0], m[1], _wild_cost(m)))
        return non_bomb[0]

    if bombs:
        bombs.sort(key=lambda m: (m[3], m[1], _wild_cost(m)))
        return bombs[0]

    return pass_move
