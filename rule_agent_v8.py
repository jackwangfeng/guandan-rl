"""Rule-based agent for v8 env (handles 6-tuple moves and flush sequences)."""
from __future__ import annotations
import numpy as np

from env_v8 import legal_moves, PASS, BOMB, FLUSH_SEQ5, NUM_TYPES


def _team(idx): return idx % 2
def _min_opp_size(hand_sizes, my_idx):
    return min(int(hand_sizes[(my_idx + 1) % 4]), int(hand_sizes[(my_idx + 3) % 4]))


def rule_choose(hand: np.ndarray, wildcards: int, level_rank: int,
                last_play, last_player, hand_sizes: np.ndarray, my_idx: int):
    legal = legal_moves(hand, int(wildcards), int(level_rank), last_play)
    if not legal:
        return None
    free = (last_play is None) or (last_play[0] == PASS)

    # Don't beat teammate
    if not free and last_player is not None and _team(last_player) == _team(my_idx) and last_player != my_idx:
        for m in legal:
            if m[0] == PASS:
                return m

    min_opp = _min_opp_size(hand_sizes, my_idx)
    threat = min_opp <= 5

    # Categorize
    non_bomb_tier = [m for m in legal if m[0] not in (BOMB, FLUSH_SEQ5, PASS)]
    bomb_tier = [m for m in legal if m[0] in (BOMB, FLUSH_SEQ5)]
    pass_move = next((m for m in legal if m[0] == PASS), None)

    def wcost(m): return m[4]

    if free:
        candidates = non_bomb_tier if non_bomb_tier else bomb_tier
        candidates.sort(key=lambda m: (m[0], m[1], wcost(m)))
        return candidates[0]

    if pass_move is not None and not threat:
        return pass_move

    if non_bomb_tier:
        non_bomb_tier.sort(key=lambda m: (m[0], m[1], wcost(m)))
        return non_bomb_tier[0]

    if bomb_tier:
        # smallest bomb first
        bomb_tier.sort(key=lambda m: (m[3], m[1], wcost(m)))
        return bomb_tier[0]

    return pass_move
