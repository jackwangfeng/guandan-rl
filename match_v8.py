"""Multi-round Guandan match wrapper for v8 (suit-aware + flush)."""
from __future__ import annotations
import random
from typing import Optional
import numpy as np

from env_v8 import (
    GuandanEnvV8, NUM_TYPES, NUM_SUITS, SJ, BJ, MAX_WILDCARDS, RANK_NAMES,
    HEART, rank_totals,
)


def level_to_rank(level_idx: int) -> int:
    """level 0='2'→rank 12, level k(1..12)='3'..'A'→rank k-1."""
    if level_idx == 0:
        return 12
    return level_idx - 1


def rank_to_level_name(level_idx: int) -> str:
    if level_idx == 0:
        return '2'
    if 1 <= level_idx <= 12:
        return RANK_NAMES[level_idx - 1]
    return f'lvl{level_idx}'


def _apply_tribute(env: GuandanEnvV8, finish_order: list) -> Optional[dict]:
    """4th-place gives biggest non-wild non-joker card to 1st-place; 1st returns lowest.

    Skipped on 双下 (1st and 4th same team).
    Selects an arbitrary non-heart suit when multiple available to preserve wildcards.
    """
    first = finish_order[0]
    fourth = finish_order[-1]
    if first % 2 == fourth % 2:
        return None
    level_rank = env.level_rank

    def _pick_highest_non_wild_non_joker(player):
        hand = env.hands[player]
        for r in range(12, -1, -1):  # '2'='12','A'='11',...,'3'=0
            if r in (SJ, BJ):
                continue
            for s in range(NUM_SUITS):
                if r == level_rank and s == HEART:
                    continue
                if int(hand[r, s]) > 0:
                    return r, s
        return None

    def _pick_lowest_non_wild_non_joker(player):
        hand = env.hands[player]
        for r in range(0, 13):
            if r in (SJ, BJ):
                continue
            for s in range(NUM_SUITS):
                if r == level_rank and s == HEART:
                    continue
                if int(hand[r, s]) > 0:
                    return r, s
        return None

    give = _pick_highest_non_wild_non_joker(fourth)
    if give is None:
        return None
    ret = _pick_lowest_non_wild_non_joker(first)
    if ret is None:
        return None
    if give == ret:
        return None
    gr, gs = give
    rr, rs = ret
    env.hands[fourth][gr, gs] -= 1
    env.hands[first][gr, gs] += 1
    env.hands[first][rr, rs] -= 1
    env.hands[fourth][rr, rs] += 1
    return {'giver': fourth, 'receiver': first,
            'give': (gr, gs), 'return': (rr, rs)}


class GuandanMatchV8:
    """v8 multi-round match (full real-rules wildcards + flush)."""

    def __init__(self, seed: Optional[int] = None, max_rounds: int = 50):
        self.rng = random.Random(seed)
        self.team_levels = [0, 0]
        self.last_finish_order: Optional[list] = None
        self.history: list = []
        self.match_done = False
        self.winner_team: Optional[int] = None
        self.max_rounds = max_rounds

    def start_round(self) -> GuandanEnvV8:
        if self.last_finish_order:
            attacker_team = self.last_finish_order[-1] % 2
        else:
            attacker_team = 0
        level_idx = min(self.team_levels[attacker_team], 12)
        env = GuandanEnvV8(seed=self.rng.randrange(2**31),
                            level_rank=level_to_rank(level_idx),
                            full_round=True)
        tribute = None
        if self.last_finish_order:
            tribute = _apply_tribute(env, self.last_finish_order)
        env._match_tribute = tribute  # type: ignore[attr-defined]
        return env

    def end_round(self, env: GuandanEnvV8):
        assert env.done and env.full_round
        finish_order = list(env.finish_order)
        if len(finish_order) != 4:
            return
        winner = finish_order[0] % 2
        partner_seat = (finish_order[0] + 2) % 4
        partner_pos = finish_order.index(partner_seat)
        if partner_pos == 1:
            jump = 3   # 双下
        elif partner_pos == 2:
            jump = 2   # 单下
        else:
            jump = 1   # 平
        self.team_levels[winner] += jump
        self.last_finish_order = finish_order
        self.history.append({
            'finish': finish_order, 'winner_team': winner, 'jump': jump,
            'tribute': getattr(env, '_match_tribute', None),
            'levels_after': list(self.team_levels),
        })
        if self.team_levels[winner] > 12 or len(self.history) >= self.max_rounds:
            self.match_done = True
            self.winner_team = winner if self.team_levels[winner] > 12 \
                                else (0 if self.team_levels[0] > self.team_levels[1] else 1)

    def status(self) -> str:
        return (f"T0={rank_to_level_name(self.team_levels[0])}, "
                f"T1={rank_to_level_name(self.team_levels[1])}, "
                f"rounds={len(self.history)}")


if __name__ == '__main__':
    rng = random.Random(0)
    m = GuandanMatchV8(seed=0, max_rounds=20)
    for r in range(20):
        env = m.start_round()
        while not env.done:
            env.step(rng.choice(env.legal()))
        m.end_round(env)
        print(f"round {r+1}: finish={env.finish_order}  →  {m.status()}")
        if m.match_done:
            print(f"MATCH OVER — team{m.winner_team} wins")
            break
