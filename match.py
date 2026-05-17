"""Multi-round Guandan Match: tribute + level-up rules.

A Match plays out rounds until one team's level passes 'A'. Each round is one
GuandanEnv with full_round=True so we get the complete 4-player finish order
(needed for 双下 / 单下 / 平 determination).

Levels are numbered 0..12 in this wrapper:
  level 0  = 起步 '2'   (env rank index 12)
  level 1  = '3'        (env rank index 0)
  level 2  = '4'        (env rank index 1)
  ...
  level 12 = 'A'        (env rank index 11)
Match ends when a team's level exceeds 12 (i.e. wins at 'A').

The level_rank for a round is determined by the ATTACKER team's level (real
Guandan: attacker = team that lost the previous round; first round defaults to
team 0). Tribute is applied between rounds before deal-modification:
  - 4th-place gives biggest non-wild, non-joker card to 1st-place
  - 1st-place returns lowest non-wild, non-joker card
  - Simplified: no anti-tribute (big-joker refusal) modelled.
"""
from __future__ import annotations
import random
from typing import Optional
import numpy as np

from env import (
    GuandanEnv, NUM_TYPES, SJ, BJ, MAX_WILDCARDS, RANK_NAMES,
)


def level_to_rank(level_idx: int) -> int:
    """Map level (0='2', 1='3', ..., 12='A') to env rank index."""
    if level_idx == 0:
        return 12  # '2'
    return level_idx - 1  # level 1..12 → rank 0..11


def rank_to_level_name(level_idx: int) -> str:
    """Human-readable level label."""
    if level_idx == 0:
        return '2'
    if 1 <= level_idx <= 12:
        return RANK_NAMES[level_idx - 1]
    return f'lvl{level_idx}'


def _apply_tribute(env: GuandanEnv, finish_order: list) -> Optional[dict]:
    """Mutate env.hands / env.wildcards to apply tribute exchange.

    Skips when 1st and 4th are on the same team (双下) — no tribute then.
    Returns a dict describing the transfer, or None if skipped.
    """
    first = finish_order[0]
    fourth = finish_order[-1]
    if first % 2 == fourth % 2:
        return None

    level_rank = env.level_rank

    def _highest_non_wild_non_joker(player):
        hand = env.hands[player]
        wild = env.wildcards[player]
        for r in range(12, -1, -1):  # iterate '2','A','K',...,'3'
            real = int(hand[r]) - (wild if r == level_rank else 0)
            if real > 0 and r not in (SJ, BJ):
                return r
        return None

    def _lowest_non_wild_non_joker(player):
        hand = env.hands[player]
        wild = env.wildcards[player]
        # Lowest '3'..'A' that isn't wildcard or joker, and not '2' / jokers.
        # Real Guandan: must return a card ≤ '10' (rank 7). Simplify: any non-wild non-joker.
        for r in range(0, 13):
            real = int(hand[r]) - (wild if r == level_rank else 0)
            if real > 0 and r not in (SJ, BJ):
                return r
        return None

    give = _highest_non_wild_non_joker(fourth)
    if give is None:
        return None
    ret = _lowest_non_wild_non_joker(first)
    if ret is None:
        return None
    if give == ret:
        return None  # degenerate; skip

    env.hands[fourth][give] -= 1
    env.hands[first][give] += 1
    env.hands[first][ret] -= 1
    env.hands[fourth][ret] += 1
    return {'giver': fourth, 'receiver': first, 'give': give, 'return': ret}


class GuandanMatch:
    """Multi-round match controller."""

    def __init__(self, seed: Optional[int] = None, max_rounds: int = 50):
        self.rng = random.Random(seed)
        self.team_levels = [0, 0]
        self.last_finish_order: Optional[list] = None
        self.history: list = []
        self.match_done = False
        self.winner_team: Optional[int] = None
        self.max_rounds = max_rounds

    def start_round(self) -> GuandanEnv:
        """Returns a new env for the next round. Apply tribute if applicable."""
        # Attacker = team of 4th-place finisher last round (else team 0 to start)
        if self.last_finish_order:
            attacker_team = self.last_finish_order[-1] % 2
        else:
            attacker_team = 0
        level_idx = self.team_levels[attacker_team]
        if level_idx > 12:
            level_idx = 12  # clamp
        env = GuandanEnv(seed=self.rng.randrange(2**31),
                         level_rank=level_to_rank(level_idx),
                         full_round=True)
        tribute = None
        if self.last_finish_order:
            tribute = _apply_tribute(env, self.last_finish_order)
        env._match_tribute = tribute  # type: ignore[attr-defined]
        return env

    def end_round(self, env: GuandanEnv):
        """Update levels + finish_order tracking after a round ends."""
        assert env.done and env.full_round
        finish_order = list(env.finish_order)
        assert len(finish_order) == 4, f"expected 4 finishers, got {len(finish_order)}"

        winner = finish_order[0] % 2
        # partner_pos: at what position did the winner's teammate finish (0..3)?
        partner_seat = (finish_order[0] + 2) % 4
        partner_pos = finish_order.index(partner_seat)

        if partner_pos == 1:
            jump = 3  # 双下
        elif partner_pos == 2:
            jump = 2  # 单下
        else:
            jump = 1  # 平

        self.team_levels[winner] += jump
        self.last_finish_order = finish_order
        self.history.append({
            'finish': finish_order,
            'winner_team': winner,
            'jump': jump,
            'tribute': getattr(env, '_match_tribute', None),
            'levels_after': list(self.team_levels),
        })
        if self.team_levels[winner] > 12:
            self.match_done = True
            self.winner_team = winner

    def play_round(self, policy_fn):
        """Convenience: play out one full round, calling policy_fn(env) -> move.
        Returns the env after termination."""
        env = self.start_round()
        while not env.done:
            move = policy_fn(env)
            env.step(move)
        self.end_round(env)
        return env

    def status(self) -> str:
        return (
            f"T0={rank_to_level_name(self.team_levels[0])}, "
            f"T1={rank_to_level_name(self.team_levels[1])}, "
            f"rounds={len(self.history)}"
        )


if __name__ == '__main__':
    # Quick smoke: random policy match, print level progression.
    rng = random.Random(0)
    def random_policy(env: GuandanEnv):
        legal = env.legal()
        return rng.choice(legal) if legal else None

    m = GuandanMatch(seed=0, max_rounds=20)
    for r in range(20):
        env = m.play_round(random_policy)
        print(f"round {r+1}: finish={env.finish_order}  →  {m.status()}")
        if m.match_done:
            print(f"MATCH OVER — team{m.winner_team} wins")
            break
