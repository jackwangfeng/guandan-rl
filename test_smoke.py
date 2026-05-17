"""Smoke tests for v6 env (level/wildcard rules)."""
from __future__ import annotations
import random
import numpy as np

from env import (
    GuandanEnv, legal_moves, _apply_move_to_hand, deal,
    NUM_TYPES, DECK_COUNTS, PASS, SINGLE, PAIR, TRIPLE, FULLHOUSE, BOMB,
    SEQ5, PSEQ3, TSEQ2, JOKER_BOMB_RANK, SJ, BJ, MAX_WILDCARDS,
    hand_str, move_str,
)


def test_deal_invariants():
    for seed in range(20):
        rng = random.Random(seed)
        for level in range(13):
            hands, wild = deal(rng, level)
            total = sum(h.sum() for h in hands)
            assert total == sum(DECK_COUNTS) == 108
            assert all(h.sum() == 27 for h in hands)
            assert sum(wild) == MAX_WILDCARDS
            for p in range(4):
                assert hands[p][level] >= wild[p], (
                    f"player {p} wildcards={wild[p]} exceed hand[{level}]={hands[p][level]}"
                )
    print("[ok] deal invariants")


def test_legal_basic():
    seen = set()
    for seed in range(200):
        env = GuandanEnv(seed=seed)
        for m in env.legal():
            seen.add(m[0])
    assert {SINGLE, PAIR}.issubset(seen), f"missing combo types: {{SINGLE, PAIR}} - {seen}"
    print(f"[ok] legal-basic — saw combo types {sorted(seen)}")


def test_run_to_end():
    for seed in range(50):
        env = GuandanEnv(seed=seed)
        rng = random.Random(1234 + seed)
        safety = 0
        while not env.done:
            safety += 1
            assert safety < 5000, "infinite loop"
            legal = env.legal()
            assert legal
            env.step(rng.choice(legal))
        assert env.winner_team in (0, 1)
    print("[ok] run-to-end 50 seeds")


def test_step_conservation():
    for seed in range(50):
        env = GuandanEnv(seed=seed)
        init_total = np.zeros(NUM_TYPES, dtype=np.int32)
        for h in env.hands:
            init_total += h.astype(np.int32)
        assert (init_total == np.array(DECK_COUNTS, dtype=np.int32)).all()
        rng = random.Random(seed + 7)
        while not env.done:
            env.step(rng.choice(env.legal()))
        post = np.zeros(NUM_TYPES, dtype=np.int32)
        for h in env.hands:
            post += h.astype(np.int32)
        for p in env.played:
            post += p
        assert (post == np.array(DECK_COUNTS, dtype=np.int32)).all(), \
            f"conservation fail seed={seed}: {post}"
        held_wild = sum(env.wildcards)
        played_wild = sum(env.played_wild)
        assert held_wild + played_wild == MAX_WILDCARDS, \
            f"wild conservation fail seed={seed}: held={held_wild} played={played_wild}"
    print("[ok] conservation (rank + wildcards) 50 seeds")


def test_wildcard_substitution():
    """Hand: 1 real '7' + 2 wildcards (level=8). Pair-of-7 with 1 wild should be legal,
    pair-of-7 with 2 wilds should NOT (need ≥1 real card)."""
    level = 5  # rank=5 → '8'
    hand = np.zeros(NUM_TYPES, dtype=np.int8)
    hand[4] = 1  # one '7'
    hand[5] = 2  # two '8's, both wildcards
    wild = 2
    moves = legal_moves(hand, wild, level, last=None)
    pair_7_subs = [m for m in moves if m[0] == PAIR and m[1] == 4]
    assert any(m[4] == 1 for m in pair_7_subs), \
        f"missing pair-of-7 with 1 wild: {[move_str(m, level) for m in moves]}"
    assert not any(m[4] == 2 for m in pair_7_subs), \
        "pair-of-7 with 2 wilds should be illegal"
    print("[ok] wildcard pair substitution")


def test_wildcard_bomb():
    """2 regular Qs + 2 wildcards = 4-card Q bomb."""
    level = 0  # level=3 → wildcards are 3s
    hand = np.zeros(NUM_TYPES, dtype=np.int8)
    hand[9] = 2  # Q
    hand[0] = 2  # 2 wildcards
    wild = 2
    moves = legal_moves(hand, wild, level, last=None)
    q_bombs = [m for m in moves if m[0] == BOMB and m[1] == 9 and m[3] == 4]
    assert q_bombs, f"missing Q-bomb-w/-wildcards: {[move_str(m, level) for m in moves]}"
    print("[ok] wildcard bomb")


def test_apply_consistency():
    for seed in range(30):
        env = GuandanEnv(seed=seed)
        rng = random.Random(seed + 99)
        for _ in range(20):
            if env.done:
                break
            legal = env.legal()
            m = rng.choice(legal)
            cur = env.cur
            old_hand = env.hands[cur].copy()
            old_wild = env.wildcards[cur]
            env.step(m)
            if m[0] == PASS:
                assert (env.hands[cur] == old_hand).all()
                assert env.wildcards[cur] == old_wild
            else:
                assert (env.hands[cur] >= 0).all()
                assert env.wildcards[cur] >= 0
                assert env.wildcards[cur] <= env.hands[cur][env.level_rank]
    print("[ok] apply consistency")


def test_legal_count_reasonable():
    max_seen = 0
    for seed in range(100):
        env = GuandanEnv(seed=seed)
        n = len(env.legal())
        max_seen = max(max_seen, n)
    assert max_seen < 4000, f"too many legal moves: {max_seen}"
    print(f"[ok] legal count reasonable (max={max_seen})")


if __name__ == '__main__':
    test_deal_invariants()
    test_legal_basic()
    test_run_to_end()
    test_step_conservation()
    test_wildcard_substitution()
    test_wildcard_bomb()
    test_apply_consistency()
    test_legal_count_reasonable()
    print("\nALL OK")
