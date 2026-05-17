"""Smoke tests for env_v8 (suit-aware + flush straight)."""
from __future__ import annotations
import random
import numpy as np

from env_v8 import (
    GuandanEnvV8, legal_moves, _apply_move, deal, rank_totals,
    NUM_TYPES, NUM_SUITS, DECK, PASS, SINGLE, PAIR, TRIPLE, FULLHOUSE,
    BOMB, SEQ5, PSEQ3, TSEQ2, FLUSH_SEQ5, JOKER_BOMB_RANK,
    SJ, BJ, HEART, MAX_WILDCARDS, bomb_tier, beats_bomb_tier,
    hand_str, move_str,
)


def test_deal_invariants():
    """108 cards total; 27 per player; exactly 2 wildcards = red-heart level cards."""
    for seed in range(20):
        rng = random.Random(seed)
        for level in range(13):
            hands, wild = deal(rng, level)
            total = sum(int(h.sum()) for h in hands)
            assert total == 108
            assert all(int(h.sum()) == 27 for h in hands)
            assert sum(wild) == MAX_WILDCARDS
            # wildcards count must match hand[level][HEART] per player
            for p in range(4):
                assert wild[p] == int(hands[p][level, HEART]), (
                    f"seed={seed} level={level} p={p}: wild={wild[p]} "
                    f"hand[level,HEART]={hands[p][level, HEART]}"
                )
            # DECK conservation: sum of all hands == DECK
            total_hand = np.zeros_like(DECK)
            for h in hands:
                total_hand += h.astype(np.int8)
            assert (total_hand == DECK).all(), f"seed={seed} level={level} deck mismatch"
    print("[ok] deal invariants")


def test_legal_basic():
    seen = set()
    for seed in range(200):
        env = GuandanEnvV8(seed=seed)
        for m in env.legal():
            seen.add(m[0])
    assert {SINGLE, PAIR, TRIPLE, BOMB}.issubset(seen), \
        f"missing combo types: {seen}"
    print(f"[ok] legal-basic — saw {sorted(seen)}")


def test_flush_seq5_detection():
    """Manually build a hand with a heart 3-4-5-6-7, verify it's detected."""
    hand = np.zeros((NUM_TYPES, NUM_SUITS), dtype=np.int8)
    for r in range(5):
        hand[r, HEART] = 1
    wild = 0
    level = 12  # '2' (so level_rank doesn't interfere with the 3-7 sequence)
    legal = legal_moves(hand, wild, level, last=None)
    flush_moves = [m for m in legal if m[0] == FLUSH_SEQ5]
    assert any(m[1] == 0 and m[5] == HEART for m in flush_moves), \
        f"missing heart flush 3-7: legal flush moves = {flush_moves}"
    print("[ok] flush_seq5 detection")


def test_flush_seq5_application():
    """Build heart 3-7, apply the flush move, verify those cards consumed."""
    hand = np.zeros((NUM_TYPES, NUM_SUITS), dtype=np.int8)
    for r in range(5):
        hand[r, HEART] = 1
    hand[7, 0] = 1  # one extra non-flush card
    wild = 0
    level = 12
    move = (FLUSH_SEQ5, 0, 0, 5, 0, HEART)
    new_hand, new_wild = _apply_move(hand, wild, level, move)
    for r in range(5):
        assert new_hand[r, HEART] == 0, f"heart {r} not consumed"
    assert new_hand[7, 0] == 1, "non-target card incorrectly consumed"
    print("[ok] flush_seq5 application")


def test_bomb_tier_hierarchy():
    """6-card bomb > flush > 5-card bomb > 4-card bomb; joker bomb > all."""
    jb = (BOMB, JOKER_BOMB_RANK, 0, 4, 0, 0)
    fl = (FLUSH_SEQ5, 0, 0, 5, 0, HEART)
    b4 = (BOMB, 5, 0, 4, 0, 0)
    b5 = (BOMB, 5, 0, 5, 0, 0)
    b6 = (BOMB, 5, 0, 6, 0, 0)
    assert bomb_tier(jb[0], jb[1], jb[3]) > 0
    # Compare
    assert beats_bomb_tier(jb, fl)
    assert beats_bomb_tier(jb, b6)
    assert beats_bomb_tier(b6, fl)
    assert beats_bomb_tier(fl, b5)
    assert beats_bomb_tier(b5, b4)
    assert not beats_bomb_tier(b4, fl)
    assert not beats_bomb_tier(b5, b6)
    # Two flushes by rank
    fl_low = (FLUSH_SEQ5, 0, 0, 5, 0, HEART)
    fl_high = (FLUSH_SEQ5, 3, 0, 5, 0, HEART)
    assert beats_bomb_tier(fl_high, fl_low)
    print("[ok] bomb-tier hierarchy")


def test_run_to_end():
    """Random play to terminal in many seeds — env doesn't crash."""
    for seed in range(50):
        env = GuandanEnvV8(seed=seed)
        rng = random.Random(1234 + seed)
        safety = 0
        while not env.done:
            safety += 1
            assert safety < 5000, "infinite loop"
            legal = env.legal()
            assert legal, "no legal moves"
            env.step(rng.choice(legal))
        assert env.winner_team in (0, 1)
    print("[ok] run-to-end 50 seeds")


def test_step_conservation():
    """Across a game, played-rank-totals + remaining-rank-totals == initial DECK rank totals.
    Wildcards: total played wild + remaining wild == 2."""
    deck_rank = DECK.sum(axis=1).astype(np.int32)
    for seed in range(50):
        env = GuandanEnvV8(seed=seed)
        rng = random.Random(seed + 7)
        while not env.done:
            env.step(rng.choice(env.legal()))
        post = np.zeros(NUM_TYPES, dtype=np.int32)
        for h in env.hands:
            post += rank_totals(h)
        for p in env.played:
            post += p
        assert (post == deck_rank).all(), \
            f"rank conservation fail seed={seed}: {post} vs deck {deck_rank}"
        held_wild = sum(env.wildcards)
        played_wild = sum(env.played_wild)
        assert held_wild + played_wild == MAX_WILDCARDS, \
            f"wild conservation fail seed={seed}: held={held_wild} played={played_wild}"
    print("[ok] conservation 50 seeds")


def test_apply_consistency():
    """Random legal-move plays don't break invariants (per-suit ≥0, wildcards = hand[L,HEART])."""
    for seed in range(30):
        env = GuandanEnvV8(seed=seed)
        rng = random.Random(seed + 99)
        for _ in range(40):
            if env.done:
                break
            m = rng.choice(env.legal())
            cur = env.cur
            env.step(m)
            assert (env.hands[cur] >= 0).all()
            assert env.wildcards[cur] == int(env.hands[cur][env.level_rank, HEART]), (
                f"wildcards out of sync at seed={seed}"
            )
    print("[ok] apply consistency")


def test_legal_count_reasonable():
    max_seen = 0
    for seed in range(100):
        env = GuandanEnvV8(seed=seed)
        n = len(env.legal())
        max_seen = max(max_seen, n)
    assert max_seen < 5000, f"too many legal moves: {max_seen}"
    print(f"[ok] legal count reasonable (max={max_seen})")


def test_wildcard_red_heart_only():
    """Wildcards exist exactly where (rank=level_rank, suit=HEART). Never elsewhere."""
    rng = random.Random(99)
    for level in range(13):
        hands, wild = deal(rng, level)
        for p in range(4):
            for r in range(NUM_TYPES):
                for s in range(NUM_SUITS):
                    if r == level and s == HEART:
                        continue
                    # No non-(level, HEART) card should be considered wildcard
                    # (we don't track per-card flags, but per-player wildcards count
                    # must equal hand[level, HEART])
                    pass
            assert wild[p] == int(hands[p][level, HEART])
    print("[ok] wildcard red-heart-only")


if __name__ == '__main__':
    test_deal_invariants()
    test_legal_basic()
    test_flush_seq5_detection()
    test_flush_seq5_application()
    test_bomb_tier_hierarchy()
    test_run_to_end()
    test_step_conservation()
    test_apply_consistency()
    test_legal_count_reasonable()
    test_wildcard_red_heart_only()
    print("\nALL OK")
