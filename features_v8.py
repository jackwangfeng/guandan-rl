"""Feature encoding for v8 env (suit-aware + 同花顺)."""
from __future__ import annotations
import numpy as np
from env_v8 import (
    NUM_TYPES, NUM_SUITS, NUM_COMBO_TYPES, SJ, BJ, JOKER_BOMB_RANK, MAX_WILDCARDS,
    PASS, SINGLE, PAIR, TRIPLE, FULLHOUSE, BOMB,
    SEQ5, PSEQ3, TSEQ2, FLUSH_SEQ5, SEQ5_LEN, PSEQ3_LEN, TSEQ2_LEN,
    rank_totals,
)

# State:
#   own_hand_rank_total: 15      (sum over suits)
#   own_hand_per_suit:   60      (15×4 — flatten)
#   last_play_counts:    15      (rank counts of last move)
#   last_combo_onehot:   10      (NUM_COMBO_TYPES)
#   last_suit_onehot:    4       (only meaningful for FLUSH_SEQ5)
#   hand_sizes (rel):    3
#   played_per_seat:     60      (15×4, rank-sum)
#   level_rank_onehot:   15
#   own_wildcards:       1
#   wildcards_left:      1
#   played_wildcards:    4
# = 15+60+15+10+4+3+60+15+1+1+4 = 188
STATE_DIM = NUM_TYPES + NUM_TYPES * NUM_SUITS + NUM_TYPES + NUM_COMBO_TYPES + NUM_SUITS \
            + 3 + 4 * NUM_TYPES + NUM_TYPES + 1 + 1 + 4

# Action:
#   combo_onehot: 10
#   rank_onehot:  16  (+1 joker bomb sentinel)
#   pair_rank:    16
#   count:        1
#   n_wild:       3   (0/1/2)
#   suit_onehot:  4
# = 50
ACTION_DIM = NUM_COMBO_TYPES + (NUM_TYPES + 1) + (NUM_TYPES + 1) + 1 \
             + (MAX_WILDCARDS + 1) + NUM_SUITS


def _move_to_rank_counts(m):
    """Approx decomposition into (15,) rank counts (suit-agnostic)."""
    combo, rank, pair_rank, count, n_wild, suit = m
    v = np.zeros(NUM_TYPES, dtype=np.float32)
    if combo == PASS:
        return v
    if combo == SINGLE:
        v[rank] = 1
    elif combo == PAIR:
        v[rank] = 2
    elif combo == TRIPLE:
        v[rank] = 3
    elif combo == FULLHOUSE:
        v[rank] = 3
        v[pair_rank] = 2
    elif combo == BOMB:
        if rank == JOKER_BOMB_RANK:
            v[SJ] = 2
            v[BJ] = 2
        else:
            v[rank] = count
    elif combo in (SEQ5, FLUSH_SEQ5):
        for k in range(SEQ5_LEN):
            v[rank + k] = 1
    elif combo == PSEQ3:
        for k in range(PSEQ3_LEN):
            v[rank + k] = 2
    elif combo == TSEQ2:
        for k in range(TSEQ2_LEN):
            v[rank + k] = 3
    return v


def encode_state(obs):
    me = obs['cur']
    hand = obs['hand']                      # (15, 4)
    feats = []

    # own hand: rank total + per-suit
    rt = rank_totals(hand).astype(np.float32)
    feats.append(rt / 8.0)
    feats.append(hand.astype(np.float32).reshape(-1) / 8.0)

    # last play
    last = obs['last']
    last_counts = np.zeros(NUM_TYPES, dtype=np.float32)
    last_combo = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)
    last_suit = np.zeros(NUM_SUITS, dtype=np.float32)
    if last is None:
        last_combo[PASS] = 1.0
    else:
        last_counts = _move_to_rank_counts(last) / 8.0
        last_combo[last[0]] = 1.0
        if last[0] == FLUSH_SEQ5:
            last_suit[last[5]] = 1.0
    feats.append(last_counts)
    feats.append(last_combo)
    feats.append(last_suit)

    # hand sizes (relative)
    teammate = (me + 2) % 4
    opp_l = (me + 1) % 4
    opp_r = (me + 3) % 4
    sizes = obs['hand_sizes'].astype(np.float32) / 27.0
    feats.append(np.array([sizes[teammate], sizes[opp_l], sizes[opp_r]], dtype=np.float32))

    # played piles (per seat, rank-sum only)
    feats.append(obs['played'][me].astype(np.float32) / 8.0)
    feats.append(obs['played'][teammate].astype(np.float32) / 8.0)
    feats.append(obs['played'][opp_l].astype(np.float32) / 8.0)
    feats.append(obs['played'][opp_r].astype(np.float32) / 8.0)

    # level_rank onehot
    level_v = np.zeros(NUM_TYPES, dtype=np.float32)
    lr = int(obs['level_rank'])
    if 0 <= lr < NUM_TYPES:
        level_v[lr] = 1.0
    feats.append(level_v)

    feats.append(np.array([obs['wildcards'] / max(MAX_WILDCARDS, 1)], dtype=np.float32))
    feats.append(np.array([obs['wildcards_left'] / max(MAX_WILDCARDS, 1)], dtype=np.float32))

    pw = obs['played_wild']
    feats.append(np.array([pw[me] / max(MAX_WILDCARDS, 1),
                           pw[teammate] / max(MAX_WILDCARDS, 1),
                           pw[opp_l] / max(MAX_WILDCARDS, 1),
                           pw[opp_r] / max(MAX_WILDCARDS, 1)], dtype=np.float32))

    return np.concatenate(feats)


def encode_action(m):
    combo, rank, pair_rank, count, n_wild, suit = m
    feats = []

    combo_v = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)
    combo_v[combo] = 1.0
    feats.append(combo_v)

    rank_v = np.zeros(NUM_TYPES + 1, dtype=np.float32)
    rank_v[min(rank, NUM_TYPES)] = 1.0
    feats.append(rank_v)

    pair_v = np.zeros(NUM_TYPES + 1, dtype=np.float32)
    pair_v[min(pair_rank, NUM_TYPES)] = 1.0
    feats.append(pair_v)

    feats.append(np.array([count / 8.0], dtype=np.float32))

    nw_v = np.zeros(MAX_WILDCARDS + 1, dtype=np.float32)
    nw_v[min(n_wild, MAX_WILDCARDS)] = 1.0
    feats.append(nw_v)

    suit_v = np.zeros(NUM_SUITS, dtype=np.float32)
    suit_v[min(suit, NUM_SUITS - 1)] = 1.0
    feats.append(suit_v)

    return np.concatenate(feats)
