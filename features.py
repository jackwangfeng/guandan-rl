"""Feature encoding for Guandan v6 (with level/wildcard rules)."""
from __future__ import annotations
import numpy as np
from env import (
    NUM_TYPES, NUM_COMBO_TYPES, SJ, BJ, JOKER_BOMB_RANK, MAX_WILDCARDS,
    PASS, SINGLE, PAIR, TRIPLE, FULLHOUSE, BOMB,
    SEQ5, PSEQ3, TSEQ2, SEQ5_LEN, PSEQ3_LEN, TSEQ2_LEN,
)

# State features:
#   own hand (15)
#   last play counts (15)
#   last combo onehot (9)
#   teammate + 2 opp hand sizes (3, normalised)
#   own + teammate + 2 opp played piles (60)
#   level_rank onehot (15)
#   own wildcards count (1, /2)
#   wildcards still floating (1, /2)
#   played wildcards per seat (4, /2)
# = 15+15+9+3+60+15+1+1+4 = 123
STATE_DIM = NUM_TYPES + NUM_TYPES + NUM_COMBO_TYPES + 3 + 4 * NUM_TYPES \
            + NUM_TYPES + 1 + 1 + 4

# Action features:
#   combo onehot (9)
#   rank onehot (16, +1 for joker bomb sentinel)
#   pair_rank onehot (16)
#   count scalar (1, /8)
#   n_wild onehot (3 — values 0/1/2)
# = 9+16+16+1+3 = 45
ACTION_DIM = NUM_COMBO_TYPES + (NUM_TYPES + 1) + (NUM_TYPES + 1) + 1 + (MAX_WILDCARDS + 1)


def _move_to_counts(m):
    """Decompose a move into a (15,) count vector of cards consumed.
    Approximation only — wildcards used by substitution are reported as the
    substituted rank, not as level_rank (which is what a "perfect" decomposition
    would attribute them to). This is fine for features.
    """
    combo, rank, pair_rank, count, _n_wild = m
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
    elif combo == SEQ5:
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
    feats = []

    # own hand
    feats.append(obs['hand'].astype(np.float32) / 8.0)

    # last play decomposition + combo onehot
    last_counts = np.zeros(NUM_TYPES, dtype=np.float32)
    last_combo = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)
    last = obs['last']
    if last is None:
        last_combo[PASS] = 1.0
    else:
        last_counts = _move_to_counts(last) / 8.0
        last_combo[last[0]] = 1.0
    feats.append(last_counts)
    feats.append(last_combo)

    # opponent + teammate hand sizes (relative: teammate, opp_l, opp_r)
    teammate = (me + 2) % 4
    opp_l = (me + 1) % 4
    opp_r = (me + 3) % 4
    sizes = obs['hand_sizes'].astype(np.float32) / 27.0
    feats.append(np.array([sizes[teammate], sizes[opp_l], sizes[opp_r]], dtype=np.float32))

    # played piles (own, teammate, opp_l, opp_r) in canonical order
    feats.append(obs['played'][me].astype(np.float32) / 8.0)
    feats.append(obs['played'][teammate].astype(np.float32) / 8.0)
    feats.append(obs['played'][opp_l].astype(np.float32) / 8.0)
    feats.append(obs['played'][opp_r].astype(np.float32) / 8.0)

    # level_rank onehot (15-dim; ranks 0..12 are valid, jokers won't be level)
    level_v = np.zeros(NUM_TYPES, dtype=np.float32)
    lr = int(obs['level_rank'])
    if 0 <= lr < NUM_TYPES:
        level_v[lr] = 1.0
    feats.append(level_v)

    # own wildcards & remaining wildcards in play
    feats.append(np.array([obs['wildcards'] / max(MAX_WILDCARDS, 1)], dtype=np.float32))
    feats.append(np.array([obs['wildcards_left'] / max(MAX_WILDCARDS, 1)], dtype=np.float32))

    # played wildcards per seat (relative)
    pw = obs['played_wild']
    feats.append(np.array([pw[me] / max(MAX_WILDCARDS, 1),
                           pw[teammate] / max(MAX_WILDCARDS, 1),
                           pw[opp_l] / max(MAX_WILDCARDS, 1),
                           pw[opp_r] / max(MAX_WILDCARDS, 1)], dtype=np.float32))

    return np.concatenate(feats)


def encode_action(m):
    combo, rank, pair_rank, count, n_wild = m
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

    return np.concatenate(feats)
