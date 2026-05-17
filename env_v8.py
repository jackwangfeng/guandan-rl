"""Guandan v8: suit-aware env with red-heart-specific wildcards and 同花顺 combo.

Key differences from v6/v7:
  - Hand is (NUM_TYPES=15, NUM_SUITS=4) — tracks rank × suit
  - Wildcards = cards with (rank == level_rank, suit == HEART); exactly 2 per deal
  - New combo: FLUSH_SEQ5 (5 consecutive ranks same suit), treated as bomb-tier
  - Bomb hierarchy: 王炸 > 6+张同点炸 > 同花顺 > 5张同点炸 > 4张同点炸
  - Move tuple extended with `suit` field (meaningful only for flush combos)

Move = (combo, rank, pair_rank, count, n_wild, suit)
"""
from __future__ import annotations
import random
from typing import Optional
import numpy as np

NUM_TYPES = 15
NUM_SUITS = 4
SPADE, HEART, DIAMOND, CLUB = 0, 1, 2, 3
SUIT_NAMES = ['♠', '♥', '♦', '♣']
RANK_NAMES = ['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', '2', 'sj', 'bj']
SJ, BJ = 13, 14
JOKER_BOMB_RANK = 15  # sentinel for joker bomb

# Build deck: for non-joker ranks, 2 copies × 4 suits each. For jokers, 2 of each (suit=0 by convention).
DECK = np.zeros((NUM_TYPES, NUM_SUITS), dtype=np.int8)
for r in range(13):
    for s in range(NUM_SUITS):
        DECK[r, s] = 2
DECK[SJ, 0] = 2
DECK[BJ, 0] = 2
assert int(DECK.sum()) == 108

PASS, SINGLE, PAIR, TRIPLE, FULLHOUSE, BOMB, SEQ5, PSEQ3, TSEQ2, FLUSH_SEQ5 = \
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9
NUM_COMBO_TYPES = 10
COMBO_NAMES = {
    0: 'pass', 1: 'single', 2: 'pair', 3: 'triple', 4: 'three+two', 5: 'bomb',
    6: 'seq5', 7: 'pair_seq3', 8: 'triple_seq2', 9: 'flush_seq5',
}

SEQ5_LEN = 5
PSEQ3_LEN = 3
TSEQ2_LEN = 2
SEQ_HIGH_RANK = 11  # 'A'; ranks 12 ('2') and jokers can't form sequences

MAX_WILDCARDS = 2

PASS_MOVE = (PASS, 0, 0, 0, 0, 0)


def deal(rng: random.Random, level_rank: int):
    """Deal 27 cards to each player. Tracks (rank, suit) → count per player.

    Returns (hands, wildcards):
      hands: list of 4 (NUM_TYPES, NUM_SUITS) int8 arrays
      wildcards: list of 4 ints (count of red-heart level-rank in player's hand)
    """
    # Token list: each card = (rank, suit). Then shuffle and deal.
    cards: list = []
    for r in range(NUM_TYPES):
        for s in range(NUM_SUITS):
            for _ in range(int(DECK[r, s])):
                cards.append((r, s))
    rng.shuffle(cards)

    hands = [np.zeros((NUM_TYPES, NUM_SUITS), dtype=np.int8) for _ in range(4)]
    wildcards = [0, 0, 0, 0]
    for j, (r, s) in enumerate(cards):
        p = j % 4
        hands[p][r, s] += 1
        if r == level_rank and s == HEART:
            wildcards[p] += 1
    return hands, wildcards


def rank_totals(hand: np.ndarray) -> np.ndarray:
    """Sum (NUM_TYPES, NUM_SUITS) hand over suits → (NUM_TYPES,)."""
    return hand.sum(axis=1).astype(np.int32)


def hand_str(hand: np.ndarray, w: int = 0, level_rank: int = -1) -> str:
    rt = rank_totals(hand)
    parts = [f"{RANK_NAMES[i]}x{c}" for i, c in enumerate(rt) if c]
    s = " ".join(parts) if parts else "(empty)"
    if w > 0 and 0 <= level_rank < NUM_TYPES:
        s += f"  [♥{w} of {RANK_NAMES[level_rank]}]"
    return s


def move_str(m, level_rank=-1):
    combo, rank, pair_rank, count, n_wild, suit = m
    if combo == PASS:
        return "PASS"
    if combo == BOMB and rank == JOKER_BOMB_RANK:
        return "BOMB(JOKER)"
    tag = f"+{n_wild}w" if n_wild > 0 else ""
    if combo == FULLHOUSE:
        return f"3+2({RANK_NAMES[rank]}x3+{RANK_NAMES[pair_rank]}x2){tag}"
    if combo == BOMB:
        return f"BOMB({RANK_NAMES[rank]}x{count}){tag}"
    if combo == FLUSH_SEQ5:
        return f"FLUSH({SUIT_NAMES[suit]}{RANK_NAMES[rank]}-{RANK_NAMES[rank + SEQ5_LEN - 1]})"
    if combo == SEQ5:
        return f"SEQ5({RANK_NAMES[rank]}-{RANK_NAMES[rank + SEQ5_LEN - 1]})"
    if combo == PSEQ3:
        return f"PSEQ3({RANK_NAMES[rank]}-{RANK_NAMES[rank + PSEQ3_LEN - 1]})"
    if combo == TSEQ2:
        return f"TSEQ2({RANK_NAMES[rank]}-{RANK_NAMES[rank + TSEQ2_LEN - 1]})"
    return f"{COMBO_NAMES[combo].upper()}({RANK_NAMES[rank]}x{count}){tag}"


# ---------------------------------------------------------------------------
# Beat hierarchy for "bomb-tier" plays (BOMB and FLUSH_SEQ5).
# Returns an integer tier; bigger tier beats smaller. Ties broken by rank.
# Tier scheme:
#   王炸:                       tier 1000
#   6+ 张同点炸:                 tier 600 + count
#   同花顺:                       tier 500 (+rank tiebreak)
#   5 张同点炸:                   tier 400
#   4 张同点炸:                   tier 300
# Same tier: bigger rank beats; for flush vs flush, bigger start rank beats.
# ---------------------------------------------------------------------------
def bomb_tier(combo, rank, count):
    if combo == BOMB and rank == JOKER_BOMB_RANK:
        return 1000
    if combo == BOMB:
        if count >= 6:
            return 600 + count
        if count == 5:
            return 400
        if count == 4:
            return 300
    if combo == FLUSH_SEQ5:
        return 500
    return -1  # not bomb-tier


def beats_bomb_tier(my, last):
    """my and last are bomb-tier moves. Return True if `my` beats `last`."""
    t_my = bomb_tier(my[0], my[1], my[3])
    t_last = bomb_tier(last[0], last[1], last[3])
    if t_my != t_last:
        return t_my > t_last
    # same tier — break by rank
    return my[1] > last[1]


def _can_take(rank_tot, wildcards, level_rank, rank, count_of_rank, n_wild):
    if rank == level_rank:
        return n_wild == 0 and rank_tot[level_rank] >= count_of_rank
    # Wildcards cannot substitute for jokers
    if rank in (SJ, BJ) and n_wild > 0:
        return False
    real_needed = count_of_rank - n_wild
    if real_needed < 1:
        return False
    if rank_tot[rank] < real_needed:
        return False
    if wildcards < n_wild:
        return False
    return True


def _add_singles(rank_tot, wildcards, level_rank, last, moves):
    free = last is None or last[0] == PASS
    last_combo = -1 if free else last[0]
    last_rank = -1 if free else last[1]
    if not (free or last_combo == SINGLE):
        return
    for r in range(NUM_TYPES):
        if rank_tot[r] >= 1 and (free or r > last_rank):
            moves.append((SINGLE, r, 0, 1, 0, 0))


def _add_pairs(rank_tot, wildcards, level_rank, last, moves):
    free = last is None or last[0] == PASS
    last_combo = -1 if free else last[0]
    last_rank = -1 if free else last[1]
    if not (free or last_combo == PAIR):
        return
    for r in range(NUM_TYPES):
        if not (free or r > last_rank):
            continue
        if r == level_rank:
            if rank_tot[r] >= 2:
                moves.append((PAIR, r, 0, 2, 0, 0))
        else:
            for nw in range(0, min(wildcards, 1) + 1):
                if _can_take(rank_tot, wildcards, level_rank, r, 2, nw):
                    moves.append((PAIR, r, 0, 2, nw, 0))


def _add_triples(rank_tot, wildcards, level_rank, last, moves):
    free = last is None or last[0] == PASS
    last_combo = -1 if free else last[0]
    last_rank = -1 if free else last[1]
    if not (free or last_combo == TRIPLE):
        return
    for r in range(13):
        if not (free or r > last_rank):
            continue
        if r == level_rank:
            if rank_tot[r] >= 3:
                moves.append((TRIPLE, r, 0, 3, 0, 0))
        else:
            for nw in range(0, min(wildcards, 2) + 1):
                if _can_take(rank_tot, wildcards, level_rank, r, 3, nw):
                    moves.append((TRIPLE, r, 0, 3, nw, 0))


def _fh_split(rank_tot, wildcards, level_rank, tr, pr, total_nw):
    H_L = int(rank_tot[level_rank])
    W = int(wildcards)
    regular_L = H_L - W

    if tr == level_rank:
        if H_L < 3:
            return None
        nw_t_options = [0]
    else:
        nw_t_min = max(0, 3 - int(rank_tot[tr]))
        nw_t_max = min(2, total_nw)
        if nw_t_min > nw_t_max:
            return None
        nw_t_options = list(range(nw_t_max, nw_t_min - 1, -1))

    for nw_t in nw_t_options:
        nw_p = total_nw - nw_t
        if nw_p < 0:
            continue
        if tr == level_rank:
            wild_t = max(0, 3 - regular_L)
            L_used_t = 3
        else:
            wild_t = nw_t
            L_used_t = nw_t
        if pr == level_rank:
            if nw_p != 0:
                continue
            remaining_L = H_L - L_used_t
            if remaining_L < 2:
                continue
            remaining_wild = W - wild_t
            remaining_regular_L = remaining_L - remaining_wild
            wild_p = max(0, 2 - remaining_regular_L)
            L_used_p = 2
        else:
            if nw_p > 1:
                continue
            if pr in (SJ, BJ) and nw_p > 0:
                continue  # wildcards can't substitute for jokers
            real_p = 2 - nw_p
            if int(rank_tot[pr]) < real_p:
                continue
            wild_p = nw_p
            L_used_p = nw_p
        if wild_t + wild_p > W:
            continue
        if L_used_t + L_used_p > H_L:
            continue
        return (nw_t, nw_p)
    return None


def _add_fullhouse(rank_tot, wildcards, level_rank, last, moves):
    free = last is None or last[0] == PASS
    last_combo = -1 if free else last[0]
    last_rank = -1 if free else last[1]
    if not (free or last_combo == FULLHOUSE):
        return
    for tr in range(13):
        if not (free or tr > last_rank):
            continue
        for pr in range(NUM_TYPES):
            if pr == tr:
                continue
            for total_nw in range(0, min(wildcards, 3) + 1):
                if _fh_split(rank_tot, wildcards, level_rank, tr, pr, total_nw) is not None:
                    moves.append((FULLHOUSE, tr, pr, 5, total_nw, 0))


def _add_sequences(rank_tot, last, moves):
    """Normal (non-flush) sequences; no wildcards (v8 first version)."""
    free = last is None or last[0] == PASS
    last_combo = -1 if free else last[0]
    last_rank = -1 if free else last[1]

    if free or last_combo == SEQ5:
        for s in range(0, SEQ_HIGH_RANK - SEQ5_LEN + 2):
            if not (free or s > last_rank):
                continue
            if all(rank_tot[s + k] >= 1 for k in range(SEQ5_LEN)):
                moves.append((SEQ5, s, 0, SEQ5_LEN, 0, 0))

    if free or last_combo == PSEQ3:
        for s in range(0, SEQ_HIGH_RANK - PSEQ3_LEN + 2):
            if not (free or s > last_rank):
                continue
            if all(rank_tot[s + k] >= 2 for k in range(PSEQ3_LEN)):
                moves.append((PSEQ3, s, 0, PSEQ3_LEN * 2, 0, 0))

    if free or last_combo == TSEQ2:
        for s in range(0, SEQ_HIGH_RANK - TSEQ2_LEN + 2):
            if not (free or s > last_rank):
                continue
            if all(rank_tot[s + k] >= 3 for k in range(TSEQ2_LEN)):
                moves.append((TSEQ2, s, 0, TSEQ2_LEN * 3, 0, 0))


def _add_flush_seq5(hand, last, moves):
    """同花顺: 5 consecutive ranks all same suit. No wildcards in v8.
    Plays under FLUSH_SEQ5 (bomb-tier — interrupts any non-bomb-tier combo).
    """
    free = last is None or last[0] == PASS
    last_combo = -1 if free else last[0]
    last_rank = -1 if free else last[1]
    last_is_bomb_tier = (not free) and last_combo in (BOMB, FLUSH_SEQ5)

    for s in range(0, SEQ_HIGH_RANK - SEQ5_LEN + 2):  # start 0..7
        for suit in range(NUM_SUITS):
            ok = all(int(hand[s + k, suit]) >= 1 for k in range(SEQ5_LEN))
            if not ok:
                continue
            move = (FLUSH_SEQ5, s, 0, SEQ5_LEN, 0, suit)
            if last_is_bomb_tier:
                if not beats_bomb_tier(move, last):
                    continue
            elif last_combo == FLUSH_SEQ5:
                # responding to same combo, can also use this branch but
                # last_is_bomb_tier already handled above
                continue
            moves.append(move)


def _add_bombs(rank_tot, wildcards, level_rank, last, moves):
    free = last is None or last[0] == PASS
    last_combo = -1 if free else last[0]
    last_is_bomb_tier = (not free) and last_combo in (BOMB, FLUSH_SEQ5)

    have_joker_bomb = rank_tot[SJ] >= 2 and rank_tot[BJ] >= 2

    for r in range(13):
        if r == level_rank:
            c_options = [(int(rank_tot[r]), 0)] if rank_tot[r] >= 4 else []
        else:
            real = int(rank_tot[r])
            c_options = []
            for nw in range(0, wildcards + 1):
                c = real + nw
                if c >= 4 and (nw == 0 or real >= 1):
                    c_options.append((c, nw))
        for c, nw in c_options:
            move = (BOMB, r, 0, c, nw, 0)
            if last_is_bomb_tier:
                if not beats_bomb_tier(move, last):
                    continue
            moves.append(move)

    if have_joker_bomb:
        move = (BOMB, JOKER_BOMB_RANK, 0, 4, 0, 0)
        if not (last_is_bomb_tier and last_combo == BOMB and last[1] == JOKER_BOMB_RANK):
            # joker bomb tops everything; only blocked by another joker bomb (impossible)
            moves.append(move)


def legal_moves(hand: np.ndarray, wildcards: int, level_rank: int,
                last: Optional[tuple]) -> list:
    """All legal moves given hand (15×4), wildcards count, level_rank, last play."""
    rank_tot = rank_totals(hand)
    moves: list = []
    free = last is None or last[0] == PASS
    if not free:
        moves.append(PASS_MOVE)

    _add_singles(rank_tot, wildcards, level_rank, last, moves)
    _add_pairs(rank_tot, wildcards, level_rank, last, moves)
    _add_triples(rank_tot, wildcards, level_rank, last, moves)
    _add_fullhouse(rank_tot, wildcards, level_rank, last, moves)
    _add_sequences(rank_tot, last, moves)
    _add_flush_seq5(hand, last, moves)
    _add_bombs(rank_tot, wildcards, level_rank, last, moves)
    return moves


# ---------------------------------------------------------------------------
# Card consumption: deduct cards from a (15,4) hand respecting suit choices.
# Convention: when consuming non-level-rank cards, prefer non-heart suits.
# Wildcards (rank==level_rank, suit==HEART) are consumed only when necessary
# (for level-rank combos) or explicitly (n_wild > 0 substitution).
# ---------------------------------------------------------------------------
def _consume_rank(hand: np.ndarray, rank: int, count: int, level_rank: int,
                  forbid_heart_at_level: bool = True) -> int:
    """Subtract `count` cards of `rank` from hand. Returns # wildcards consumed.

    forbid_heart_at_level: when True (default), do NOT consume heart cards if
    rank == level_rank UNLESS no other suits suffice (preserves wildcards).
    """
    wild_consumed = 0
    if rank == level_rank and forbid_heart_at_level:
        order = [SPADE, DIAMOND, CLUB, HEART]
    elif rank in (SJ, BJ):
        order = [0]  # jokers use suit slot 0
    else:
        order = [SPADE, DIAMOND, CLUB, HEART]
    for s in order:
        if count == 0:
            break
        take = min(int(hand[rank, s]), count)
        if take > 0:
            hand[rank, s] -= take
            count -= take
            if rank == level_rank and s == HEART:
                wild_consumed += take
    if count > 0:
        raise ValueError(f"can't take {count} more of rank {rank} from hand")
    return wild_consumed


def _consume_wild(hand: np.ndarray, level_rank: int, n: int):
    """Consume exactly n wildcards (level_rank × HEART)."""
    if hand[level_rank, HEART] < n:
        raise ValueError(f"not enough wildcards: have {hand[level_rank, HEART]}, need {n}")
    hand[level_rank, HEART] -= n


def _apply_move(hand: np.ndarray, wildcards: int, level_rank: int,
                move: tuple) -> tuple:
    """Return (new_hand, new_wildcards) after applying move. Raises on invalid."""
    combo, rank, pair_rank, count, n_wild, suit = move
    new_hand = hand.copy()
    new_wild = int(wildcards)

    if combo == PASS:
        return new_hand, new_wild

    if combo == FLUSH_SEQ5:
        # No explicit substitution. But if suit==HEART and rank+k passes through
        # level_rank, the wildcard at that slot is physically consumed.
        for k in range(SEQ5_LEN):
            r = rank + k
            if new_hand[r, suit] < 1:
                raise ValueError(f"flush seq5 missing rank {r} suit {suit}")
            new_hand[r, suit] -= 1
            if r == level_rank and suit == HEART:
                new_wild -= 1
        return new_hand, new_wild

    if combo == BOMB and rank == JOKER_BOMB_RANK:
        new_hand[SJ, 0] -= 2
        new_hand[BJ, 0] -= 2
        return new_hand, new_wild

    if combo == FULLHOUSE:
        split = _fh_split(rank_totals(hand), wildcards, level_rank,
                          rank, pair_rank, n_wild)
        if split is None:
            raise ValueError(f"infeasible fullhouse: {move}")
        nw_t, nw_p = split
        # Substitution wildcards: nw_t (only if tr != level_rank), nw_p (only if pr != level_rank)
        sub_wild = (nw_t if rank != level_rank else 0) + (nw_p if pair_rank != level_rank else 0)
        if sub_wild > 0:
            _consume_wild(new_hand, level_rank, sub_wild)
            new_wild -= sub_wild
        # Triple part
        if rank == level_rank:
            wc = _consume_rank(new_hand, rank, 3, level_rank)
            new_wild -= wc
        else:
            real_t = 3 - nw_t
            if real_t > 0:
                wc = _consume_rank(new_hand, rank, real_t, level_rank)
                new_wild -= wc
        # Pair part
        if pair_rank == level_rank:
            wc = _consume_rank(new_hand, pair_rank, 2, level_rank)
            new_wild -= wc
        else:
            real_p = 2 - nw_p
            if real_p > 0:
                wc = _consume_rank(new_hand, pair_rank, real_p, level_rank)
                new_wild -= wc
    else:
        # Single-rank combos (SINGLE, PAIR, TRIPLE, BOMB) and sequences.
        # For these, n_wild > 0 requires rank != level_rank.
        if n_wild > 0:
            if rank == level_rank:
                raise ValueError(f"n_wild>0 not allowed when rank==level_rank, move={move}")
            _consume_wild(new_hand, level_rank, n_wild)
            new_wild -= n_wild

        if combo == SINGLE:
            wc = _consume_rank(new_hand, rank, 1 - n_wild, level_rank)
            new_wild -= wc
        elif combo == PAIR:
            wc = _consume_rank(new_hand, rank, 2 - n_wild, level_rank)
            new_wild -= wc
        elif combo == TRIPLE:
            wc = _consume_rank(new_hand, rank, 3 - n_wild, level_rank)
            new_wild -= wc
        elif combo == BOMB:
            wc = _consume_rank(new_hand, rank, count - n_wild, level_rank)
            new_wild -= wc
        elif combo == SEQ5:
            for k in range(SEQ5_LEN):
                wc = _consume_rank(new_hand, rank + k, 1, level_rank)
                new_wild -= wc
        elif combo == PSEQ3:
            for k in range(PSEQ3_LEN):
                wc = _consume_rank(new_hand, rank + k, 2, level_rank)
                new_wild -= wc
        elif combo == TSEQ2:
            for k in range(TSEQ2_LEN):
                wc = _consume_rank(new_hand, rank + k, 3, level_rank)
                new_wild -= wc
        else:
            raise ValueError(f"unknown combo {combo}")

    if (new_hand < 0).any() or new_wild < 0:
        raise ValueError(f"negative after apply: hand={new_hand}, wild={new_wild}")
    return new_hand, new_wild


class GuandanEnvV8:
    """v8 round/match-capable env. full_round=True keeps playing past first finisher."""

    def __init__(self, seed: Optional[int] = None, level_rank: Optional[int] = None,
                 full_round: bool = False):
        self.rng = random.Random(seed)
        self._init_level_rank = level_rank
        self.full_round = full_round
        self.reset()

    def reset(self, seed: Optional[int] = None, level_rank: Optional[int] = None,
              hands: Optional[list] = None, wildcards: Optional[list] = None,
              starter: Optional[int] = None):
        if seed is not None:
            self.rng = random.Random(seed)
        if level_rank is None:
            level_rank = self._init_level_rank
        if level_rank is None:
            level_rank = self.rng.randrange(13)
        self.level_rank = int(level_rank)
        if hands is not None and wildcards is not None:
            self.hands = [h.copy() for h in hands]
            self.wildcards = list(wildcards)
        else:
            self.hands, self.wildcards = deal(self.rng, self.level_rank)
        self.played = [np.zeros(NUM_TYPES, dtype=np.int32) for _ in range(4)]
        self.played_wild = [0, 0, 0, 0]
        self.cur = self.rng.randrange(4) if starter is None else int(starter)
        self.last_play: Optional[tuple] = None
        self.last_player: Optional[int] = None
        self.passes_in_a_row = 0
        self.done = False
        self.winner_team: Optional[int] = None
        self.finish_order: list = []
        self.steps = 0
        return self.obs()

    def obs(self):
        return {
            'cur': self.cur,
            'hand': self.hands[self.cur].copy(),
            'wildcards': int(self.wildcards[self.cur]),
            'level_rank': self.level_rank,
            'last': self.last_play,
            'last_player': self.last_player,
            'hand_sizes': np.array([int(h.sum()) for h in self.hands], dtype=np.int32),
            'wildcards_left': MAX_WILDCARDS - sum(self.played_wild),
            'played': [p.copy() for p in self.played],
            'played_wild': list(self.played_wild),
        }

    def legal(self):
        return legal_moves(self.hands[self.cur], int(self.wildcards[self.cur]),
                           self.level_rank, self.last_play)

    def _next_active(self, start: int) -> int:
        if not self.full_round:
            return start
        c = start
        for _ in range(4):
            if int(self.hands[c].sum()) > 0:
                return c
            c = (c + 1) % 4
        return start

    def step(self, move):
        assert not self.done
        self.steps += 1
        combo = move[0]

        if combo == PASS:
            self.passes_in_a_row += 1
            if self.passes_in_a_row >= 3:
                self.last_play = None
                self.passes_in_a_row = 0
                if (self.last_player is not None
                        and int(self.hands[self.last_player].sum()) > 0):
                    self.cur = self.last_player
                else:
                    self.cur = self._next_active((self.cur + 1) % 4)
                return self.obs(), 0.0, False, {}
            self.cur = self._next_active((self.cur + 1) % 4)
            return self.obs(), 0.0, False, {}

        old_hand = self.hands[self.cur]
        old_wild = self.wildcards[self.cur]
        new_hand, new_wild = _apply_move(old_hand, old_wild, self.level_rank, move)
        diff_rank = (rank_totals(old_hand) - rank_totals(new_hand))
        self.played[self.cur] += diff_rank
        self.played_wild[self.cur] += (old_wild - new_wild)
        self.hands[self.cur] = new_hand
        self.wildcards[self.cur] = new_wild
        self.last_play = move
        self.last_player = self.cur
        self.passes_in_a_row = 0

        if int(self.hands[self.cur].sum()) == 0:
            self.finish_order.append(self.cur)
            if not self.full_round:
                self.done = True
                self.winner_team = self.cur % 2
                self.cur = (self.cur + 1) % 4
                return self.obs(), 0.0, True, {'winner_team': self.winner_team,
                                                'finish_order': list(self.finish_order)}
            if len(self.finish_order) >= 3:
                remaining = [p for p in range(4) if p not in self.finish_order]
                if remaining:
                    self.finish_order.append(remaining[0])
                self.done = True
                self.winner_team = self.finish_order[0] % 2
                self.cur = self._next_active((self.cur + 1) % 4)
                return self.obs(), 0.0, True, {'winner_team': self.winner_team,
                                                'finish_order': list(self.finish_order)}

        self.cur = self._next_active((self.cur + 1) % 4)
        return self.obs(), 0.0, False, {}
